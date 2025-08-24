#!/bin/bash
#SBATCH --job-name=example_run_name
#SBATCH --partition=your_partition
#SBATCH --time=your_time
#SBATCH --gres=gpu:your_gpu_num
#SBATCH --constraint='your_gpu_type'
#SBATCH --cpus-per-task=your_cpu_num
#SBATCH --mem=your_mem
#SBATCH --mail-type=END,FAIL,RUNNING
#SBATCH --mail-user=your_email
#SBATCH --output=your_output_dir/output_report-%j.out
#SBATCH --requeue

export CUDA_VISIBLE_DEVICES=your_gpu_ids
export WANDB_API_KEY=your_wandb_api_key

export MASTER_PORT=$((RANDOM%16384+49152))  # 49152-65535 random
ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
BUILD_NAME="build_${ARCH}"
export BUILD_NAME

VARIANT=example_run_name
echo "Running variant: $VARIANT"

KILL_SWITCH_NAME=kill-switch-$VARIANT
CHECKPOINT_NAME=checkpoints_$VARIANT
TENSORBOARD_NAME=tensorboard_$VARIANT
WANDB_NAME=wandb_$VARIANT
WANDB_GROUP=$VARIANT
WANDB_PROJECT="example project name"
WANDB_EXP_NAME=$VARIANT

CHECKPOINT_PATH=/path/to/your/checkpoint/dir/$CHECKPOINT_NAME
TENSORBOARD_PATH=/path/to/your/tensorboard/dir/$TENSORBOARD_NAME
KILL_SWITCH_PATH=/path/to/your/kill-switch/dir/$KILL_SWITCH_NAME
WANDB_PATH=/path/to/your/wandb/dir/$WANDB_NAME

VOCAB_FILE=utils/data/gpt2-vocab.json
MERGE_FILE=utils/data/gpt2-merges.txt
TRAIN_DATA_PATH=utils/datapaths/example_trainset.txt
VALID_DATA_PATH=utils/datapaths/val_c4.txt

PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=micro_batch_size
GLOBAL_BATCH_SIZE=256

# Model parameters
source utils/model_params.sh
MODEL_PARAM=("${PARAM_model_size[@]}")
NHIDDEN=${MODEL_PARAM[0]}
FFN_HIDDEN_SIZE=${MODEL_PARAM[1]}
KV_SIZE=${MODEL_PARAM[2]}
NHEADS=${MODEL_PARAM[3]}
NLAYERS=${MODEL_PARAM[4]}
SEQ_LEN=2048

echo "Model parameters: d_model $NHIDDEN ffw_size $FFN_HIDDEN_SIZE kv_size $KV_SIZE n_heads $NHEADS n_layers $NLAYERS"

SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
LOG_INTERVAL=10
EVAL_ITERS=10

# Calculate training samples based on flops
FLOPS_FACTOR=10 # choose from 10(1e19), 30(3e19), 100(1e20)
BILLION=1000000000
FLOPS_BILLION=$((FLOPS_FACTOR*BILLION))
VOCAB_SIZE=50257
NEW_FFN_HIDDEN_SIZE=$(python3 -c "print(int((4 * $NHIDDEN * 2 / 3) / 64) * 64)")
MODEL_PARAM_CNT=$(python3 -c "print(4 * $NLAYERS * ($NHIDDEN ** 2) + 3 * $NLAYERS * $NHIDDEN * $NEW_FFN_HIDDEN_SIZE + 6*$NLAYERS*$NHIDDEN + ($VOCAB_SIZE * $NHIDDEN))")
DATA_CNT=$(python3 -c "print(int(($FLOPS_BILLION/6) * ($BILLION / $MODEL_PARAM_CNT)))")

echo "Training FLOPS(Billion): $FLOPS_BILLION, Model parameters: $MODEL_PARAM_CNT, Data count: $DATA_CNT"

# Tokens: 1516071000
# -> Samples: 740269, sample=token/seq_len
TRAIN_SAMPLES=$(python3 -c "print(round($DATA_CNT / $SEQ_LEN))")
WARMUP_SAMPLES=$(python3 -c "print(max(round($TRAIN_SAMPLES / 100), 100))")

echo "Training samples: $TRAIN_SAMPLES, warmup samples: $WARMUP_SAMPLES"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 2e-4 \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $TRAIN_SAMPLES \
    --lr-warmup-samples $WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --kv-channels $KV_SIZE \
    --seq-length $SEQ_LEN \
    --use-rotary-position-embeddings \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --disable-bias-linear \
    --kill-switch-file $KILL_SWITCH_PATH \
    --normalization rmsnorm \
    --swiglu \
    --bf16 \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_PATH \
    --wandb-group $WANDB_GROUP \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-data-path-file $TRAIN_DATA_PATH \
    --valid-data-path-file $VALID_DATA_PATH \
    --data-impl mmap \
    "

ZERO_STAGE=0
mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$VARIANT.json"

cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "bf16": {
        "enabled": true
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE
    "

CMD=" \
    pretrain_diff_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    "

LAUNCHER="deepspeed --master_port $MASTER_PORT"

$LAUNCHER $CMD