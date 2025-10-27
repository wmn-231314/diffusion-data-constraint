#!/bin/bash
export MASTER_PORT=$((RANDOM%16384+49152))  # 49152-65535 random
ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
BUILD_NAME="build_${ARCH}"
export BUILD_NAME

VARIANT=mdm_best_on_100m
echo "Running variant: $VARIANT"

LOAD_TAG=latest

KILL_SWITCH_NAME=kill-switch-$VARIANT
CHECKPOINT_NAME=checkpoints_$VARIANT
TENSORBOARD_NAME=tensorboard_$VARIANT
WANDB_NAME=wandb_$VARIANT
WANDB_GROUP=$VARIANT
WANDB_PROJECT="diffusion_data_constraint"
WANDB_EXP_NAME=$VARIANT

CHECKPOINT_PATH=output/$CHECKPOINT_NAME
TENSORBOARD_PATH=output/$TENSORBOARD_NAME
KILL_SWITCH_PATH=output/$KILL_SWITCH_NAME
WANDB_PATH=output/$WANDB_NAME

VOCAB_FILE=utils/data/gpt2-vocab.json
MERGE_FILE=utils/data/gpt2-merges.txt
TRAIN_DATA_PATH=utils/datapaths/train_c4_100m.txt
VALID_DATA_PATH=utils/datapaths/val_c4.txt

PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256

# Model parameters
source utils/model_params.sh
MODEL_PARAM=("${PARAM_425M[@]}")
NHIDDEN=${MODEL_PARAM[0]}
FFN_HIDDEN_SIZE=${MODEL_PARAM[1]}
KV_SIZE=${MODEL_PARAM[2]}
NHEADS=${MODEL_PARAM[3]}
NLAYERS=${MODEL_PARAM[4]}
SEQ_LEN=2048

# Data Count
source utils/epoch_tokens.sh
DATA_CNT=${DATA_100M[@]}
EPOCH_CNT=500

echo "Model parameters: d_model $NHIDDEN ffw_size $FFN_HIDDEN_SIZE kv_size $KV_SIZE n_heads $NHEADS n_layers $NLAYERS"

SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
LOG_INTERVAL=10
EVAL_ITERS=10

# Tokens: 1516071000
# -> Samples: 740269
TRAIN_SAMPLES=$((DATA_CNT*EPOCH_CNT/SEQ_LEN))
WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))
echo "Training samples: $TRAIN_SAMPLES, Number of epochs: $EPOCH_CNT, Data count: $DATA_CNT"

VOCAB_SIZE=50257
NEW_FFN_HIDDEN_SIZE=$(python3 -c "print(int((4 * $NHIDDEN * 2 / 3) / 64) * 64)")
MODEL_PARAM_CNT=$(python3 -c "print(4 * $NLAYERS * ($NHIDDEN ** 2) + 3 * $NLAYERS * $NHIDDEN * $NEW_FFN_HIDDEN_SIZE + 6*$NLAYERS*$NHIDDEN + ($VOCAB_SIZE * $NHIDDEN))")

echo "Model parameters: $MODEL_PARAM_CNT, New ffn hidden size: $NEW_FFN_HIDDEN_SIZE"

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
    "

DATA_ARGS=" \
    --load $CHECKPOINT_PATH \
    --load-tag $LOAD_TAG \
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

TASKS="arc_easy,boolq,copa,hellaswag,piqa,race,winogrande,sciq"
MC_NUM=32
RESULT_PATH="results/${VARIANT}_${LOAD_TAG}_mc${MC_NUM}.json"

EVAL_ARGS=" \
    --inference \
    --no-load-optim \
    --no-load-rng \
    --eval_fp32 \
    --task_list $TASKS\
    --results_path $RESULT_PATH \
    --eval_method mc \
    --only_mc_nll \
    --num-mc $MC_NUM \
    "

CMD=" \
    tasks/eval_harness/evaluate_diff.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    $EVAL_ARGS \
    "

LAUNCHER="deepspeed --master_port $MASTER_PORT"

$LAUNCHER $CMD