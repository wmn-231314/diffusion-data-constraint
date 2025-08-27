# Training Instructions

This folder contains training examples and configurations for different experimental setups. The main training script `example_training.sh` demonstrates the standard training configuration, while various experimental configurations can be implemented by modifying specific parameters and adding additional training arguments.

## Detailed Explanation of the Training Script

In this section, we will provide detailed explanations of the main configuration components in the standard training script.

### Slurm Setup (Optional)
We provide a slurm setup script on top of the standard training script for reference. Use 'sbatch' instead of 'bash' to submit the job when using slurm.

```bash
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
```

### Data Paths

Filled in all the paths for checkpoints, logs, and execution files. For AR, we use the execution file 'pretrain_gpt.py', for MDM, we use the execution file 'pretrain_diff_gpt.py'. In addition, there are some other experimental configurations with different execution files, we will explain them in the following sections.

```bash
CHECKPOINT_PATH=/path/to/your/checkpoint/dir/$CHECKPOINT_NAME
TENSORBOARD_PATH=/path/to/your/tensorboard/dir/$TENSORBOARD_NAME
KILL_SWITCH_PATH=/path/to/your/kill-switch/dir/$KILL_SWITCH_NAME
WANDB_PATH=/path/to/your/wandb/dir/$WANDB_NAME

CMD=" \
    /path/to/your/project/Megatron-DeepSpeed/example_execution.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    "
```

### Optimizer

```bash
WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))

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
```

As mentioned in the paper Section 3.4, we use AdamW optimizer with cosine learning rate schedule. We set the learning rate to 2e-4 and the minimum learning rate to 2e-5. The warmup samples is set to 1% of the training samples. The clip grad is set to 1.0 and the weight decay is set to 1e-1. **Note: All other experiments in this work adopt these exact optimizer settings to control variables, except for our learning rate exploration experiments.**


### Model Parameters

```bash
VOCAB_FILE=utils/data/gpt2-vocab.json
MERGE_FILE=utils/data/gpt2-merges.txt

# Model parameters
source utils/model_params.sh
MODEL_PARAM=("${PARAM_425M[@]}") # use 425M as example
NHIDDEN=${MODEL_PARAM[0]}
FFN_HIDDEN_SIZE=${MODEL_PARAM[1]}
KV_SIZE=${MODEL_PARAM[2]}
NHEADS=${MODEL_PARAM[3]}
NLAYERS=${MODEL_PARAM[4]}
SEQ_LEN=2048

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
    $OPTIMIZER_ARGS \
    "
```

In Section 10 of our paper, we discuss the computational details of model parameters. In the example script, our model parameters are referenced from [datablations](https://github.com/huggingface/datablations), where we recalculate the FFN hidden layer size and use a new model parameter estimation function to estimate the updated model parameters. Therefore, the model parameters may not be exactly matched with their names. For vocabulary size, we adopt the GPT2Tokenizer size, and the code automatically aligns it to multiples of 128 for computational efficiency reasons.


### Data Count

```bash
TRAIN_DATA_PATH=utils/datapaths/train_c4_100m.txt
VALID_DATA_PATH=utils/datapaths/val_c4.txt

# Data Count
source utils/epoch_tokens.sh
DATA_CNT=${DATA_100M[@]} # use 100m unique token and 100 epoch as example
EPOCH_CNT=100

TRAIN_SAMPLES=$((DATA_CNT*EPOCH_CNT/SEQ_LEN)) # use k-epoch experiments as example

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-data-path-file $TRAIN_DATA_PATH \
    --valid-data-path-file $VALID_DATA_PATH \
    --data-impl mmap \
    "
```

After completing data preprocessing, we recommend placing the data in `utils/datapaths/` and referencing the examples provided therein. For experiments requiring multiple epochs, this script provides train_samples calculation based on multi-epoch training. Additionally, the current epoch tokens only contain the token count for the specific dataset subset required by the corresponding experiment. If you need to try other training configurations, you'll need to calculate the total tokens yourself. We recommend checking "Tokens per epoch" during the first run to obtain this information.


### Logger

```bash
SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
LOG_INTERVAL=10
EVAL_ITERS=100

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
```

By default, we simultaneously output logs to both Weights & Biases (wandb) and TensorBoard. We adopt the same interval for both save and evaluation operations. During training, we use 100 eval iters, which ensures we obtain results similar to those from the complete validation set while maintaining training efficiency.

### DeepSpeed Configuration

```bash
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
```
In this work, we use DeepSpeed to manage memory and optimize the training process. For model smaller than 1B, we use zero stage 0 as default, and for model larger than 1B, we use zero stage > 0 and cpu offloading to improve the training efficiency, but detailed settings depend on the gpus we use.

### Large Model Training

Based on the inherent features of the Megatron-DeepSpeed codebase, the code supports large model training and is equipped with **DP** (data parallelism), **TP** (tensor parallelism), **PP** (pipeline parallelism), as well as DeepSpeed's three zero stages and CPU offloading optimizations. It's important to note that during our experiments with models larger than 1B, we tried using zero stage > 0, and based on empirical results, zero stage > 0 combined with TP and PP leads to a certain degree of performance degradation.

In addition, for model parameters size larger than 4B, we suggested to change the global batch size to larger such as 512, etc. This is also mentioned in [datablations experiments](https://huggingface.co/datablations/lm1-misc), because larger models exhibit higher gradient variance and benefit from larger global batch sizes, which stabilize optimization especially under TP and PP configurations.

### Multi-Node Training

We provide a multi-node training script that modifies the slurm header for multi-node requests and changes the final execution script to use `launch.sh` as shown in `example_multi_node_training.sh`. Please note that this script is primarily designed for training larger models and therefore adopts zero stage 1 with CPU offloading.


## Different Experimental Configurations

### LR Exploration

We tested different combinations of beta2, learning rate, minimum learning rate, and warmup step proportions.

**Results:**
- AR models perform best with default settings
- Diffusion models show slight improvement with higher LR and warmup proportion

**Note:** We use identical optimization parameters for both AR and Diffusion to ensure fair comparison across experiments.

### Chinchilla Scaling Law
In this work, we fitted a "data-constrained" scaling law for both Autoregressive and Diffusion models, which is introduced in Section 3.3 of our paper. To get this equation, we first need to find the variables for the chinchilla scaling law, and based on our experiments, we found that the compute-optimal settings follow the chinchilla scaling law and the diffusion scaling relationship found in the paper "Scaling up Masked Diffusion Models on Text". The experiments are conducted on 4 different compute scales: 6e18, 1e19, 3e19, 1e20. To verify this result, we provide example scripts in [mdm/example_training_chinchilla.sh](./mdm/example_training_chinchilla.sh) and [arm/example_training_chinchilla.sh](./arm/example_training_chinchilla.sh). Since the chinchilla scaling law is a single epoch experiment, we directly use the full C4 dataset for training.

**NOTE**: In addition to the unfilled path, the only thing you need to change is the 'MODEL_PARAM' and the 'FLOPS_FACTOR' in the example script.


### Data Utilization Ability
In this work, we explore the data utilization ability of Diffusion and Autoregressive models. For each model architecture, we use the compute-optimal setting under compute 1e19, 3e19, and 1e20 to find the data value decay with epoch repetition. Based on the previous experiment, we found that the compute-optimal setting under 1e19 is 217M for AR and 117M for Diffusion; the compute-optimal setting under 3e19 is 425M for AR and 217M for Diffusion; the compute-optimal setting under 1e20 is 724M for AR and 425M for Diffusion. In addition, we conduct experiments on 1, 2, 4, 10, 20, 50, 100 epochs (or 100%, 50%, 25%, 10%, 5%, 2%, 1% unique percentage). To verify the result shown in the paper Figures 3, 4, and 5, we provide example scripts in [mdm/example_training_data_value.sh](./mdm/example_training_data_value.sh) and [arm/example_training_data_value.sh](./arm/example_training_data_value.sh) and the corresponding data subsets [here](https://huggingface.co/datasets/ZahlenReal/diffusion_data_constraint_c4subsets).

**NOTE**: The variables you need to change are the same as those in the chinchilla scaling law experiment.

**How to calculate the corresponding dataset size:**
- Given a fixed FLOPS, we can directly calculate the total unique token count for single epoch training using the following equation:
    - `DATA_CNT = FLOPS_FACTOR * BILLION / 6 / MODEL_SIZE`
- For multi-epoch training, we create subsets by `DATA_CNT_PER_EPOCH = DATA_CNT / EPOCH_CNT`, and all of them are named with a postfix of `_EPOCH_CNT`, e.g. `c4_7B6_2ep.json`.
- For example, if you want to try a data value experiment with 100 epochs in 1e19 compute:
    - For AR, we use model 217M (217.5M): `DATA_CNT = 1e19 / 6 / 217.5M = 7.6B`, so choose dataset `c4_7B6_100ep.json`
    - For Diffusion, we use model 117M (123.6M): `DATA_CNT = 1e19 / 6 / 123.6M = 13.5B`, so choose dataset `c4_13B5_100ep.json`
- For the exact model parameter of each model name, please refer to Section 10 of our paper.
- To create these datasets, we use the accurate model parameters to calculate an accurate sample count.
- Some datasets are too large to be directly uploaded, so we slice them into pieces, use `cat data_line_shards/c4_12B.part-*\.jsonl > c4_12B.jsonl` to concatenate them.


### Left-to-Right Masking on MDM

TODO

### Attention Dropout

TODO


### Token Masking

TODO


### Predefined Order Perturbation

TODO









