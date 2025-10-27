# Downstream Evaluation Instructions

This folder contains downstream evaluation examples and configurations for evaluating models on standard NLP benchmarks. The downstream evaluation scripts demonstrate how to assess model performance on various language understanding tasks.

## Extra Installation

For downstream evaluation, you need to install additional dependencies:

```bash
pip install lm-eval==0.3.0
```

**Troubleshooting**: If you encounter issues related to the `datasets` library during runtime, try reinstalling it to version 4.2.0:

```bash
pip install datasets==4.2.0
```

## Dataset Preparation

Most datasets can be automatically downloaded through the `datasets` library. However, there is one exception:

### PIQA Dataset

The PIQA dataset's current format is not supported by the `datasets` library and cannot be directly loaded. You need to manually download it and place it in the HuggingFace cache directory or your designated data folder.

**Steps:**
1. Download the PIQA dataset from the official source
2. Place the downloaded files in one of the following locations:
   - HuggingFace cache directory (typically `~/.cache/huggingface/datasets/`)
   - Your custom data folder configured for the project

## Evaluation Settings

### Hardware Configuration

**Important**: Downstream evaluation currently does not support data parallelism. When not using pipeline parallelism (PP), tensor parallelism (TP) or other parallelism, please use a single GPU for evaluation.

### Diffusion Model Evaluation Modes

For diffusion models, we provide two distinct evaluation approaches:

#### 1. Left-to-Right Generation
Standard autoregressive-style generation from left to right.

#### 2. Random Generation (Confidence-Based)
Generation based on confidence ordering, which requires Monte Carlo sampling for robust evaluation (similar to the approach described in the evaluation documentation).

### Monte Carlo Sampling for Downstream Tasks

Similar to the main evaluation setup, Monte Carlo sampling can be enabled by setting `--num-mc > 1`. This provides more reliable estimates but increases computational requirements by a factor of `num-mc`.

**Implementation Details:**
- **Maximum Support**: For downstream evaluation, Monte Carlo sampling is supported up to infinite number of samples.
- **Research Configuration**: In our paper experiments, we utilized `num-mc = 32` for optimal balance between computational efficiency and statistical reliability.

### Efficient Evaluation Modes

To improve computational efficiency when using Monte Carlo sampling, we have implemented two specialized evaluation modes:

#### 1. Generation-Only Mode (`--only_generate`)
Designed for generation exact match tasks (lambda standard tasks):
- Focuses solely on generation without computing likelihood scores
- Significantly reduces computational overhead
- Ideal for tasks that only require exact match evaluation

#### 2. Monte Carlo NLL-Only Mode (`--only_mc_nll`)
Designed for rank-based tasks that require logits comparison:
- Computes only the negative log-likelihood scores using Monte Carlo sampling
- Bypasses generation step when not needed
- Optimal for multiple-choice tasks that rely on perplexity comparison

## Examples

We provide several example scripts demonstrating downstream evaluation for different model types corresponding to the quickstart training settings:

- **`downstream_arm.sh`**: Autoregressive model (ARM) evaluation on standard downstream tasks
- **`downstream_mdm.sh`**: MDM evaluation with left-to-right generation mode (`--eval_method ar`)
- **`downstream_mdm_mc.sh`**: MDM evaluation with Monte Carlo sampling for rank-based tasks (`--only_mc_nll`, `--eval_method mc`, `--num-mc 32`)
- **`downstream_mdm_onlygen.sh`**: MDM evaluation with generation-only mode for exact match tasks like `lambada_standard` (`--only_generate`)

These scripts correspond to the models trained in the quickstart examples and demonstrate the different evaluation strategies discussed above.

