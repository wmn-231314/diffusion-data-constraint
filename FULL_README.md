## Installation

This project uses micromamba for environment management and requires ninja for C++ compilation. The supported base environment includes Python 3.10, CUDA Toolkit 12.4, and PyTorch 2.5.0. For AArch64 architecture or B200s (sm_100 architecture) GPU users, additional installation steps are required.

### Prerequisites

- **Environment Manager**: micromamba/conda/etc.
- **Python**: 3.10
- **CUDA Toolkit**: 12.4 (or 12.8 for B200s)
- **PyTorch**: 2.5.0 (or 2.7.0 for B200s)
- **Compiler**: ninja 1.11.1
- **C++ Compiler**: gcc 11.2

### 1. Environment Setup

Install micromamba if not already installed:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Create the project environment:
```bash
micromamba create -n deepspeed_scaling python=3.10 -y
micromamba activate deepspeed_scaling
```

### 2. CUDA Toolkit Installation

**For standard CUDA 12.4:**
```bash
micromamba install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
```

**For B200s GPU (CUDA 12.8):**
```bash
micromamba install -c "nvidia/label/cuda-12.8.0" cuda-toolkit -y
```

**Configure CUDA environment variables:**
```bash
# Create activation/deactivation directories
ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_D="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

# Create activation script
cat > "$ACTIVATE_D/cuda_setup.sh" << 'EOL'
#!/bin/bash
export _OLD_CUDA_HOME=$CUDA_HOME
export _OLD_PATH=$PATH
export _OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOL

# Create deactivation script
cat > "$DEACTIVATE_D/cuda_reset.sh" << 'EOL'
#!/bin/bash
export CUDA_HOME=$_OLD_CUDA_HOME
export PATH=$_OLD_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH
EOL

# Make scripts executable
chmod +x "$ACTIVATE_D/cuda_setup.sh" "$DEACTIVATE_D/cuda_reset.sh"

**Note:** After configuring the CUDA environment variables, you need to restart the environment by running `micromamba deactivate` and then `micromamba activate deepspeed` again for the changes to take effect.
```

### 3. PyTorch Installation

**For x86_64 architecture:**
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

**For AArch64 architecture:**
```bash
pip install https://download.pytorch.org/whl/nightly/cu124/torch-2.5.0.dev20240902%2Bcu124-cp310-cp310-linux_aarch64.whl
```

**For B200s GPU:**

Install PyTorch 2.7.0 with CUDA 12.8 support:
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Project Dependencies

```bash
pip install -r requirements.txt
```

### 5. Apex Installation

**Clone and prepare Apex:**
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 22.04-dev
```

**Apply PyTorch 2.5.0 compatibility patches:**
```bash
# Comment out line 2
sed -i '2s/^/#/' apex/amp/_initialize.py

# Replace "string_classes" with str in line 42
sed -i '42s/string_classes/str/' apex/amp/_initialize.py
```

**Install Apex:**
```bash
# Set MAX_JOBS before pip install if CPU resources are limited, example:
# export MAX_JOBS=4  # Adjust based on your CPU cores
pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" ./

```

**Note for B200s GPU users:** If using SM_100 architecture with PyTorch 2.7.0, you may need to manually replace `.type().scalarType()` calls with `.scalarType()` in Apex C++ files before installing Apex. For example:
```cpp
// Before (may fail on SM_100):
AT_ASSERTM(input.type().scalarType() == at::ScalarType::Half, ...)

// After:
AT_ASSERTM(input.scalarType() == at::ScalarType::Half, ...)
```

### 6. Verification

Verify the installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import apex; print('Apex installed successfully')"
python -c "import torch; import amp_C; print('amp_C installed successfully')"
```

### 7. Tips
If you encounter C++ compilation errors during installation, try installing environment-based GCC/G++ compilers:
```bash
micromamba install -c conda-forge gcc_linux-64==11.2 gxx_linux-64==11.2 -y
```


## Data Preparation

This section describes how to prepare the datasets used in our experiments: C4 and Slimpajama. We provide instructions for downloading, subsetting, and preprocessing these datasets.

### Dataset Download

We use the Hugging Face `datasets` library to download the datasets. For each dataset, download and save as separate JSON files:

**C4 Dataset:**
```bash
python -c "
from datasets import load_dataset
dataset = load_dataset('allenai/c4', 'en', split='train', num_proc=N)
dataset.to_json('/path/to/your/data/c4.jsonl')
"
```

**Slimpajama Dataset:**
```bash
python -c "
from datasets import load_dataset
dataset = load_dataset('cerebras/SlimPajama-627B', split='train', num_proc=N)
dataset.to_json('/path/to/your/data/slimpajama.jsonl')
"
```
Validation follows the same procedure here. We use the full validation set of C4 and Slimpajama so for validation set, you can skip the subsetting step.


### Dataset Subsetting

To create consistent subsets with specific token counts, we use the first K samples from each dataset. Based on token statistics from [datablations](https://github.com/huggingface/datablations):

- **C4**: 478.625835 tokens per sample (using GPT2Tokenizer)
- **Slimpajama**: 1,058.057342 tokens per sample (using GPT2Tokenizer)

For a 100M token subset:
- **C4**: 100M / 478.625835 = 208,931 samples (rounded up)
- **Slimpajama**: 100M / 1058.057342 = 94,513 samples (rounded up)

**Example for C4 100M subset:**
```bash
head -n 208931 c4.jsonl > c4_100m.jsonl
```

**Example for Slimpajama 100M subset:**
```bash
head -n 94513 slimpajama.jsonl > slimpajama_100m.jsonl
```

**Note:** To facilitate testing of the paper's experimental results, we provide all the subsets used in the paper on [Hugging Face](https://huggingface.co/datasets/your-username/paper-subsets) for easy access.



### Data Preprocessing

Use the provided preprocessing script to convert JSON files to the required format for training and validation:

```bash
python tools/preprocess_data_many_cores.py \
    --input <input_file.json> \
    --output-prefix gpt2tok_<dataset_name>_<token_count> \
    --dataset-impl mmap \
    --tokenizer-type HFTokenizer \
    --tokenizer-model gpt2 \
    --append-eod \
    --workers 64 \
    --root /path/to/output/directory
```

**Parameters:**
- `--input`: Path to your JSON input file
- `--output-prefix`: Prefix for output files
- `--dataset-impl mmap`: Use memory-mapped dataset implementation
- `--tokenizer-type HFTokenizer`: Use HuggingFace tokenizer
- `--tokenizer-model gpt2`: Use GPT-2 tokenizer
- `--append-eod`: Append end-of-document tokens
- `--workers`: Number of worker processes for parallel processing
- `--root`: Output directory for processed files

**Adding Processed Datasets to Configuration:**

After preprocessing, it's recommended to add your processed datasets to `utils/datapaths` using the same format as the existing examples. The previous preprocessing will generate two files (binary and index) with the same prefix, fill in the prefix in the `datapaths` file. This allows the training/evaluation scripts to easily locate and use your datasets.

**Note:** To facilitate testing of the paper's experimental results, we provide all the preprocessed datasets used in the paper on [Hugging Face](https://huggingface.co/datasets/your-username/paper-preprocessed) for easy access. For evaluation, we use the full validation set of C4 and Slimpajama.


## Training

We provide training examples in the `examples_scaling/training` directory. These examples demonstrate how to train models using the prepared datasets and include various configurations for different model sizes and training scenarios.

For detailed training instructions and all the experimental configurations, please refer to the [TRAIN.md](examples_scaling/training/TRAIN.md) file in `examples_scaling/training/`.

## Evaluation

We provide evaluation examples in the `examples_scaling/evaluation` directory. These examples demonstrate how to evaluate models on validation sets using the prepared datasets and include various configurations for different model sizes and evaluation scenarios.

For detailed evaluation instructions and all the experimental configurations, please refer to the [EVAL.md](examples_scaling/evaluation/EVAL.md) file in `examples_scaling/evaluation/`.

## Downstream Tasks

We provide downstream task evaluation examples in the `examples_scaling/downstream` directory. These examples demonstrate how to evaluate models on various downstream tasks and include configurations for different model sizes and task scenarios.

For detailed downstream task instructions and all the experimental configurations, please refer to the [DOWNSTREAM.md](examples_scaling/downstream/DOWNSTREAM.md) file in `examples_scaling/downstream/`.