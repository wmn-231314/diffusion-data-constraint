#!/bin/bash
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# create conda environment
micromamba create -n dba python=3.10 -y

# Initialize micromamba for this shell session and activate the environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate dba

# install cuda toolkit
micromamba install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y


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

# reactivate the environment to load the scripts
micromamba deactivate
micromamba activate dba

# For x86_64 architecture:
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# install requirements
pip install -r requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 22.04-dev

# Apply PyTorch 2.5.0 compatibility patches:
# Comment out line 2
sed -i '2s/^/#/' apex/amp/_initialize.py

# Replace "string_classes" with str in line 42
sed -i '42s/string_classes/str/' apex/amp/_initialize.py

# Install Apex:
# Set MAX_JOBS before pip install if CPU resources are limited, example:
# export MAX_JOBS=4  # Adjust based on your CPU cores

pip install -v --disable-pip-version-check --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" ./

# Verify the installation:
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import apex; print('Apex installed successfully')"
python -c "import torch; import amp_C; print('amp_C installed successfully')"

