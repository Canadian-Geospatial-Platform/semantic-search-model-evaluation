#!/bin/bash
################################################################################
# Script Name package_installation_HPC.sh
# Description: This script installs the required applications and python packages for fine-tuning semantic search models on NRCan HPC.
# Author: Xinli Cai 
# Date: April 7, 2024
################################################################################

################################################################################
# Additional Description: 
# This scripts Install the PyTorch through conda by matching the cuda-toolkit version of driver currently available (12.2) on HPC.
# conda-forge is set to the default channel for permission reasons. 
# This also needs to be performmed inside a job launched on a server with gpus in order to select the right packages.
################################################################################


# -----------------------------------------------------------------------------
# Section 1: launch a job inside the compute node 
# -----------------------------------------------------------------------------
# ssh to the compute node with gpus and launch a job for 6 hrs 
ssh inter-nrcan-ubuntu2204.science.gc.ca
salloc --qos=low --cluster=gpsc7 --partition=gpu_a100 --account=nrcan_geobase__gpu_a100 --nodes=1 --gpus=1 --time=06:00:00

export http_proxy=http://webproxy.science.gc.ca:8888/
export https_proxy=http://webproxy.science.gc.ca:8888/

# Verify CUDA support version 
nvidia-smi

# -----------------------------------------------------------------------------
# Section 2: miniconda installation
# -----------------------------------------------------------------------------
# This section install miniconda, and set conda-forge as the default channel
cd /space/partner/nrcan/geobase/work/oatt

# Download miniconda for Linux from https://docs.conda.io/en/latest/miniconda.html 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo "Miniconda downloaded successfully."

mv Miniconda3-latest-Linux-x86_64.sh /space/partner/nrcan/geobase/work/oatt
chmod +x Miniconda3-py39_24.1.2-0-Linux-x86_64.sh

# Run the installer scripts and make sure the installer is in the opt/miniconda3 path 
./Miniconda3-py39_24.1.2-0-Linux-x86_64.sh -b -p opt/miniconda3
echo "Miniconda installed successfully."

conda init 
source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh #or source ~/.bashrc if miniconda path is added as env path 

# Creaet conda-forge channel 
conda config --add channels conda-forge
echo "Conda-forge channel added successfully."

# Clean up 
rm Miniconda3-py39_24.1.2-0-Linux-x86_64.sh

# -----------------------------------------------------------------------------
# Section 3: Install AWS CLI  
# -----------------------------------------------------------------------------
# Dowbload the AWS CLI, insturctions here: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
echo "AWS CLI downloaded and unzipped successfully."

# Install the aws in a local folder that you have permission to write to 
./aws/install -i ~/aws-cli -b ~/bin
echo "AWS CLI installed successfully."

# Set aws executable path (~/bin) to the system's PATH 
export PATH="$HOME/bin:$PATH"
echo "AWS CLI path set successfully."

echo "AWS CLI version:"
aws --version 

#Set up the aws config and credential files
aws configure 
echo "Check AWS S3 connection"
aws s3 ls 

# -----------------------------------------------------------------------------
# Section 4: Create a new env and install Pytorch with CUDA (11.2) support
# -----------------------------------------------------------------------------
source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh
conda create -c conda-forge -n semantic-finetune python=3.9
conda activate semantic-finetune
echo "Semantic-finetune environment created and activated successfully."


# Install Pytroch with CUDA 12.2. Instruction:https://pytorch.org/get-started/locally/
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12
echo "PyTorch with CUDA 12.2 installed successfully."

# Check the the cuda availiblity in the computer node 
python << END
import torch

print("CUDA is available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())
print("Current CUDA device index:", torch.cuda.current_device())
print("CUDA device at index 0:", torch.cuda.device(0))
print("Name of CUDA device at index 0:", torch.cuda.get_device_name(0))
END

# -----------------------------------------------------------------------------
# Section 5: Install the reqirements.txt for the semantic search fine-tuning scripts 
# -----------------------------------------------------------------------------
# Install the packages on existing conda env pytorch-py
conda env update -f dev/semanticsearch/code/src/finetune/environment.yml
echo "Packages required for semantic search fine-tuning installed at conda env semantic-finetune successfully."
