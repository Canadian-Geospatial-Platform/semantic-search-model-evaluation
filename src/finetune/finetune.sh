#!/bin/bash

#SBATCH --job-name=model-finetune
#SBATCH --output=result.out
#SBATCH --error=error.err
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB

# Load any modules or set environment variables here if necessary

# Activate Python environment
source /path/to/your/env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Variables for command-line arguments
DATA_PATH="$1"
SAVE_DIRECTORY="$2"
NUM_TRAIN_EPOCHS="$3"

# Run the Python script with command-line arguments
python finetune.py "$DATA_PATH" "$SAVE_DIRECTORY" "$NUM_TRAIN_EPOCHS"

# Deactivate Python environment
deactivate
