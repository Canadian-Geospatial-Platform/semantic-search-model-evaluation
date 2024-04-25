#!/bin/bash
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=semantic_finetune_gpu_job
#SBATCH --output=~/%x-%j.out
#SBATCH --error=~/%x-%j.err
#SBATCH --no-requeue
#SBATCH --qos=low
#SBATCH --account=nrcan_geobase__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=2 
#SBATCH --time=06:00:00
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"



# -----------------------------------------------------------------------------
# GPU is only availible on compute nodes for some specific partitions like gpu_a100 on the gpsc7 cluster 
# This section is required if you are in a landing node that has no GPU devices availible 
# -----------------------------------------------------------------------------
#ssh inter-nrcan-ubuntu2204.science.gc.ca
#salloc --qos=low --cluster=gpsc7 --partition=gpu_a100 --account=nrcan_geobase__gpu_a100 --nodes=1 --gpus=1 --time=06:00:00
#export http_proxy=http://webproxy.science.gc.ca:8888/
#export https_proxy=http://webproxy.science.gc.ca:8888/

cd /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch

# See environment_setup.sh for conda env "semantic-finetune" setup
source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh 
conda activate semantic-finetune

# Run the scripts 
python code/src/finetune/finetune.py data/raw/records.parquet results/finetune_full_datasets 15

