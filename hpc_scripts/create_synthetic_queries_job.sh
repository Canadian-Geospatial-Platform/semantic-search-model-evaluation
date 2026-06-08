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
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

cd /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch

source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh 
conda activate semantic-finetune

python code/src/data_processing/create_synthetic_queries.py \
    --input-data-dir="./data/processed-se/" \
    --output-data-dir="./data/processed-se/with_synthetic_queries/" \
    --base-column-name="text_en" \
    --new-column-name="query_en" \