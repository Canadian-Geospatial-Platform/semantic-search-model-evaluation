#!/bin/bash
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=semantic_finetune_gpu_job
#SBATCH --output=$HOME/%x-%j.out
#SBATCH --error=$HOME/%x-%j.err
#SBATCH --no-requeue
#SBATCH --qos=low
#SBATCH --account=nrcan_geobase__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

export WORKDIR="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch"
export EXP_NAME="all-mpnet-base-v2-finetune-test"
export TRAINING_LOG_FILE="./results/finetune_via_trainer/${EXP_NAME}"

cd $WORKDIR
mkdir $TRAINING_LOG_FILE

source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh 
conda activate semantic-finetune

python code/src/finetune/finetune_via_trainer.py \
    --train_data_path="${WORKDIR}/data/processed-se/train.parquet" \
    --eval_data_path="${WORKDIR}/data/processed-se/eval.parquet" \
    --model_name="sentence-transformers/all-mpnet-base-v2" \
    --model_save_directory="${WORKDIR}/results/finetune_via_trainer/${EXP_NAME}/model" \
    --data_anchor_column="query_en" \
    --data_doc_column="text_en" \
    --data_restrict_num_records_to="100" \
    --train_losstype="MNRL"
    # --data_mix_languages \  # uncomment to expand dataset for multilingual training
