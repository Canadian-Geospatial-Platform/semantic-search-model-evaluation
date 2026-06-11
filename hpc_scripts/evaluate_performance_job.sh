#!/bin/bash
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=evaluate_performance_mpnet_base
#SBATCH --output=/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/slurm_logs/%x-%j.out
#SBATCH --no-requeue
#SBATCH --qos=low
#SBATCH --account=nrcan_geobase__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04"

export WORKDIR="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch"
export MODEL_NAME="all-mpnet-base-v2"
export MODEL_PATH="sentence-transformers/${MODEL_NAME}"
export EXP_NAME="${MODEL_NAME}-baseline"
export LOGGER_OUTPUT="${WORKDIR}/results/finetune_via_trainer/${EXP_NAME}"

export http_proxy=http://webproxy.science.gc.ca:8888/
export https_proxy=http://webproxy.science.gc.ca:8888/

cd $WORKDIR
mkdir -p $LOGGER_OUTPUT

source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh 
conda activate semantic-finetune

echo Starting evaluation on text_en
python code/src/evaluate_performance.py \
    --query2doc_dataset_path="${WORKDIR}/data/preprocessed-se/with_synthetic_queries/test.parquet" \
    --additional_corpus_filepaths='["'"${WORKDIR}"'/data/preprocessed-se/with_synthetic_queries/train.parquet", "'"${WORKDIR}"'/data/preprocessed-se/with_synthetic_queries/eval.parquet"]' \
    --document_col_names='["text_en", "text_seq", "text_para"]' \
    --model_path="${MODEL_PATH}" \
    --save_filedir="${LOGGER_OUTPUT}/performance_evaluation/"