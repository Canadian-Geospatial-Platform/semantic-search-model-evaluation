# Finetune-via-Trainer on HPC Guide
This guide details the end-to-end process for how finetune-via-trainer was used to finetune SentenceTransformer models on HPC

## Table of Contents
[Setting up working environment](#setting-up-working-environment)
* [1. Logging into cluster and working with SLURM](#1-logging-into-cluster-and-working-with-slurm)
    * [Common Bash commands](#common-bash-commands)
* [2. Navigating directories and virtual environments](#2-navigating-directories-and-virtual-environments)

[Loading datasets](#loading-datasets)
* [Configure AWS](#configure-aws)
* [Generating synthetic queries](#generating-synthetic-queries)

[Finetuning workflow](#finetuning-workflow)
* [Changes required for every experiment](#changes-required-for-every-experiment)
* [Results from finetuning](#results-from-finetuning)

[Evaluation workflow](#evaluation-workflow)
* [Selecting the best configuration for each experiment](#selecting-the-best-configuration-for-each-experiment)
* [Final comparison of models](#final-comparison-of-models)
    * [In-domain test](#in-domain-test)
    * [Real user query test](#real-user-query-test)
* [Post-evaluation](#post-evaluation)

[Special note for General Text Embedding (GTE) models](#special-note-gte-models)

[Common pitfalls](#common-pitfalls)
* [Out of space](#out-of-space)
* [Libtiff Import error on Salloc session](#libtiff-import-error-on-salloc-session)
* [Torch version upgrade](#torch-version-upgrade)

[Theoretical Analysis of Experiments](#theoretical-analysis-of-experiments)
* [Existing Methodology](#existing-methodology)
* [New Methodology Implemented](#new-methodologies-implemented)
* [Stage 2: 2026-08-11](#stage-2-2026-08-11)

---
## Setting up working environment
### 1. Logging into cluster and working with SLURM
First off, you want to connect to the remote server that you will be working on to obtain access to resources like GPU. Depending on the access you have, you may be able to directly connect to the server that has a GPU you can use, or you may need to go through the landing page (my case). Connecting to the server involves setting up an SSH tunnel to the system through your own system, which can be done using the following steps:

1. Open CommandLine or a terminal on your system
2. Use the following template to connect to the remote server: `ssh <username>@<server_name>`. If you need to first connect to the landing page and then to the server you case use: `ssh -J <username>@<landing_page> <username>@<server_name>` For my case, it is: 
    ```
    ssh -J <username>@inter-nrcan-lp-gccloud.science.gc.ca <username>@inter-nrcan-ubuntu2204.science.gc.ca
    ```
    *Note*: you will be prompted to enter your password. If you are using a jump server, you will need to input your password twice: once for the jump server and once for the actual server you want to reach.
3. If this is you first time running a job through SLURM, you want to direct what default cluster to use for your jobs. Please consult the [HPC documentation](https://portal.science.gc.ca/xwiki/bin/view/Projects/Science/Home/) for the technical capabilities of all available clusters. In my case, `gpsc7` fits my use case. So, what we will do is add the following section (replacing the cluster name with the one you need) to the `~/.bashrc` file. I am going to use `vim` for my example.

    a) Open file:
    ```
    vim ~/.bashrc
    ```
    b) Go into insert mode by pressing `i` and paste the following in the file:
    ```
    export SLURM_DEFAULT_CLUSTER=gpsc7.science.gc.ca

    if [[ -z $SLURM_CONF ]]; then
    if [[ -f "/etc/slurm-llnl/${SLURM_DEFAULT_CLUSTER}.conf" ]]; then
        export SLURM_CONF="/etc/slurm-llnl/${SLURM_DEFAULT_CLUSTER}.conf"
    fi
    f
    ```
    c) Save and exit file through `Esc`, `:wq`, then `Enter`.

4. Activate the bashrc file to set a default cluster
    ```
    source ~/.bashrc
    ```
    If you are encountering a `invalid partition specified: gpu_a100` error, you probably just forgot to activate the bashrc file, so please try running the command above before running the job again.

Now you are ready to schedule and run jobs through `sbatch`. Below are some other commands that you may find yourself commonly using on HPC.

#### Common Bash commands
* Check status of job:
    ```
    squeue --user=$USER
    ```
* Start an interactive session for 2 hours:
    ```
    salloc --qos=low --cluster=gpsc7 --partition=gpu_a100 --account=nrcan_geobase__gpu_a100 --nodes=1 --gpus=1 --time=02:00:00
    ```
* Allowing internet access to run commands like `git pull`:
    ```
    export http_proxy=http://webproxy.science.gc.ca:8888/
    export https_proxy=http://webproxy.science.gc.ca:8888/
    ```

### 2. Navigating directories and virtual environments
For the semantic search project, the working directory is shared and can be found under:
```
cd /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/
```

In this directory, the subdirectory `code/` contains a [GitHub repository](https://github.com/Canadian-Geospatial-Platform/semantic-search-model-evaluation) to all the scripts and executable jobs that are available for use. Another subdirectory is `data/` and it contains records that were originally stored in AWS S3 buckets for easy access by the scripts. An often-used subdirectory is `results/` which contains all the outcomes generated by the scripts, jobs, etc. Inside, it is split up by project. For finetuning through SentenceTransformer's Trainer, the relevant subdirectory is `results/finetune_via_trainer/`.

Before executing any jobs, you need to make sure your virtual environment is properly set up. It is configured through Anaconda:
1. Conda is already installed, you just need to point to the source via:
    ```
    source /space/partner/nrcan/geobase/work/oatt/opt/miniconda3/etc/profile.d/conda.sh
    ```
2. Environment should already be configured, you just need to activate it:
    ```
    conda activate semantic-finetune
    ```

## Loading datasets
The datasets used for finetuning and checking the performance of the model are housed under `./data/`. In the current implementation, these datasets are manually downloaded from their AWS S3 bucket through the following aws cli command:
```
aws s3 sync s3://webpresence-nlp-data-preprocessing-stage/semantic_search_se/data/ /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/preprocessed-se/
```
*Note*: internet access is required to run the above command.

### Configure aws
If aws is not configured on your account, the above command will throw an error resembling `-bash: aws: command not found`. Alternatively, if aws has been wrongly configured, you will see an error resembling `fatal error: An error occurred (InvalidToken) when calling the ListObjectsV2 operation: The provided token is malformed or otherwise invalid`. For both of these cases, you would want to reconfigure aws using the following steps (extracted from `hpc_scripts/environment_setup.sh`). If you have already installed aws previously, you want to start from step 4; no need to redownload it again.

1. Make sure you are in the home directory via
    ```
    cd ~
    ```
2. Download aws-cli through CURL and unzip the contents
    ```
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ```
3. Install aws in a local folder like `bin`
    ```
    ./aws/install -i ~/aws-cli -b ~/bin
    ```
4. Set aws executable path (~/bin) to the system's PATH.
    ```
    export PATH="$HOME/bin:$PATH"
    ```
5. Check aws installation was successful. There should be no error returned by:
    ```
    aws --version
    ```
6. Set up aws credentials
    ```
    aws configure
    ```
7. Done! Check that you can read the contents of s3 buckets through `aws s3 ls`.

### Generating synthetic queries
Further, these datasets are used to create synthetic queries that mimic real user inquiries. For this, you will need to start an interactive session with an internet proxy and run the following command. The process takes no more than approximately 3 mins.
```
python code/src/data_processing/create_synthetic_queries.py \
    --input-data-dir="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/preprocessed-se/" \
    --output-data-dir="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/preprocessed-se/with_synthetic_queries/" \
    --base-column-name="text_en" \
    --new-column-name="query_en" 
```
To explain the above command, `base-column-name` takes a column from the loaded dataframe as basis for its query generation, generates the query through a doc2query model and stores it in the column defined by `new-column-name`.

Additionally to running it on English documents to acquire English queries, for evaluating cross-lingual performance, you will need to run the script a second time on the French documents.
```
python code/src/data_processing/create_synthetic_queries.py \
    --input-data-dir="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/preprocessed-se/with_synthetic_queries/" \
    --output-data-dir="/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/preprocessed-se/with_synthetic_queries/" \
    --base-column-name="text_fr" \
    --new-column-name="query_fr" 
```


## Finetuning workflow
Finetuning is configured such that you can change the variables (models, hyperparameters, etc.) being set in a bash file and execute the job through:
```
sbatch code/hpc_scripts/finetune_via_trainer_job.sh
```
The job will execute in the background once resources become available. Logs outtputed through this job are saved in `./results/finetune_via_trainer/slurm-logs/`. Let's talk about what changes took place in the bash scripts to acquire the finetuned models, so that you can replicate the results.

### Changes required for every experiment
- **For model change** -- 1) change the `MODEL_NAME` environment variable in the bash scripts accordingly, and 2) change the SBATCH job name on top of the bash script to reflect in the job name what model is being used for good logging practice
- **For script argument value change** -- 1) document it in the `EXP_NAME` environment variable in the bash scripts accordingly (the file directory where the results are saved will be called this), and 2) change the script argument values in the python script call.
    
    Example: Finetuning the model with data_mix_languages, with `para` document representation, with sequence length at 512, batch size at 64, and max training steps at 2048. 
    
    1) Changed experiment name to reflect configuration change:
    ```
    export EXP_NAME="finetune-mnrl-mix-para-sq512b64ms2048"
    ```

    2) Changed argument values in script call:
    ```
    python code/src/finetune/finetune_via_trainer.py \
    --train_data_path="${WORKDIR}/data/preprocessed-se/with_synthetic_queries/train.parquet" \
    --eval_data_path="${WORKDIR}/data/preprocessed-se/with_synthetic_queries/eval.parquet" \
    --model_path="${MODEL_PATH}" \
    --model_save_directory="${LOGGER_OUTPUT}" \
    --data_anchor_column="query" \
    --data_doc_column="text_para" \ # CHANGED
    --train_losstype="MNRL" \
    --model_enforce_max_seq="512" \  # CHANGED
    --train_batch_size="64" \  # CHANGED
    --train_max_steps="2048" \  # CHANGED
    --data_mix_languages \  # CHANGED
    ```

    *Note*: when not specified, changing the batch size is accompanied by changing the logging steps and max steps for a comparative training cycle. E.g. if the default uses a batch size of 32, logging steps at 64 and max steps at 1024, then for a batch size of 64, the logging steps are set to 32 and the max steps to 512
- **For multilingual training** -- the idea is that the model should be exposed to both english and french queries during the training loop. Hence, the anchor column should be abstracted to a prefix (e.g. `query_en` -> `query`) and the mix-languages flag should be set for the script call (i.e. `--data_mix_languages`). This will use both Engish and French queries, appending the (anchor, doc) pairs to the dataset and, therefore, doubling the size of the training dataset, for the model to learn both query representations at the same time.
- **For GTE model use** -- please see [Special note for General Text Embedding (GTE) models](#special-note-gte-models)
- **For training with hard negatives** - hard negatives will be generated using the base model if the flag `--data_mine_hard_negatives` is set. Additionally, you can configure the parameters for mining the negative documents using all the CLI arguements available under `--hard_negatives_*`.

---

*Note:* The best model saved by the finetuning script is the one that acheived the best MRR@10 score on the evaluation dataset. The finetuning job script tests the implementation of the finetuned model after it has been saved. This is not required so you can comment out this section in the finetuning job if it is not needed.

---

### Results from finetuning
All results including metrics tracked during training, model checkpoints and best model saves, performance on validation split, etc. are saved in the directory dictated by the `LOGGER_OUTPUT` environment variable. To acquire the MRR@10 on the validation split via which the model configurations/hyperparameter sets are compared and verify the integrity of the training run, I manually copied the training logs to my local machine, opened them via Excel and graphed the train and validation loss curves. To do this, I used the following command template on my local machine:
```
scp -J <username>@<jump server> <username>@<server name>:<path to experiment directory on server>/training_logs.csv <local file path to save the training logs as>
```


## Evaluation workflow
### Selecting the best configuration for each experiment
To determine the best hyperparameter configuration for each model, the MRR@10 metric value on the validation split is used to compare the training runs. Specifically, we examine the metrics produced by the InformationRetrievalEvaluator during the training run to select the best model, resulting in a two-fold evaluation: 

1. only the best model checkpoint (the one with the highest MRR@10 on the validation split) **during a single run** is retained for every experiment, and

2. only the best model experiment (the one with the highest MRR@10 on the validation split) out of the ones that **share the same experimental setup, i.e. base model, but varying batch size, learning rate, etc.** is considered for further evaluation on test. 

### Final Comparison of Models
To determine which model is best, we consider several test datasets: the test on in-domain anchors (i.e. synthetic queries on documents the model has not seen before that are generated in the same manner as the ones for training) and the test on real-user queries. While the former acts as a simple sanity check that the model is not over-tuned to the validation split, the latter is the one that will best reflect how the model will act in a production environment.

#### In-domain test
At the point of writing this guide, the evaluation on the test dataset is appended to the end of every finetune job execution for ease of processing - it does not take long to compute so it does not add much overhead to a finetuning run. However, it is important to note that the results on the test dataset were only examined for the "best" finetuned model as dictated by the MRR@10 on the validation split to avoid evaluation bias.

The evaluation script can be found under `./code/src/evaluate_performance.py` and it combines documents found in the train, validation and test splits to create the corpus. Queries are only used from the test split as ones the models have never seen before. They are synthetically generated so they should mimic typical user queries, but with no guarantee that they are all realistic.

For the baseline models, the following steps were followed to check the performance of the models on the test dataset:

1. Ensure you are in the working directory and have set the default cluster through `.bashrc`.
2. Change SBATCH job_name, MODEL_NAME, and MODEL_PATH appropriately to the model being loaded in `./code/hpc_scripts/evaluate_performance_job.sh`
3. Kickstart the job via command line:
    ```
    sbatch ./code/hpc_scripts/evaluate_performance_job.sh
    ```
4. Once the job has finished running, you can verify performance metrics under `cat ./results/finetune_via_trainer/{MODEL_EXP}/performance_evaluation/results.csv` alongside the specific predictions under `cat ./results/finetune_via_trainer/{MODEL_EXP}/performance_evaluation/*_predictions_cosine.jsonl`.

##### Summarizing findings
With multiple experiments and each having its own individual `.csv` with the metrics generated on test, it becomes challenging to compare them against each other. As such, there is a separate script that can be run to combine all the metrics into one file. This script can be run via:
```
python code/src/collect_results.py
```
- To specify which experiment's test metrics you want to consider, modification to `src/finetune/configs/finetune-via-trainer-best-eval-se.json` is required or the creation of a sibling configuration file. This config file contains a list of experiments with their parent model name under `model` and experiment configuration under `training_configs`, with both reflective of the filepath on the HPC cluster that contains the model weights.
- At the time of writing this guide, the summarization python script does not allow for configuration file or output filename specification via CLI arguements. As such, you would need to manually edit the variables `config_path` and `save_path` within the python script itself if you would like to run it.
- To run the python script, you do not need to request extra resources via the `salloc` command; the login node's resources should be sufficient.


#### Real-user query test
Realistically, there are two real-user query benchmarks at play: the original one, and a new benchmark created from a survey (since there are no French queries in the original one). In this guide, we will refer to the two benchmarks as benchmark 1 and benchmark 2, respectively. Both can be found on HPC under 
```
cd /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/benchmark/
```
- On HPC, benchmark 1 is found under the `by_geo_theme` directory, where specifically the one with the `_updated` suffix has been manually modified and extended by me (Svetlana Esina). It does include more records but the relevancy of the dataset identified per query is debateable (as it was not checked by an expert).
- Benchmark 2 is found in the root of the above specified absolute path.


##### Benchmark 2 creation
The benchmark with real-user queries is composed of user-defined relevant documents from the PostHog survey results gathered on the following website from July 9th to the 22nd:
```
https://canadian-geospatial-platform.github.io/semantic-search-demo/
```

To evaluate performance on both benchmarks, you can simply run the following script in HPC:
```
python code/src/evaluate_performance_on_benchmark.py
```
- Modify the python script directly if you want to change which training configs are being evaluated on what corpus data with what benchmarks
- Results for best model via recall@3 will be printed in the output of the script.
- Corpus saved under `perf_on_benchmark` for each experiment to enable manual exploration of the retrieved documents.

### Post-evaluation
After the best model is determined, you can transfer it to your local system via:

1. Compressing all the model files together on the cluster: 
    ```
    tar -czvf ./results/finetune_via_trainer/gte-multi-ft-se.tar.gz --exclude='./results/finetune_via_trainer/gte-multilingual-base-baseline/finetune-mnrl-mix-seq-sq512/gte-multilingual-base/checkpoint-*' ./results/finetune_via_trainer/gte-multilingual-base-baseline/finetune-mnrl-mix-seq-sq512/gte-multilingual-base/
    ```
    or
    ```
    tar -czvf ./results/finetune_via_trainer/gte-multi-ft2-se.tar.gz --exclude='*/checkpoint-*' --exclude='*/perf_on_benchmark/*' ./results/finetune_via_trainer/gte-multilingual-base-baseline/2026_08_11/finetune-mnrl-mix-seq-sq1024/gte-multilingual-base/
    ```
2. Transferring file to local system:
    ```
    scp -J sve000@inter-nrcan-lp-gccloud.science.gc.ca sve000@inter-nrcan-ubuntu2204.science.gc.ca:/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/gte-multi-ft-se.tar.gz ./
    ```
3. Upload to Sagemaker AI notebook that has a clone of the `semantic-search-with-amazon-opensearch` repository
4. Open a terminal and un-tar the compressed file using:
    ```
    cd SageMaker/
    mkdir ./semantic-search-with-amazon-opensearch/model/gte-multi-ft-se
    tar -xvf ./semantic-search-with-amazon-opensearch/model/gte-multi-ft-se.tar.gz -C ./semantic-search-with-amazon-opensearch/model/gte-multi-ft-se --strip-components=6
    ```
5. Do not forget to add any necessary LICENSE files into the model directory. E.g. GTE models are under the Apache2.0 license that requires a copy of the license to be present in the derivative of the work.


## Special note: GTE models
GTE models require a custom script for modelling and configuring the model. This script is public on huggingface but in order for the sentence-transformer library to use it, it must have `"trust_remote_code"` set to `True` in the `SentenceTransformer()` initialization. This, however, presents a security concern - any code retrieved from this public repository will be automatically used without notifying the user when the model is loaded. The community had suggested a workaround for this particular situation which is detailed [here](https://huggingface.co/Alibaba-NLP/new-impl/discussions/2). This workaround suggests to download the Huggingface model locally, alongside the custom scripts, where you will gain full control over the code being loaded for your model. The caveat is that `"trust_remote_code"` should still be set to `True`, for the custom scripts to be properly referenced and used. As such, the current code is set up to be able to set `"trust_remote_code"` to `True` only if the model is being loaded from a local directory and not from HuggingFace itself.

### Workaround - Downloading the model
1. Start an interactive session via `salloc` to get access to gpu for faster model loading speeds.
    ```
    salloc --qos=low --cluster=gpsc7 --partition=gpu_a100 --account=nrcan_geobase__gpu_a100 --nodes=1 --gpus=1 --time=01:00:00
    ```
2. Route internet through the proxies.
    ```
    export http_proxy=http://webproxy.science.gc.ca:8888/
    export https_proxy=http://webproxy.science.gc.ca:8888/
    ```
3. Activate conda environment.
    ```
    conda activate semantic-finetune
    ```
4. Navigate to the working directory for this project and model. If the model folder does not exist, create it using `mkdir`.
    ```
    cd /space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/gte-multilingual-base-baseline
    ``` 
5. Download the model from HuggingFace through the command line and store it in a local directory (it will be created if it does not exist)
    ```
    hf download Alibaba-NLP/gte-multilingual-base --local-dir ./gte-multilingual-base/
    ```
6. Download the custom scripts from HuggingFace
    ```
    hf download Alibaba-NLP/new-impl --local-dir ./new-impl/
    ```
7. Move the following custom scripts into the directory where the model weights are saved
    ```
	mv ./new-impl/modeling.py ./gte-multilingual-base/
	mv ./new-impl/configuration.py ./gte-multilingual-base/
    ```
8. Modify the following lines in the model `config.json`. I will be using `vim` as my editor in this example.
    1) Open file contents:
    ```
    vim gte-multilingual-base/config.json
    ```
    2) Replace the following portion of the file with the local path. You can use `dd` to delete entire lines from the file.
    
    Original portion of text (to be changed):
    ```{json}
    "auto_map": {
        "AutoConfig": "Alibaba-NLP/new-impl--configuration.NewConfig",
        "AutoModel": "Alibaba-NLP/new-impl--modeling.NewModel",
        "AutoModelForMaskedLM": "Alibaba-NLP/new-impl--modeling.NewForMaskedLM",
        "AutoModelForMultipleChoice": "Alibaba-NLP/new-impl--modeling.NewForMultipleChoice",
        "AutoModelForQuestionAnswering": "Alibaba-NLP/new-impl--modeling.NewForQuestionAnswering",
        "AutoModelForSequenceClassification": "Alibaba-NLP/new-impl--modeling.NewForSequenceClassification",
        "AutoModelForTokenClassification": "Alibaba-NLP/new-impl--modeling.NewForTokenClassification"
    },
    ```
    Modifion portion of text (after change):

    ```{json}
    "auto_map": {
        "AutoConfig": "configuration.NewConfig",
        "AutoModel": "modeling.NewModel",
        "AutoModelForMaskedLM": "modeling.NewForMaskedLM",
        "AutoModelForMultipleChoice": "modeling.NewForMultipleChoice",
        "AutoModelForQuestionAnswering": "modeling.NewForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling.NewForSequenceClassification",
        "AutoModelForTokenClassification": "modeling.NewForTokenClassification"
    },
    ```
    3) Save changes in file via `Esc` then `:wq` then `Enter`.

---

*Note:* implementing this workaround affects the finetuning and evaluation frameworks in the following manner: a) the `MODEL_PATH` variable in the HPC bash scripts will need to be manually modified so that it does not start with `sentence-transformers/`, but instead points to the local directory where the model was saved, and b) after finetuning the model, when it is being saved, the custom scripts for modelling and configuration need to be saved alongside it.

Example:
```
export MODEL_PATH="${WORKDIR}/results/finetune_via_trainer/${MODEL_NAME}-baseline/${MODEL_NAME}"
```
---

# Common pitfalls

## Out of space
When testing multiple models/experiments, you may find that you get an out of space error. This typically refers to the space allocated to the home `~/` directory, which gets filled because all the models are cached in `~\.cache\huggingface\hub`. You can just clear this directory in its entirety and huggingface will recreate it when it downloads the model from the web again.

## Libtiff import error on salloc session
If you get an error trying to import the sentence-transformer library in python that starts with `ImportError: libtiff.so.5: cannot open shared object file: No such file or directory`, try the following steps:
1. It may be a mismatch between the os library libtiff.so.6 vs libtiff.so.5 and the version the pillow library points to (pillow is used by transformers, and by extension sentence-transformers). So we will try to reinstall the pillow and libtiff packages via conda to hopefully resolve this error:
    
    1. Ensure no pillow package is installed: `pip uninstall pillow -y` and `conda remove pillow -y`
    2. Reinstall: `conda install -c conda-forge pillow libtiff --force-reinstall`
    3. Check sentence-transformer version that may have been downgraded during these processes via `conda list sentence-transformers`. Expected to be 5.1.0 at the time of writing this guide.
    4. If the version is off, upgrade: `conda install -c conda-forge sentence-transformers --force-reinstall`
    5. Deactivate and reactivate the environment and try running the script again

## Torch version upgrade
Error on generating synthetic queries:
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function.
```

Fix through `pip install`. "Conda" installs are no longer supported through official Pytorch installation webpage.

1. Install updated version
    ```
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```
    - Confirm command through [PyTorch](https://pytorch.org/get-started/previous-versions/) webpage
    - Get appropriate CUDA version by checking [Cluster](https://portal.science.gc.ca/xwiki/bin/view/Projects/Science/Home/) resources. In my case, I am using the `a100` for `gpsc7`


2. Check proper installation of Pytorch
    ```
    python
    import torch
    print(torch.__version__)
    ```

3. [Optional] Update conda environment yaml to reflect new package install.
    ```
    conda env export > code/src/finetune/environment.yml
    ``` 

# Theoretical Analysis of Experiments
## Existing Methodology
1. Masked Language Modelling: 
    * File location: https://github.com/Canadian-Geospatial-Platform/semantic-search-model-evaluation/blob/main/src/finetune/finetune.py
    * Loads model usings AutoModelForCausalLM or AutoModelForMaskedLM
    * Trains using masked language modelling, so model learns to predict the masked word in a given phrase
        * Typical for models learning how to process language from scratch or for refining a models vocabulary / language understanding to a particular domain
    * Applicability:
        * We are using pre-trained models, so they should already be great at language modelling i.e. interpreting and generating natural language
        * Will not help with retrieval of relevant documents, but will help if you want to expand vocabulary/knowledge of LLM to another domain (i.e. if current domain uses highly technical language)

2. Finetuning with Multiple Negatives Ranking Loss (MNRL)
    * File location: https://github.com/Canadian-Geospatial-Platform/semantic-search-model-evaluation/blob/main/src/finetune/multiple_positive_finetune.py
    * Loads model using SentenceTransformer
    * Finetunes using MNRL : given an anchor-document pair, attempts to minimize distance between them in the vector space while maximizing distance between all other documents in the batch (authomatically creates in-batch negatives)
    * Anchor-document is determined by splitting the record in half sentence-wise
    * Uses older implementation of training via model.fit()
    * Applicability:
        * This is the typical way to fine-tune a model for an information retrieval context. It is akin to the sentence-similarity finetuning method described by Sentence Transformers, but eliminates the need for explicitly defining negatives or providing a score to determine “closeness” of documents.
        * Disadvantage #1: MNRL treats all other documents found within a batch equally as negatives, so an anchor cannot be linked to multiple documents and a document cannot be linked to multiple anchors within the same batch
        * Disadvantage #2: Splitting the document equally creates an unrealistic training context for the model. In our use case, the search queries are typically shorter than the document contents and use similar keywords/phrases as the document’s most important sections

## New Methodologies Implemented
Demo of Old vs New Semantic Search systems can be found: https://canadian-geospatial-platform.github.io/semantic-search-demo/ 

Name: **Finetuning with Multiple Negatives Ranking Loss (MNRL) via Trainer**
* File location: https://github.com/Canadian-Geospatial-Platform/semantic-search-model-evaluation/blob/feat/gistembedloss/src/finetune/finetune_via_trainer.py

Details:
* Loads model using Sentence Transformer
* Finetunes using MNRL, but also has support for finetuning with GISTEmbedLoss: similar to MNRL but uses similarity to select the best negative documents/anchors from the batch
* Anchor-document is not set in script, but usage of script involves generating the anchor-document pair first before finetuning. 
    * Document houses all document record contents and anchor is a synthetically generated query from the document contents using a doc2query model: https://github.com/Canadian-Geospatial-Platform/semantic-search-model-evaluation/blob/feat/gistembedloss/src/data_processing/create_synthetic_queries.py
* Multilingual support: allows for training dataset to be extended to include query in English and query in French as anchors pointing to the same document
    * Adds both pairs to the dataset and shuffles. BatchSampler with strict requirement to form batches without duplicates is used when passing dataset to Trainer to ensure document isn’t treated as a negative by MNRL 
* Updates implementation to use SentenceTransformerTrainer instead of model.fit(), which allows more extensive control of the fine-tuning configurations

### Models Attempted

| # | Family | Monolingual | Multilingual |
| - | --- | --- | --- |
| 1 | mpnet | sentence-transformers/all-mpnet-base-v2 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| 2 | miniLM | sentence-transformers/all-MiniLM-L12-v2 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| 3 | GTE | Alibaba-NLP/gte-base-en-v1.5 | Alibaba-NLP/gte-multilingual-base |

Mpnet and miniLM families were originally attempted using “Finetuning with MNRL” but these were specific versions: 
* "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
* However, preprocessing was different, training dataset was different, and evaluation records were different making direct comparison of the existing and new approaches not feasible

Therefore, some models were kept for these experiments but new ones were also added

### Configurations
* Monolingual models were only trained in a monolingual setting. They lack the vocabulary/ language knowledge to understand French, as they were only pre-trained on English text. Therefore, there is no reason to fine-tune or embed French text directly.
* Multilingual models were trained in both monolingual and multilingual settings. Since they have prior understanding of English and French (through pre-training), they may be utilized in all contexts.
* Multilingual document representations involve either appending the French translation of the document after the English, or interweaving the two translations.
* MNRL and GISTEmbedLoss were both tested
* Hyperparameters modified: learning rate, batch size, max steps, sequence length, warmup steps

### Experiment Pipeline
1.	Synthetic queries were generated for all dataset splits.
2.	All 6 baseline models were evaluated on the test split and all information retrieval metrics were captured 
a.	Corpus for testing is composed of all documents from training, eval and testing
3.	Models were trained on the training split with periodic evaluation on the evaluation split. Best model checkpoint during training according to mrr@10 score on the evaluation split.
a.	Corpus used to obtain mrr@10 includes documents from the train and eval splits.
4.	Models extracted from step 2 were all evaluated on the test split where all information retrieval metrics were captured
5.	Models were compared using recall@3 for more interpretability

### Experiment Key Findings
Monolingual models:
*	Best-performing baseline model on EN queries and EN doc representations: gte with >50%, but lowest mpnet is around 48%
    *	Best on FR queries: gte with 19%, but lowest mpnet is around 8%
*	Finetuning improved performance in all cases
    *	On EN queries: largest improvement on mpnet, all models performance stabilized around 55%
    *	On FR queries: largest improvement on mpnet, gte is still best at 24% but mpnet and miniLM around 21%
*	No multilingual models (base or finetuned) trained in monolingual context outperform monolingual models on EN queries and EN document representations
    *	Gte has smallest performance difference between the finetuned mono and multilingual models at ~2%, miniLM has largest at ~5%
    *	On FR queries, multi-ft variants perform best with a largest (20%) increase on gte and lowest (10%) increase on miniLM
*	Only gte mulilingual model (fintetuned on text-seq) performance beats best monolingual model on EN queries

Multilingual models:
*	Finetuning on text-seq bumps performance for both EN and FR queries for all models, yielding a minimum 10% overall increase, composed of a ~7% EN increase and a ~12% FR increase (gte)
    *	Effect on FR performance is much larger than effect on EN performance (potentially because of its low-resource representation during training)
    *   Allows discrepancy between EN and FR performances to fall from 10-15% to 3-9%
*	GTE multi ft cross-lingual performance is around 3%

Other:
*	Whether multilingual document representation uses sequential or parallel join on the languages or not does not affect the performance/training of the model
*	Best performing model: 
    *	Model name: `gte-multilingual-base`
    *	Config: `finetune-mnrl-mix-seq-sq512`

## Stage 2 [2026-08-11]
As a continuation of the above experiment, training query refinement and hard-negative mining was further explored. In the 2026-08-11 versions of the model, some datasets used to train and test the model were supplemented with additional location data in the keyword list (courtesy of Bo Lu: `bo.lu@NRCan-RNCan.gc.ca`). 

This keyword expansion meant that the model could potentially learn to rely on more on the keyword data for location-specific retrieval. Additionally, the synthetic queries generated for the models were made more specific by relying on the assumption that more words in the query equated to zeroing in on the unique quality of each dataset description, thereby decreasing the number of potential positive documents that may be mistakenly labelled as negative during training. This finetuning variant is named `finetuned_2026_08_11` to recognize the training data difference and is crowned the best model at the point of writing this guide.

*Finding: Performance is comparable but language discrepancy is much better (<5% down to <1%)*

Furthermore, threshold mining of hard negatives was tested in an attempt to decrease the number of false negatives used within the training loop. This technique augments the training data with "challenging distractors" (semantically-close-but-irrelevant documents) to force finer semantic discrimination. 

*Finding: Causes denser clusterings of documents which is not reflective of real "closeness" of relevant documents to a query*

### Best configuration
* Scope: from original experiment and stage 2 experiments
* Base model: `gte-multilingual-base` 
* Finetuning configuration: `2026_08_11/finetune-mnrl-mix-seq-sq1024`
