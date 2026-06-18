# Crawl given directory that has the following form:
# parent dir/
#   model/
#       finetune-config1/
#           performance_evaluation
#               results.csv
#       finetune-config2/
#       performance_evaluation/
#           results.csv
import pandas as pd
import os

def generate_model_summary(workdir, training_configs, performance_path):
    # 1. Get perf eval for baseline
    path = os.path.join(workdir, performance_path)
    df_baseline = pd.read_csv(path)
    df_baseline["config"] = "baseline"

    model_summary_list = [df_baseline]
    for config in training_configs:
        config_path = os.path.join(workdir, config, performance_path)
        df_config = pd.read_csv(config_path)
        df_config["config"] = config
        model_summary_list.append(df_config)
    
    # 4. Combine into single df
    model_summary_df = pd.concat(model_summary_list, ignore_index=True)

    # 5. Save.
    model_summary_df.to_csv(f"{workdir}/test_summary.csv")

    return model_summary_df

if __name__ == "__main__":
    all_results = []

    # defaults
    model = "all-mpnet-base-v2"
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    training_configs = ['finetune-mnrl-b64']
    performance_path = "performance_evaluation/results.csv"

    df = generate_model_summary(workdir, training_configs, performance_path)
    df['model'] = model
    all_results.append(df)

    model = "paraphrase-multilingual-mpnet-base-v2"
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    training_configs = ['finetune-mnrl', 'finetune-mnrl-mix-seq', 'finetune-gist-mix-para']

    df = generate_model_summary(workdir, training_configs, performance_path)
    df['model'] = model
    all_results.append(df)

    model = "all-MiniLM-L12-v2"
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    training_configs = ['finetune-mnrl']

    df = generate_model_summary(workdir, training_configs, performance_path)
    df['model'] = model
    all_results.append(df)

    model = "paraphrase-multilingual-MiniLM-L12-v2"
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    training_configs = ['finetune-mnrl', 'finetune-mnrl-mix-seq-ms2048', 'finetune-gist-mix-para-b64ms1024']

    df = generate_model_summary(workdir, training_configs, performance_path)
    df['model'] = model
    all_results.append(df)

    model = "gte-base-en-v1.5"
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    training_configs = ['finetune-mnrl-b16']

    df = generate_model_summary(workdir, training_configs, performance_path)
    df['model'] = model
    all_results.append(df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv("/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/summary_on_test.csv")
    print("Done")

