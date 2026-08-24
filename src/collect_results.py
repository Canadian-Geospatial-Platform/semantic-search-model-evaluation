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
import json

def generate_model_summary(workdir, training_configs, performance_path):
    # 1. Get perf eval for baseline
    path = os.path.join(workdir, performance_path)
    if not os.path.exists(path):
        print(f"Attempted to retreive results from {path}. But path does not exist. Skipping")
        return
    
    df_baseline = pd.read_csv(path)
    df_baseline["config"] = "baseline"

    model_summary_list = [df_baseline]
    for ft_config in training_configs:
        ft_config_path = os.path.join(workdir, ft_config, performance_path)
        if not os.path.exists(ft_config_path):
            print(f"Attempted to retreive results from {ft_config_path}. But path does not exist. Skipping")
            continue
        df_ft_config = pd.read_csv(ft_config_path)
        df_ft_config["config"] = ft_config
        model_summary_list.append(df_ft_config)
    
    model_summary_df = pd.concat(model_summary_list, ignore_index=True)

    model_summary_df.to_csv(f"{workdir}/test_summary.csv", index=False)

    return model_summary_df

def run_generate_model_summary(model, training_configs):
    workdir = f"/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/{model}-baseline"
    performance_path = "performance_evaluation/results.csv"

    df = generate_model_summary(workdir, training_configs, performance_path)
    if df is None:
        return df
    
    df['model'] = model

    return df


def main(config_path, save_path):
    print(f"Loading configurations from {config_path}")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    all_results = []

    for model_config in config:
        if not isinstance(model_config['training_configs'], list) or len(model_config['training_configs']) == 0:
            print(f"Skipping model: {model_config['model']}. Invalid or empty training_configs to capture.")
            continue
        
        df = run_generate_model_summary(model_config['model'], model_config['training_configs'])
        if df is not None:
            all_results.append(df)


    all_results_df = pd.concat(all_results).reset_index()
    print(f"Writing final summary to {save_path}")
    all_results_df.to_csv(save_path, index=False)

    print("Done")

if __name__ == "__main__":
    # config_path = "/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/code/src/finetune/configs/finetune-via-trainer-best-eval-se.json"
    # save_path = "/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/summary_on_test.csv"
    
    config_path = "/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/code/src/finetune/configs/finetune-via-trainer-best-eval-se-2026_08_11.json"
    save_path = "/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/results/finetune_via_trainer/summary-2026_08_11_on_test.csv"

    main(config_path, save_path)
    