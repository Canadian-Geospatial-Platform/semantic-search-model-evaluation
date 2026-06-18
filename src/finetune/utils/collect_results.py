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

    df_model_summary_list = [df_baseline]
    for config in training_configs:
        config_path = os.path.join(workdir, config, performance_path)
        df_config = pd.read_csv(config_path)
        df_config["config"] = config
        df_model_summary_list.append(df_config)
    
    # 4. Combine into single df
    df_model_summary_df = pd.concat(df_model_summary_list, ignore_index=True)

    # 5. Save.
    df_model_summary_df.to_csv(f"{workdir}/test_summary.csv")

if __name__ == "__main__":

    # defaults
    workdir = "/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch"
    training_configs = ['finetune-mnrl', 'finetune-mnrl-b64']
    performance_path = "performance_evaluator/results.csv"

    generate_model_summary(workdir, training_configs, performance_path)
