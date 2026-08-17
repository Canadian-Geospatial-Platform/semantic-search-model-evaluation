import os
import pandas as pd
from finetune.utils.ir_evaluate import extract_query_corpus_relevant_docs
from finetune.utils.extract_dataset import extract_dataset
from data_processing.utils.auxilliary_preprocessing import load_data_and_combine
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import json

parent_dir = "./results/finetune_via_trainer/"
models_to_compare = [
    {
        "alias": "gte-multilingual-base-baseline",
        "model_name": "gte-multilingual-base-baseline",
        "model_ft_config": "gte-multilingual-base"
    },
    {
        "alias": "gte-multilingual-base-finetuned",
        "model_name": "gte-multilingual-base-baseline",
        "model_ft_config": "finetune-mnrl-mix-seq-sq512/gte-multilingual-base"
    },
    {
        "alias": "gte-multilingual-base-finetuned-2026_08_11",
        "model_name": "gte-multilingual-base-baseline",
        "model_ft_config": "2026_08_11/finetune-mnrl-mix-seq-sq1024/gte-multilingual-base"
    },
]

# scp -J sve000@inter-nrcan-lp-gccloud.science.gc.ca ../geoca_real_eval_updated.xlsx sve000@inter-nrcan-ubuntu2204.science.gc.ca:/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/benchmark/by_geo_theme/query2title_survey_results.xlsx
benchmark_filepaths = [
    "./data/benchmark/by_geo_theme/query2title_updated.xlsx",
    "./data/benchmark/query2title_survey_results.xlsx",
]

corpus_filepath = "./data/preprocessed-se/2026_08_11/with_synthetic_queries/"
print(f"Obtaining corpus of documents from {corpus_filepath}")
corpus_df = load_data_and_combine(corpus_filepath)
print(f"Corpus shape: {corpus_df.shape}")

# [{model: model_alias, benchmark_file: benchmark file, recall@3: recall_value}, ...]
best_recall = []

for model_info in models_to_compare:
    model_name = model_info["model_name"]
    model_ft_config = model_info.get("model_ft_config", "")
    print(f"Evaluating model: {model_name} with config: {model_ft_config}")

    model_path = f"{parent_dir}{model_name}/{model_ft_config}"
    print(f"Acquiring model from path {model_path}")
    isModelLocal = os.path.exists(model_path)
    model = SentenceTransformer(model_path, trust_remote_code=isModelLocal, local_files_only=isModelLocal)
    print(f"Loaded model: {model}")

    # Set up save directory for results
    save_dir = f"{parent_dir}{model_name}/{model_ft_config}/perf_on_benchmark/"
    os.makedirs(save_dir, exist_ok=True)

    # Load benchmark data
    for benchmark_file in benchmark_filepaths:
        bm_name = os.path.basename(benchmark_file).split(".")[0]
        os.makedirs(save_dir+bm_name, exist_ok=True)
        print(f"Loading benchmark data from: {benchmark_file}")
        real_eval = pd.read_excel(benchmark_file, sheet_name=None)

        # Convert to DataFrame
        df_list = []
        for key, val in real_eval.items():
            df = real_eval[key]
            df['geo_theme'] = key
            df_list.append(df)

        full_df = pd.concat(df_list, ignore_index=True)
        print(f"Converted real queries to pandas DataFrame: {full_df.shape}")

        # Filter entries with labels
        print(f"Removing entries without related documents.")
        print(f"\tBefore: {full_df.shape}")
        full_df = full_df[~full_df['related documents'].isna()]
        print(f"\tAfter: {full_df.shape}")

        # Link 'related_documents' titles with d_ids
        query2doc = pd.merge(full_df, corpus_df, left_on="related documents", right_on="features_properties_title_en", how="left")
        query2doc = query2doc.dropna(subset=["text_seq"])
        print(f"Merged queries to real documents: {query2doc.shape}")

        # Remove documents in query2doc from corpus_df to avoid duplicates
        corpus_df_copy = corpus_df[~corpus_df['features_properties_id'].isin(query2doc['features_properties_id'])].copy(deep=True)
        print(f"Fixing corpus to omit related documents from queries dataset: {corpus_df_copy.shape}")

        main_ds = extract_dataset(query2doc, 'query', 'text_seq', mix_languages=False)
        additional_corpus_ds = extract_dataset(corpus_df_copy, "features_properties_id", 'text_seq', mix_languages=False)
        queries, corpus, qid2did_mapping = extract_query_corpus_relevant_docs(main_ds, "anchor", "doc", [additional_corpus_ds])
        print(f"Acquired queries, corpus, and query to document mapping for IR Evaluator")

        # save ds
        with open(f"{save_dir+bm_name}/corpus.json", "w") as file:
            json.dump(corpus, file, indent=4)
        # with open(f"{save_dir}/queries.parquet", "w") as file:
        #     json.dump(queries, file, indent=4)
        print(f"Saved copy of corpus in {save_dir}")

        # load evaluator
        evaluator_name = f"{model_info['alias']}_performance_on_benchmark"
        evaluator = InformationRetrievalEvaluator(
            queries=queries, #q_id:query
            corpus=corpus, #d_id:doc
            relevant_docs=qid2did_mapping, #q_id -> set(d_id).
            write_predictions=True,
            name=evaluator_name
        )
        print(f"Loaded evaluator: {evaluator}")

        # save predictions
        results = evaluator(model, output_path=save_dir+bm_name)

        best_recall.append({
            "model": model_info["alias"],
            "benchmark_file": benchmark_file,
            "recall@3": results[f"{evaluator_name}_cosine_recall@3"]
        })

print(f"Best recall@3 for each benchmark")
best_recall_df = pd.DataFrame(best_recall)
best_models = (
    best_recall_df.loc[
        best_recall_df.groupby("benchmark_file")["recall@3"].idxmax(),
        ["benchmark_file", "model", "recall@3"]
    ].reset_index(drop=True)
    )
print(best_models)