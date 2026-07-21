import os
import pandas as pd
from finetune.utils.ir_evaluate import extract_query_corpus_relevant_docs
from finetune.utils.extract_dataset import extract_dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import json

parent_dir = "./results/finetune_via_trainer/"
model_name = "gte-multilingual-base"
model_ft_config = "finetune-mnrl-mix-seq-sq512"
save_dir = f"{parent_dir}{model_name}-baseline/perf_on_benchmark/"
os.makedirs(save_dir, exist_ok=True)

# scp -J sve000@inter-nrcan-lp-gccloud.science.gc.ca ../geoca_real_eval_updated.xlsx sve000@inter-nrcan-ubuntu2204.science.gc.ca:/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch/data/benchmark/by_geo_theme/query2title_updated.xlsx
real_eval = pd.read_excel("./data/benchmark/by_geo_theme/query2title_updated.xlsx", sheet_name=None)
print(f"Obtained real queries jsonl")

# convert to df
df_list = []
for key, val in real_eval.items():
    df = real_eval[key]
    df['geo_theme'] = key
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
print(f"Converted real queries jsonl to pandas DataFrame: {full_df.shape}")

# selecting only ones with labels
full_df = full_df[~full_df['related documents'].isna()]
print(f"Removing entries without related documents: {full_df.shape}")

# obtaining corpus of documents
corpus_df = pd.read_parquet("./data/preprocessed-se/with_synthetic_queries/full_corpus.parquet")
print(f"Obtained document corpus: {corpus_df.shape}")

# link 'related_documents' titles with d_ids
query2doc = pd.merge(full_df, corpus_df, left_on="related documents", right_on="features_properties_title_en", how="left")
# dropping all entried with NA i.e. unable to find match in corpus
query2doc = query2doc.dropna(subset=["text_seq"])
print(f"Merged queries to real documents, dropped rows with empty values: {query2doc.shape}")

# removing documents in query2doc from corpus_df to avoid creating duplicates
corpus_df = corpus_df[~corpus_df['features_properties_id'].isin(query2doc['features_properties_id'])]
print(f"Fixing corpus to omit related documents from queries dataset: {corpus_df.shape}")

# extract queries and pass to irevaluator for predictions
main_ds = extract_dataset(query2doc, 'query', 'text_seq', mix_languages=False)
additional_corpus_ds = extract_dataset(corpus_df, "features_properties_id", 'text_seq', mix_languages=False)
queries, corpus, qid2did_mapping = extract_query_corpus_relevant_docs(main_ds, "anchor", "doc", [additional_corpus_ds])
print(f"Acquired queries, corpus, and query to document mapping for IR Evaluator")

# save ds
with open(f"{save_dir}/corpus.parquet", "w") as file:
    json.dump(corpus, file, indent=4)
with open(f"{save_dir}/queries.parquet", "w") as file:
    json.dump(queries, file, indent=4)
print(f"Saved copy of corpus and queries in {save_dir}")

# load evaluator
evaluator = InformationRetrievalEvaluator(
    queries=queries, #q_id:query
    corpus=corpus, #d_id:doc
    relevant_docs=qid2did_mapping, #q_id -> set(d_id)
    name=f"{model_name}_{model_ft_config}_performance_on_benchmark"
)
print(f"Loaded evaluator: {evaluator}")

# load model
model_path = f"{parent_dir}{model_name}-baseline/{model_ft_config}/{model_name}"
print(f"Acquiring model from path {model_path}")
isModelLocal = os.path.exists(model_path)
model = SentenceTransformer(model_path, trust_remote_code=isModelLocal, local_files_only=isModelLocal)
print(f"Loaded model: {model}")

# save predictions
evaluator(model, output_path=save_dir)

print("Done.")


# OUTPUT TRACE
# 
# (semantic-finetune) sve000@ib14gpu-001:/space/partner/nrcan/geobase/work/oatt/dev/semanticsearch$ python code/src/evaluate_performance_on_benchmark.py
# Obtained real queries jsonl
# Converted real queries jsonl to pandas DataFrame: (82, 3)
# Removing entries without related documents: (20, 3)
# Obtained document corpus: (9786, 14)
# Merged queries to real documents, dropped rows with empty values: (13, 17)
# Fixing corpus to omit related documents from queries dataset: (9775, 14)
# Acquired queries, corpus, and query to document mapping for IR Evaluator
# Saved copy of corpus and queries in ./results/finetune_via_trainer/gte-multilingual-base-baseline/perf_on_benchmark/
# Loaded evaluator: <sentence_transformers.evaluation.InformationRetrievalEvaluator.InformationRetrievalEvaluator object at 0x7ff4a3270790>
# Acquiring model from path ./results/finetune_via_trainer/gte-multilingual-base-baseline/finetune-mnrl-mix-seq-sq512/gte-multilingual-base
# Loaded model: SentenceTransformer(
#   (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'NewModel'})
#   (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
#   (2): Normalize()
# )
# Done.

# OUTPUT RESULTS for FTed gte-mulitilingual
# ===========================================
# cosine-Accuracy@1: 0.6923076923076923,
# cosine-Accuracy@3: 0.9230769230769231,
# cosine-Accuracy@5: 0.9230769230769231,
# cosine-Accuracy@10: 1.0,
# cosine-Precision@1: 0.6923076923076923,
# cosine-Recall@1: 0.6923076923076923,
# cosine-Precision@3: 0.3076923076923077,
# cosine-Recall@3: 0.9230769230769231,
# cosine-Precision@5: 0.18461538461538465,
# cosine-Recall@5: 0.9230769230769231,
# cosine-Precision@10: 0.10000000000000002,
# cosine-Recall@10: 1.0,
# cosine-MRR@10: 0.8044871794871795,
# cosine-NDCG@10: 0.8521018756868188,
# cosine-MAP@100: 0.8044871794871795

# OUTPUT RESULTS for FTed all-MiniLM-L6-v2-baseline
# ===========================================
# cosine-Accuracy@1: 0.5384615384615384,
# cosine-Accuracy@3: 0.8461538461538461,
# cosine-Accuracy@5: 0.9230769230769231,
# cosine-Accuracy@10: 0.9230769230769231,
# cosine-Precision@1: 0.5384615384615384,
# cosine-Recall@1: 0.5384615384615384,
# cosine-Precision@3: 0.28205128205128205,
# cosine-Recall@3: 0.8461538461538461,
# cosine-Precision@5: 0.18461538461538465,
# cosine-Recall@5: 0.9230769230769231,
# cosine-Precision@10: 0.09230769230769233,
# cosine-Recall@10: 0.9230769230769231,
# cosine-MRR@10: 0.6987179487179487,
# cosine-NDCG@10: 0.7556512168298283,
# cosine-MAP@100:0.7003901895206244