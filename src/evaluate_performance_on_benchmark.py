import os
import pandas as pd
from finetune.utils.ir_evaluate import extract_query_corpus_relevant_docs
from finetune.utils.extract_dataset import extract_dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

parent_dir = "./results/finetune_via_trainer/"
model_name = "gte-multilingual-base-baseline"
model_ft_config = "finetune-mnrl-mix-seq-sq512"
save_dir = f"{parent_dir}{model_name}/perf_on_benchmark/"
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
query2doc = query2doc.dropna()
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
corpus.to_parquet(f"{save_dir}/corpus.parquet")
queries.to_parquet(f"{save_dir}/queries.parquet")
print(f"Saved copy of corpus and queries in {save_dir}")

# load evaluator
evaluator = InformationRetrievalEvaluator(
    queries=eval_queries, #q_id:query
    corpus=eval_corpus, #d_id:doc
    relevant_docs=eval_rel_docs, #q_id -> set(d_id)
    name=f"{query_col}_{doc_col}"
)
print(f"Loaded evaluator: {evaluator}")

# load model
model_path = f"{parent_dir}{model_name}/{model_ft_config}"
print(f"Acquiring model from path {model_path}")
isModelLocal = os.path.exists(args.model_path)
model = SentenceTransformer(args.model_path, trust_remote_code=isModelLocal, local_files_only=isModelLocal)
print(f"Loaded model: {model}")

# save predictions
evaluator(model, output_path=save_dir)

print("Done.")
