from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

def extract_query_corpus_relevant_docs(dataset, query_col, doc_col, additional_corpus_datasets=[]):
    '''
    Extracts queries, corpus, and relevant documents from the evaluation dataset for InformationRetrievalEvaluator.

    Args:
    - dataset: HuggingFace Dataset object containing the evaluation data
    - query_col: Name of the column to use as query (anchor)
    - doc_col: Name of the column to use as document
    - additional_corpus_datasets: List of datasets to add to corpus via doc_col

    Returns:
    - queries: Dictionary mapping query IDs to query strings
    - corpus: Dictionary mapping document IDs to document strings
    - relevant_docs: Dictionary mapping query IDs to sets of relevant document IDs
    '''
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for idx in range(len(dataset)):
        row = dataset[idx]
        q_id = idx
        d_id = idx

        queries[q_id] = row[query_col]
        corpus[d_id] = row[doc_col]

        if q_id not in relevant_docs:
            relevant_docs[q_id] = set()
        relevant_docs[q_id].add(d_id)
    
    for additional_dataset in additional_corpus_datasets:
        for doc_row in additional_dataset:
            corpus[len(corpus)+1] = doc_row[doc_col]
    
    return queries, corpus, relevant_docs

def get_ir_evaluator(ds, anchor_col="anchor", doc_col="doc", additional_corpus_datasets=[], **ir_evaluator_kwargs):
    print(ds)
    eval_queries, eval_corpus, eval_rel_docs = extract_query_corpus_relevant_docs(ds, anchor_col, doc_col, additional_corpus_datasets)
    return InformationRetrievalEvaluator(
        queries=eval_queries, #q_id:query
        corpus=eval_corpus, #d_id:doc
        relevant_docs=eval_rel_docs, #q_id -> set(d_id)
        **ir_evaluator_kwargs,
    )


# class PerformanceEvaluator:
#     def __init__(corpus_df, corpus_doc_col_name, model_name):
#         self.corpus_df = corpus_df
#         self.corpus_doc_col_name = corpus_doc_col_name
#         self.model = SentenceTransformer(model_name)
    
#     def load_test_queries(self, query2doc_df, query_col_name, relevant_docs_col_name):
#         if self.relevant_docs_col_name != self.corpus_doc_col_name:
#             # relevant docs point to a portion of the doc, not the full doc
#             # mapping to full doc

#             # lookup tables
#             title2id_en_map = dict(zip(self.corpus_df["features_properties_title_en"], self.corpus_df["features_properties_id"]))
#             title2id_fr_map = dict(zip(self.corpus_df["features_properties_title_fr"], self.corpus_df["features_properties_id"]))
#             id2doc_map = dict(zip(self.corpus_df["features_properties_id"], self.corpus_df[self.corpus_doc_col_name]))      
            
#             def map_to_full_doc_repr(entry):
#                 if entry in title2id_en_map:
#                     return id2doc_map.get(title2id_en_map[entry], entry)
#                 elif entry in title2id_fr_map:
#                     return id2doc_map.get(title2id_fr_map[entry], entry)
#                 else:
#                     entry

#             query2doc_df[relevant_docs_col_name] = query2doc_df[relevant_docs_col_name].apply(map_to_full_doc_repr)
        
#         self.query2doc_df = query2doc_df
#         self.evaluator = get_ir_evaluator(self.query2doc_df, query_col_name, relevant_docs_col_name, additional_corpus_datasets=[self.corpus_df])
        
    
#     def run_performance_evaluation(self):
#         if not self.query2doc_df:
#             raise ValueError("Query and relevant document mapping has not been set yet. Please run load_test_queries() first.")

#         self.perf_results = self.evaluator(self.model)
        
#     def get_corpus_embeddings(self):
#         self.corpus_df['doc_embeddings'] = self.model.encode(self.corpus[self.corpus_doc_col_name])
#         return self.corpus_df