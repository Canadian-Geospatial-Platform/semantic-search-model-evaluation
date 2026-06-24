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
    eval_queries, eval_corpus, eval_rel_docs = extract_query_corpus_relevant_docs(ds, anchor_col, doc_col, additional_corpus_datasets)
    return InformationRetrievalEvaluator(
        queries=eval_queries, #q_id:query
        corpus=eval_corpus, #d_id:doc
        relevant_docs=eval_rel_docs, #q_id -> set(d_id)
        **ir_evaluator_kwargs,
    )