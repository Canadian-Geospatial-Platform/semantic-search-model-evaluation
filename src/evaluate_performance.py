import pandas as pd
import logging
import argparse
import os
from sentence_transformers import SentenceTransformer
import json

from finetune.utils.extract_dataset import extract_dataset
from finetune.utils.ir_evaluate import get_ir_evaluator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query2doc_dataset_path", type=str, required=True, help="Filepath to .parquet that includes query-to-relevant-document mapping")
    parser.add_argument("--additional_corpus_filepaths", type=str, default="[]", help="List of filepaths to .parquet that need to be included in corpus consideration")
    parser.add_argument("--document_col_name", type=str, default="text_en", help="Column in datasets to be used as document representation")

    parser.add_argument("--model_name", type=str, required=True, help="Name or local path to model to evaluate")

    parser.add_argument("--save_filedir", type=str, required=True, help="Filepath directory to save evaluation results and corpus embeddings to")

    return parser.parse_args()

def run_performance_evaluation(model, query2doc_df, query_col, doc_col, additional_corpus_dfs, **ir_evaluator_kwargs):
    # extracting huggingface dataset, set mixLanguages to False to disable dataset expansion
    query2doc_dataset = extract_dataset(query2doc_df, query_col, doc_col, mix_languages=False)
    additional_corpus_datasets = [extract_dataset(df, "features_properties_id", doc_col, mix_languages=False) for df in additional_corpus_dfs]

    ir_evaluator = get_ir_evaluator(query2doc_dataset, "anchor", "doc", additional_corpus_datasets, **ir_evaluator_kwargs)

    logger.info("Starting performance evaluation on model")
    results = ir_evaluator(model)
    logger.info("Performance evaluation completed.")

    return results

def main(args):
    logger.info("Running performance evaluation script")

    logger.info("Loading datasets")
    if not os.path.isfile(args.query2doc_dataset_path):
        logger.error(f"Unable to locate query-to-document dataset at {args.query2doc_dataset_path}.")
        return
    
    query2doc_df = pd.read_parquet(args.query2doc_dataset_path)

    try:
        args.additional_corpus_filepaths = json.loads(args.additional_corpus_filepaths)
    except Exception:
        logger.warning(f"Failed to parse additional corpus filepaths. Expected to receive a stringified list. Setting to [].")
        args.additional_corpus_filepaths = []
    
    extra_dfs=[]    
    for filepath in args.additional_corpus_filepaths:
        if not os.path.isfile(filepath):
            logger.warning(f"Unable to locate additional corpus dataset at {filepath}. Skipping...")
            continue
        extra_dfs.append(pd.read_parquet(filepath))
    
    # loading model
    logger.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # running performance evaluation
    logger.info("Running performance evaluation on English queries")
    results_en = run_performance_evaluation(model, query2doc_df, 'query_en', args.document_col_name, extra_dfs, write_csv=False)
    logger.info("Running performance evaluation on French queries")
    results_fr = run_performance_evaluation(model, query2doc_df, 'query_fr', args.document_col_name, extra_dfs, write_csv=False)

    results = {
        'en': results_en,
        'fr': results_fr
    }
    logger.info("Performance results for both English and French queries acquired. Saving results...")

    # saving results
    os.makedirs(args.save_filedir, exist_ok=True)

    results_path = os.path.join(args.save_filedir, 'results.json')
    with open(results_path, 'w') as file:
        json.dump(results, file)
    
    logger.info('Performance evaluation results saved successfully')

    logger.info(f"Generating embeddings for corpus on {args.document_col_name}")
    full_corpus = pd.concat([query2doc_df] + extra_dfs)
    full_corpus[f"{args.document_col_name}_embeddings"] = model.encode(full_corpus[args.document_col_name].tolist())

    logger.info("Finished generating embeddings. Saving corpus...")
    corpus_path = os.path.join(args.save_filedir, 'document_corpus.parquet')
    full_corpus.to_parquet(corpus_path)  
    logger.info('Embeddings saved successfully')


if __name__ == '__main__':
    args = parse_args()

    main(args)

