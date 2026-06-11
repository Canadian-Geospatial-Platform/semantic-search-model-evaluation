import pandas as pd
import numpy as np
import logging
import argparse
import os
from sentence_transformers import SentenceTransformer
import json
import torch
import gc

from finetune.utils.extract_dataset import extract_dataset
from finetune.utils.ir_evaluate import get_ir_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers.evaluation.InformationRetrievalEvaluator").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query2doc_dataset_path", type=str, required=True, help="Filepath to .parquet that includes query-to-relevant-document mapping")
    parser.add_argument("--additional_corpus_filepaths", type=str, default="[]", help="List of filepaths to .parquet that need to be included in corpus consideration")
    parser.add_argument("--document_col_names", type=str, default='["text_en", "text_seq", "text_para"]', help="Column in datasets to be used as document representation. Default is [\"text_en\", \"text_seq\", \"text_para\"]")
    parser.add_argument("--model_path", type=str, required=True, help="Name or local path to model to evaluate")
    parser.add_argument("--save_filedir", type=str, required=True, help="Filepath directory to save evaluation results and corpus embeddings to")
    
    parser.add_argument("--generate_corpus_embeddings", action="store_true", default=False, help="Column in datasets to be used as document representation")

    return parser.parse_args()

def run_performance_evaluation(model, query2doc_df, query_col, doc_col, additional_corpus_dfs, output_path=None, **ir_evaluator_kwargs):
    # extracting huggingface dataset, set mixLanguages to False to disable dataset expansion
    query2doc_dataset = extract_dataset(query2doc_df, query_col, doc_col, mix_languages=False)
    additional_corpus_datasets = [extract_dataset(df, "features_properties_id", doc_col, mix_languages=False) for df in additional_corpus_dfs]

    ir_evaluator = get_ir_evaluator(query2doc_dataset, "anchor", "doc", additional_corpus_datasets, name=f"{query_col}_{doc_col}", **ir_evaluator_kwargs)

    results = ir_evaluator(model, output_path=output_path)
    logger.info("Performance evaluation completed.")
    
    # clean up
    del ir_evaluator
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame([results])

def main(args):
    logger.info("Running performance evaluation script")
    
    os.makedirs(args.save_filedir, exist_ok=True)

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
    logger.info(f"Loading model: {args.model_path}")
    model = SentenceTransformer(args.model_path, trust_remote_code=True)

    # running performance evaluation
    
    logger.info(f"Loading document names: {args.document_col_names}")
    try:
        args.document_col_names = json.loads(args.document_col_names)
    except Exception:
        logger.warning(f"Failed to parse document_col_names. Expected to receive a stringified list. Setting to [].")
        args.document_col_names = []
    
    all_results_list = []
    for doc_col_name in args.document_col_names:
        logger.info(f"Running performance evaluation on {doc_col_name}.")
        logger.info("Query: EN")
        results_en_df = run_performance_evaluation(model, query2doc_df, 'query_en', doc_col_name, extra_dfs, output_path=args.save_filedir, write_csv=False, write_predictions=True)
        results_en_df['lang'] = 'en'
        results_en_df['document_repr'] = doc_col_name
        results_en_df = results_en_df.rename(columns=lambda col: col[col.find("cosine"):] if "cosine" in col else col)
        all_results_list.append(results_en_df)

        logger.info("Query: FR")
        results_fr_df = run_performance_evaluation(model, query2doc_df, 'query_fr', doc_col_name, extra_dfs, output_path=args.save_filedir, write_csv=False, write_predictions=True)
        results_fr_df['lang'] = 'fr'
        results_fr_df['document_repr'] = doc_col_name
        results_fr_df = results_fr_df.rename(columns=lambda col: col[col.find("cosine"):] if "cosine" in col else col)

        all_results_list.append(results_fr_df)

    if len(all_results_list) == 0:
        logger.info("No results computed, exitting.")
        return

    results_combined = pd.concat(all_results_list).reset_index()
    logger.info(f"Summary of results: {results_combined}")
                
    # saving results
    logger.info("Saving results...")

    results_path = os.path.join(args.save_filedir, 'results.csv')
    results_combined.to_csv(results_path)
    
    logger.info('Performance evaluation results saved successfully')

    if args.generate_corpus_embeddings:
        logger.info(f"Generating embeddings for corpus on {args.document_col_name}")
        full_corpus = pd.concat([query2doc_df] + extra_dfs)
        logger.info(f"Full corpus shape: {full_corpus.shape}")
        embeddings = model.encode(full_corpus[args.document_col_name].tolist())
        full_corpus[f"{args.document_col_name}_embeddings"] = list(embeddings)

        logger.info("Finished generating embeddings. Saving corpus...")
        corpus_path = os.path.join(args.save_filedir, 'document_corpus.parquet')
        full_corpus.to_parquet(corpus_path)  
        logger.info('Embeddings saved successfully')
    else:
        logger.info(f"Generate corpus embeddings is set to: {args.generate_corpus_embeddings}. Skipping.")

    
    logger.info(f"Evaluation complete.")

if __name__ == '__main__':
    args = parse_args()

    main(args)

