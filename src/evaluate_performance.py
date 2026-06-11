import pandas as pd
import numpy as np
import logging
import argparse
import os
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

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
    parser.add_argument("--document_col_name", type=str, default="text_en", help="Column in datasets to be used as document representation")
    parser.add_argument("--model_path", type=str, required=True, help="Name or local path to model to evaluate")
    parser.add_argument("--save_filedir", type=str, required=True, help="Filepath directory to save evaluation results and corpus embeddings to")
    
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials to run IR Evaluator. Default: 5")
    parser.add_argument("--generate_corpus_embeddings", action="store_true", default=False, help="Column in datasets to be used as document representation")

    return parser.parse_args()

def run_performance_evaluation(model, query2doc_df, query_col, doc_col, additional_corpus_dfs, num_trials, output_path=None, **ir_evaluator_kwargs):
    # extracting huggingface dataset, set mixLanguages to False to disable dataset expansion
    query2doc_dataset = extract_dataset(query2doc_df, query_col, doc_col, mix_languages=False)
    additional_corpus_datasets = [extract_dataset(df, "features_properties_id", doc_col, mix_languages=False) for df in additional_corpus_dfs]

    ir_evaluator = get_ir_evaluator(query2doc_dataset, "anchor", "doc", additional_corpus_datasets, **ir_evaluator_kwargs)

    # logger.info(f"Starting performance evaluation on model for {num_trials} trials")
    results_list = []
    for trial_i in tqdm(range(num_trials), desc="Trials"):
        ir_evaluator.write_predictions = (trial_i == 0) # only save predictions for first trial run
        ir_evaluator.predictions_file = f"predictions_trial_{trial_i}.jsonl"
        results = ir_evaluator(model, output_path=output_path)
        results_list.append(results)
    logger.info("Performance evaluation completed.")

    return pd.DataFrame(results_list)

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
    model = SentenceTransformer(args.model_path)

    # running performance evaluation
    logger.info("Running performance evaluation on English queries")
    results_en_df = run_performance_evaluation(model, query2doc_df, 'query_en', args.document_col_name, extra_dfs, args.num_trials, output_path=args.save_filedir)
    results_en_df['lang'] = 'en'
    
    logger.info("Running performance evaluation on French queries")
    results_fr_df = run_performance_evaluation(model, query2doc_df, 'query_fr', args.document_col_name, extra_dfs, args.num_trials, output_path=args.save_filedir)
    results_fr_df['lang'] = 'fr'

    results_combined = pd.concat([results_en_df, results_fr_df]).reset_index()
    logger.info("Performance results for both English and French queries acquired.")

    summary_en = results_combined[results_combined['lang'] == 'en']["cosine_mrr@10"].agg(
        mean="mean",
        stderr=lambda x: x.std(ddof=1) / np.sqrt(len(x))
    )
    logger.info(f"Summary query_en using {args.document_col_name} for doc representation: {summary_en}")

    summary_fr = results_combined[results_combined['lang'] == 'fr']["cosine_mrr@10"].agg(
        mean="mean",
        stdev=lambda x: x.std(ddof=1),
        stderr=lambda x: x.std(ddof=1) / np.sqrt(len(x))
    )
    logger.info(f"Summary query_fr using {args.document_col_name} for doc representation: {summary_fr}")
                
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



if __name__ == '__main__':
    args = parse_args()

    main(args)

