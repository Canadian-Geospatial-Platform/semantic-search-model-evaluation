import argparse

import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, GISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
import sys
import os 

from utils.extract_dataset import extract_dataset

LOGGER_OUTPUT = os.getenv("TRAINING_LOG_FILE", "./results") + "/logger_output.log"

# Configure logging to file and console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOGGER_OUTPUT),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data in parquet format")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to the evaluation data in parquet format")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model to fine-tune (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--model_save_directory", type=str, required=True, help="Directory to save the fine-tuned model")

    # configuration
    parser.add_argument("--data_anchor_column", type=str, default='query_en', help="Name of the column to use as anchor (query) in training. Default is 'query_en'")
    parser.add_argument("--data_doc_column", type=str, default='text_en', help="Name of the column to use as document in training. Default is 'features_properties_text_en'")
    parser.add_argument("--data_mix_languages", action="store_true", default=False, help="If set, uses bilingual document expansion for training by treating the specified anchor column as a prefix and looking for corresponding columns with _en and _fr suffixes. The document column is expected to be the same for both languages. By default, this is set to False.")
    parser.add_argument("--data_restrict_num_records_to", type=int, default=None, help="If not None, restricts number of records in training dataset to the number specified")
    
    # training specific
    parser.add_argument("--train_num_epochs", type=int, default=2, help="Number of training epochs. Default is 2.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training. Default is 32.")
    parser.add_argument("--train_learning_rate", type=float, default=2e-5, help="Learning rate for training. Default is 2e-5.")
    parser.add_argument("--train_losstype", type=str, default="MNRL", help="Loss function to use for training. Options are 'MNRL' for MultipleNegativesRankingLoss and 'GIST' for GISTEmbedLoss. Default is 'MNRL'.")

    return parser.parse_args()

def extract_query_coprus_relevant_docs(dataset, query_col, doc_col):
    '''
    Extracts queries, corpus, and relevant documents from the evaluation dataset for InformationRetrievalEvaluator.

    Args:
    - dataset: HuggingFace Dataset object containing the evaluation data
    - query_col: Name of the column to use as query (anchor)
    - doc_col: Name of the column to use as document

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

    return queries, corpus, relevant_docs

def main(args):
    logger.info(f"Starting fine-tuning for {args.model_name}")

    # 1. SETUP
    # Check if the model save directory exists, create if not
    if not os.path.exists(args.model_save_directory):
        os.makedirs(args.model_save_directory)
        logger.info(f"Created model save directory: {args.model_save_directory}")
    
    # Load train and eval datasets
    logger.info(f"Loading data from {args.train_data_path} and {args.eval_data_path}")
    train_df = pd.read_parquet(args.train_data_path)
    eval_df = pd.read_parquet(args.eval_data_path)
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Eval data shape: {eval_df.shape}")

    if args.data_restrict_num_records_to:
        logger.warning(f"Restricting number of records to {args.data_restrict_num_records_to}")
        train_df = train_df.iloc[:args.data_restrict_num_records_to]
        logger.info(f"Train data shape is now: {train_df.shape}")

    # Check if specified columns exist in the datasets
    anchor_col = args.data_anchor_column
    doc_col = args.data_doc_column

    if anchor_col not in train_df.columns or doc_col not in train_df.columns:
        logger.error(f"Anchor column '{anchor_col}' or document column '{doc_col}' not found in the training data.")
        return
    if anchor_col not in eval_df.columns or doc_col not in eval_df.columns:
        logger.error(f"Anchor column '{anchor_col}' or document column '{doc_col}' not found in the evaluation data.")
        return

    logger.info(f"Using anchor column: {anchor_col} and document column: {doc_col} for training and evaluation.")

    # 2. PREPARE DATASETS
    # Restrict datasets to anchor and doc columns and expand dataset if mix_languages is set
    logger.info(f"Preparing training dataset with mix_languages set to {args.data_mix_languages}")
    train_dataset = extract_dataset(train_df, anchor_col, doc_col, mix_languages=args.data_mix_languages)
    eval_dataset = extract_dataset(eval_df, anchor_col, doc_col, mix_languages=args.data_mix_languages)

    # 3. INITIALIZE MODEL
    logger.info(f"Initializing model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    logger.info("Base model loaded successfully")

    # Define output path for model saving
    model_output_path = os.path.join(args.model_save_directory, args.model_name.split('/')[-1])
    logger.info(f"Set up model output path: {model_output_path}")
    
    # 4. INITIALIZE TRAINING ARGUMENTS AND LOSS
    logger.info(f"Setting up training arguments with loss type: {args.train_losstype}, learning rate: {args.train_learning_rate}, batch size: {args.train_batch_size}, and number of epochs: {args.train_num_epochs}")
    # MultipleNegativesRankingLoss
    if args.train_losstype == "MNRL":
        train_loss = MultipleNegativesRankingLoss(model)
    elif args.train_losstype == "GIST":
        train_loss = GISTEmbedLoss(model)
    else:
        logger.error(f"Loss type, {args.train_losstype}, is not supported. Please use either 'MNRL' or 'GIST'")
        return

    training_args = SentenceTransformerTrainingArguments(
        output_dir = model_output_path,
        num_train_epochs = args.train_num_epochs,
        per_device_train_batch_size = args.train_batch_size,
        learning_rate=args.train_learning_rate,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_strategy="epoch",
        eval_on_start=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cosine_mrr@10",
        greater_is_better=True
    )

    # Configure InformationRetrievalEvaluator for the eval dataset
    logger.info("Extracting queries, corpus, and relevant documents for evaluation")
    eval_queries, eval_corpus, eval_rel_docs = extract_query_coprus_relevant_docs(eval_dataset, 'anchor', 'doc')
    ir_evaluator = InformationRetrievalEvaluator(
        queries=eval_queries, #q_id:query
        corpus=eval_corpus, #d_id:doc
        relevant_docs=eval_rel_docs, #q_id -> set(d_id)
    )
    
    logger.info("Setting up trainer with model, train and eval datasets (subset of only anchor and doc columns), loss function, and evaluator")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=ir_evaluator,
    )

    # 5. START TRAINING
    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return
    
    logger.info(f"Finished fine-tuning for {args.model_name}")

    # 6. SAVE MODEL
    try:
        trainer.save_model(model_output_path)
        logger.info(f"Best model saved successfully at {model_output_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    # 7. Save training logs
    training_logs_path = os.path.join(args.model_save_directory, "training_logs.csv")
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(training_logs_path, index=False)


if __name__ == '__main__':
    args = parse_args()

    main(args)
    
# Example usage:
#     python finetune_on_columns.py \
#         --train_data_path data/train.parquet \
#         --eval_data_path data/eval.parquet \
#         --model_name sentence-transformers/all-MiniLM-L6-v2 \
#         --model_save_directory ./models \
#         --anchor_column features_properties_title_en \
#         --doc_column features_properties_text_en \
#         --train_num_epochs 2 \
#         --train_batch_size 32 \
#         --train_learning_rate 2e-5 \
#         --train_losstype MNRL