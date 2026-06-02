import argparse

import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, GISTEmbedLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
import logging
import sys
import os 

from data_loader import subset_extraction, InterleaveBatchSampler
from torch.utils.data import DataLoader

# Configure logging to file and console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_log_MPR.log"),
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
    parser.add_argument("--data_anchor_column", type=str, default='features_properties_title_en', help="Name of the column to use as anchor (query) in training. Default is 'features_properties_title_en'")
    parser.add_argument("--data_doc_column", type=str, default='text_en', help="Name of the column to use as document in training. Default is 'features_properties_text_en'")
    parser.add_argument("--data_mix_languages", action="store_true", default=False, help="If set, uses bilingual document expansion for training by treating the specified anchor column as a prefix and looking for corresponding columns with _en and _fr suffixes. The document column is expected to be the same for both languages. By default, this is set to False.")
    parser.add_argument("--data_sampler_mode", type=str, default='sequential', help="Mode for sampling batches during training. Default is 'sequential'. Other supported option is 'interleave' which alternates between English and French samples in each batch when mix_languages is enabled.")

    # training specific
    parser.add_argument("--train_num_epochs", type=int, default=2, help="Number of training epochs. Default is 2.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training. Default is 32.")
    parser.add_argument("--train_learning_rate", type=float, default=2e-5, help="Learning rate for training. Default is 2e-5.")
    parser.add_argument("--train_losstype", type=str, default="MNRL", help="Loss function to use for training. Options are 'MNRL' for MultipleNegativesRankingLoss and 'GIST' for GISTEmbedLoss. Default is 'MNRL'.")

    return parser.parse_args()

def extract_query_coprus_relevant_docs(df, query_col, doc_col):
    queries = {}
    corpus = {}
    relevant_docs = {}

    for idx, row in df.iterrows():
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

    # Check if the model save directory exists, create if not
    if not os.path.exists(args.model_save_directory):
        os.makedirs(args.model_save_directory)
        logger.info(f"Created model save directory: {args.model_save_directory}")
    
    logger.info(f"Loading data from {args.train_data_path} and {args.eval_data_path}")
    train_df = pd.read_parquet(args.train_data_path)
    eval_df = pd.read_parquet(args.eval_data_path)
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Eval data shape: {eval_df.shape}")

    anchor_col = args.data_anchor_column
    doc_col = args.data_doc_column

    if anchor_col not in train_df.columns or doc_col not in train_df.columns:
        logger.error(f"Anchor column '{anchor_col}' or document column '{doc_col}' not found in the training data.")
        return
    if anchor_col not in eval_df.columns or doc_col not in eval_df.columns:
        logger.error(f"Anchor column '{anchor_col}' or document column '{doc_col}' not found in the evaluation data.")
        return

    logger.info(f"Using anchor column: {anchor_col} and document column: {doc_col} for training and evaluation.")

    logger.info(f"Preparing training dataset with mix_languages set to {args.data_mix_languages}")
    train_dataset = subset_extraction(train_df, anchor_col, doc_col, mix_languages=args.data_mix_languages)
    eval_dataset = subset_extraction(eval_df, anchor_col, doc_col, mix_languages=args.data_mix_languages)

    train_sampler = InterleaveBatchSampler(
        lenngths=[train_df.shape[0]]*2 if args.data_mix_languages else [train_df.shape[0]],
        batch_size=args.train_batch_size, # sampler batch size must match training batch size to avoid multiple query-doc pairs sharing the same document in same batch (will be falsely treated as a negative in MNRL)
        mode=args.data_sampler_mode  # "interleave"or "sequential"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler)
    logger.info(f"Training dataset prepared with {len(train_dataset)} samples and batch size {args.train_batch_size}. Sampler mode: {args.data_sampler_mode}")

    logger.info(f"Initializing model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    logger.info("Base model loaded successfully")

    # Define output path for model saving
    model_output_path = os.path.join(args.model_save_directory, args.model_name.split('/')[-1])
    logger.info(f"Set up model output path: {model_output_path}")
    
    logger.info(f"Setting up training arguments with loss type: {args.train_losstype}, learning rate: {args.train_learning_rate}, batch size: {args.train_batch_size}, and number of epochs: {args.train_num_epochs}")
    # MultipleNegativesRankingLoss
    if args.train_losstype == "MNRL":
        train_loss = MultipleNegativesRankingLoss(model)
    elif args.train_losstype == "GIST":
        train_loss = GISTEmbedLoss(model)
    else:
        logger.error(f"Loss type, {args.train_losstype}, is not supported. Please use either 'MNRL' or 'GIST'")
        return

    args = SentenceTransformerTrainingArguments(
        output_dir = model_output_path,
        num_train_epochs = args.train_num_epochs,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.train_batch_size,
        learning_rate=args.train_learning_rate,

        logging_first_step=True,
        logging_strategy="epoch",
        log_level="info",
        eval_on_start=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
    )

    logger.info("Extracting queries, corpus, and relevant documents for evaluation")
    eval_queries, eval_corpus, eval_rel_docs = extract_query_coprus_relevant_docs(eval_dataset, anchor_col, doc_col)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=eval_queries, #q_id:query
        corpus=eval_corpus, #d_id:doc
        relevant_docs=eval_rel_docs, #q_id -> set(d_id)
    )
    ir_evaluator(model)

    logger.info("Setting up trainer with model, train and eval datasets (subset of only anchor and doc columns), loss function, and evaluator")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataloader=train_dataloader
        loss=train_loss,
        evaluator=ir_evaluator,
    )

    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return


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