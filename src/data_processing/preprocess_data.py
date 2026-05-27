import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.auxilliary_preprocessing import *
from utils.full_processing import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--keep-eoCollections", action="store_true", default=False,
                        help="If set, keeps records with defined eoCollection values. By default, these records are removed from the dataset as they are not relevant for semantic search training.")

    return parser.parse_args()


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("Splitting dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    return train_df, test_df

def main():
    logger.info("Starting preprocessing job")
    args = parse_args()

    # Load data
    df = load_data(args.input_data_dir)

    logger.info("Complete preprocessing disabled. Applying preprocessing for natural language processing and training preparation without feature engineering.")
    
    required_col_list = [
        'features_properties_id','features_properties_title_en','features_properties_title_fr',
        'features_properties_description_en','features_properties_description_fr',
        'features_properties_keywords_en','features_properties_keywords_fr','features_properties_eoCollection',
    ]
    df = df[required_col_list]
    logger.info(f"Selected required columns. Dataset shape: {df.shape}")
    
    df = process_data_text_only(df)

    if not args.keep_eoCollections:
        logger.info(f"Removing records with defined eoCollection values. Current shape before removal: {df.shape}")
        df = df[df['features_properties_eoCollection'].isna()]
        logger.info(f"Successfully removed records with eoCollection values. Current shape: {df.shape}")
    
    # splitting data
    logger.info(f"Splitting dataset into train and test sets based on ratio: {args.train_test_split_ratio}")
    train_df, test_df = split_data(df, args.train_test_split_ratio, args.random_state)
    logger.info(f"Successfully split data.")
    
    save_data(
        train_df,
        test_df,
        args.output_path,
    )

    logger.info("Preprocessing job completed.")

if __name__ == "__main__":
    main()