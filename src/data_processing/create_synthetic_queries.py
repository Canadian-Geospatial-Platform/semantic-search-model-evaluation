import logging
import argparse
import os

from utils.query_generator import *
from utils.auxilliary_preprocessing import load_data, save_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-dir", type=str, help="Path to the input data directory containing the raw datasets in parquet format")
    parser.add_argument("--output-data-dir", type=str, help="Path to the output directory where the preprocessed data will be saved in parquet")
    parser.add_argument("--base-column-name", type=str, default="text_en" help="Column in input data to use as basis for synthetic query. Default: 'text_en'")
    parser.add_argument("--new-column-name", type=str, default="query_en" help="New column name under which to save the synthetic queries. Default: 'query_en'")
    
    return parser.parse_args()

def main():
    logger.info("Starting create_syntetic_queries script")
    args = parse_args()
    logger.info(f"Configurations set to: {args}")

    # Load data
    dfs, dfs_names = load_data(args.input_data_dir)

    # Setting up save destination
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)
        logger.info(f"Created output save directory: {args.output_data_dir}")
    
    new_dfs = []
    for df, name in zip(dfs, dfs_names):
        logger.info(f"Processing {name} using {args.base_column_name} as basis...")
        df = create_queries(df, args.base_column_name, args.new_column_name)
        new_dfs.append(df)
    
    save_data(new_dfs, dfs_names, args.output_data_dir)

    logger.info("Script run completed.")

if __name__ == "__main__":
    main()