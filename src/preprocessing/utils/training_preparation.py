import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def preprocess_records_into_text(df, languages=['en', 'fr'], keyword_prefixes=['keywords:', 'mots-clés associés:'], output_format='sequential'):
    '''
    Combines text columns from the DataFrame to create a text representation of the records in the specified languages.

    Parameters:
    - df: pandas DataFrame containing the records.
    - languages: List of languages to include in the text representation. Supported values are 'en' for English and 'fr' for French. Default is [en, fr].
    - keyword_prefixes: List of prefixes to use for the keywords in each language. The length of this list should match the number of languages specified. Default is ['keywords:', 'mots-clés associés:'] for English and French respectively.
    - output_format: Method to combine the text columns. Supported values are 'sequential' for combining in the order of languages and 'parallel' for combining all languages together. Default is 'sequential'.

    Returns:
    - pandas DataFrame: DataFrame containing the text representations of the records.
    '''
    # confirm that the length of keyword_prefixes matches the number of languages
    if len(keyword_prefixes) != len(languages):
        raise ValueError("Length of keyword_prefixes must match the number of languages specified.")

    selected_columns = []
    for lang in languages:
        selected_columns.extend([f'features_properties_title_{lang}', f'features_properties_description_{lang}_normalized', f'features_properties_keywords_{lang}'])
    
    # send warning if any of the selected columns are not in the DataFrame
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Columns {missing_columns} used for text processing are not found in the DataFrame.")
        return pd.Series(["\n".join(["\n\n\n" for _ in languages])] * len(df))

    df = df[selected_columns]
    if output_format == 'sequential':
        return df.apply(lambda x: "\n".join([f"{x[f'features_properties_title_{lang}']}\n{keyword_prefixes[i]}{', '.join(x[f'features_properties_keywords_{lang}'])}\n{x[f'features_properties_description_{lang}_normalized']}" for i, lang in enumerate(languages)]), axis=1)
    elif output_format == 'parallel':
        return df.apply(lambda x: "\n".join( 
            [f"{x[f'features_properties_title_{lang}']}" for lang in languages] + 
            [f"{keyword_prefixes[i]}{', '.join(x[f'features_properties_keywords_{lang}'])}" for i, lang in enumerate(languages)] +
            [f"{x[f'features_properties_description_{lang}_normalized']}" for lang in languages]),
            axis=1)
    else:
        raise ValueError("Invalid processing method. Supported values are 'sequential' and 'parallel'.")


def deduplicate_data(df: pd.DataFrame, subset_columns: list[str]) -> pd.DataFrame:
    '''
    Removes duplicate records from the DataFrame based on the specified subset of columns.

    Parameters:
    - df: pandas DataFrame containing the records.
    - subset_columns: List of column names to consider independently for identifying duplicates. E.g., ['text_en', 'text_fr'] to drop duplicates based on English and French text columns separately.

    Returns:
    - pandas DataFrame: DataFrame with duplicates removed based on the specified subset of columns.
    '''
    for subset in subset_columns:
        initial_shape = df.shape
        df = df.drop_duplicates(subset=subset)
        logger.info(f"Dropped duplicates based on column {subset}. Initial shape: {initial_shape}, new shape: {df.shape}")
    
    return df

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