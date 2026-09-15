import json
import pandas as pd
import logging

from sklearn.model_selection import train_test_split

import glob
import os

logger = logging.getLogger(__name__)


def load_data(input_data_dir: str) -> pd.DataFrame:
    # obtaining all files saved in bucket
    files = glob.glob(f"{input_data_dir}/*.parquet")

    if not files:
        raise ValueError("No parquet files found. Check the input path.")

    logger.info(f"Found {len(files)} parquet files. Loading...")
    dfs = [pd.read_parquet(f) for f in files]
    
    logger.info(f"Loaded {len(dfs)} datasets.")
    return dfs, [os.path.basename(f) for f in files]

def load_data_and_combine(input_data_dir: str) -> pd.DataFrame:
    dfs, _ = load_data(input_data_dir)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Done. Combined dataset shape: {combined_df.shape}")
    return combined_df

def save_data(
    df_list: list[pd.DataFrame],
    filenames: list[str],
    output_path: str,
):
    logger.info("Saving processed datasets")

    if len(df_list) != len(filenames):
        logger.error("Length of df_list and filenames must be the same. Skipping save.")
        return

    os.makedirs(output_path, exist_ok=True)

    for df, filename in zip(df_list, filenames):
        file_path = os.path.join(output_path, filename)
        df.to_parquet(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    
    logger.info(f"{len(df_list)} datasets saved to {output_path}")

def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info("Splitting dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # further split test_df into eval and test sets 50-50
    eval_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Eval shape: {eval_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    return train_df, eval_df, test_df

def extract_unique_desc(options_str: str, language: str = 'en') -> list[str]:
    '''
    Extracts unique English descriptions from a stringified JSON of options.
    
    Parameters:
    - options_str: Stringified JSON containing the options data.
    - language: The language for which to extract descriptions. Default is 'en'.
    
    Returns:
    - List of str: Unique descriptions found first in every option in the specified language.
    '''
    try:
        options = json.loads(options_str) if isinstance(options_str, str) else options_str

        descs = []
        if isinstance(options, list):
            for opt in options:
                if isinstance(opt, dict):
                    desc = opt.get("description", {}).get(language)
                    if isinstance(desc, str):
                        # keep only text before first semicolon
                        descs.append(desc.split(";")[0].strip())
        elif isinstance(options, dict):
            desc = options.get("description", {}).get(language)
            if isinstance(desc, str):
                descs.append(desc.split(";")[0].strip())

        return list(set(descs))  # unique
    except Exception as e:
        logger.error(f"Error parsing options: {e}")
        return []

# organisation extraction
def get_second_segment(s):
    """Extract second segment (index 1) from semicolon-separated string."""
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(";")]
        if len(parts) >= 2:
            return parts[1]
    return None

def extract_org(contact_list):
    """Extract org dict with 'en' and 'fr' second segments."""
    if isinstance(contact_list, str):
        try:
            contact_list = json.loads(contact_list)
        except Exception:
            return {"en": None, "fr": None}

    if not isinstance(contact_list, list) or not contact_list:
        return {"en": None, "fr": None}

    # filter out empty/nulls
    contact_list = [c for c in contact_list if isinstance(c, dict) and c]
    if not contact_list:
        return {"en": None, "fr": None}

    org = contact_list[0].get("organisation", {})
    if not isinstance(org, dict):
        return {"en": None, "fr": None}

    return {
        "en": get_second_segment(org.get("en")),
        "fr": get_second_segment(org.get("fr"))
    }

def choose_org(cited, contact):
    """Use cited if valid, else fallback to contact."""
    if not(cited in [None, [], {}, [None], "[]", "[null]"] or (
        isinstance(cited, float) and pd.isna(cited)
    )):
        # try to extract org from cited
        org_from_cited = extract_org(cited)
        if org_from_cited.get("en") or org_from_cited.get("fr"):
            return org_from_cited
    # found no valid info in cited, fallback to contact
    return extract_org(contact)

def is_mappable_from_str(options_str):
    '''
    Extracts whether *esri* or *ogc* in protocol from options json

    Parameters:
    - options_str: Stringified JSON containing options information.

    Returns:
    - str(bool): 'true' if any protocol contains *esri* or *ogc*, 'false' otherwise
    '''
    try:
        options = json.loads(options_str) if isinstance(options_str, str) else options_str
        
        protocols = []
        if isinstance(options, list):
            protocols = [opt.get("protocol", "").strip().lower() for opt in options if isinstance(opt, dict)]
        elif isinstance(options, dict):
            protocol = options.get("protocol", "").strip().lower()
            protocols = [protocol]
            
        return str(any("esri" in p or "ogc" in p for p in protocols)).lower()
    except Exception as e:
        logger.error(f"Error parsing options: {e}")
        return str(False).lower()

def map_topics_to_themes(topic_list, theme_bins):
    '''
    Maps items in topic list string to theme via theme_bins

    Parameters:
    - topic_list: List containing topic categories.
    - theme_bins: Dict mapping lowercased topic category to theme

    Returns:
    - sorted list of str: all unique themes corresponding to the topic categories
    '''
    if not isinstance(topic_list, list):
        logging.warning(f"Expected topic_list to be a list, got {type(topic_list)} with value: {topic_list}. Returning empty theme list.")
        return []

    topics = [t.lower().strip() for t in topic_list]
    themes = {theme_bins.get(topic) for topic in topics if theme_bins.get(topic)}

    # Convert everything to strings explicitly (just in case)
    return sorted(set([str(theme) for theme in themes]))


def is_foundational(theme):
    '''
    Checks if the theme is foundational based on the presence of "foundational" in the theme tags.
    Parameters:
    - theme: The theme tags which can be a stringified JSON, a list, or a plain string.
    Returns:
    - str(bool): 'true' if "foundational" is found in the theme tags, 'false' otherwise.
    '''
    try:        
        # If it's a JSON string, parse it
        if isinstance(theme, str):
            try:
                theme_parsed = json.loads(theme)
            except Exception:
                # fallback: treat as plain string
                theme_parsed = theme
        else:
            theme_parsed = theme

        # Normalize to a list
        if isinstance(theme_parsed, list):
            themes = [str(t).strip().lower() for t in theme_parsed if t is not None]
        elif isinstance(theme_parsed, str):
            themes = [theme_parsed.strip().lower()]
        else:
            themes = []

        return str("foundational" in themes).lower()
    except Exception as e:
        logger.error(f"Error parsing theme: {e}")
        return str(False).lower()

def deduplicate_data(df: pd.DataFrame, subset_columns: list[str]) -> pd.DataFrame:
    '''
    Removes duplicate records from the DataFrame based on any of the specified subset of columns.

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