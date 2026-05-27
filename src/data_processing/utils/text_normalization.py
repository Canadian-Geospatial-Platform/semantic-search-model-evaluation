import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

def mask_emails(text: str, mask: str = "[email]") -> str:
    '''
    Masks email addresses in the given text with a specified mask.

    Parameters:
    - text: The input text containing potential email addresses.
    - mask: The string to replace email addresses with. Default is "[email]".

    Returns:
    - str: The text with email addresses masked.
    '''
    if text is None:
        logger.warning("Invalid text being passed for email masking: {text}. Returning text without masking emails.")
        return text
    
    if not mask:
        logger.warning("Invalid mask being passed for email masking: {mask}. Returning text without masking emails.")
        return text

    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    masked_text = re.sub(email_pattern, mask, text)
    return masked_text

def mask_urls(text: str, mask: str = "[url]") -> str:
    '''
    Masks URLs in the given text with a specified mask.

    Parameters:
    - text: The input text containing potential URLs.
    - mask: The string to replace URLs with. Default is "[url]".

    Returns:
    - str: The text with URLs masked.
    '''
    if text is None:
        logger.warning("Invalid text being passed for url masking: {text}. Returning text without masking urls.")
        return text
    
    if not mask:
        logger.warning("Invalid mask being passed for url masking: {mask}. Returning text without masking urls.")
        return text
    
    # Match http/https or www urls, capture only the FQDN (domain.tld) and drop any path/query/fragment.
    url_pattern = r"(?:https?:\/\/|www\.|ftp\.)((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,63})(?:\:[0-9]+)?(?:[\/\?#][^\s()]*)?"
    masked_text = re.sub(url_pattern, lambda m: m.group(1) + ' ' + mask, text)
    return masked_text

def convert_tables_to_text(text: str) -> str:
    '''
    Converts tables in the given text to a plain text representation.

    Parameters:
    - text: The input text containing potential tables.

    Returns:
    - str: The text with tables converted to plain text.
    '''
    if text is None:
        logger.warning("Invalid text being passed for table conversion: {text}. Skipping.")
        return text
    
    # remove +--------------+ table borders
    text = re.sub(r'\s*\+\-+(?:\+\-+)?\+\s*', '\n', text)
    # print('after table borders:', repr(text))

    # pattern to replace escaped \ within tables e.g. | (word)?\ | becomes | (word)? |
    text = re.sub(r'\\ *\|', ' |', text)
    
    # Clearing empty rows
    text = re.sub(r'\n\| *(\| *)*\|\n', '\n', text)
    
    return text

def remove_section_and_onwards(text: str, section_header: str) -> str:
    '''
    Removes the specified section header and all text that follows it in the given text.

    Parameters:
    - text: The input text containing the section to be removed.
    - section_header: The header of the section to remove. The function will remove this header and all text that follows it.

    Returns:
    - str: The text with the specified section and all following text removed.
    '''
    if text is None:
        logger.warning("Invalid text being passed for section removal: {text}. Skipping.")
        return text
    
    if not section_header:
        logger.warning("Invalid mask being passed for section removal: {section_header}. Skipping.")
        return text
    
    pattern = re.escape(section_header) + r'.*'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    return cleaned_text.strip()

def remove_unnatural_punctuation(text: str) -> str:
    '''
    Removes unnatural punctuation from the given text. Unnatural punctuation is defines as multiple consecutive punctuation marks (e.g., "!!!", "??", ",,,") which are often not useful for text processing and can be reduced to a single punctuation mark. Note: consecutive single- or double-quotations or periods are not replaced as they may be intentional (e.g., for indicating inches or ellipses).

    Parameters:
    - text: The input text containing unnatural punctuation.

    Returns:
    - str: The text with unnatural punctuation removed.
    '''
    if text is None:
        logger.warning("Invalid text being passed for punctuation removal: {text}. Skipping.")
        return text
    
    # Replace multiple consecutive punctuation marks with a single one (e.g., "!!!" -> "!")
    cleaned_text = re.sub(r'([,!?;:\-\>\<\#\*\+\=\|"])\1{1,}', r'\1', text)

    return cleaned_text.strip()

def remove_extra_whitespace(text: str) -> str:
    '''
    Removes extra whitespace from the given text, including leading, trailing, and multiple consecutive spaces.

    Parameters:
    - text: The input text containing extra whitespace.

    Returns:
    - str: The text with extra whitespace removed.
    '''
    if text is None:
        logger.warning("Invalid text being passed for whitespace removal: {text}. Skipping.")
        return text
    
    cleaned_text = re.sub(r' {2,}', ' ', text)
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    cleaned_text = re.sub(r' *\n +\n', '\n', cleaned_text)

    return cleaned_text.strip()

def map_symbol_to_word(text: str, symbol: str, word: str) -> str:
    '''
    Maps a symbol to a word in the given text.

    Parameters:
    - text: The input text containing the symbol.
    - symbol: The symbol to be replaced.
    - word: The word to replace the symbol with.

    Returns:
    - str: The text with the symbol replaced by the word.
    '''
    if text is None or not symbol or word is None:
        logger.warning("Invalid text, symbol, or word being passed for symbol-to-word mapping:\nSymbol: {symbol}\nWord: {word}\nText: {text}. Skipping.")
        return text

    if len(symbol) < 1:
        logger.warning("Symbol to replace is empty. Returning original text.")
        return text
    
    return text.replace(symbol, word)

def apply_fn_and_count_affected_rows(df: pd.DataFrame, column: str, fn, **kwargs) -> pd.DataFrame:
    '''
    Applies a given function to a specified column in the DataFrame and counts the number of rows affected by the transformation.

    Parameters:
    - df: pandas DataFrame containing the records.
    - column: The name of the column to which the function will be applied.
    - fn: The function to apply to the specified column.
    - kwargs: Additional keyword arguments to pass to the function.

    Returns:
    - pandas DataFrame: DataFrame with the function applied to the specified column.
    '''
    initial_values = df[column].copy()
    df[column] = df[column].apply(fn, **kwargs)
    affected_rows = (initial_values != df[column]).sum()
    
    logger.info(f"Applied function {fn.__name__} to column {column}. Rows affected: {affected_rows}.")
    
    return df

def normalize_text_columns(df: pd.DataFrame, text_columns: list[str], language_config: dict) -> pd.DataFrame:
    '''
    Normalizes text columns in the DataFrame by converting removing unnatural punctuation, extra whitespace, emails, etc.

    Parameters:
    - df: pandas DataFrame containing the records.
    - text_columns: List of column names that contain text to be normalized.
    - language_config: Dictionary containing language-specific configuration for text normalization.
        E.g. language_config = {
            'en': {
                'section_header_to_remove': "References",
                'map_symbol_to_word': [('-->', ', then '),
                                    ('°', ' degrees ')],
            },
            'fr': {
                'section_header_to_remove': "Références",
                'map_symbol_to_word': [('-->', ', ensuite '),
                                    ('°', ' degrés ')],
            }
        }

    Returns:
    - pandas DataFrame: DataFrame with normalized text columns.
    '''
    logger.info(f"Normalizing text in columns: {text_columns}")

    if not all(col[-3] == '_' and col[-2:] in language_config for col in text_columns):
        logger.warning("Some columns in text_columns do not end with '_en' or '_fr'. Expanding column list to include all language variants.")
        # adding _en and _fr variants of any columns that don't already have them
        expanded_columns = []
        for col in text_columns:
            if not (col[-3] == '_' and col[-2:] in language_config):
                [expanded_columns.append(f"{col}_{lang}") for lang in language_config.keys()]
            else:
                expanded_columns.append(col)
    else:
        expanded_columns = text_columns

    for col in expanded_columns:
        new_col = f"{col[:-3]}_normalized_{col[-2:]}"  # create a new column for the normalized text
        df[new_col] = df[col]  # create a new column for the normalized text
        logger.info(f"Normalizing column {col} into new column {new_col}")
        section_header = language_config[col[-2:]].get('section_header_to_remove')
        if section_header:
            df = apply_fn_and_count_affected_rows(df, new_col, remove_section_and_onwards, section_header=section_header)
        df = apply_fn_and_count_affected_rows(df, new_col, mask_urls)
        df = apply_fn_and_count_affected_rows(df, new_col, mask_emails)
        for symbol, word in language_config[col[-2:]].get('map_symbol_to_word', []):
            df = apply_fn_and_count_affected_rows(df, new_col, map_symbol_to_word, symbol=symbol, word=word)
        df = apply_fn_and_count_affected_rows(df, new_col, convert_tables_to_text)
        df = apply_fn_and_count_affected_rows(df, new_col, remove_unnatural_punctuation)
        df = apply_fn_and_count_affected_rows(df, new_col, remove_extra_whitespace)
    
    return df


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
        selected_columns.extend([f'features_properties_title_{lang}', f'features_properties_description_normalized_{lang}', f'features_properties_keywords_{lang}'])
    
    # send warning if any of the selected columns are not in the DataFrame
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Columns {missing_columns} used for text processing are not found in the DataFrame.")
        return pd.Series(["\n".join(["\n\n\n" for _ in languages])] * len(df))

    df = df[selected_columns]
    if output_format == 'sequential':
        return df.apply(lambda x: "\n".join(
            [f"{x[f'features_properties_title_{lang}']}\n{keyword_prefixes[i]}{x[f'features_properties_keywords_{lang}'] if isinstance(x[f'features_properties_keywords_{lang}'], str) else ', '.join(x[f'features_properties_keywords_{lang}']) if isinstance(x[f'features_properties_keywords_{lang}'], list) else ''}\n{x[f'features_properties_description_normalized_{lang}']}" for i, lang in enumerate(languages)]), axis=1)
    elif output_format == 'parallel':
        return df.apply(lambda x: "\n".join( 
            [f"{x[f'features_properties_title_{lang}']}" for lang in languages] + 
            [f"{keyword_prefixes[i]}{x[f'features_properties_keywords_{lang}'] if isinstance(x[f'features_properties_keywords_{lang}'], str) else ', '.join(x[f'features_properties_keywords_{lang}']) if isinstance(x[f'features_properties_keywords_{lang}'], list) else ''}" for i, lang in enumerate(languages)] +
            [f"{x[f'features_properties_description_normalized_{lang}']}" for lang in languages]),
            axis=1)
    else:
        raise ValueError("Invalid processing method. Supported values are 'sequential' and 'parallel'.")