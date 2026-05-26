import logging
import pandas as pd

from utils.data_enrichment import DynamoDBEnricher
from utils.auxilliary_preprocessing import *
from utils.text_normalization import *

logger = logging.getLogger(__name__)

def process_data_text_only(df: pd.DataFrame) -> pd.DataFrame:
    affected_col_list = ['features_properties_keywords_en', 'features_properties_keywords_fr']
    logger.info(f"Converting values to lists for columns: {affected_col_list}")
    def convert_to_list(value):
        return [s.strip() for s in value.split(',')] if isinstance(value, str) else value if isinstance(value, list) else []

    for col in affected_col_list:
        df[col] = df[col].apply(convert_to_list)
    logger.info("Successfully converted")

    logger.info("Starting text normalization for semantic search")
    text_columns = ['features_properties_description']
    text_norm_config = {
            'en': {
                'section_header_to_remove': "References",
                'map_symbol_to_word': [('-->', ' then '),
                                    ('°', ' degrees '),
                                    ('\\n', '\n')],
            },
            'fr': {
                'section_header_to_remove': "Références",
                'map_symbol_to_word': [('->', ' ensuite '),
                                    ('°', ' degrés '),
                                    ('\\n', '\n')],
            }
        }
    df = normalize_text_columns(df, text_columns, text_norm_config)
    logger.info("Successfully completed text normalization")

    logger.info("Creating text for semantic search by concatenating title, description, and keywords")
    df['text_en'] = preprocess_records_into_text(df, languages=['en'])
    df['text_fr'] = preprocess_records_into_text(df, languages=['fr'])
    df['text_seq'] = preprocess_records_into_text(df, languages=['en', 'fr'], output_format='sequential')
    df['text_para'] = preprocess_records_into_text(df, languages=['en', 'fr'], output_format='parallel')
    logger.info("Successfully created text features")

    logger.info(f"Removing missing values in text features. Current shape before dropping: {df.shape}")
    df = df.dropna(subset=['text_en', 'text_fr'], how='any')
    logger.info(f"Successfully removed missing values. Current shape: {df.shape}")

    logger.info(f"Removing duplicates in dataset.")
    logger.info(f"Initial shape before deduplication: {df.shape}")
    df = deduplicate_data(df, subset_columns=['text_en', 'text_fr'])
    logger.info(f"Dropped duplicates based on text_en, and based on text_fr. Current shape: {df.shape}")

    return df

def process_data_e2e(df: pd.DataFrame, region: str, keep_eoCollections: bool) -> pd.DataFrame:
    logger.info("Starting processing job. Applying full preprocessing to entire dataset before text_normalization and train-test split.")
    # Select required columns based on geo api
    required_col_list = [
        'features_properties_id','features_geometry_coordinates','features_properties_title_en','features_properties_title_fr',
        'features_properties_description_en','features_properties_description_fr','features_properties_date_published_date',
        'features_properties_keywords_en','features_properties_keywords_fr','features_properties_options','features_properties_contact','features_properties_cited',
        'features_properties_topicCategory','features_properties_date_created_date',
        'features_properties_spatialRepresentation','features_properties_type',
        'features_properties_temporalExtent_begin','features_properties_temporalExtent_end',
        'features_properties_graphicOverview','features_properties_language','features_popularity',
        'features_properties_sourceSystemName','features_properties_eoCollection',
        'features_properties_eoFilters'
    ]
    df = df[required_col_list]
    logger.info(f"Selected required columns. Dataset shape: {df.shape}")

    logger.info("Starting data normalization")

    values_to_replace = {'Not Available; Indisponible': None}
    columns_to_replace = ['features_properties_date_published_date', 'features_properties_date_created_date']
    logger.info(f"Replacing '{list(values_to_replace.keys())[0]}' with None in columns: {columns_to_replace}")
    df[columns_to_replace] = df[columns_to_replace].replace(values_to_replace)
    logger.info(f"Successfully replaced")

    logger.info("Updating source system name from 'cgp' or None to 'geo-ca'")
    df.loc[
        (df['features_properties_sourceSystemName'] == 'cgp') |
        (df['features_properties_sourceSystemName'].isna()),
        'features_properties_sourceSystemName'
    ] = 'geo-ca'
    logger.info("Successfully updated")

    affected_col_list = ['features_properties_topicCategory', 'features_properties_keywords_en', 'features_properties_keywords_fr']
    logger.info(f"Converting values to lists for columns: {affected_col_list}")
    def convert_to_list(value):
        return [s.strip() for s in value.split(',')] if isinstance(value, str) else value if isinstance(value, list) else []

    for col in affected_col_list:
        df[col] = df[col].apply(convert_to_list)
    logger.info("Successfully converted")

    logger.info("Extracting unique descriptions for features_properties_type")
    df["features_properties_type"] = df["features_properties_options"].apply(extract_unique_desc)
    logger.info("Successfully extracted")

    logger.info("Starting feature engineering")

    logger.info("Creating temporalExtent feature")
    values_to_replace = {'Present': None, 'Not Available; Indisponible': None}
    columns_to_replace = ['features_properties_temporalExtent_begin', 'features_properties_temporalExtent_end']
    df[columns_to_replace] = df[columns_to_replace].replace(values_to_replace)

    df['temporalExtent'] = df.apply(lambda row: {'begin': row['features_properties_temporalExtent_begin'], 'end': row['features_properties_temporalExtent_end']}, axis=1)
    logger.info("Dropping original temporalExtent columns")
    df = df.drop(columns=['features_properties_temporalExtent_begin', 'features_properties_temporalExtent_end'])
    logger.info("Successfully created temporalExtent feature")
    
    logger.info("Extracting organization information into features_properties_org. Prioritizing cited over contact.")
    df["features_properties_org"] = [
        choose_org(cited, contact)
        for cited, contact in zip(df["features_properties_cited"], df["features_properties_contact"])
    ]
    logger.info("Successfully extracted organization information")

    logger.info("Creating features_properties_mappable based on features_properties_options")
    df["features_properties_mappable"] = df["features_properties_options"].apply(is_mappable_from_str)
    logger.info("Successfully created mappable feature")

    logger.info("Adding geo theme")
    theme_bins = {
        'boundaries': 'administration',
        'planningcadastre': 'administration',
        'location': 'administration',
        'transportation': 'administration',

        'economy': 'economy',
        'farming': 'economy',

        'biota': 'environment',
        'environment': 'environment',
        'elevation': 'environment',
        'inlandwaters': 'environment',
        'oceans': 'environment',
        'climatologymeteorologyatmosphere': 'environment',  # lowercase key

        'imagerybasemapsearthcover': 'imagery',
        'earthobservation;syntheticaperatureradar': 'imagery',

        'structure': 'infrastructure',
        'transport': 'infrastructure',
        'utilitiescommunication': 'infrastructure',

        'geoscientificinformation': 'science',  # lowercase key

        'health': 'society',
        'society': 'society',
        'intelligencemilitary': 'society',
    }
    df["features_properties_geo_theme"] = df["features_properties_topicCategory"].apply(map_topics_to_themes, theme_bins=theme_bins)
    logger.info("Successfully added geo theme via mapping")
    logger.info("Enriching geo theme with DynamoDB data")
    theme_db = DynamoDBEnricher(table_name='theme', region_name=region, item_key='tag')
    df = theme_db.merge_with_df(df, subset_values=['emergency', 'legal'])
    foundational_db = DynamoDBEnricher(table_name='foundational', region_name=region, item_key='loc')
    df = foundational_db.merge_with_df(df, subset_values=None)
    
    logger.info("Creating is_foundational feature based on geo theme")
    df["features_properties_foundational"] = df["features_properties_geo_theme"].apply(is_foundational)
    logger.info("Successfully created is_foundational feature")
    logger.info("Completed DynamoDB enrichment.")
    
    logger.info("Complete preprocessing disabled. Applying preprocessing for natural language processing and training preparation without feature engineering.")

    df = process_data_text_only(df)

    if not keep_eoCollections:
        logger.info(f"Removing records with defined eoCollection values. Current shape before removal: {df.shape}")
        df = df[df['features_properties_eoCollection'].isna()]
        logger.info(f"Successfully removed records with eoCollection values. Current shape: {df.shape}")
    
    logger.info("Processing job completed.")

    return df