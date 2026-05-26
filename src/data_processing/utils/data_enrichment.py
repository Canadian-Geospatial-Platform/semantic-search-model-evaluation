import boto3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DynamoDBEnricher():
    def __init__(self, table_name: str, region_name:str, item_key:str):
        self.table_name = table_name
        self.item_key = item_key
        try:
            self.dynamodb = boto3.client('dynamodb', region_name=region_name)
        except Exception as e:
            logger.error(f"Error initializing DynamoDB client: {e}")

    def get_items(self):
        """Batch read all items from the DynamoDB table."""
        try:
            paginator = self.dynamodb.get_paginator('scan')
            items = []
            for page in paginator.paginate(TableName=self.table_name):
                items.extend(page['Items'])
            return items
        except Exception as e:
            logger.error(f"Error retrieving items from DynamoDB: {e}")
            return []

    def get_clean_items(self):
        """Get cleaned items with only the key and tag."""
        all_items = self.get_items()
        cleaned = [{'uuid': item['uuid']['S'], 'tag': item[self.item_key]['S']} for item in all_items]

        return pd.DataFrame(cleaned)

    @staticmethod
    def append_theme(existing, tag, subset=None):
        """
        Appends a theme tag to the existing list of themes if it's not already present.

        Parameters:
        - existing: The current list of theme tags
        - tag: The theme tag to append (e.g., 'emergency' or 'legal').
        - subset: List of theme tags that are allowed to be appended e.g. ['emergency', 'legal']. Default is None which means all tags are allowed.

        Returns:
        - List of theme tags with the new tag appended if it was not already present.
        """
        try:
            if not isinstance(existing, list):
                existing = []
            if isinstance(tag, str) and tag and (subset is None or tag in subset) and tag not in existing:
                existing.append(tag)
            return existing
        except Exception:
            return existing
    
    def merge_with_df(self, df: pd.DataFrame, subset_values: list[str]) -> pd.DataFrame:
        """Merge the cleaned DynamoDB items with the input DataFrame."""
        clean_items_df = self.get_clean_items()
        df = df.merge(clean_items_df, how='left', left_on='features_properties_id', right_on='uuid')
        df["features_properties_geo_theme"] = df.apply(
            lambda row: DynamoDBEnricher.append_theme(row["features_properties_geo_theme"], row["tag"], subset_values),
            axis=1
        )
        df.drop(columns=['uuid', 'tag'], inplace=True)
        return df
