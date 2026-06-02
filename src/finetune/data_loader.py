import logging

import pandas as pd
from datasets import Dataset
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

def extract_dataset(df, query_col, document_col, mix_languages=False):
    # if mix languages is True, query col becomes a prefix
    if mix_languages:
        return _build_expanded_dataset(df, anchor_col_prefix=query_col, doc_col=document_col)

    if query_col not in df.columns or document_col not in df.columns:
        logger.error(f"Requested {query_col} or {document_col} not found in dataset.")
        return

    subset_df = df[[query_col, document_col]].rename(columns={
        query_col: "anchor",
        document_col: "doc",
    }).sample(frac=1).reset_index(drop=True)

    return Dataset.from_pandas(subset_df, preserve_index=False)


def _build_expanded_dataset(df, anchor_col_prefix, doc_col):
    df = df[[f"{anchor_col_prefix}_en", f"{anchor_col_prefix}_fr", doc_col]]

    df_en = (
        df[[f"{anchor_col_prefix}_en", doc_col]]
        .rename(columns={f"{anchor_col_prefix}_en": "anchor", doc_col: "doc"})
        .reset_index(drop=True)
    )
    df_fr = (
        df[[f"{anchor_col_prefix}_fr", doc_col]]
        .rename(columns={f"{anchor_col_prefix}_fr": "anchor", doc_col: "doc"})
        .reset_index(drop=True)
    )

    combined = pd.concat([df_en, df_fr], ignore_index=True)
    return Dataset.from_pandas(combined.sample(frac=1), preserve_index=False)

