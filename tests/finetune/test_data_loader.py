import pytest
import pandas as pd
import numpy as np
from datasets import Dataset
from finetune.data_loader import (
    extract_dataset,
)


class TestExtractDataset:
    """Test the extract_dataset function."""

    @pytest.fixture
    def sample_df_basic(self):
        """Create a sample DataFrame with basic columns."""
        return pd.DataFrame({
            "query": ["question 1", "question 2", "question 3"],
            "document": ["doc 1", "doc 2", "doc 3"],
        })

    @pytest.fixture
    def sample_df_bilingual(self):
        """Create a sample DataFrame with bilingual columns."""
        return pd.DataFrame({
            "anchor_en": ["question 1", "question 2", "question 3"],
            "anchor_fr": ["question 1 FR", "question 2 FR", "question 3 FR"],
            "document": ["doc 1", "doc 2", "doc 3"],
        })

    def test_extract_dataset_basic_returns_dataset(self, sample_df_basic):
        """Test that extract_dataset returns a Dataset when mix_languages=False."""
        dataset = extract_dataset(
            sample_df_basic, "query", "document", mix_languages=False
        )
        assert isinstance(dataset, Dataset)

    def test_extract_dataset_bilingual_returns_dataset(
        self, sample_df_bilingual
    ):
        """Test that extract_dataset returns a Dataset when mix_languages=True."""
        dataset = extract_dataset(
            sample_df_bilingual, "anchor", "document", mix_languages=True
        )
        assert isinstance(dataset, Dataset)

    def test_extract_dataset_missing_column_returns_none(self, sample_df_basic):
        """Test that extract_dataset returns None when required columns are missing."""
        dataset = extract_dataset(
            sample_df_basic, "nonexistent", "document", mix_languages=False
        )
        assert dataset is None

    def test_extract_dataset_missing_document_column_returns_none(
        self, sample_df_basic
    ):
        """Test that extract_dataset returns None when document column is missing."""
        dataset = extract_dataset(
            sample_df_basic, "query", "nonexistent", mix_languages=False
        )
        assert dataset is None

    def test_extract_dataset_renames_columns_correctly(self, sample_df_basic):
        """Test that extract_dataset renames columns to 'anchor' and 'doc'."""
        dataset = extract_dataset(
            sample_df_basic, "query", "document", mix_languages=False
        )
        assert dataset.column_names == ["anchor", "doc"]
        assert "query" not in dataset.column_names
        assert "document" not in dataset.column_names

