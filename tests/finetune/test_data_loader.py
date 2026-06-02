import pytest
import pandas as pd
import numpy as np
from finetune.data_loader import (
    subset_extraction,
    BasicDataset,
    ExpandedDataset,
    InterleaveBatchSampler,
)


class TestSubsetExtraction:
    """Test the subset_extraction function."""

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

    def test_subset_extraction_basic_returns_basic_dataset(self, sample_df_basic):
        """Test that subset_extraction returns BasicDataset when mix_languages=False."""
        dataset = subset_extraction(
            sample_df_basic, "query", "document", mix_languages=False
        )
        assert isinstance(dataset, BasicDataset)

    def test_subset_extraction_bilingual_returns_expanded_dataset(
        self, sample_df_bilingual
    ):
        """Test that subset_extraction returns ExpandedDataset when mix_languages=True."""
        dataset = subset_extraction(
            sample_df_bilingual, "anchor", "document", mix_languages=True
        )
        assert isinstance(dataset, ExpandedDataset)

    def test_subset_extraction_missing_column_returns_none(self, sample_df_basic):
        """Test that subset_extraction returns None when required columns are missing."""
        dataset = subset_extraction(
            sample_df_basic, "nonexistent", "document", mix_languages=False
        )
        assert dataset is None

    def test_subset_extraction_missing_document_column_returns_none(
        self, sample_df_basic
    ):
        """Test that subset_extraction returns None when document column is missing."""
        dataset = subset_extraction(
            sample_df_basic, "query", "nonexistent", mix_languages=False
        )
        assert dataset is None

    def test_subset_extraction_renames_columns_correctly(self, sample_df_basic):
        """Test that subset_extraction renames columns to 'anchor' and 'doc'."""
        dataset = subset_extraction(
            sample_df_basic, "query", "document", mix_languages=False
        )
        assert "anchor" in dataset.df.columns
        assert "doc" in dataset.df.columns
        assert "query" not in dataset.df.columns
        assert "document" not in dataset.df.columns


class TestBasicDataset:
    """Test the BasicDataset class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            "anchor": ["text 1", "text 2", "text 3", "text 4"],
            "doc": ["doc 1", "doc 2", "doc 3", "doc 4"],
        })

    @pytest.fixture
    def basic_dataset(self, sample_df):
        """Create a BasicDataset instance."""
        return BasicDataset(sample_df)

    def test_basic_dataset_initialization(self, sample_df):
        """Test that BasicDataset initializes correctly."""
        dataset = BasicDataset(sample_df)
        assert hasattr(dataset, "df")
        assert len(dataset.df) == len(sample_df)

    def test_basic_dataset_shuffles_data(self, sample_df):
        """Test that BasicDataset shuffles the data during initialization."""
        # Set random seed for reproducibility
        np.random.seed(42)
        df_copy = sample_df.copy()
        
        dataset = BasicDataset(df_copy)
        # Check that df has been reset (index should be 0-based)
        assert list(dataset.df.index) == list(range(len(dataset.df)))

    def test_basic_dataset_length(self, basic_dataset, sample_df):
        """Test that BasicDataset returns correct length."""
        assert len(basic_dataset) == len(sample_df)

    def test_basic_dataset_getitem_returns_series(self, basic_dataset):
        """Test that __getitem__ returns a pandas Series."""
        item = basic_dataset[0]
        assert isinstance(item, pd.Series)

    def test_basic_dataset_getitem_contains_anchor_and_doc(self, basic_dataset):
        """Test that __getitem__ returns rows with correct columns."""
        item = basic_dataset[0]
        assert "anchor" in item.index
        assert "doc" in item.index

    def test_basic_dataset_getitem_valid_indices(self, basic_dataset):
        """Test that __getitem__ works for all valid indices."""
        for i in range(len(basic_dataset)):
            item = basic_dataset[i]
            assert isinstance(item, pd.Series)
            assert len(item) == 2  # anchor and doc

    def test_basic_dataset_getitem_out_of_range_raises_error(self, basic_dataset):
        """Test that __getitem__ raises error for out-of-range index."""
        with pytest.raises(IndexError):
            _ = basic_dataset[len(basic_dataset) + 10]


class TestExpandedDataset:
    """Test the ExpandedDataset class."""

    @pytest.fixture
    def sample_df_bilingual(self):
        """Create a sample bilingual DataFrame."""
        return pd.DataFrame({
            "query_en": ["q1", "q2", "q3"],
            "query_fr": ["q1_fr", "q2_fr", "q3_fr"],
            "document": ["d1", "d2", "d3"],
        })

    @pytest.fixture
    def expanded_dataset(self, sample_df_bilingual):
        """Create an ExpandedDataset instance."""
        return ExpandedDataset(sample_df_bilingual, "query", "document")

    def test_expanded_dataset_initialization(self, sample_df_bilingual):
        """Test that ExpandedDataset initializes correctly."""
        dataset = ExpandedDataset(sample_df_bilingual, "query", "document")
        assert hasattr(dataset, "df_en")
        assert hasattr(dataset, "df_fr")
        assert hasattr(dataset, "len_en")
        assert hasattr(dataset, "len_fr")

    def test_expanded_dataset_selects_correct_columns(self, sample_df_bilingual):
        """Test that ExpandedDataset selects the correct columns."""
        dataset = ExpandedDataset(sample_df_bilingual, "query", "document")
        assert "query" in dataset.df_en.columns
        assert "query_en" not in dataset.df_en.columns
        assert "query_fr" not in dataset.df_en.columns
        assert "doc" in dataset.df_en.columns
        assert "document" not in dataset.df_en.columns
        assert "query" in dataset.df_fr.columns
        assert "query_en" not in dataset.df_fr.columns
        assert "query_fr" not in dataset.df_fr.columns
        assert "doc" in dataset.df_fr.columns
        assert "document" not in dataset.df_fr.columns

    def test_expanded_dataset_length_equals_sum(self, sample_df_bilingual):
        """Test that __len__ returns sum of en and fr lengths."""
        dataset = ExpandedDataset(sample_df_bilingual, "query", "document")
        assert len(dataset) == dataset.len_en + dataset.len_fr
        assert len(dataset) == 6  # 3 + 3

    def test_expanded_dataset_getitem_en_rows(self, expanded_dataset):
        """Test that __getitem__ returns en rows for indices < len_en."""
        item = expanded_dataset[0]
        assert isinstance(item, pd.Series)
        assert item['query'] == "q1"

    def test_expanded_dataset_getitem_fr_rows(self, expanded_dataset):
        """Test that __getitem__ returns fr rows for indices >= len_en."""
        len_en = expanded_dataset.len_en
        item = expanded_dataset[len_en]
        assert isinstance(item, pd.Series)
        assert item['query'] == "q1_fr"

    def test_expanded_dataset_getitem_out_of_range_raises_error(
        self, expanded_dataset
    ):
        """Test that __getitem__ raises error for out-of-range index."""
        with pytest.raises(IndexError):
            _ = expanded_dataset[len(expanded_dataset) + 10]

    def test_expanded_dataset_with_different_sizes(self):
        """Test ExpandedDataset with different DataFrame sizes."""
        df = pd.DataFrame({
            "anchor_en": ["q1", "q2"],
            "anchor_fr": ["q1_fr", "q2_fr"],
            "doc": ["d1", "d2"],
        })
        dataset = ExpandedDataset(df, "anchor", "doc")
        assert dataset.len_en == 2
        assert dataset.len_fr == 2
        assert len(dataset) == 4


class TestInterleaveBatchSampler:
    """Test the InterleaveBatchSampler class."""

    @pytest.fixture
    def sampler_single_dataset(self):
        """Create a sampler with a single dataset."""
        return InterleaveBatchSampler([10], batch_size=3, mode="sequential")

    @pytest.fixture
    def sampler_two_datasets_sequential(self):
        """Create a sampler with two datasets in sequential mode."""
        return InterleaveBatchSampler([5, 5], batch_size=2, mode="sequential")

    @pytest.fixture
    def sampler_two_datasets_interleave(self):
        """Create a sampler with two datasets in interleave mode."""
        return InterleaveBatchSampler([5, 5], batch_size=2, mode="interleave")

    def test_sampler_initialization(self):
        """Test that InterleaveBatchSampler initializes correctly."""
        sampler = InterleaveBatchSampler([5, 5], batch_size=2, mode="sequential")
        assert sampler.lengths == [5, 5]
        assert sampler.batch_size == 2
        assert sampler.mode == "sequential"

    def test_sampler_builds_index_groups(self):
        """Test that InterleaveBatchSampler builds correct index groups."""
        sampler = InterleaveBatchSampler([5, 3], batch_size=2, mode="sequential")
        assert len(sampler.index_groups) == 2
        assert sampler.index_groups[0] == [0, 1, 2, 3, 4]
        assert sampler.index_groups[1] == [5, 6, 7]

    def test_sampler_length_single_dataset(self, sampler_single_dataset):
        """Test that __len__ returns correct length for single dataset."""
        assert len(sampler_single_dataset) == 10

    def test_sampler_length_multiple_datasets(self, sampler_two_datasets_sequential):
        """Test that __len__ returns sum of all dataset lengths."""
        assert len(sampler_two_datasets_sequential) == 10

    def test_sampler_iter_single_dataset_returns_all_indices(
        self, sampler_single_dataset
    ):
        """Test that iterator returns all indices for single dataset."""
        indices = list(sampler_single_dataset)
        assert indices == list(range(10))

    def test_sampler_iter_sequential_mode_all_en_then_fr(
        self, sampler_two_datasets_sequential
    ):
        """Test that sequential mode yields all EN indices then all FR indices."""
        indices = list(sampler_two_datasets_sequential)
        # First 5 should be from first dataset (0-4)
        assert indices[:5] == [0, 1, 2, 3, 4]
        # Next 5 should be from second dataset (5-9)
        assert indices[5:] == [5, 6, 7, 8, 9]

    def test_sampler_iter_interleave_mode_batches_alternately(
        self, sampler_two_datasets_interleave
    ):
        """Test that interleave mode yields batches alternately from each dataset."""
        indices = list(sampler_two_datasets_interleave)
        # With batch_size=2, should interleave batches:
        # [0, 1] from first, [5, 6] from second, [2, 3] from first, [7, 8] from second, [4] from first, [9] from second
        expected = [0, 1, 5, 6, 2, 3, 7, 8, 4, 9]
        assert indices == expected

    def test_sampler_iter_interleave_different_batch_size(self):
        """Test interleave mode with different batch sizes."""
        sampler = InterleaveBatchSampler([6, 4], batch_size=2, mode="interleave")
        indices = list(sampler)
        assert len(indices) == 10
        # Should have batches of size 2 interleaved from each dataset
        # [0,1] from first, [6,7] from second, [2,3] from first, [8,9] from second, [4,5] from first
        expected = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5]
        assert indices == expected

    def test_sampler_iter_interleave_unequal_lengths(self):
        """Test interleave mode with unequal dataset lengths."""
        sampler = InterleaveBatchSampler([7, 3], batch_size=2, mode="interleave")
        indices = list(sampler)
        assert len(indices) == 10
        # With unequal lengths and batch_size=2:
        # [0,1] from first (0-6), [7,8] from second (7-9), [2,3] from first, [9] from second (exhausted), [4,5] from first, [6] from first
        expected = [0, 1, 7, 8, 2, 3, 9, 4, 5, 6]
        assert indices == expected

    def test_sampler_iter_sequential_mode_returns_all_indices(
        self, sampler_two_datasets_sequential
    ):
        """Test that sequential mode returns all indices."""
        indices = list(sampler_two_datasets_sequential)
        assert len(indices) == 10
        assert set(indices) == set(range(10))

    def test_sampler_iter_interleave_mode_returns_all_indices(
        self, sampler_two_datasets_interleave
    ):
        """Test that interleave mode returns all indices with correct alternating pattern."""
        indices = list(sampler_two_datasets_interleave)
        assert len(indices) == 10
        # Verify alternating pattern: batches of 2 from first (0-4), then second (5-9)
        # Expected: [0,1] from first, [5,6] from second, [2,3] from first, [7,8] from second, [4] from first, [9] from second
        expected = [0, 1, 5, 6, 2, 3, 7, 8, 4, 9]
        assert indices == expected

    def test_sampler_with_batch_size_one(self):
        """Test sampler with batch_size=1."""
        sampler = InterleaveBatchSampler([4, 4], batch_size=1, mode="interleave")
        indices = list(sampler)
        assert len(indices) == 8
        assert set(indices) == set(range(8))

    def test_sampler_with_large_batch_size(self):
        """Test sampler with batch_size larger than datasets."""
        sampler = InterleaveBatchSampler([5, 5], batch_size=10, mode="interleave")
        indices = list(sampler)
        assert len(indices) == 10
        assert set(indices) == set(range(10))

    def test_sampler_default_mode_is_sequential(self):
        """Test that default mode is sequential."""
        sampler = InterleaveBatchSampler([5, 5], batch_size=2)
        assert sampler.mode == "sequential"

    def test_sampler_sequential_specific_order(self):
        """Test exact order in sequential mode."""
        sampler = InterleaveBatchSampler([3, 2], batch_size=1, mode="sequential")
        indices = list(sampler)
        # First 3 from first dataset, then 2 from second
        assert indices == [0, 1, 2, 3, 4]

    def test_sampler_interleave_specific_order(self):
        """Test order in interleave mode with specific batch size."""
        sampler = InterleaveBatchSampler([4, 4], batch_size=2, mode="interleave")
        indices = list(sampler)
        # Should interleave batches of 2: [0,1] from first, [4,5] from second, [2,3] from first, [6,7] from second
        expected = [0, 1, 4, 5, 2, 3, 6, 7]
        assert indices == expected

