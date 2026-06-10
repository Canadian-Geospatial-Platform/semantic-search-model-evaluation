import pytest
import pandas as pd
from datasets import Dataset
from finetune.utils.ir_evaluate import (
    extract_query_coprus_relevant_docs,
)


class TestExtractQueryCorpusRelevantDocs:
    """Test the extract_query_coprus_relevant_docs function."""

    @pytest.fixture
    def sample_dataset_basic(self):
        """Create a sample DataFrame with basic columns."""
        return Dataset.from_pandas(pd.DataFrame({
            "query": ["question 1", "question 2", "question 3"],
            "document": ["doc 1", "doc 2", "doc 3"],
        }))
    
    @pytest.fixture
    def additional_dataset_basic(self):
        """Create a sample DataFrame with basic columns."""
        return Dataset.from_pandas(pd.DataFrame({
            "query": ["question 4", "question 5", "question 6"],
            "document": ["doc 4", "doc 5", "doc 6"],
        }))
    
    def test_extract_qcr_with_one_to_one_mapping(self, sample_dataset_basic):
        q, c, r = extract_query_coprus_relevant_docs(sample_dataset_basic, 'query', 'document')

        assert len(q) == 3
        assert len(c) == 3
        assert len(r) == 3

        assert all([len(v) == 1 and k in v for k,v in r.items()]) == True
    
    def test_extract_qcr_with_additional_corpus(self, sample_dataset_basic, additional_dataset_basic):
        q, c, r = extract_query_coprus_relevant_docs(sample_dataset_basic, 'query', 'document', additional_corpus_datasets=[additional_dataset_basic])

        assert len(q) == 3
        assert len(c) == 6
        assert len(r) == 3

        assert all([len(v) == 1 and k in v for k,v in r.items()]) == True
        assert set([k for k in r]) == set(range(3)) # no mappings to additional corpus ds
