import pytest
import pandas as pd
from preprocessing.utils.training_preparation import preprocess_records_into_text

class TestPreprocessRecordsIntoText:
    
    @pytest.fixture
    def sample_df_en(self):
        """Create sample DataFrame with English columns."""
        return pd.DataFrame({
            'features_properties_title_en': ['Product A', 'Product B'],
            'features_properties_description_en': ['Description A', 'Description B'],
            'features_properties_keywords_en': ['keyword1, keyword2', 'keyword3, keyword4']
        })
    
    @pytest.fixture
    def sample_df_bilingual(self):
        """Create sample DataFrame with English and French columns."""
        return pd.DataFrame({
            'features_properties_title_en': ['Product A', 'Product B'],
            'features_properties_description_en': ['Description A', 'Description B'],
            'features_properties_keywords_en': ['keyword1, keyword2', 'keyword3, keyword4'],
            'features_properties_title_fr': ['Produit A', 'Produit B'],
            'features_properties_description_fr': ['Description A FR', 'Description B FR'],
            'features_properties_keywords_fr': ['motcle1, motcle2', 'motcle3, motcle4']
        })
    
    def test_sequential_format_en(self, sample_df_en):
        """Test sequential output format with English language."""
        result = preprocess_records_into_text(sample_df_en, languages=['en'], keyword_prefixes=['keywords:'], output_format='sequential')
        
        assert len(result) == 2
        assert 'Product A' in result.iloc[0]
        assert 'keywords:keyword1, keyword2' in result.iloc[0]
        assert 'Description A' in result.iloc[0]
        
        assert 'Produit A' not in result.iloc[0]  # should not contain French text
    
    def test_sequential_format_fr(self, sample_df_bilingual):
        """Test sequential output format with French language."""
        result = preprocess_records_into_text(sample_df_bilingual, languages=['fr'], keyword_prefixes=['mots-clés associés:'], output_format='sequential')
        
        assert len(result) == 2
        assert 'Produit A' in result.iloc[0]
        assert 'mots-clés associés:motcle1, motcle2' in result.iloc[0]
        assert 'Description A FR' in result.iloc[0]
        assert 'Product A' not in result.iloc[0]  # should not contain English text

    def test_sequential_format_bilingual(self, sample_df_bilingual):
        """Test sequential output format with multiple languages."""
        result = preprocess_records_into_text(sample_df_bilingual, languages=['en', 'fr'], keyword_prefixes=['keywords:', 'mots-clés associés:'], output_format='sequential')
        
        assert len(result) == 2
        assert 'Product A' in result.iloc[0]
        assert 'Produit A' in result.iloc[0]
        assert 'keywords:keyword1, keyword2' in result.iloc[0]
        assert 'mots-clés associés:motcle1, motcle2' in result.iloc[0]
        assert 'Description A' in result.iloc[0]
        assert 'Description A FR' in result.iloc[0]

        # check order of languages in sequential format
        assert max(result.iloc[0].index('Product A'), result.iloc[0].index('keywords:keyword1, keyword2'), result.iloc[0].index('Description A')) < min(result.iloc[0].index('Produit A'), result.iloc[0].index('mots-clés associés:motcle1, motcle2'), result.iloc[0].index('Description A FR'))

    
    def test_parallel_format_en(self, sample_df_en):
        """Test parallel output format with English language."""
        result = preprocess_records_into_text(sample_df_en, languages=['en'], keyword_prefixes=['keywords:'], output_format='parallel')
        
        assert len(result) == 2
        assert 'Product A' in result.iloc[0]
        assert 'keywords:keyword1, keyword2' in result.iloc[0]
        assert 'Description A' in result.iloc[0]
        assert 'Produit A' not in result.iloc[0]  # should not contain French text
    
    def test_parallel_format_bilingual(self, sample_df_bilingual):
        """Test parallel output format with multiple languages."""
        result = preprocess_records_into_text(sample_df_bilingual, languages=['en', 'fr'], keyword_prefixes=['keywords:', 'mots-clés associés:'], output_format='parallel')
        
        assert len(result) == 2
        assert 'Product A' in result.iloc[0]
        assert 'Produit A' in result.iloc[0]
        assert 'keywords:keyword1, keyword2' in result.iloc[0]
        assert 'mots-clés associés:motcle1, motcle2' in result.iloc[0]
        assert 'Description A' in result.iloc[0]
        assert 'Description A FR' in result.iloc[0]

        # check order of languages in parallel format
        assert result.iloc[0].index('Product A') < result.iloc[0].index('Produit A')
        assert result.iloc[0].index('Produit A') < result.iloc[0].index('keywords:keyword1, keyword2')
        assert result.iloc[0].index('keywords:keyword1, keyword2') < result.iloc[0].index('mots-clés associés:motcle1, motcle2')
        assert result.iloc[0].index('mots-clés associés:motcle1, motcle2') < result.iloc[0].index('Description A')
        assert result.iloc[0].index('Description A') < result.iloc[0].index('Description A FR')

    def test_invalid_output_format(self, sample_df_en):
        """Test that invalid output format raises ValueError."""
        with pytest.raises(ValueError):
            preprocess_records_into_text(sample_df_en, languages=['en'], output_format='invalid')
    
    def test_missing_columns(self, sample_df_en):
        """Test that missing columns raise ValueError."""
        result = preprocess_records_into_text(sample_df_en, languages=['fr'], keyword_prefixes=['mots-clés associés:'], output_format='sequential')

        assert len(result) == 2
        assert result.iloc[0] == "\n\n\n"  # should return empty text with newlines for missing columns
    
    def test_output_is_series(self, sample_df_en):
        """Test that output is a pandas Series."""
        result = preprocess_records_into_text(sample_df_en, languages=['en'], keyword_prefixes=['keywords:'], output_format='sequential')
        
        assert isinstance(result, pd.Series)
    
