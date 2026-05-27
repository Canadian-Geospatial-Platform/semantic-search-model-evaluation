import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import logging

from src.data_processing.utils.text_normalization import (
    mask_emails,
    mask_urls,
    normalize_text_columns,
    remove_section_and_onwards,
    remove_unnatural_punctuation,
    remove_extra_whitespace,
    map_symbol_to_word,
    apply_fn_and_count_affected_rows,
    convert_tables_to_text,
    preprocess_records_into_text,
)

class TestMaskEmails:
    def test_mask_single_email(self):
        text = "Contact us at support@example.com for more info."
        assert mask_emails(text) == "Contact us at [email] for more info."

    def test_mask_email_in_brackets(self):
        text = "Contact us at [support.help@example.ca] for more info."
        assert mask_emails(text) == "Contact us at [[email]] for more info."

    def test_mask_multiple_emails(self):
        text = "alice@example.com, bob.smith+news@sub.domain.co.uk are contacts"
        expected = "[email], [email] are contacts"
        assert mask_emails(text) == expected
    
    def test_change_mask_for_emails(self):
        text = "Contact us at support@example.com for more info."
        assert mask_emails(text, 'EMAIL') == "Contact us at EMAIL for more info."

    def test_mask_no_email(self):
        text = "There is no email address here."
        assert mask_emails(text) == text

    def test_mask_empty_string(self):
        assert mask_emails("") == ""

    def test_no_mask_for_incomplete_emails(self):
        # Incomplete emails, should not be matched
        text = "user@localhost or user@domain"
        assert mask_emails(text) == text
    
    def test_no_mask_with_invalid_text(self):
        text = None
        assert mask_emails(text) == None
    
    def test_no_mask_with_invalid_mask(self):
        text = "This is normal text"
        invalid_mask = None
        assert mask_emails(text, invalid_mask) == text

class TestMaskUrls:
    def test_mask_single_url(self):
        text = "Visit our website at https://www.example.com for details."
        assert mask_urls(text) == "Visit our website at www.example.com [url] for details."
    
    def test_mask_url_with_port(self):
        text = "Visit our website at https://example.com:8080/other for details."
        assert mask_urls(text) == "Visit our website at example.com [url] for details."
    
    def test_mask_url_with_query(self):
        text = "Visit our website at https://www.example.com/other?param=value for details."
        assert mask_urls(text) == "Visit our website at www.example.com [url] for details."

    def test_mask_multiple_urls(self):
        text = "Check out http://example.com and https://sub.domain.com/page"
        expected = "Check out example.com [url] and sub.domain.com [url]"
        assert mask_urls(text) == expected
    
    def test_mask_url_in_brackets(self):
        text = "Our site (https://www.example.com/something-else/other) has more info."
        assert mask_urls(text) == "Our site (www.example.com [url]) has more info."
    
    def test_change_mask_for_url(self):
        text = "Visit our website at https://www.example.com for details."
        assert mask_urls(text, 'URL_MASK') == "Visit our website at www.example.com URL_MASK for details."

    def test_mask_no_url(self):
        text = "This text does not contain any URLs."
        assert mask_urls(text) == text

    def test_mask_empty_string(self):
        assert mask_urls("") == ""

    def test_mask_url_like_but_invalid(self):
        # Incomplete URLs, should not be matched
        text = "Visit ww.example or http:/example.com"
        assert mask_urls(text) == text
    
    def test_no_mask_with_invalid_text(self):
        text = None
        assert mask_urls(text) == None
    
    def test_no_mask_with_invalid_mask(self):
        text = "This is normal text"
        invalid_mask = None
        assert mask_urls(text, invalid_mask) == text

class TestRemoveSectionAndOnwards:
    def test_remove_section_at_middle(self):
        text = "This is the introduction. Section: This is the section to remove. More text."
        expected = "This is the introduction."
        assert remove_section_and_onwards(text, "Section:") == expected

    def test_keep_section_when_header_not_found(self):
        text = "This is some text without the section header."
        assert remove_section_and_onwards(text, "Section:") == text

    def test_remove_section_at_start(self):
        text = "Section: This is the section to remove. More text."
        expected = ""
        assert remove_section_and_onwards(text, "Section:") == expected

    def test_remove_section_at_end(self):
        text = "This is the introduction. Section: This is the section to remove."
        expected = "This is the introduction."
        assert remove_section_and_onwards(text, "Section:") == expected
    
    def test_skip_with_empty_text(self):
        text = ""
        assert remove_section_and_onwards(text, "Section") == ""
    
    def test_skip_with_invalid_text(self):
        text = None
        assert remove_section_and_onwards(text, "Section") == None
    
    def test_skip_with_invalid_section_header(self):
        text = "This is normal text"
        invalid_header = None
        assert remove_section_and_onwards(text, invalid_header) == text

class TestRemoveUnnaturalPunctuation:
    def test_remove_consecutive_punctuation(self):
        text = "This is a test!!! Does it work?? \"Yes\", it does,,,,,"
        expected = "This is a test! Does it work? \"Yes\", it does,"
        assert remove_unnatural_punctuation(text) == expected

    def test_no_unnatural_punctuation(self):
        text = "This is a normal sentence with proper punctuation."
        assert remove_unnatural_punctuation(text) == text

    def test_empty_string(self):
        assert remove_unnatural_punctuation("") == ""

    def test_remove_only_unnatural_punctuation(self):
        text = "!!!???##***-----++=="
        assert remove_unnatural_punctuation(text) == "!?#*-+="

    def test_keep_intentional_punctuation(self):
        text = "The provided text is '' (empty)... Does it work? \"Yes\", it does."
        assert remove_unnatural_punctuation(text) == text

    def test_skip_with_empty_text(self):
        text = ""
        assert remove_unnatural_punctuation(text) == text

    def test_skip_with_invalid_text(self):
        text = None
        assert remove_unnatural_punctuation(text) == text

class TestRemoveExtraWhitespace:
    def test_remove_consecutive_whitespace(self):
        text = "This   is a   test.\n\nNew line with  extra spaces.\tTabs too."
        expected = "This is a test.\nNew line with extra spaces.\tTabs too."
        assert remove_extra_whitespace(text) == expected
    
    def test_remove_leading_trailing_whitespace(self):
        text = "   This is a test with leading and trailing whitespace.   "
        expected = "This is a test with leading and trailing whitespace."
        assert remove_extra_whitespace(text) == expected
    
    def test_remove_trailing_space_with_newline(self):
        text = "Hello   \n"
        expected = "Hello"
        assert remove_extra_whitespace(text) == expected

    def test_with_no_extra_whitespace(self):
        text = "This is a normal sentence."
        assert remove_extra_whitespace(text) == text

    def test_empty_string(self):
        assert remove_extra_whitespace("") == ""

    def test_only_whitespace(self):
        text = "   \n\t  "
        assert remove_extra_whitespace(text) == ""

    def test_skip_with_empty_text(self):
        text = ""
        assert remove_extra_whitespace(text) == ""

    def test_skip_with_invalid_text(self):
        text = None
        assert remove_extra_whitespace(text) == text

class TestMapSymbolToWord:
    def test_map_symbol_to_word(self):
        text = "I have 2 apples and 3 oranges."
        expected = "I have two apples and 3 oranges."
        assert map_symbol_to_word(text, "2", "two") == expected

    def test_map_symbol_not_found(self):
        text = "This text does not contain the symbol."
        assert map_symbol_to_word(text, "#", "number") == text

    def test_empty_string(self):
        assert map_symbol_to_word("", "@", "at") == ""

    def test_empty_symbol(self):
        text = "This text should remain unchanged."
        assert map_symbol_to_word(text, "", "word") == text

    def test_empty_word(self):
        text = "I have $2 apples and $3 oranges."
        expected = "I have 2 apples and 3 oranges."
        assert map_symbol_to_word(text, "$", "") == expected

    def test_map_multiple_occurrences(self):
        text = "Item is #5. Item B is #3."
        expected = "Item is number 5. Item B is number 3."
        assert map_symbol_to_word(text, "#", "number ") == expected

    def test_with_invalid_symbol(self):
        text = "Item is #5. Item B is #3."
        assert map_symbol_to_word(text, None, "number ") == text

    def test_with_invalid_word(self):
        text = "Item is #5. Item B is #3."
        assert map_symbol_to_word(text, "#", None) == text

    def test_with_invalid_text(self):
        text = None
        assert map_symbol_to_word(text, "#", "Number") == text

class TestApplyFnAndCountAffectedRows:
    @pytest.fixture
    def sample_df(self):
        data = {
            'text': [
                "This is a sentence.",
                "This has WORD_TO_REPLACE.",
            ]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_df_no_match(self):
        data = {
            'text': [
                "This is a normal sentence.",
                "This is another normal sentence."
            ]
        }
        return pd.DataFrame(data)

    def test_apply_fn_and_count_affected_rows(self, sample_df):
        def fn(x):
            return x.replace('WORD_TO_REPLACE', 'REPLACED')

        result_df = apply_fn_and_count_affected_rows(sample_df, 'text', fn)
        assert result_df.loc[0, 'text'] == "This is a sentence."
        assert result_df.loc[1, 'text'] == "This has REPLACED."
    
    def test_no_affected_rows(self, sample_df_no_match):
        def fn(x):
            return x.replace('WORD_TO_REPLACE', 'REPLACED')

        result_df = apply_fn_and_count_affected_rows(sample_df_no_match, 'text', fn)
        assert result_df.loc[0, 'text'] == "This is a normal sentence."
        assert result_df.loc[1, 'text'] == "This is another normal sentence."
    
    def test_kwargs_passed_to_function(self, sample_df):
        def fn(x, replacement):
            return x.replace('WORD_TO_REPLACE', replacement)

        result_df = apply_fn_and_count_affected_rows(sample_df, 'text', fn, replacement='REPLACED2')
        assert result_df.loc[0, 'text'] == "This is a sentence."
        assert result_df.loc[1, 'text'] == "This has REPLACED2."

class TestConvertTablesToText:
    def test_convert_tables_to_text(self):
        table_to_convert = '+-----------------------------------+-----------------------------------+\n| Field\\ | Description\\ |\n| \\ | \\ |\n+-----------------------------------+-----------------------------------+\n| SOME_FIELD_NAME\\ | Value for field |\n| \\ | overfills to next line\\ |\n| | \\ |\n+-----------------------------------+-----------------------------------+'

        result = convert_tables_to_text(table_to_convert)
        expected = "\n| Field | Description |\n| SOME_FIELD_NAME | Value for field |\n|  | overfills to next line |\n"

        assert result == expected
    
    def test_preserve_info_around_table(self):
        table_to_convert = 'This is some text before the table.\n+-----------------------------------+-----------------------------------+\n| Field\\ | Description\\ |\n| SOME_FIELD_NAME\\ | Value for field |\n+-----------------------------------+-----------------------------------+\nThis is some text after the table.'

        result = convert_tables_to_text(table_to_convert)
        expected = "This is some text before the table.\n| Field | Description |\n| SOME_FIELD_NAME | Value for field |\nThis is some text after the table."

        assert result == expected
    
    def test_no_table_in_text(self):
        text_without_table = 'This is some text without a table.'

        result = convert_tables_to_text(text_without_table)
        assert result == text_without_table
    
    def test_invalid_text(self):
        invalid_text = None

        result = convert_tables_to_text(invalid_text)
        assert result == invalid_text
    
    def test_heading_in_text(self):
        text_without_table = 'This is some text without a table.\n-----------However--------\nThis test does include a subheading surrounded by -'

        result = convert_tables_to_text(text_without_table)
        assert result == text_without_table

class TestNormalizeTextColumns:    
    @pytest.fixture
    def sample_df_en(self):
        return pd.DataFrame({
            'description_en': ['Description A (control)', 'Description B with http://example.com/pages and     email@example.com that also has a table!!!\n+---+---+\n| Col1 | Col2 |\n| \\ | \\ |\n| Val1 | Val2 |\n+---+---+\nEnd of table.   And some extra spaces.\n\nMore spaces\n'],
            'features_properties_keywords_en': ['keyword1, keyword2', 'keyword3, keyword4']
        })
    
    def test_normalize_text_columns(self, sample_df_en):
        language_config = {
            'en': {},
        }
        text_columns = ['description_en']
        
        result_df = normalize_text_columns(sample_df_en, text_columns, language_config)
                
        # Check that the original columns are unchanged
        assert result_df['description_en'].iloc[0] == 'Description A (control)'
        assert result_df['description_en'].iloc[1] == 'Description B with http://example.com/pages and     email@example.com that also has a table!!!\n+---+---+\n| Col1 | Col2 |\n| \\ | \\ |\n| Val1 | Val2 |\n+---+---+\nEnd of table.   And some extra spaces.\n\nMore spaces\n'

        assert result_df['description_normalized_en'].iloc[0] == 'Description A (control)'
        
        expected = 'Description B with example.com [url] and [email] that also has a table!\n| Col1 | Col2 |\n| Val1 | Val2 |\nEnd of table. And some extra spaces.\nMore spaces'
        assert result_df['description_normalized_en'].iloc[1] == expected


class TestPreprocessRecordsIntoText:
    
    @pytest.fixture
    def sample_df_en(self):
        """Create sample DataFrame with English columns."""
        return pd.DataFrame({
            'features_properties_title_en': ['Product A', 'Product B'],
            'features_properties_description_normalized_en': ['Description A', 'Description B'],
            'features_properties_keywords_en': ['keyword1, keyword2', 'keyword3, keyword4']
        })
    
    @pytest.fixture
    def sample_df_bilingual(self):
        """Create sample DataFrame with English and French columns."""
        return pd.DataFrame({
            'features_properties_title_en': ['Product A', 'Product B'],
            'features_properties_description_normalized_en': ['Description A', 'Description B'],
            'features_properties_keywords_en': ['keyword1, keyword2', 'keyword3, keyword4'],
            'features_properties_title_fr': ['Produit A', 'Produit B'],
            'features_properties_description_normalized_fr': ['Description A FR', 'Description B FR'],
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
    