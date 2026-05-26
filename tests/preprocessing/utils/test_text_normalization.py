import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import logging

from preprocessing.utils.text_normalization import mask_emails, mask_urls, normalize_text_columns, remove_section_and_onwards, remove_unnatural_punctuation, remove_extra_whitespace, map_symbol_to_word, apply_fn_and_count_affected_rows, convert_tables_to_text

class TestMaskEmails:
    def test_mask_single_email(self):
        text = "Contact us at support@example.com for more info."
        assert mask_emails(text) == "Contact us at [email] for more info."

    def test_mask_nested_email(self):
        text = "Contact us at [support.help@example.ca] for more info."
        assert mask_emails(text) == "Contact us at [[email]] for more info."

    def test_mask_multiple_emails(self):
        text = "alice@example.com, bob.smith+news@sub.domain.co.uk are contacts"
        expected = "[email], [email] are contacts"
        assert mask_emails(text) == expected

    def test_mask_no_email(self):
        text = "There is no email address here."
        assert mask_emails(text) == text

    def test_mask_empty_string(self):
        assert mask_emails("") == ""

    def test_mask_email_like_but_invalid(self):
        # Incomplete emails, should not be matched
        text = "user@localhost or user@domain"
        assert mask_emails(text) == text

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

    def test_mask_no_url(self):
        text = "This text does not contain any URLs."
        assert mask_urls(text) == text

    def test_mask_empty_string(self):
        assert mask_urls("") == ""

    def test_mask_url_like_but_invalid(self):
        # Incomplete URLs, should not be matched
        text = "Visit ww.example or http:/example.com"
        assert mask_urls(text) == text

class TestRemoveSectionAndOnwards:
    def test_remove_section_at_middle(self):
        text = "This is the introduction. Section: This is the section to remove. More text."
        expected = "This is the introduction."
        assert remove_section_and_onwards(text, "Section:") == expected

    def test_remove_section_not_found(self):
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

    def test_only_unnatural_punctuation(self):
        text = "!!!???##***-----++=="
        assert remove_unnatural_punctuation(text) == "!?#*-+="

    def test_keep_intentional_punctuation(self):
        text = "The provided text is '' (empty)... Does it work? \"Yes\", it does."
        assert remove_unnatural_punctuation(text) == text

class TestRemoveExtraWhitespace:
    def test_remove_extra_whitespace(self):
        text = "This   is a   test.\n\nNew line with  extra spaces.\tTabs too."
        expected = "This is a test.\nNew line with extra spaces.\tTabs too."
        assert remove_extra_whitespace(text) == expected
    
    def test_remove_leading_trailing_whitespace(self):
        text = "   This is a test with leading and trailing whitespace.   "
        expected = "This is a test with leading and trailing whitespace."
        assert remove_extra_whitespace(text) == expected

    def test_no_extra_whitespace(self):
        text = "This is a normal sentence."
        assert remove_extra_whitespace(text) == text

    def test_empty_string(self):
        assert remove_extra_whitespace("") == ""

    def test_only_whitespace(self):
        text = "   \n\t  "
        assert remove_extra_whitespace(text) == ""

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

    def test_map_multiple_occurrences(self):
        text = "Item is #5. Item B is #3."
        expected = "Item is number 5. Item B is number 3."
        assert map_symbol_to_word(text, "#", "number ") == expected

class TestApplyFnAndCountAffectedRows:
    @pytest.fixture
    def sample_df(self):
        data = {
            'text': [
                "This has WORD_TO_REPLACE.",
                "This is another sentence."
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
        assert result_df.loc[0, 'text'] == "This has REPLACED."
        assert result_df.loc[1, 'text'] == "This is another sentence."
    
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
        assert result_df.loc[0, 'text'] == "This has REPLACED2."
        assert result_df.loc[1, 'text'] == "This is another sentence."

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
        expected = 'This is some text without a table.'

        assert result == expected

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
        
        # Check that the normalized columns are created
        for col in text_columns:
            assert f"{col}_normalized" in result_df.columns
        
        # Check that the original columns are unchanged
        assert result_df['description_en'].iloc[0] == 'Description A (control)'
        assert result_df['description_en'].iloc[1] == 'Description B with http://example.com/pages and     email@example.com that also has a table!!!\n+---+---+\n| Col1 | Col2 |\n| \\ | \\ |\n| Val1 | Val2 |\n+---+---+\nEnd of table.   And some extra spaces.\n\nMore spaces\n'

        assert result_df['description_en_normalized'].iloc[0] == 'Description A (control)'
        
        expected = 'Description B with example.com [url] and [email] that also has a table!\n| Col1 | Col2 |\n| Val1 | Val2 |\nEnd of table. And some extra spaces.\nMore spaces'
        assert result_df['description_en_normalized'].iloc[1] == expected