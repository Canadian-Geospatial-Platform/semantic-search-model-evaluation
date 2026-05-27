import pytest
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import logging

from data_processing.utils.auxilliary_preprocessing import (
    load_data,
    save_data,
    extract_unique_desc,
    get_second_segment,
    extract_org,
    choose_org,
    is_mappable_from_str,
    map_topics_to_themes,
    is_foundational,
    deduplicate_data,
)


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_single_parquet(self, tmp_path):
        """Test loading a single parquet file."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)
        
        result = load_data(str(tmp_path))
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        pd.testing.assert_frame_equal(result, df)
    
    def test_load_data_multiple_parquets(self, tmp_path):
        """Test loading multiple parquet files."""
        df1 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df2 = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        parquet_file1 = tmp_path / "test1.parquet"
        parquet_file2 = tmp_path / "test2.parquet"
        df1.to_parquet(parquet_file1, index=False)
        df2.to_parquet(parquet_file2, index=False)
        
        result = load_data(str(tmp_path))
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 4
        assert list(result.columns) == ["col1", "col2"]
    
    def test_load_data_no_parquet_files(self, tmp_path):
        """Test error when no parquet files are found."""
        with pytest.raises(ValueError, match="No parquet files found"):
            load_data(str(tmp_path))
    
    def test_load_data_nonexistent_directory(self):
        """Test error with nonexistent directory."""
        with pytest.raises(ValueError, match="No parquet files found"):
            load_data("/nonexistent/path")


class TestSaveData:
    """Tests for save_data function."""
    
    def test_save_data_creates_files(self, tmp_path):
        """Test that save_data creates train.parquet and eval.parquet files."""
        train_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        test_df = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        
        save_data(train_df, test_df, str(tmp_path))
        
        train_file = tmp_path / "train.parquet"
        test_file = tmp_path / "eval.parquet"
        
        assert train_file.exists()
        assert test_file.exists()
    
    def test_save_data_content_correctness(self, tmp_path):
        """Test that saved data can be read back correctly."""
        train_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        test_df = pd.DataFrame({"col1": [4, 5], "col2": ["d", "e"]})
        
        save_data(train_df, test_df, str(tmp_path))
        
        loaded_train = pd.read_parquet(tmp_path / "train.parquet")
        loaded_test = pd.read_parquet(tmp_path / "eval.parquet")
        
        pd.testing.assert_frame_equal(loaded_train, train_df)
        pd.testing.assert_frame_equal(loaded_test, test_df)
    
    def test_save_data_creates_directory(self, tmp_path):
        """Test that save_data creates output directory if it doesn't exist."""
        nested_path = tmp_path / "output" / "nested"
        train_df = pd.DataFrame({"col1": [1]})
        test_df = pd.DataFrame({"col1": [2]})
        
        save_data(train_df, test_df, str(nested_path))
        
        assert nested_path.exists()
        assert (nested_path / "train.parquet").exists()
        assert (nested_path / "eval.parquet").exists()


class TestExtractUniqueDesc:
    """Tests for extract_unique_desc function."""
    
    def test_extract_desc_from_list_of_options(self):
        """Test extracting descriptions from list of options."""
        options_str = json.dumps([
            {"description": {"en": "First option; extra info"}},
            {"description": {"en": "Second option; more info"}},
        ])
        
        result = extract_unique_desc(options_str)
        
        assert isinstance(result, list)
        assert set(result) == {"First option", "Second option"}
    
    def test_extract_desc_from_dict_option(self):
        """Test extracting description from a single dict option."""
        options_str = json.dumps({"description": {"en": "Single option; more text"}})
        
        result = extract_unique_desc(options_str)
        
        assert result == ["Single option"]
    
    def test_extract_desc_removes_semicolon_content(self):
        """Test that content after semicolon is removed."""
        options_str = json.dumps([
            {"description": {"en": "Main desc; extra1; extra2"}},
        ])
        
        result = extract_unique_desc(options_str)
        
        assert result == ["Main desc"]
    
    def test_extract_desc_ignores_other_languages(self):
        """Test that only specified language is extracted."""
        options_str = json.dumps([
            {"description": {"en": "English desc", "fr": "French desc"}},
        ])
        
        result = extract_unique_desc(options_str, language='en')
        
        assert result == ["English desc"]
    
    def test_extract_desc_different_language(self):
        """Test extracting descriptions in different language."""
        options_str = json.dumps([
            {"description": {"en": "English", "fr": "Français"}},
        ])
        
        result = extract_unique_desc(options_str, language='fr')
        
        assert result == ["Français"]
    
    def test_extract_desc_handles_duplicates(self):
        """Test that duplicate descriptions are removed."""
        options_str = json.dumps([
            {"description": {"en": "Same desc"}},
            {"description": {"en": "Same desc"}},
        ])
        
        result = extract_unique_desc(options_str)
        
        assert result == ["Same desc"]
        assert len(result) == 1
    
    def test_extract_desc_from_non_string(self):
        """Test extracting from already parsed JSON (not string)."""
        options = [{"description": {"en": "Test desc"}}]
        
        result = extract_unique_desc(options)
        
        assert result == ["Test desc"]
    
    def test_extract_desc_missing_language(self):
        """Test handling missing language key."""
        options_str = json.dumps([
            {"description": {"en": "English"}},
            {"description": {"fr": "French"}},
        ])
        
        result = extract_unique_desc(options_str, language='en')
        
        assert result == ["English"]
    
    def test_extract_desc_invalid_json(self):
        """Test handling invalid JSON."""
        result = extract_unique_desc("invalid json")
        
        assert result == []
    
    def test_extract_desc_empty_list(self):
        """Test handling empty options list."""
        options_str = json.dumps([])
        
        result = extract_unique_desc(options_str)
        
        assert result == []
    
    def test_extract_desc_strips_whitespace(self):
        """Test that whitespace is stripped."""
        options_str = json.dumps([
            {"description": {"en": "  Text with spaces  ; extra"}},
        ])
        
        result = extract_unique_desc(options_str)
        
        assert result == ["Text with spaces"]
    
    def test_extract_desc_missing_value(self):
        """Test handling missing description value."""
        options_str = json.dumps([
            {"description": {"en": None}},
        ])
        
        result = extract_unique_desc(options_str)
        
        assert result == []


class TestGetSecondSegment:
    """Tests for get_second_segment function."""
    
    def test_get_second_segment_basic(self):
        """Test extracting second segment from semicolon-separated string."""
        result = get_second_segment("first;second;third")
        assert result == "second"
    
    def test_get_second_segment_two_parts(self):
        """Test with exactly two parts."""
        result = get_second_segment("first;second")
        assert result == "second"
    
    def test_get_second_segment_single_part(self):
        """Test with single part (no semicolon)."""
        result = get_second_segment("single")
        assert result is None
    
    def test_get_second_segment_with_whitespace(self):
        """Test with whitespace around semicolons."""
        result = get_second_segment("first ; second ; third")
        assert result == "second"
    
    def test_get_second_segment_non_string(self):
        """Test with non-string input."""
        result = get_second_segment(123)
        assert result is None
    
    def test_get_second_segment_empty_string(self):
        """Test with empty string."""
        result = get_second_segment("")
        assert result is None
    
    def test_get_second_segment_empty_second_part(self):
        """Test when second part is empty."""
        result = get_second_segment("first;")
        assert result == ""

class TestExtractOrg:
    """Tests for extract_org function."""
    
    def test_extract_org_from_contact_list(self):
        """Test extracting org from contact list."""
        contact_list = json.dumps([
            {
                "organisation": {
                    "en": "org1; English Org; extra",
                    "fr": "org1; Org Français; extra"
                }
            }
        ])
        
        result = extract_org(contact_list)
        
        assert result == {"en": "English Org", "fr": "Org Français"}
    
    def test_extract_org_from_list(self):
        """Test extracting org from already parsed list."""
        contact_list = [
            {
                "organisation": {
                    "en": "first; English",
                    "fr": "first; Français"
                }
            }
        ]
        
        result = extract_org(contact_list)
        
        assert result == {"en": "English", "fr": "Français"}
    
    def test_extract_org_empty_list(self):
        """Test with empty contact list."""
        result = extract_org([])
        
        assert result == {"en": None, "fr": None}
    
    def test_extract_org_invalid_json(self):
        """Test with invalid JSON string."""
        result = extract_org("invalid json")
        
        assert result == {"en": None, "fr": None}
        
    def test_extract_org_missing_organisation_key(self):
        """Test when organisation key is missing."""
        contact_list = [{"name": "John"}]
        
        result = extract_org(contact_list)
        
        assert result == {"en": None, "fr": None}
    
    def test_extract_org_organisation_not_dict(self):
        """Test when organisation is not a dict."""
        contact_list = [{"organisation": "not a dict"}]
        
        result = extract_org(contact_list)
        
        assert result == {"en": None, "fr": None}


class TestChooseOrg:
    """Tests for choose_org function."""
    
    def test_choose_org_uses_cited_when_valid(self):
        """Test that choose_org uses cited when it's valid."""
        cited = json.dumps([
            {"organisation": {"en": "first; Cited Org", "fr": "first; Org Cité"}}
        ])
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org(cited, contact)
        
        assert result["en"] == "Cited Org"
        assert result["fr"] == "Org Cité"
        
    def test_choose_org_fallback_to_contact_when_cited_values_are_none(self):
        """Test that choose_org uses cited when it's valid."""
        cited = json.dumps([
            {"organisation": {"en": None, "fr": None}}
        ])
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org(cited, contact)
        
        assert result["en"] == "Contact Org"
        assert result["fr"] == "Org Contact"
    
    def test_choose_org_fallback_to_contact_when_cited_none(self):
        """Test fallback to contact when cited is None."""
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org(None, contact)
        
        assert result["en"] == "Contact Org"
        assert result["fr"] == "Org Contact"
    
    def test_choose_org_fallback_when_cited_empty_list(self):
        """Test fallback when cited is empty list."""
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org([], contact)
        
        assert result["en"] == "Contact Org"
    
    def test_choose_org_fallback_when_cited_nan(self):
        """Test fallback when cited is NaN."""
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org(float('nan'), contact)
        
        assert result["en"] == "Contact Org"
    
    def test_choose_org_fallback_when_cited_empty_dict(self):
        """Test fallback when cited is empty dict."""
        contact = json.dumps([
            {"organisation": {"en": "first; Contact Org", "fr": "first; Org Contact"}}
        ])
        
        result = choose_org({}, contact)
        
        assert result["en"] == "Contact Org"


class TestIsMappableFromStr:
    """Tests for is_mappable_from_str function."""
    
    def test_is_mappable_esri_in_protocol(self):
        """Test detection of 'esri' in protocol."""
        options_str = json.dumps([
            {"protocol": "ESRI REST API"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "true"
    
    def test_is_mappable_ogc_in_protocol(self):
        """Test detection of 'ogc' in protocol."""
        options_str = json.dumps([
            {"protocol": "OGC WMS"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "true"
    
    def test_is_mappable_neither_esri_nor_ogc(self):
        """Test when neither 'esri' nor 'ogc' is present."""
        options_str = json.dumps([
            {"protocol": "Custom Protocol"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "false"
    
    def test_is_mappable_case_insensitive(self):
        """Test that protocol check is case insensitive."""
        options_str = json.dumps([
            {"protocol": "esri"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "true"
    
    def test_is_mappable_dict_option(self):
        """Test with single dict option instead of list."""
        options_str = json.dumps({"protocol": "OGC"})
        
        result = is_mappable_from_str(options_str)
        
        assert result == "true"
    
    def test_is_mappable_multiple_options(self):
        """Test with multiple options where one has esri/ogc."""
        options_str = json.dumps([
            {"protocol": "Custom"},
            {"protocol": "ESRI REST"},
            {"protocol": "Other"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "true"
    
    def test_is_mappable_invalid_json(self):
        """Test with invalid JSON."""
        result = is_mappable_from_str("invalid json")
        
        assert result == "false"
    
    def test_is_mappable_missing_protocol_key(self):
        """Test when protocol key is missing."""
        options_str = json.dumps([
            {"name": "option1"}
        ])
        
        result = is_mappable_from_str(options_str)
        
        assert result == "false"


class TestMapTopicsToThemes:
    """Tests for map_topics_to_themes function."""
    
    def test_map_topics_basic(self):
        """Test basic topic to theme mapping."""
        topic_list = ["topic1", "topic2"]
        theme_bins = {"topic1": "theme_a", "topic2": "theme_b"}
        
        result = map_topics_to_themes(topic_list, theme_bins)
        
        assert set(result) == {"theme_a", "theme_b"}
        assert isinstance(result, list)
    
    def test_map_topics_case_insensitive(self):
        """Test that mapping is case insensitive."""
        topic_list = ["TOPIC1", "Topic2 "]
        theme_bins = {"topic1": "theme_a", "topic2": "theme_b"}
        
        result = map_topics_to_themes(topic_list, theme_bins)
        
        assert set(result) == {"theme_a", "theme_b"}
    
    def test_map_topics_removes_duplicates(self):
        """Test that duplicate themes are removed."""
        topic_list = ["topic1", "topic2"]
        theme_bins = {"topic1": "same_theme", "topic2": "same_theme"}
        
        result = map_topics_to_themes(topic_list, theme_bins)
        
        assert result == ["same_theme"]
    
    def test_map_topics_sorted_output(self):
        """Test that output is sorted."""
        topic_list = ["topic1", "topic2", "topic3"]
        theme_bins = {"topic1": "z_theme", "topic2": "a_theme", "topic3": "m_theme"}
        
        result = map_topics_to_themes(topic_list, theme_bins)
        
        assert result == ["a_theme", "m_theme", "z_theme"]
    
    def test_map_topics_unmapped_topics_ignored(self):
        """Test that unmapped topics are ignored."""
        topic_list = ["topic1", "unmapped"]
        theme_bins = {"topic1": "theme_a"}
        
        result = map_topics_to_themes(topic_list, theme_bins)
        
        assert result == ["theme_a"]
    
    def test_map_topics_empty_list(self):
        """Test with empty topic list."""
        result = map_topics_to_themes([], {})
        
        assert result == []
    
    def test_map_topics_non_list_input(self):
        """Test with non-list input."""
        result = map_topics_to_themes("not a list", {})
        
        assert result == []
    
    def test_map_topics_none_input(self):
        """Test with None input."""
        result = map_topics_to_themes(None, {})
        
        assert result == []


class TestIsFoundational:
    """Tests for is_foundational function."""
    
    def test_is_foundational_in_list(self):
        """Test when 'foundational' is in a list."""
        theme = json.dumps(["foundational", "other_tag"])
        
        result = is_foundational(theme)
        
        assert result == "true"
    
    def test_is_foundational_in_string(self):
        """Test when theme is a single string with 'foundational'."""
        result = is_foundational("foundational")
        
        assert result == "true"
    
    def test_is_foundational_not_present(self):
        """Test when 'foundational' is not present."""
        theme = json.dumps(["other_tag", "another_tag"])
        
        result = is_foundational(theme)
        
        assert result == "false"
    
    def test_is_foundational_case_insensitive(self):
        """Test that check is case insensitive."""
        theme = json.dumps(["FOUNDATIONAL"])
        
        result = is_foundational(theme)
        
        assert result == "true"
    
    def test_is_foundational_case_mixed(self):
        """Test with mixed case."""
        theme = json.dumps(["FoUnDaTiOnAl"])
        
        result = is_foundational(theme)
        
        assert result == "true"
    
    def test_is_foundational_with_spaces(self):
        """Test handling of whitespace."""
        theme = json.dumps(["  foundational  ", "other"])
        
        result = is_foundational(theme)
        
        assert result == "true"
    
    def test_is_foundational_parsed_list(self):
        """Test with already parsed list (not JSON string)."""
        theme = ["foundational", "tag"]
        
        result = is_foundational(theme)
        
        assert result == "true"
    
    def test_is_foundational_invalid_json(self):
        """Test with invalid JSON."""
        result = is_foundational("invalid json")
        
        assert result == "false"
    
    def test_is_foundational_none_input(self):
        """Test with None in list."""
        theme = json.dumps([None, "other"])
        
        result = is_foundational(theme)
        
        assert result == "false"
    
    def test_is_foundational_empty_list(self):
        """Test with empty list."""
        theme = json.dumps([])
        
        result = is_foundational(theme)
        
        assert result == "false"


class TestDeduplicateData:
    """Tests for deduplicate_data function."""
    
    def test_deduplicate_single_column(self):
        """Test deduplication on a single column."""
        df = pd.DataFrame({
            "col1": [1, 1, 2, 3],
            "col2": ["a", "b", "c", "d"]
        })
        
        result = deduplicate_data(df, ["col1"])
        
        assert result.shape[0] == 3
        assert list(result["col1"].values) == [1, 2, 3]
        assert list(result["col2"].values) == ["a", "c", "d"]
    
    def test_deduplicate_multiple_columns(self):
        """Test deduplication on multiple columns independently."""
        df = pd.DataFrame({
            "col1": [1, 1, 2, 2],
            "col2": ["a", "b", "a", "b"]
        })
        
        result = deduplicate_data(df, ["col1", "col2"])
        
        assert result.shape[0] == 1
    
    def test_deduplicate_no_duplicates(self):
        """Test when there are no duplicates."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        
        result = deduplicate_data(df, ["col1"])
        
        assert result.shape == df.shape
        pd.testing.assert_frame_equal(result, df)
    
    def test_deduplicate_all_duplicates(self):
        """Test when all rows are duplicates."""
        df = pd.DataFrame({
            "col1": [1, 1, 1],
            "col2": ["a", "a", "a"]
        })
        
        result = deduplicate_data(df, ["col1", "col2"])
        
        assert result.shape[0] == 1
    
    def test_deduplicate_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({"col1": [], "col2": []})
        
        result = deduplicate_data(df, ["col1"])
        
        assert result.shape[0] == 0
    
    def test_deduplicate_preserves_other_columns(self):
        """Test that non-subset columns are preserved."""
        df = pd.DataFrame({
            "col1": [1, 1, 2],
            "col2": ["x", "y", "x"],
            "col3": [10, 20, 30]
        })
        
        result = deduplicate_data(df, ["col1"])
        
        assert list(result.columns) == ["col1", "col2", "col3"]
        assert result.shape[0] == 2

        expected = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["x", "x"],
            "col3": [10, 30]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_deduplicate_with_nan_values(self):
        """Test deduplication with NaN values."""
        df = pd.DataFrame({
            "col1": [1, 1, None, None],
            "col2": ["a", "b", "c", "d"]
        })
        
        result = deduplicate_data(df, ["col1"])
        
        assert result.shape[0] == 2