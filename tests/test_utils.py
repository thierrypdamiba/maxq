"""
Unit tests for utility functions.
"""
import pytest


class TestNormalizeDatasetName:
    """Tests for normalize_dataset_name utility."""

    def test_empty_string(self):
        """Test empty string returns empty."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name(None) == ""

    def test_strips_whitespace(self):
        """Test whitespace is stripped."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name("  dataset/name  ") == "dataset/name"

    def test_preserves_valid_name(self):
        """Test valid name is preserved."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name("fka/awesome-chatgpt-prompts") == "fka/awesome-chatgpt-prompts"

    def test_leading_whitespace(self):
        """Test leading whitespace is stripped."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name("   name") == "name"

    def test_trailing_whitespace(self):
        """Test trailing whitespace is stripped."""
        from maxq.utils import normalize_dataset_name
        assert normalize_dataset_name("name   ") == "name"
