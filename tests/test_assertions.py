"""Tests for the assertion framework."""

import pytest
from maxq.core.assertions.base import SearchResult, EvalContext, AssertionResult
from maxq.core.assertions import (
    get_assertion,
    list_assertions,
    NotEmptyAssertion,
    ContainsIdAssertion,
    LatencyAssertion,
    NDCGAssertion,
    RecallAssertion,
    MRRAssertion,
    PrecisionAssertion,
    HitRateAssertion,
    ContainsTextAssertion,
    RegexAssertion,
    FieldEqualsAssertion,
)


@pytest.fixture
def sample_results():
    """Create sample search results for testing."""
    return [
        SearchResult(id="doc1", score=0.95, text="Vector databases are fast", metadata={"category": "tech"}),
        SearchResult(id="doc2", score=0.85, text="Qdrant is a vector database", metadata={"category": "tech"}),
        SearchResult(id="doc3", score=0.75, text="Machine learning models", metadata={"category": "ai"}),
        SearchResult(id="doc4", score=0.65, text="Natural language processing", metadata={"category": "ai"}),
        SearchResult(id="doc5", score=0.55, text="Python programming tutorial", metadata={"category": "code"}),
    ]


@pytest.fixture
def context_with_ground_truth():
    """Create evaluation context with ground truth."""
    return EvalContext(
        query="vector database",
        ground_truth=["doc1", "doc2", "doc6"],
        latency_ms=50.0,
        top_k=10,
    )


class TestAssertionRegistry:
    """Tests for assertion registry."""

    def test_list_assertions(self):
        """Test that all expected assertions are registered."""
        assertions = list_assertions()
        expected = [
            "not-empty",
            "contains-id",
            "count",
            "latency",
            "ndcg",
            "mrr",
            "recall",
            "precision",
            "hit-rate",
        ]
        for name in expected:
            assert name in assertions, f"Missing assertion: {name}"

    def test_get_assertion(self):
        """Test getting an assertion by name."""
        assertion = get_assertion("not-empty", {})
        assert isinstance(assertion, NotEmptyAssertion)

    def test_get_unknown_assertion(self):
        """Test that unknown assertion raises error."""
        with pytest.raises(ValueError, match="Unknown assertion type"):
            get_assertion("nonexistent", {})


class TestBasicAssertions:
    """Tests for basic assertions."""

    def test_not_empty_pass(self, sample_results):
        """Test not-empty assertion passes with results."""
        assertion = NotEmptyAssertion({})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed
        assert "5 results" in result.message

    def test_not_empty_fail(self):
        """Test not-empty assertion fails without results."""
        assertion = NotEmptyAssertion({})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate([], context)
        assert not result.passed
        assert "No results" in result.message

    def test_contains_id_pass(self, sample_results):
        """Test contains-id assertion passes when ID is present."""
        assertion = ContainsIdAssertion({"value": "doc1"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed

    def test_contains_id_multiple(self, sample_results):
        """Test contains-id with multiple IDs."""
        assertion = ContainsIdAssertion({"value": ["doc1", "doc2"]})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed

    def test_contains_id_fail(self, sample_results):
        """Test contains-id fails when ID is missing."""
        assertion = ContainsIdAssertion({"value": "doc999"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert not result.passed
        assert "doc999" in result.message

    def test_latency_pass(self, sample_results):
        """Test latency assertion passes within bounds."""
        assertion = LatencyAssertion({"max_ms": 100})
        context = EvalContext(query="test", latency_ms=50.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed

    def test_latency_fail(self, sample_results):
        """Test latency assertion fails when exceeded."""
        assertion = LatencyAssertion({"max_ms": 10})
        context = EvalContext(query="test", latency_ms=50.0)
        result = assertion.evaluate(sample_results, context)
        assert not result.passed
        assert "exceeds" in result.message


class TestMetricAssertions:
    """Tests for IR metric assertions."""

    def test_ndcg_pass(self, sample_results, context_with_ground_truth):
        """Test NDCG assertion passes above threshold."""
        assertion = NDCGAssertion({"threshold": 0.5})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        assert result.passed
        assert "NDCG" in result.message

    def test_ndcg_fail(self, sample_results, context_with_ground_truth):
        """Test NDCG assertion fails below threshold."""
        assertion = NDCGAssertion({"threshold": 0.99})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        assert not result.passed

    def test_ndcg_no_ground_truth(self, sample_results):
        """Test NDCG fails without ground truth."""
        assertion = NDCGAssertion({"threshold": 0.5})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert not result.passed
        assert "ground_truth" in result.message

    def test_recall_pass(self, sample_results, context_with_ground_truth):
        """Test recall assertion passes above threshold."""
        assertion = RecallAssertion({"threshold": 0.5})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        # 2 of 3 ground truth found = 0.67
        assert result.passed

    def test_recall_fail(self, sample_results, context_with_ground_truth):
        """Test recall assertion fails below threshold."""
        assertion = RecallAssertion({"threshold": 0.9})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        assert not result.passed

    def test_mrr_pass(self, sample_results, context_with_ground_truth):
        """Test MRR assertion passes (first result is relevant)."""
        assertion = MRRAssertion({"threshold": 0.5})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        # First result (doc1) is in ground truth, so MRR = 1.0
        assert result.passed
        assert result.actual == 1.0

    def test_precision_pass(self, sample_results, context_with_ground_truth):
        """Test precision assertion passes."""
        assertion = PrecisionAssertion({"threshold": 0.3})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        # 2 of 5 results are relevant = 0.4
        assert result.passed

    def test_hit_rate_pass(self, sample_results, context_with_ground_truth):
        """Test hit-rate assertion passes (any relevant in results)."""
        assertion = HitRateAssertion({"threshold": 1.0})
        result = assertion.evaluate(sample_results, context_with_ground_truth)
        assert result.passed
        assert "Hit" in result.message


class TestContentAssertions:
    """Tests for content-based assertions."""

    def test_contains_text_pass(self, sample_results):
        """Test contains-text finds matching text."""
        assertion = ContainsTextAssertion({"value": "vector"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed
        assert "2 results" in result.message  # doc1 and doc2

    def test_contains_text_case_insensitive(self, sample_results):
        """Test contains-text is case insensitive by default."""
        assertion = ContainsTextAssertion({"value": "VECTOR"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed

    def test_contains_text_fail(self, sample_results):
        """Test contains-text fails when text not found."""
        assertion = ContainsTextAssertion({"value": "blockchain"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert not result.passed

    def test_regex_pass(self, sample_results):
        """Test regex assertion matches pattern."""
        assertion = RegexAssertion({"pattern": r"vector\s+database"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed

    def test_regex_fail(self, sample_results):
        """Test regex assertion fails on no match."""
        assertion = RegexAssertion({"pattern": r"^xyz"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert not result.passed

    def test_field_equals_pass(self, sample_results):
        """Test field-equals matches metadata field."""
        assertion = FieldEqualsAssertion({"field": "category", "value": "tech"})
        context = EvalContext(query="test", latency_ms=10.0)
        result = assertion.evaluate(sample_results, context)
        assert result.passed
        assert "2/5" in result.message


class TestTestConfig:
    """Tests for config loading and validation."""

    def test_load_config_from_string(self, tmp_path):
        """Test loading a config from YAML."""
        from maxq.core.testconfig import load_config

        config_content = """
description: Test config
provider:
  collection: test_collection
  model: sentence-transformers/all-MiniLM-L6-v2
defaults:
  top_k: 5
tests:
  - query: "test query"
    assert:
      - type: not-empty
"""
        config_file = tmp_path / "maxq.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        assert config.description == "Test config"
        assert config.provider.collection == "test_collection"
        assert config.defaults.top_k == 5
        assert len(config.tests) == 1

    def test_variable_expansion(self, tmp_path):
        """Test that {{var}} syntax is expanded."""
        from maxq.core.testconfig import load_config

        config_content = """
description: Variable test
provider:
  collection: test
tests:
  - query: "{{color}} shoes"
    vars:
      - { color: "red" }
      - { color: "blue" }
    assert:
      - type: not-empty
"""
        config_file = tmp_path / "maxq.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        # Should expand to 2 tests
        assert len(config.tests) == 2
        assert config.tests[0].query == "red shoes"
        assert config.tests[1].query == "blue shoes"

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Test that ${VAR} is substituted from environment."""
        from maxq.core.testconfig import load_config

        monkeypatch.setenv("TEST_COLLECTION", "my_collection")

        config_content = """
provider:
  collection: ${TEST_COLLECTION}
tests: []
"""
        config_file = tmp_path / "maxq.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)
        assert config.provider.collection == "my_collection"

    def test_validate_config_valid(self, tmp_path):
        """Test validation of a valid config."""
        from maxq.core.testconfig import validate_config

        config_content = """
provider:
  collection: test
tests:
  - query: "test"
    assert:
      - type: not-empty
"""
        config_file = tmp_path / "maxq.yaml"
        config_file.write_text(config_content)

        is_valid, errors = validate_config(config_file)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_invalid_assertion(self, tmp_path):
        """Test validation catches invalid assertion type."""
        from maxq.core.testconfig import validate_config

        config_content = """
provider:
  collection: test
tests:
  - query: "test"
    assert:
      - type: invalid-assertion-type
"""
        config_file = tmp_path / "maxq.yaml"
        config_file.write_text(config_content)

        is_valid, errors = validate_config(config_file)
        assert not is_valid
        assert any("unknown type" in e for e in errors)
