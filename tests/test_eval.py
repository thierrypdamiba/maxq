"""
Tests for evaluation framework functionality.
Tests both ID-Matching (default) and LLM-as-Judge (Ragas) evaluators.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ============================================
# Tests for ID-Matching Evaluator (Default)
# ============================================

class TestEvaluator:
    """Tests for the default ID-Matching Evaluator."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mocked engine for evaluator tests."""
        engine = MagicMock()
        engine.openai_api_key = "test-key"
        engine.client.count.return_value = MagicMock(count=10)
        engine.client.scroll.return_value = ([
            MagicMock(id=1, payload={"_text": "Sample document text " * 50}),
            MagicMock(id=2, payload={"_text": "Another sample document " * 50})
        ], None)
        return engine

    def test_evaluator_init(self, mock_engine):
        """Test Evaluator initialization."""
        from maxq.eval import Evaluator

        evaluator = Evaluator(mock_engine, openai_api_key="test-key")
        assert evaluator.engine == mock_engine
        assert evaluator.openai_api_key == "test-key"

    def test_evaluator_lazy_loads_llm(self, mock_engine):
        """Test that LLM client is lazy loaded."""
        from maxq.eval import Evaluator

        evaluator = Evaluator(mock_engine, openai_api_key="test-key")
        assert evaluator._llm_client is None

        with patch('openai.OpenAI') as mock_openai:
            _ = evaluator.llm_client
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_calculate_ndcg_hit_at_1(self):
        """Test nDCG calculation when item is at rank 1."""
        from maxq.eval import Evaluator

        retrieved = [1, 2, 3, 4, 5]
        relevant = 1

        ndcg = Evaluator.calculate_ndcg(retrieved, relevant, k=5)
        assert ndcg == 1.0  # Perfect score

    def test_calculate_ndcg_hit_at_3(self):
        """Test nDCG calculation when item is at rank 3."""
        from maxq.eval import Evaluator
        import math

        retrieved = [2, 3, 1, 4, 5]
        relevant = 1

        ndcg = Evaluator.calculate_ndcg(retrieved, relevant, k=5)
        expected = (1.0 / math.log2(3 + 1)) / (1.0 / math.log2(2))
        assert abs(ndcg - expected) < 0.001

    def test_calculate_ndcg_miss(self):
        """Test nDCG calculation when item is not found."""
        from maxq.eval import Evaluator

        retrieved = [2, 3, 4, 5, 6]
        relevant = 1

        ndcg = Evaluator.calculate_ndcg(retrieved, relevant, k=5)
        assert ndcg == 0.0

    def test_calculate_mrr_hit_at_1(self):
        """Test MRR calculation when item is at rank 1."""
        from maxq.eval import Evaluator

        retrieved = [1, 2, 3]
        relevant = 1

        mrr = Evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_calculate_mrr_hit_at_3(self):
        """Test MRR calculation when item is at rank 3."""
        from maxq.eval import Evaluator

        retrieved = [2, 3, 1]
        relevant = 1

        mrr = Evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 1.0 / 3

    def test_calculate_mrr_miss(self):
        """Test MRR calculation when item is not found."""
        from maxq.eval import Evaluator

        retrieved = [2, 3, 4]
        relevant = 1

        mrr = Evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 0.0

    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        from maxq.eval import Evaluator

        retrieved = [1, 2, 3, 4, 5]

        # Hit at k=5
        assert Evaluator.calculate_hit_rate(retrieved, 5, k=5) == 1.0
        # Hit at k=3
        assert Evaluator.calculate_hit_rate(retrieved, 3, k=5) == 1.0
        # Miss
        assert Evaluator.calculate_hit_rate(retrieved, 10, k=5) == 0.0

    def test_generate_testset_from_collection_empty(self, mock_engine):
        """Test testset generation with empty collection."""
        from maxq.eval import Evaluator

        mock_engine.client.scroll.return_value = ([], None)

        evaluator = Evaluator(mock_engine)
        testset = evaluator.generate_testset_from_collection(
            "test_collection", num_samples=5, use_llm=False
        )
        assert testset == []

    def test_generate_testset_from_docs(self, mock_engine):
        """Test generating testset from document list."""
        from maxq.eval import Evaluator

        docs = [
            {"text": "This is a long enough document for testing purposes. " * 5, "id": 1},
            {"text": "Another sufficiently long document for the test suite. " * 5, "id": 2},
        ]

        evaluator = Evaluator(mock_engine)
        testset = evaluator.generate_testset_from_docs(
            docs, text_field="text", id_field="id", num_samples=2, use_llm=False
        )

        assert len(testset) == 2
        assert all("question" in t for t in testset)
        assert all("expected_id" in t for t in testset)

    def test_extract_pseudo_query(self, mock_engine):
        """Test pseudo-query extraction from text."""
        from maxq.eval import Evaluator

        evaluator = Evaluator(mock_engine)

        text = "This is the first sentence. This is the second sentence."
        query = evaluator._extract_pseudo_query(text)

        assert "This is the first sentence" in query


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from maxq.eval import RetrievalMetrics

        metrics = RetrievalMetrics(
            hit_rate_1=0.8,
            hit_rate_5=0.95,
            mrr=0.85,
            ndcg_5=0.9,
            latency_p50=50.0
        )

        d = metrics.to_dict()
        assert d["hit_rate@1"] == 0.8
        assert d["hit_rate@5"] == 0.95
        assert d["mrr"] == 0.85
        assert d["ndcg@5"] == 0.9


class TestModelEvalResult:
    """Tests for ModelEvalResult dataclass."""

    def test_convenience_accessors(self):
        """Test convenience property accessors."""
        from maxq.eval import ModelEvalResult, RetrievalMetrics

        metrics = RetrievalMetrics(
            hit_rate_5=0.9,
            mrr=0.85,
            ndcg_5=0.88,
            latency_p50=45.0
        )

        result = ModelEvalResult(
            model_name="test-model",
            collection_name="test-collection",
            metrics=metrics
        )

        assert result.ndcg_at_k == 0.88
        assert result.hit_at_k == 0.9
        assert result.mrr == 0.85
        assert result.latency_p50 == 45.0


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_enhanced_evaluator_alias(self):
        """Test EnhancedEvaluator alias exists."""
        from maxq.eval import EnhancedEvaluator, Evaluator
        assert EnhancedEvaluator is Evaluator


# ============================================
# Tests for LLM-as-Judge Evaluator (Ragas)
# ============================================

class TestLLMJudgeEvaluator:
    """Tests for the LLM-as-Judge (Ragas) evaluator."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mocked engine for evaluator tests."""
        engine = MagicMock()
        engine.openai_api_key = "test-key"
        engine.client.count.return_value = MagicMock(count=10)
        engine.client.scroll.return_value = ([
            MagicMock(id=1, payload={"_text": "Sample document text " * 50}),
            MagicMock(id=2, payload={"_text": "Another sample document " * 50})
        ], None)
        return engine

    @pytest.fixture
    def mock_config(self):
        """Create a mocked collection strategy."""
        config = MagicMock()
        config.collection_name = "test_collection"
        return config

    def test_evaluator_init(self, mock_engine, mock_config):
        """Test LLMJudgeEvaluator initialization."""
        from maxq.eval_llm_judge import LLMJudgeEvaluator

        with patch('openai.OpenAI') as mock_openai:
            evaluator = LLMJudgeEvaluator(mock_engine, mock_config)
            assert evaluator.engine == mock_engine
            assert evaluator.config == mock_config

    def test_evaluator_sets_env_var(self, mock_engine, mock_config):
        """Test that evaluator sets OPENAI_API_KEY env var."""
        from maxq.eval_llm_judge import LLMJudgeEvaluator
        import os

        with patch('openai.OpenAI'):
            LLMJudgeEvaluator(mock_engine, mock_config)
            assert os.environ.get("OPENAI_API_KEY") == "test-key"

    def test_generate_testset_empty_collection(self, mock_engine, mock_config):
        """Test testset generation with empty collection."""
        from maxq.eval_llm_judge import LLMJudgeEvaluator

        mock_engine.client.count.return_value = MagicMock(count=0)

        with patch('openai.OpenAI'):
            evaluator = LLMJudgeEvaluator(mock_engine, mock_config)
            testset = evaluator.generate_testset(num_samples=5)
            assert testset == []

    def test_evaluate_empty_testset(self, mock_engine, mock_config):
        """Test evaluate handles empty testset gracefully."""
        from maxq.eval_llm_judge import LLMJudgeEvaluator

        with patch('openai.OpenAI'):
            evaluator = LLMJudgeEvaluator(mock_engine, mock_config)
            result = evaluator.evaluate([])
            assert result is None

    def test_backward_compatibility_alias(self):
        """Test MaxQEvaluator alias exists for backward compatibility."""
        from maxq.eval_llm_judge import MaxQEvaluator, LLMJudgeEvaluator
        assert MaxQEvaluator is LLMJudgeEvaluator

    def test_ragas_available_flag(self):
        """Test that RAGAS_AVAILABLE flag exists."""
        from maxq.eval_llm_judge import RAGAS_AVAILABLE
        # Should be a boolean
        assert isinstance(RAGAS_AVAILABLE, bool)


# ============================================
# Tests for RagasManager
# ============================================

class TestRagasManager:
    """Tests for RagasManager class."""

    def test_ragas_available_flag(self):
        """Test RAGAS_AVAILABLE flag is exported."""
        from maxq.ragas_utils import RAGAS_AVAILABLE
        assert isinstance(RAGAS_AVAILABLE, bool)

    def test_ragas_manager_raises_when_unavailable(self):
        """Test RagasManager raises ImportError when Ragas is unavailable."""
        from maxq.ragas_utils import RagasManager, RAGAS_AVAILABLE

        if not RAGAS_AVAILABLE:
            with pytest.raises(ImportError):
                RagasManager("test-key")

    def test_ragas_manager_init_when_available(self):
        """Test RagasManager initialization when Ragas is available."""
        from maxq.ragas_utils import RagasManager, RAGAS_AVAILABLE
        import os

        if RAGAS_AVAILABLE:
            # If Ragas works, test full initialization
            manager = RagasManager("test-api-key")
            assert os.environ.get("OPENAI_API_KEY") == "test-api-key"
        else:
            # If not available, just verify the import doesn't crash
            assert True

    def test_generate_testset_returns_empty_for_empty_collection(self):
        """Test testset generation with no points."""
        from maxq.ragas_utils import RagasManager, RAGAS_AVAILABLE

        if not RAGAS_AVAILABLE:
            pytest.skip("Ragas not available in this environment")

        mock_engine = MagicMock()
        mock_engine.client.scroll.return_value = ([], None)

        manager = RagasManager("test-key")
        result = manager.generate_testset(mock_engine, "test_collection")
        assert result == []

    def test_evaluate_results_empty(self):
        """Test evaluate_results with empty results."""
        from maxq.ragas_utils import RagasManager, RAGAS_AVAILABLE

        if not RAGAS_AVAILABLE:
            pytest.skip("Ragas not available in this environment")

        manager = RagasManager("test-key")
        result = manager.evaluate_results([])
        assert result == {}


# ============================================
# Tests for Metric Calculations
# ============================================

class TestMetricCalculations:
    """Tests for evaluation metric calculations."""

    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        hits = [1, 0, 1, 1, 0]  # 3 hits out of 5
        expected_hit_rate = 0.6

        calculated = np.mean(hits)
        assert calculated == expected_hit_rate

    def test_mrr_calculation(self):
        """Test MRR (Mean Reciprocal Rank) is calculated correctly."""
        reciprocal_ranks = [1.0, 0.5, 1/3, 0, 0.25]  # ranks 1, 2, 3, miss, 4
        expected_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        calculated = np.mean(reciprocal_ranks)
        assert abs(calculated - expected_mrr) < 0.01

    def test_ndcg_vs_mrr_difference(self):
        """Test that nDCG and MRR differ in how they weight positions."""
        from maxq.eval import Evaluator
        import math

        retrieved = [2, 1, 3, 4, 5]  # relevant doc at position 2
        relevant = 1

        mrr = Evaluator.calculate_mrr(retrieved, relevant)  # 1/2 = 0.5
        ndcg = Evaluator.calculate_ndcg(retrieved, relevant, k=5)

        # nDCG uses log discounting, MRR uses linear
        # They should give different values for non-first positions
        assert mrr == 0.5
        assert ndcg != mrr  # Different formulas


# Module-level fixtures for backward compatibility
@pytest.fixture
def mock_engine():
    """Module-level mock engine fixture."""
    engine = MagicMock()
    engine.openai_api_key = "test-key"
    engine.client.count.return_value = MagicMock(count=10)
    engine.client.scroll.return_value = ([
        MagicMock(id=1, payload={"_text": "Sample document text " * 50})
    ], None)
    return engine


@pytest.fixture
def mock_config():
    """Module-level mock config fixture."""
    config = MagicMock()
    config.collection_name = "test_collection"
    return config
