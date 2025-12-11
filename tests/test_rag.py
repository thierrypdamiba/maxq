"""
Tests for RAG pipeline functionality.

Tests the StandardRAG and SpeculativeRAG pipelines, as well as the LLM Judge.
"""

import pytest
from unittest.mock import MagicMock, patch
from maxq.core.rag.pipeline import RAGPipeline, RAGResult, RAGMetrics, RetrievedDocument
from maxq.core.rag.standard import StandardRAG
from maxq.core.rag.speculative import SpeculativeRAG, Draft
from maxq.core.judge.llm_judge import LLMJudge, JudgeResult


# --- Test Fixtures ---


@pytest.fixture
def mock_retriever():
    """Create a mock retriever that returns sample documents.

    Uses spec to ensure hasattr() returns False for query_points and query,
    so the retrieve() method falls through to the generic search() path.
    """
    # Create a simple class spec with only search method
    class RetrieverSpec:
        def search(self, **kwargs): pass

    mock = MagicMock(spec=RetrieverSpec)
    mock.search.return_value = [
        MagicMock(
            id="doc1",
            score=0.95,
            payload={"text": "Vector databases store embeddings for similarity search."},
        ),
        MagicMock(
            id="doc2",
            score=0.85,
            payload={"text": "Qdrant is a high-performance vector database."},
        ),
        MagicMock(
            id="doc3",
            score=0.75,
            payload={"text": "Embeddings capture semantic meaning of text."},
        ),
    ]
    return mock


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Vector databases store embeddings as high-dimensional vectors for fast similarity search."
            )
        )
    ]
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = mock_openai_response
    return mock


# --- RetrievedDocument Tests ---


class TestRetrievedDocument:
    """Tests for RetrievedDocument model."""

    def test_create_retrieved_document(self):
        """Test creating a retrieved document."""
        doc = RetrievedDocument(
            doc_id="doc1",
            text="Sample document text",
            score=0.95,
            metadata={"source": "test"},
        )
        assert doc.doc_id == "doc1"
        assert doc.text == "Sample document text"
        assert doc.score == 0.95
        assert doc.metadata == {"source": "test"}

    def test_retrieved_document_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        doc = RetrievedDocument(doc_id="doc1", text="Text", score=0.5)
        assert doc.metadata == {}


# --- RAGResult Tests ---


class TestRAGResult:
    """Tests for RAGResult model."""

    def test_create_rag_result(self):
        """Test creating a RAG result."""
        result = RAGResult(
            query="What is a vector database?",
            answer="A vector database stores embeddings.",
            total_latency_ms=150.5,
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.5,
            pipeline_type="StandardRAG",
            model_used="gpt-4o-mini",
            num_docs_retrieved=5,
        )
        assert result.query == "What is a vector database?"
        assert result.answer == "A vector database stores embeddings."
        assert result.total_latency_ms == 150.5
        assert result.pipeline_type == "StandardRAG"

    def test_rag_result_with_drafts(self):
        """Test RAG result with speculative drafts."""
        result = RAGResult(
            query="Test query",
            answer="Test answer",
            rationale="Because the context supports this.",
            drafts=[
                {"answer": "Draft 1", "score": 0.8},
                {"answer": "Draft 2", "score": 0.7},
            ],
            drafting_latency_ms=200.0,
            verification_latency_ms=100.0,
        )
        assert result.rationale == "Because the context supports this."
        assert len(result.drafts) == 2
        assert result.drafting_latency_ms == 200.0


# --- RAGMetrics Tests ---


class TestRAGMetrics:
    """Tests for RAGMetrics model."""

    def test_create_rag_metrics(self):
        """Test creating RAG metrics."""
        metrics = RAGMetrics(
            faithfulness=0.85,
            relevance=0.90,
            correctness=0.80,
            context_precision=0.75,
            latency_p50_ms=100.0,
            latency_p95_ms=250.0,
            total_queries=100,
            successful_queries=98,
            failed_queries=2,
        )
        assert metrics.faithfulness == 0.85
        assert metrics.relevance == 0.90
        assert metrics.total_queries == 100
        assert metrics.failed_queries == 2

    def test_rag_metrics_defaults(self):
        """Test RAG metrics default values."""
        metrics = RAGMetrics()
        assert metrics.faithfulness == 0.0
        assert metrics.total_queries == 0
        assert metrics.latency_p50_ms == 0.0


# --- StandardRAG Tests ---


class TestStandardRAG:
    """Tests for StandardRAG pipeline."""

    def test_init_standard_rag(self, mock_retriever):
        """Test initializing StandardRAG."""
        pipeline = StandardRAG(
            retriever=mock_retriever,
            generator_model="gpt-4o-mini",
            collection_name="test_collection",
            top_k=10,
        )
        assert pipeline.generator_model == "gpt-4o-mini"
        assert pipeline.collection_name == "test_collection"
        assert pipeline.top_k == 10
        assert pipeline.pipeline_type == "StandardRAG"

    def test_format_context(self, mock_retriever):
        """Test context formatting."""
        pipeline = StandardRAG(retriever=mock_retriever, collection_name="test")
        docs = [
            RetrievedDocument(doc_id="1", text="First document", score=0.9),
            RetrievedDocument(doc_id="2", text="Second document", score=0.8),
        ]
        context = pipeline.format_context(docs)
        assert "[1] First document" in context
        assert "[2] Second document" in context

    def test_run_standard_rag(self, mock_retriever, mock_openai_client):
        """Test running StandardRAG pipeline."""
        pipeline = StandardRAG(
            retriever=mock_retriever,
            generator_model="gpt-4o-mini",
            collection_name="test_collection",
            top_k=3,
            api_key="test-key",
        )

        # Mock the openai_client property
        pipeline._openai_client = mock_openai_client

        result = pipeline.run("What is a vector database?")

        assert isinstance(result, RAGResult)
        assert result.query == "What is a vector database?"
        assert result.answer is not None
        assert result.pipeline_type == "StandardRAG"
        assert result.total_latency_ms > 0
        assert result.num_docs_retrieved == 3

    def test_retrieve_documents(self, mock_retriever):
        """Test document retrieval."""
        pipeline = StandardRAG(
            retriever=mock_retriever,
            collection_name="test_collection",
            top_k=3,
        )

        docs, latency = pipeline.retrieve("test query")

        assert len(docs) == 3
        assert latency > 0
        assert docs[0].doc_id == "doc1"
        assert docs[0].score == 0.95


# --- SpeculativeRAG Tests ---


class TestSpeculativeRAG:
    """Tests for SpeculativeRAG pipeline."""

    def test_init_speculative_rag(self, mock_retriever):
        """Test initializing SpeculativeRAG."""
        pipeline = SpeculativeRAG(
            retriever=mock_retriever,
            drafter_model="gpt-4o-mini",
            verifier_model="gpt-4o",
            collection_name="test_collection",
            top_k=10,
            num_drafts=5,
            docs_per_draft=2,
        )
        assert pipeline.drafter_model == "gpt-4o-mini"
        assert pipeline.verifier_model == "gpt-4o"
        assert pipeline.num_drafts == 5
        assert pipeline.docs_per_draft == 2
        assert pipeline.pipeline_type == "SpeculativeRAG"

    def test_cluster_documents(self, mock_retriever):
        """Test document clustering."""
        pipeline = SpeculativeRAG(
            retriever=mock_retriever,
            collection_name="test",
            num_clusters=2,
        )

        docs = [
            RetrievedDocument(doc_id="1", text="Doc 1", score=0.9),
            RetrievedDocument(doc_id="2", text="Doc 2", score=0.8),
            RetrievedDocument(doc_id="3", text="Doc 3", score=0.7),
            RetrievedDocument(doc_id="4", text="Doc 4", score=0.6),
        ]

        clusters = pipeline.cluster_documents(docs, "test query")

        assert len(clusters) == 2
        # Each cluster should have docs
        for cluster in clusters:
            assert len(cluster) >= 1

    def test_sample_subsets(self, mock_retriever):
        """Test diverse subset sampling."""
        pipeline = SpeculativeRAG(
            retriever=mock_retriever,
            collection_name="test",
            num_drafts=3,
            docs_per_draft=2,
            num_clusters=2,
        )

        clusters = [
            [
                RetrievedDocument(doc_id="1", text="Doc 1", score=0.9),
                RetrievedDocument(doc_id="2", text="Doc 2", score=0.8),
            ],
            [
                RetrievedDocument(doc_id="3", text="Doc 3", score=0.7),
                RetrievedDocument(doc_id="4", text="Doc 4", score=0.6),
            ],
        ]

        subsets = pipeline.sample_subsets(clusters)

        # Should have up to num_drafts subsets
        assert len(subsets) <= 3
        # Each subset should have docs_per_draft documents
        for subset in subsets:
            assert len(subset) == 2

    def test_draft_model(self):
        """Test Draft model."""
        draft = Draft(
            answer="Test answer",
            rationale="Test rationale",
            doc_subset=["doc1", "doc2"],
            draft_score=0.8,
            self_consistency_score=0.7,
            self_reflection_score=0.9,
            final_score=0.5,
            latency_ms=100.0,
        )
        assert draft.answer == "Test answer"
        assert draft.final_score == 0.5
        assert len(draft.doc_subset) == 2

    def test_generate_draft(self, mock_retriever, mock_openai_client):
        """Test draft generation."""
        # Configure mock for drafting response
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="## Rationale:\nThe evidence shows that vector databases are useful.\n\n## Answer:\nVector databases enable fast similarity search."
                    )
                )
            ]
        )

        pipeline = SpeculativeRAG(
            retriever=mock_retriever,
            drafter_model="gpt-4o-mini",
            collection_name="test",
            api_key="test-key",
        )

        # Mock the client
        pipeline._openai_client = mock_openai_client

        docs = [
            RetrievedDocument(doc_id="1", text="Doc 1", score=0.9),
            RetrievedDocument(doc_id="2", text="Doc 2", score=0.8),
        ]

        draft = pipeline.generate_draft("What are vector databases?", docs)

        assert isinstance(draft, Draft)
        assert draft.answer is not None
        assert draft.latency_ms > 0


# --- LLMJudge Tests ---


class TestLLMJudge:
    """Tests for LLM Judge."""

    def test_init_judge(self):
        """Test initializing LLM Judge."""
        judge = LLMJudge(model="gpt-4o", api_key="test-key")
        assert judge.model == "gpt-4o"
        assert judge.api_key == "test-key"

    def test_judge_faithfulness(self, mock_openai_client):
        """Test faithfulness judgment."""
        # Configure mock to return JSON response
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"score": 0.85, "reasoning": "Answer is grounded"}')
                )
            ]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_faithfulness(
            question="What is Qdrant?",
            answer="Qdrant is a vector database used for searching similar items.",
            context="Qdrant is a vector database for similarity search.",
        )

        assert isinstance(result, JudgeResult)
        assert result.score == 0.85
        assert result.metric == "faithfulness"

    def test_judge_relevance(self, mock_openai_client):
        """Test relevance judgment."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"score": 0.90, "reasoning": "Answer addresses question"}'
                    )
                )
            ]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_relevance(
            question="How does vector search work?",
            answer="Vector search finds similar items by comparing their embeddings.",
        )

        assert isinstance(result, JudgeResult)
        assert result.metric == "relevance"

    def test_judge_correctness(self, mock_openai_client):
        """Test correctness judgment."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"score": 0.75, "reasoning": "Mostly correct"}')
                )
            ]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_correctness(
            question="What is 2+2?",
            answer="The answer is 4.",
            expected_answer="4",
        )

        assert isinstance(result, JudgeResult)
        assert result.metric == "correctness"

    def test_judge_context_precision(self, mock_openai_client):
        """Test context precision judgment."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"score": 0.80, "reasoning": "Most context is relevant"}'
                    )
                )
            ]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_context_precision(
            question="What is RAG?",
            context="RAG stands for Retrieval Augmented Generation. Weather today is sunny.",
            expected_answer="RAG combines retrieval with generation.",
        )

        assert isinstance(result, JudgeResult)
        assert result.metric == "context_precision"

    def test_parse_json_response(self, mock_openai_client):
        """Test parsing JSON response from LLM."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"score": 0.95, "reasoning": "Excellent answer"}')
                )
            ]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_relevance(question="Test?", answer="Test answer")

        assert result.score == 0.95
        assert result.reasoning == "Excellent answer"

    def test_parse_invalid_response(self, mock_openai_client):
        """Test handling of invalid LLM response."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Invalid response - not JSON"))]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        result = judge.judge_relevance(question="Test?", answer="Test answer")

        # Should default to 0.0 on parse error
        assert result.score == 0.0
        assert result.error is not None


# --- Integration Tests ---


class TestRAGIntegration:
    """Integration tests for RAG pipelines."""

    def test_pipeline_type_property(self, mock_retriever):
        """Test that pipeline type is correctly reported."""
        standard = StandardRAG(retriever=mock_retriever, collection_name="test")
        speculative = SpeculativeRAG(retriever=mock_retriever, collection_name="test")

        assert standard.pipeline_type == "StandardRAG"
        assert speculative.pipeline_type == "SpeculativeRAG"

    def test_rag_result_serialization(self):
        """Test that RAG results can be serialized to JSON."""
        result = RAGResult(
            query="Test query",
            answer="Test answer",
            retrieved_docs=[
                RetrievedDocument(doc_id="1", text="Doc 1", score=0.9),
            ],
            total_latency_ms=100.0,
            pipeline_type="StandardRAG",
        )

        # Should serialize without error
        json_data = result.model_dump()
        assert json_data["query"] == "Test query"
        assert len(json_data["retrieved_docs"]) == 1

    def test_rag_metrics_serialization(self):
        """Test that RAG metrics can be serialized to JSON."""
        metrics = RAGMetrics(
            faithfulness=0.85,
            relevance=0.90,
            latency_p50_ms=100.0,
            total_queries=50,
        )

        # Should serialize without error
        json_data = metrics.model_dump()
        assert json_data["faithfulness"] == 0.85
        assert json_data["total_queries"] == 50


# --- Edge Case Tests ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_documents(self, mock_retriever):
        """Test handling of empty document list."""
        mock_retriever.search.return_value = []
        pipeline = StandardRAG(retriever=mock_retriever, collection_name="test")

        docs, latency = pipeline.retrieve("test query")
        assert docs == []
        assert latency >= 0

    def test_cluster_with_few_docs(self, mock_retriever):
        """Test clustering with fewer docs than clusters."""
        pipeline = SpeculativeRAG(
            retriever=mock_retriever,
            collection_name="test",
            num_clusters=5,
        )

        docs = [
            RetrievedDocument(doc_id="1", text="Doc 1", score=0.9),
            RetrievedDocument(doc_id="2", text="Doc 2", score=0.8),
        ]

        clusters = pipeline.cluster_documents(docs, "test query")

        # Should return each doc as its own cluster when fewer docs than clusters
        assert len(clusters) == 2

    def test_rag_result_with_empty_answer(self):
        """Test RAG result with empty answer."""
        result = RAGResult(query="Test", answer="")
        assert result.answer == ""
        assert result.pipeline_type == "unknown"

    def test_judge_result_model(self):
        """Test JudgeResult model creation."""
        result = JudgeResult(
            metric="test",
            score=0.85,
            reasoning="Test reasoning",
        )
        assert result.score == 0.85
        assert result.metric == "test"
        assert result.reasoning == "Test reasoning"

    def test_evaluate_rag_response(self, mock_openai_client):
        """Test full RAG response evaluation."""
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"score": 0.85, "reasoning": "Good"}'))]
        )

        judge = LLMJudge(api_key="test-key")
        judge._client = mock_openai_client

        results = judge.evaluate_rag_response(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context="RAG combines retrieval with generation.",
            expected_answer="RAG is a technique for augmenting LLMs.",
        )

        assert "faithfulness" in results
        assert "relevance" in results
        assert "correctness" in results
        assert "context_precision" in results
        assert "context_recall" in results
