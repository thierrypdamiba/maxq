"""Base RAG Pipeline interface and common data structures."""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A document retrieved from the vector store."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Result from a RAG pipeline execution."""

    query: str
    answer: str
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)

    # Optional fields for advanced pipelines
    rationale: Optional[str] = None  # For Speculative RAG
    drafts: list[dict[str, Any]] = Field(default_factory=list)  # For Speculative RAG

    # Timing information (milliseconds)
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0

    # For Speculative RAG
    drafting_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0

    # Metadata
    pipeline_type: str = "unknown"
    model_used: str = ""
    num_docs_retrieved: int = 0


class RAGMetrics(BaseModel):
    """Aggregated metrics from RAG evaluation."""

    # Answer quality (LLM-judged)
    faithfulness: float = 0.0
    relevance: float = 0.0
    correctness: float = 0.0

    # Context quality (LLM-judged)
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Retrieval quality (computed)
    retrieval_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    retrieval_recall_at_k: dict[int, float] = Field(default_factory=dict)
    retrieval_mrr_at_k: dict[int, float] = Field(default_factory=dict)

    # Latency (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0

    # Pipeline-specific latencies
    retrieval_latency_mean_ms: float = 0.0
    generation_latency_mean_ms: float = 0.0

    # Speculative RAG specific
    drafting_latency_mean_ms: float = 0.0
    verification_latency_mean_ms: float = 0.0
    rationale_quality: float = 0.0

    # Counts
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0


class RAGPipeline(ABC):
    """Abstract base class for RAG pipelines."""

    def __init__(
        self,
        retriever,  # QdrantClient or similar
        generator_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        collection_name: str = "",
        top_k: int = 10,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: The retrieval client (e.g., QdrantClient)
            generator_model: Model to use for generation
            api_key: API key for the generator model
            collection_name: Name of the collection to search
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.generator_model = generator_model
        self.api_key = api_key
        self.collection_name = collection_name
        self.top_k = top_k

    @abstractmethod
    def run(self, query: str) -> RAGResult:
        """
        Run the RAG pipeline on a query.

        Args:
            query: The user's question

        Returns:
            RAGResult with the answer and metadata
        """
        pass

    def retrieve(self, query: str) -> tuple[list[RetrievedDocument], float]:
        """
        Retrieve relevant documents.

        Args:
            query: The query to search for

        Returns:
            Tuple of (documents, latency_ms)
        """
        start = time.perf_counter()

        # Support different retriever interfaces
        # 1. QdrantClient - use query_points with embedding
        # 2. MaxQEngine - use query method
        # 3. Generic - use search method

        results = []

        if hasattr(self.retriever, "query_points"):
            # QdrantClient - need to embed the query first
            try:
                from qdrant_client.models import models

                # Use Qdrant's built-in query (assumes collection has dense vectors)
                response = self.retriever.query_points(
                    collection_name=self.collection_name,
                    query=query,  # Qdrant Cloud can handle text queries with FastEmbed
                    limit=self.top_k,
                    with_payload=True,
                )
                results = response.points if hasattr(response, "points") else []
            except Exception:
                # Fallback: Try scroll for basic retrieval (no ranking)
                points, _ = self.retriever.scroll(
                    collection_name=self.collection_name,
                    limit=self.top_k,
                    with_payload=True,
                )
                results = points
        elif hasattr(self.retriever, "query"):
            # MaxQEngine style
            results = self.retriever.query(
                collection_name=self.collection_name,
                query=query,
                limit=self.top_k,
            )
        else:
            # Generic search method
            results = self.retriever.search(
                collection_name=self.collection_name,
                query_text=query,
                limit=self.top_k,
            )

        docs = []
        for r in results:
            # Get text from payload - try multiple common field names
            payload = r.payload or {}
            text = (
                payload.get("_text")
                or payload.get("text")
                or payload.get("content")
                or payload.get("prompt")
                or ""
            )

            docs.append(
                RetrievedDocument(
                    doc_id=str(r.id),
                    text=text,
                    score=getattr(r, "score", 0.0) or 0.0,
                    metadata=payload,
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000
        return docs, latency_ms

    def format_context(self, docs: list[RetrievedDocument]) -> str:
        """Format retrieved documents into context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] {doc.text}")
        return "\n\n".join(parts)

    @property
    def pipeline_type(self) -> str:
        """Return the pipeline type name."""
        return self.__class__.__name__
