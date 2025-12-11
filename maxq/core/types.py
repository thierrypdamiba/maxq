"""Core data types for MaxQ."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Status of a run."""

    QUEUED = "queued"
    INDEXING = "indexing"
    INDEXED = "indexed"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    REPORTING = "reporting"
    DONE = "done"
    FAILED = "failed"


class JobType(str, Enum):
    """Type of job."""

    INDEX = "index"
    EVAL = "eval"
    REPORT = "report"


class JobStatus(str, Enum):
    """Status of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Document(BaseModel):
    """A source document."""

    id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of text from a document."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalQuery(BaseModel):
    """An evaluation query with ground truth."""

    id: str
    query: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)  # chunk IDs
    expected_answer: str = ""  # Ground truth answer for RAG evaluation
    metadata: dict[str, Any] = Field(default_factory=dict)  # Additional metadata


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    score: float
    doc_id: str
    text: str = ""


class QueryResult(BaseModel):
    """Results for a single query."""

    query_id: str
    query: str
    results: list[SearchResult]
    relevant_doc_ids: list[str]
    relevant_ids: list[str]


class Metrics(BaseModel):
    """IR evaluation metrics."""

    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    total_queries: int = 0
    queries_with_hits: int = 0
    # Latency metrics (in milliseconds)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    latency_mean_ms: float | None = None


class RunConfig(BaseModel):
    """Configuration for a run (index-time settings only)."""

    collection: str
    dataset_path: str
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Note: top_k is NOT here - it's an eval-time parameter, not index-time


class Run(BaseModel):
    """A complete run record."""

    run_id: str
    status: RunStatus
    config: RunConfig
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    metrics: Metrics | None = None
