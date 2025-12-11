"""
Enhanced indexing models for professional-grade index building.
Supports multi-step jobs, resumability, and comprehensive configuration.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class JobStageStatus(str, Enum):
    """Status of a job stage"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStage(BaseModel):
    """Represents a single stage in the indexing job pipeline"""
    name: str
    status: JobStageStatus = JobStageStatus.QUEUED
    active_form: str  # Present continuous form (e.g., "Validating plan")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Metrics
    docs_processed: int = 0
    chunks_created: int = 0
    points_upserted: int = 0
    errors: int = 0

    # Logs
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


# Data Source Configuration
class DataSourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    UPLOAD = "upload"
    S3 = "s3"
    URL = "url"
    JSONL = "jsonl"


class VectorizationStrategy(str, Enum):
    """How to convert data into text for embedding"""
    SINGLE_FIELD = "single_field"  # Use one field as-is
    COMBINE_FIELDS = "combine_fields"  # Combine multiple fields with template
    LLM_ENRICH = "llm_enrich"  # Use LLM to generate rich text
    IMAGE = "image"  # Handle image data
    MULTIMODAL = "multimodal"  # Mix of text and images


class VectorizationConfig(BaseModel):
    """Configuration for how to vectorize data"""
    strategy: VectorizationStrategy = VectorizationStrategy.SINGLE_FIELD

    # Single field strategy
    text_field: Optional[str] = "text"

    # Combine fields strategy
    combine_template: Optional[str] = None  # e.g., "{{title}}: {{description}}"
    combine_fields: Optional[List[str]] = None

    # LLM enrich strategy
    llm_enrich_prompt: Optional[str] = None
    llm_enrich_fields: Optional[List[str]] = None
    llm_enrich_model: str = "gpt-4o-mini"

    # Image strategy
    image_field: Optional[str] = None
    image_embedding_model: Optional[str] = "clip"

    # Multimodal strategy
    text_fields: Optional[List[str]] = None
    image_fields: Optional[List[str]] = None


class DataSourceConfig(BaseModel):
    """Configuration for data source"""
    source_type: DataSourceType

    # HuggingFace specific
    dataset_id: Optional[str] = None
    split: str = "train"
    streaming: bool = True

    # Upload/S3/URL specific
    path: Optional[str] = None

    # Vectorization strategy
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)

    # Legacy field mapping (for backward compatibility)
    text_field: Optional[str] = "text"
    id_field: Optional[str] = None
    title_field: Optional[str] = None

    # Sampling
    sample_limit: Optional[int] = None
    sample_offset: int = 0


# Chunking Configuration
class ChunkingStrategy(str, Enum):
    NONE = "none"
    FIXED_TOKENS = "fixed_tokens"
    SENTENCES = "sentences"
    MARKDOWN_HEADINGS = "markdown_headings"
    CODE_AWARE = "code_aware"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_TOKENS
    size: int = 512
    overlap: int = 50
    keep_parent_id: bool = True
    store_both_doc_and_chunk: bool = False


# Payload Schema Configuration
class PayloadFieldType(str, Enum):
    KEYWORD = "keyword"
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class PayloadField(BaseModel):
    """Definition of a payload field"""
    field_name: str
    field_type: PayloadFieldType
    indexed: bool = False
    full_text_indexed: bool = False
    notes: Optional[str] = None


class PayloadSchemaConfig(BaseModel):
    """Payload schema configuration"""
    fields: List[PayloadField] = Field(default_factory=list)
    auto_index_keywords: bool = True  # Auto-index fields like tenant_id, source, etc.


# Vector Space Configuration
class VectorProvider(str, Enum):
    LOCAL = "local"  # FastEmbed
    QCI = "qci"
    OPENAI = "openai"
    JINA = "jina"
    HF_INFERENCE = "hf_inference"
    CUSTOM = "custom"


class VectorDistance(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class VectorDataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    UINT8 = "uint8"


class DenseVectorConfig(BaseModel):
    """Dense vector space configuration"""
    name: str = "dense"
    provider: VectorProvider = VectorProvider.QCI  # Qdrant Cloud Inference
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Qdrant Cloud supported
    expected_dims: Optional[int] = None  # Auto-detect if None
    distance: VectorDistance = VectorDistance.COSINE
    datatype: VectorDataType = VectorDataType.FLOAT32
    batch_size: int = 32
    concurrency: int = 1


class SparseVectorConfig(BaseModel):
    """Sparse vector space configuration"""
    name: str = "sparse"
    enabled: bool = True
    generator: Literal["bm25", "splade", "precomputed"] = "bm25"  # Qdrant Cloud default
    model_name: Optional[str] = "Qdrant/bm25"  # Qdrant Cloud supported
    tokenizer_settings: Optional[Dict[str, Any]] = None


class LateInteractionConfig(BaseModel):
    """Late interaction vector space configuration"""
    name: str = "late"
    enabled: bool = False
    multi_vector: bool = True
    model_name: Optional[str] = None


class VectorSpacesConfig(BaseModel):
    """All vector spaces configuration"""
    dense: DenseVectorConfig
    sparse: Optional[SparseVectorConfig] = None
    late: Optional[LateInteractionConfig] = None


# Performance & Storage Configuration
class PresetType(str, Enum):
    DEV = "dev"
    PROD_SMALL = "prod_small"
    PROD_FAST_WRITES = "prod_fast_writes"
    PROD_FAST_READS = "prod_fast_reads"
    CUSTOM = "custom"


class HNSWConfig(BaseModel):
    """HNSW index configuration"""
    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10000
    on_disk: Optional[bool] = None


class QuantizationConfig(BaseModel):
    """Quantization configuration"""
    enabled: bool = False
    type: Literal["int8", "binary", "product"] = "int8"
    always_ram: bool = True


class OptimizersConfig(BaseModel):
    """Optimizers configuration"""
    indexing_threshold: int = 20000
    memmap_threshold: int = 50000


class PerformanceStorageConfig(BaseModel):
    """Performance and storage configuration"""
    preset: PresetType = PresetType.DEV

    # Sharding & Replication
    shard_number: int = 1
    replication_factor: int = 1
    write_consistency_factor: int = 1

    # HNSW
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)

    # Quantization
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    # Optimizers
    optimizers: OptimizersConfig = Field(default_factory=OptimizersConfig)

    # Storage
    on_disk_payload: bool = False


# Run Configuration
class RunConfig(BaseModel):
    """Configuration for running the indexing job"""
    dry_run: bool = False
    build_new_collection: bool = False  # Build into new collection
    swap_alias: bool = False  # Swap alias after successful build
    create_snapshot: bool = False
    run_verification: bool = True


# Complete Index Plan
class IndexPlan(BaseModel):
    """Complete index plan configuration"""
    id: str
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime

    # Configuration
    data_source: DataSourceConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    payload_schema: PayloadSchemaConfig = Field(default_factory=PayloadSchemaConfig)
    vector_spaces: VectorSpacesConfig
    performance: PerformanceStorageConfig = Field(default_factory=PerformanceStorageConfig)
    run_config: RunConfig = Field(default_factory=RunConfig)

    # Metadata
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# Job Status
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# Index Job
class IndexJob(BaseModel):
    """Represents a running instance of an index plan"""
    id: str
    plan_id: str
    project_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime

    status: JobStatus = JobStatus.QUEUED

    # Progress
    stages: List[JobStage] = Field(default_factory=list)
    current_stage_index: int = 0

    # Metrics
    total_docs: int = 0
    total_chunks: int = 0
    total_points: int = 0
    total_errors: int = 0

    # Timing
    estimated_duration_ms: Optional[int] = None
    actual_duration_ms: Optional[int] = None

    # Resume support
    checkpoint: Optional[Dict[str, Any]] = None  # Last processed doc offset, batch id, etc.

    # Results
    collection_name: Optional[str] = None
    verification_results: Optional[Dict[str, Any]] = None
    error_summary: Optional[Dict[str, Any]] = None


# Request/Response Models
class CreatePlanRequest(BaseModel):
    """Request to create a new index plan"""
    project_id: str
    name: str
    data_source: DataSourceConfig
    chunking: Optional[ChunkingConfig] = None
    payload_schema: Optional[PayloadSchemaConfig] = None
    vector_spaces: VectorSpacesConfig
    performance: Optional[PerformanceStorageConfig] = None
    run_config: Optional[RunConfig] = None
    description: Optional[str] = None


class StartJobRequest(BaseModel):
    """Request to start an indexing job"""
    plan_id: str
    dry_run: bool = False


class DryRunEstimate(BaseModel):
    """Estimate from a dry run"""
    estimated_docs: int
    estimated_chunks: int
    estimated_points: int
    estimated_storage_gb: float
    estimated_embed_time_minutes: float
    estimated_cost_usd: Optional[float] = None
    sample_chunks: List[Dict[str, Any]] = Field(default_factory=list)  # Preview of chunking


class JobProgress(BaseModel):
    """Real-time job progress"""
    job_id: str
    status: JobStatus
    current_stage: Optional[JobStage] = None
    progress_percent: float
    estimated_time_remaining_ms: Optional[int] = None
    recent_logs: List[str] = Field(default_factory=list)
