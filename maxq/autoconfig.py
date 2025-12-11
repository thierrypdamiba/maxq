"""
MaxQ Autoconfig Module (Cloud Inference)

Configuration presets for Qdrant Cloud Inference.
All embeddings are generated server-side by Qdrant Cloud - no API keys needed.
"""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field


# =============================================================================
# QDRANT CLOUD INFERENCE MODELS (No API Key Required)
# =============================================================================

CLOUD_DENSE_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "size": 384,
        "description": "Fast, general purpose English",
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "size": 1024,
        "description": "High accuracy, large model",
    },
}

CLOUD_SPARSE_MODELS = {
    "Qdrant/bm25": {
        "description": "Classic BM25 keyword search (free unlimited tokens)",
    },
    "prithivida/Splade_PP_en_v1": {
        "description": "SPLADE - Learned sparse embeddings",
    },
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HnswConfig(BaseModel):
    """HNSW index configuration."""
    m: int = 32
    ef_construct: int = 200


class QuantizationConfig(BaseModel):
    """Quantization configuration."""
    type: Literal["scalar", "binary"] = "scalar"
    always_ram: bool = True


class CloudConfig(BaseModel):
    """Configuration for Qdrant Cloud Inference collection."""
    name: str = "default"
    description: str = ""
    use_case: str = ""

    # Cloud inference models
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"

    # Collection infrastructure
    shard_number: int = 1
    replication_factor: int = 1
    multi_tenant: bool = False
    partition_key: Optional[str] = "tenant_id"

    # Index settings
    hnsw: HnswConfig = Field(default_factory=lambda: HnswConfig())
    quantization: Optional[QuantizationConfig] = Field(default_factory=lambda: QuantizationConfig())
    on_disk: bool = True

    @property
    def dense_size(self) -> int:
        return CLOUD_DENSE_MODELS.get(self.dense_model, {}).get("size", 384)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

PRESETS: Dict[str, CloudConfig] = {
    # === MiniLM Configs (384 dims, fast) ===
    "base": CloudConfig(
        name="Base",
        description="MiniLM + BM25 - Fast general purpose",
        use_case="Quick prototyping, low-latency apps",
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        sparse_model="Qdrant/bm25",
        hnsw=HnswConfig(m=16, ef_construct=100),
        quantization=QuantizationConfig(type="scalar"),
        on_disk=False,
    ),

    "base_splade": CloudConfig(
        name="Base + SPLADE",
        description="MiniLM + SPLADE - Learned sparse",
        use_case="Technical documents, better keyword matching",
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        sparse_model="prithivida/Splade_PP_en_v1",
        hnsw=HnswConfig(m=16, ef_construct=100),
        quantization=QuantizationConfig(type="scalar"),
        on_disk=False,
    ),

    # === Mixedbread Configs (1024 dims, accurate) ===
    "mxbai_bm25": CloudConfig(
        name="Mixedbread + BM25",
        description="mxbai-embed-large + BM25 - High accuracy",
        use_case="Production RAG, semantic search",
        dense_model="mixedbread-ai/mxbai-embed-large-v1",
        sparse_model="Qdrant/bm25",
        hnsw=HnswConfig(m=32, ef_construct=200),
        quantization=QuantizationConfig(type="scalar"),
        on_disk=True,
    ),

    "mxbai_splade": CloudConfig(
        name="Mixedbread + SPLADE",
        description="mxbai-embed-large + SPLADE - Best hybrid",
        use_case="Technical/scientific documents",
        dense_model="mixedbread-ai/mxbai-embed-large-v1",
        sparse_model="prithivida/Splade_PP_en_v1",
        hnsw=HnswConfig(m=32, ef_construct=200),
        quantization=QuantizationConfig(type="scalar"),
        on_disk=True,
    ),

    # === Infrastructure Configs ===
    "multi_tenant": CloudConfig(
        name="Multi-Tenant",
        description="Optimized for multi-tenant SaaS",
        use_case="SaaS platforms with tenant isolation",
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        sparse_model="Qdrant/bm25",
        multi_tenant=True,
        partition_key="tenant_id",
        shard_number=3,
        replication_factor=2,
        hnsw=HnswConfig(m=32, ef_construct=200),
        quantization=QuantizationConfig(type="scalar"),
    ),

    "high_scale": CloudConfig(
        name="High Scale",
        description="Configured for 1M+ documents",
        use_case="Enterprise, high-volume indexing",
        dense_model="mixedbread-ai/mxbai-embed-large-v1",
        sparse_model="Qdrant/bm25",
        shard_number=4,
        replication_factor=2,
        hnsw=HnswConfig(m=32, ef_construct=200),
        quantization=QuantizationConfig(type="scalar"),
        on_disk=True,
    ),

    "max_accuracy": CloudConfig(
        name="Max Accuracy",
        description="No quantization, dense HNSW index",
        use_case="Highest retrieval quality",
        dense_model="mixedbread-ai/mxbai-embed-large-v1",
        sparse_model="prithivida/Splade_PP_en_v1",
        hnsw=HnswConfig(m=64, ef_construct=400),
        quantization=None,
        on_disk=True,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> CloudConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name].model_copy()


def list_presets() -> List[Dict[str, str]]:
    """List available presets with descriptions."""
    return [
        {"name": k, "description": v.description, "use_case": v.use_case}
        for k, v in PRESETS.items()
    ]


def list_dense_models() -> List[Dict[str, Any]]:
    """List available cloud dense models."""
    return [{"model": k, **v} for k, v in CLOUD_DENSE_MODELS.items()]


def list_sparse_models() -> List[Dict[str, Any]]:
    """List available cloud sparse models."""
    return [{"model": k, **v} for k, v in CLOUD_SPARSE_MODELS.items()]


def estimate_shards(doc_count: int) -> int:
    """Estimate optimal shard count based on document count."""
    if doc_count < 100_000:
        return 1
    elif doc_count < 1_000_000:
        return 2
    elif doc_count < 10_000_000:
        return 4
    else:
        return max(4, doc_count // 2_500_000)


def create_config(
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sparse_model: str = "Qdrant/bm25",
    estimated_docs: int = 100_000,
    priority: Literal["speed", "balanced", "accuracy"] = "balanced",
    multi_tenant: bool = False,
) -> CloudConfig:
    """Create a custom configuration based on requirements."""
    hnsw_presets = {
        "speed": HnswConfig(m=16, ef_construct=100),
        "balanced": HnswConfig(m=32, ef_construct=200),
        "accuracy": HnswConfig(m=64, ef_construct=400),
    }

    quant = None if priority == "accuracy" else QuantizationConfig(type="scalar")

    return CloudConfig(
        name="custom",
        description=f"Custom config: {priority} priority",
        dense_model=dense_model,
        sparse_model=sparse_model,
        shard_number=estimate_shards(estimated_docs),
        replication_factor=2 if estimated_docs > 500_000 else 1,
        multi_tenant=multi_tenant,
        hnsw=hnsw_presets[priority],
        quantization=quant,
        on_disk=priority != "speed",
    )


def print_config_summary(config: CloudConfig) -> str:
    """Generate a human-readable summary of a configuration."""
    lines = [
        f"Configuration: {config.name}",
        f"Description: {config.description}",
        "",
        "Embedding Models:",
        f"  Dense: {config.dense_model} ({config.dense_size} dims)",
        f"  Sparse: {config.sparse_model}",
        "",
        "Index Settings:",
        f"  HNSW: m={config.hnsw.m}, ef_construct={config.hnsw.ef_construct}",
        f"  Quantization: {config.quantization.type if config.quantization else 'None'}",
        f"  On Disk: {config.on_disk}",
        "",
        "Infrastructure:",
        f"  Shards: {config.shard_number}",
        f"  Replication: {config.replication_factor}",
        f"  Multi-tenant: {config.multi_tenant}",
    ]
    return "\n".join(lines)
