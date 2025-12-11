"""
MaxQ Search Engine - Cloud Inference Version

Uses Qdrant Cloud Inference for all embeddings.
No local model dependencies required.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, computed_field
from qdrant_client import QdrantClient, models
from qdrant_client.models import Document
from datasets import load_dataset
from openai import OpenAI

from .autoconfig import (
    CLOUD_DENSE_MODELS,
    CloudConfig,
    get_preset,
    create_config,
)

# Cloud inference model defaults
DEFAULT_DENSE_MODEL = os.getenv("MAXQ_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SPARSE_MODEL = os.getenv("MAXQ_SPARSE_MODEL", "Qdrant/bm25")


class CollectionStrategy(BaseModel):
    """Configuration for a Qdrant Cloud collection."""

    collection_name: str
    estimated_doc_count: int = Field(default=100_000)
    dense_model_name: str = DEFAULT_DENSE_MODEL
    sparse_model_name: str = DEFAULT_SPARSE_MODEL
    use_quantization: bool = True

    # HNSW settings
    hnsw_m: int = 32
    hnsw_ef_construct: int = 200

    # Infrastructure
    shard_number: Optional[int] = None  # Auto-calculated if None
    replication_factor: int = 1
    multi_tenant: bool = False
    on_disk: bool = True

    @computed_field
    def calculated_shards(self) -> int:
        if self.shard_number is not None:
            return self.shard_number
        if self.estimated_doc_count < 100_000:
            return 1
        elif self.estimated_doc_count < 1_000_000:
            return 2
        else:
            return max(2, self.estimated_doc_count // 500_000)

    @property
    def dense_size(self) -> int:
        return CLOUD_DENSE_MODELS.get(self.dense_model_name, {}).get("size", 384)

    @classmethod
    def from_cloud_config(cls, collection_name: str, config: CloudConfig) -> "CollectionStrategy":
        """Create CollectionStrategy from a CloudConfig preset."""
        return cls(
            collection_name=collection_name,
            dense_model_name=config.dense_model,
            sparse_model_name=config.sparse_model,
            use_quantization=config.quantization is not None,
            hnsw_m=config.hnsw.m,
            hnsw_ef_construct=config.hnsw.ef_construct,
            shard_number=config.shard_number,
            replication_factor=config.replication_factor,
            multi_tenant=config.multi_tenant,
            on_disk=config.on_disk,
        )


class SearchRequest(BaseModel):
    """Search query configuration."""

    query: str
    limit: int = 10
    strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    score_threshold: float = 0.0


class MaxQEngine:
    """
    MaxQ Engine using Qdrant Cloud Inference.
    All embeddings are generated server-side by Qdrant Cloud.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "Qdrant Cloud credentials required. "
                "Set QDRANT_URL and QDRANT_API_KEY environment variables."
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True,
            timeout=60,
            cloud_inference=True,  # Enable server-side embeddings
        )

        self.openai_api_key = openai_api_key
        self.llm_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    def collection_exists(self, name: str) -> bool:
        return self.client.collection_exists(name)

    @staticmethod
    def get_collection_name(project_id: str, model_name: str) -> str:
        """Generate consistent collection name from project and model."""
        safe_model = model_name.replace("/", "_").replace("-", "_").lower()
        return f"{project_id}_{safe_model}"

    def initialize_collection(self, config: CollectionStrategy):
        """Create collection with dense + sparse vectors for cloud inference."""

        # Build HNSW config
        hnsw_config = models.HnswConfigDiff(m=config.hnsw_m, ef_construct=config.hnsw_ef_construct)

        # Build quantization config
        quantization_config = None
        if config.use_quantization:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            )

        # Dense vector params
        dense_params = models.VectorParams(
            size=config.dense_size,
            distance=models.Distance.COSINE,
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
            on_disk=config.on_disk,
        )

        # Sparse vector params with IDF modifier for BM25
        # Use model name to derive sparse vector name (e.g., "Qdrant/bm25" -> "bm25")
        sparse_name = config.sparse_model_name.split("/")[-1].lower().replace("-", "_")
        sparse_params = models.SparseVectorParams(
            modifier=models.Modifier.IDF, index=models.SparseIndexParams(on_disk=config.on_disk)
        )

        self.client.recreate_collection(
            collection_name=config.collection_name,
            vectors_config={"dense": dense_params},
            sparse_vectors_config={sparse_name: sparse_params},
            shard_number=config.calculated_shards,  # type: ignore
            replication_factor=config.replication_factor,
        )

    def _upload_batch(
        self, config: CollectionStrategy, texts: List[str], payloads: List[Dict], start_id: int
    ):
        """Upload batch using cloud inference."""
        # Use model name to derive sparse vector name (e.g., "Qdrant/bm25" -> "bm25")
        sparse_name = config.sparse_model_name.split("/")[-1].lower().replace("-", "_")
        points = []
        for i, text in enumerate(texts):
            meta = payloads[i].copy()
            meta["_text"] = text

            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={
                        "dense": Document(text=text, model=config.dense_model_name),
                        sparse_name: Document(text=text, model=config.sparse_model_name),
                    },
                    payload=meta,
                )
            )

        self.client.upload_points(
            collection_name=config.collection_name, points=points, batch_size=8, wait=True
        )

    def ingest_hf(
        self,
        dataset_name: str,
        config: CollectionStrategy,
        limit: int,
        embedding_column: str = None,
        callback=None,
    ) -> int:
        """Ingest HuggingFace dataset using cloud inference."""
        self.initialize_collection(config)

        dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        batch_text, batch_payload = [], []
        count = 0

        for row in dataset:
            if count >= limit:
                break

            text = None
            if embedding_column and embedding_column in row:
                text = str(row[embedding_column])
            else:
                text = next(
                    (str(v) for k, v in row.items() if isinstance(v, str) and len(str(v)) > 20),
                    None,
                )

            if not text:
                continue

            batch_text.append(text)
            batch_payload.append(row)
            count += 1

            if len(batch_text) >= 50:
                self._upload_batch(config, batch_text, batch_payload, count - 50)
                if callback:
                    callback(50)
                batch_text, batch_payload = [], []

        if batch_text:
            self._upload_batch(config, batch_text, batch_payload, count)
            if callback:
                callback(len(batch_text))

        return count

    def ingest_local(
        self,
        folder_path: str,
        config: CollectionStrategy,
        limit: int,
        glob_pattern: str = "**/*.*",
        callback=None,
    ) -> int:
        """Ingest local files using cloud inference."""
        import glob as globlib

        self.initialize_collection(config)

        files = globlib.glob(os.path.join(folder_path, glob_pattern), recursive=True)
        batch_text, batch_payload = [], []
        count = 0

        for file_path in files:
            if count >= limit:
                break
            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if len(text) < 10:
                        continue

                    batch_text.append(text)
                    batch_payload.append(
                        {"source": file_path, "filename": os.path.basename(file_path)}
                    )
                    count += 1

                    if len(batch_text) >= 50:
                        self._upload_batch(config, batch_text, batch_payload, count - 50)
                        if callback:
                            callback(50)
                        batch_text, batch_payload = [], []
            except Exception:
                continue

        if batch_text:
            self._upload_batch(config, batch_text, batch_payload, count)
            if callback:
                callback(len(batch_text))

        return count

    def query(self, config: CollectionStrategy, request: SearchRequest) -> List[models.ScoredPoint]:
        """Hybrid search using cloud inference."""
        # Use model name to derive sparse vector name (e.g., "Qdrant/bm25" -> "bm25")
        sparse_name = config.sparse_model_name.split("/")[-1].lower().replace("-", "_")

        if request.strategy == "hybrid":
            prefetch = [
                models.Prefetch(
                    query=Document(text=request.query, model=config.dense_model_name),
                    using="dense",
                    limit=request.limit,
                ),
                models.Prefetch(
                    query=Document(text=request.query, model=config.sparse_model_name),
                    using=sparse_name,  # Use dynamic name (e.g., "bm25")
                    limit=request.limit,
                ),
            ]
            return self.client.query_points(
                collection_name=config.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=request.limit,
                score_threshold=request.score_threshold,
                with_payload=True,
            ).points

        elif request.strategy == "dense":
            return self.client.query_points(
                collection_name=config.collection_name,
                query=Document(text=request.query, model=config.dense_model_name),
                using="dense",
                limit=request.limit,
                score_threshold=request.score_threshold,
                with_payload=True,
            ).points

        elif request.strategy == "sparse":
            return self.client.query_points(
                collection_name=config.collection_name,
                query=Document(text=request.query, model=config.sparse_model_name),
                using=sparse_name,  # Use dynamic name (e.g., "bm25")
                limit=request.limit,
                score_threshold=request.score_threshold,
                with_payload=True,
            ).points

        return []

    def generate_answer(self, query: str, context_points: List[Any]) -> str:
        """Generate answer using retrieved context."""
        if not self.llm_client:
            return "Error: OpenAI API Key not configured."

        context = "\n\n".join(
            [
                f"Document {i + 1}:\n{p.payload.get('_text', '')}"
                for i, p in enumerate(context_points)
            ]
        )

        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.3,
            stream=True,
        )
        return response


# Backwards compatibility
SearchEngine = MaxQEngine


# =============================================================================
# Engine/Sidecar Management Functions
# =============================================================================

import platform
import subprocess
import shutil


def get_engine_status() -> Dict[str, Any]:
    """Get the status of the optional Rust engine binary."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Determine platform
    if system == "darwin":
        plat = f"macos-{'arm64' if machine == 'arm64' else 'x64'}"
    elif system == "linux":
        plat = f"linux-{'arm64' if machine in ('aarch64', 'arm64') else 'x64'}"
    elif system == "windows":
        plat = "windows-x64"
    else:
        plat = "unknown"

    # Check if platform is supported
    supported = plat in ("macos-arm64", "macos-x64", "linux-x64", "linux-arm64", "windows-x64")

    # Check for binary
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "maxq")
    binary_name = "maxq-engine.exe" if system == "windows" else "maxq-engine"
    binary_path = os.path.join(cache_dir, binary_name)
    installed = os.path.exists(binary_path)

    return {
        "platform": plat,
        "version": "0.1.0",
        "supported": supported,
        "installed": installed,
        "binary_path": binary_path if installed else None,
        "cache_dir": cache_dir,
    }


def get_engine_mode() -> str:
    """Get the engine mode from environment variable."""
    return os.getenv("MAXQ_USE_ENGINE", "auto")


def is_engine_available() -> bool:
    """Check if the engine is currently running and available."""
    try:
        import socket

        port = int(os.getenv("MAXQ_ENGINE_PORT", "50051"))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def ensure_engine_binary(force_download: bool = False) -> Optional[str]:
    """
    Ensure the engine binary is available, downloading if necessary.

    Returns the path to the binary, or None if not available.
    """
    status = get_engine_status()

    if not status["supported"]:
        return None

    if status["installed"] and not force_download:
        return status["binary_path"]

    # Create cache directory
    os.makedirs(status["cache_dir"], exist_ok=True)

    # TODO: Download binary from releases
    # For now, just check if it exists
    if status["installed"]:
        return status["binary_path"]

    return None


def start_engine(qdrant_url: str = None, grpc_port: int = 50051) -> Optional[subprocess.Popen]:
    """
    Start the engine process.

    Returns the subprocess.Popen object, or None if failed.
    """
    binary_path = ensure_engine_binary()

    if not binary_path:
        return None

    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6334")

    try:
        process = subprocess.Popen(
            [binary_path, "--qdrant-url", qdrant_url, "--port", str(grpc_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process
    except Exception:
        return None


# Alias for CLI compatibility
start_engine_fn = start_engine
