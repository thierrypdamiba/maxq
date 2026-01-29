"""Application configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment."""

    # MaxQ Mode: cloud, local, tei, or auto
    # - cloud: Use Qdrant Cloud Inference (embeddings server-side)
    # - local: Use FastEmbed for local embeddings (limited models)
    # - tei: Use Text Embeddings Inference server (any HuggingFace model)
    # - auto: Detect based on environment (TEI_URL > cloud URL > local)
    mode: Literal["cloud", "local", "tei", "auto"] = "auto"

    # TEI (Text Embeddings Inference) server URL
    # Run any HuggingFace model via: docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest --model-id MODEL
    tei_url: str = ""

    # Qdrant - Cloud (recommended), Local Docker, or In-Memory
    # For Cloud: set MAXQ_QDRANT_URL and MAXQ_QDRANT_API_KEY
    # For Local Docker: set MAXQ_QDRANT_HOST and MAXQ_QDRANT_PORT
    # For In-Memory: set MAXQ_QDRANT_MODE=memory (for testing)
    qdrant_mode: str = ""  # Set to "memory" for in-memory mode (testing only)
    qdrant_url: str = ""  # e.g., "https://xyz-example.us-east-1-0.aws.cloud.qdrant.io:6333"
    qdrant_api_key: str = ""  # API key from Qdrant Cloud dashboard
    qdrant_host: str = "localhost"  # For local Docker instance
    qdrant_port: int = 6333  # For local Docker instance

    # Database
    db_path: str = "maxq.db"

    # Runs
    runs_dir: str = "runs"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Worker
    worker_poll_interval: float = 1.0

    # Cloud Inference Model Defaults
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"

    # Model dimensions for cloud inference
    model_dimensions: dict[str, int] = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-minilm-l6-v2": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "mixedbread-ai/mxbai-embed-large-v1": 1024,
    }

    # Optional services
    linkup_api_key: str = ""
    openai_api_key: str = ""

    # App directory
    app_dir: Path = Path.home() / ".maxq"

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_prefix="MAXQ_",
        env_file=".env",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure app directory exists
        self.app_dir.mkdir(exist_ok=True)

    def get_model_dimension(self, model: str) -> int:
        """Get dimension for a model, with fallback."""
        return self.model_dimensions.get(model, 384)

    def get_effective_mode(self) -> str:
        """Get the effective mode (resolves 'auto' to cloud, local, or tei)."""
        if self.mode != "auto":
            return self.mode
        # Auto-detect based on environment
        if self.tei_url:
            return "tei"
        if "cloud.qdrant.io" in self.qdrant_url:
            return "cloud"
        return "local"

    def is_cloud_mode(self) -> bool:
        """Check if running in cloud mode."""
        return self.get_effective_mode() == "cloud"

    def is_local_mode(self) -> bool:
        """Check if running in local mode (FastEmbed)."""
        return self.get_effective_mode() == "local"

    def is_tei_mode(self) -> bool:
        """Check if running in TEI mode (any HuggingFace model)."""
        return self.get_effective_mode() == "tei"


settings = Settings()
