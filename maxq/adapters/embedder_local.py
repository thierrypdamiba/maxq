"""Local embedder using FastEmbed for Docker/self-hosted deployments."""

from typing import Optional

from maxq.core.embedding import Embedder


# Model dimensions for FastEmbed-supported models
FASTEMBED_MODELS = {
    # Default / recommended
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    # Sentence transformers
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    # Multilingual
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    # Nomic
    "nomic-ai/nomic-embed-text-v1": 768,
    # Jina
    "jinaai/jina-embeddings-v2-small-en": 512,
    "jinaai/jina-embeddings-v2-base-en": 768,
}

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder(Embedder):
    """Local embedder using FastEmbed.

    FastEmbed is a lightweight, fast embedding library that runs models
    locally using ONNX runtime. No GPU required.

    Usage:
        embedder = LocalEmbedder()  # Uses default model
        embedder = LocalEmbedder("BAAI/bge-base-en-v1.5")  # Custom model

        vectors = embedder.embed(["text 1", "text 2"])
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, cache_dir: Optional[str] = None):
        """Initialize local embedder.

        Args:
            model: Model name (must be FastEmbed-compatible)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model
        self._dimension = FASTEMBED_MODELS.get(model, 384)
        self._cache_dir = cache_dir
        self._model = None

    def _ensure_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Install with:\n"
                "  pip install fastembed\n"
                "Or use cloud mode instead."
            )

        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=self._cache_dir,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._ensure_model()
        # FastEmbed returns a generator, convert to list
        embeddings = list(self._model.embed(texts))
        return [list(emb) for emb in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed([text])[0]


class LocalSparseEmbedder:
    """Local sparse embedder using FastEmbed's BM25 implementation."""

    def __init__(self, model: str = "Qdrant/bm25"):
        """Initialize sparse embedder.

        Args:
            model: Sparse model name (currently only BM25 supported locally)
        """
        self.model_name = model
        self._model = None

    def _ensure_model(self):
        """Lazy-load the sparse model."""
        if self._model is not None:
            return

        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Install with:\n"
                "  pip install fastembed"
            )

        self._model = SparseTextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[dict]:
        """Embed texts to sparse vectors.

        Args:
            texts: List of texts to embed

        Returns:
            List of sparse vectors as {indices: [...], values: [...]}
        """
        self._ensure_model()
        embeddings = list(self._model.embed(texts))
        return [
            {"indices": list(emb.indices), "values": list(emb.values)}
            for emb in embeddings
        ]

    def embed_single(self, text: str) -> dict:
        """Embed a single text to sparse vector."""
        return self.embed([text])[0]


def get_embedder(mode: str = "auto", model: Optional[str] = None, tei_url: Optional[str] = None) -> Embedder:
    """Get the appropriate embedder based on mode.

    Args:
        mode: "cloud", "local", "tei", or "auto"
              - cloud: Use Qdrant Cloud Inference (server-side)
              - local: Use FastEmbed locally (limited models)
              - tei: Use Text Embeddings Inference server (any HF model)
              - auto: Detect based on environment
        model: Optional model name override
        tei_url: Optional TEI server URL (for tei mode)

    Returns:
        Embedder instance
    """
    import os

    if mode == "auto":
        # Auto-detect based on environment
        tei_url_env = os.getenv("MAXQ_TEI_URL")
        qdrant_url = os.getenv("QDRANT_URL", "")

        if tei_url_env:
            mode = "tei"
        elif "cloud.qdrant.io" in qdrant_url:
            mode = "cloud"
        else:
            mode = "local"

    if mode == "cloud":
        from maxq.adapters.embedder_cloud import CloudEmbedder
        return CloudEmbedder(model=model)
    elif mode == "tei":
        from maxq.adapters.embedder_tei import TEIEmbedder
        return TEIEmbedder(url=tei_url)
    else:
        return LocalEmbedder(model=model or DEFAULT_LOCAL_MODEL)
