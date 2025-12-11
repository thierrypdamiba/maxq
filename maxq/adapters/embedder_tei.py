"""Text Embeddings Inference (TEI) embedder for any HuggingFace model.

TEI is a Rust-based server that can run ANY HuggingFace embedding model.
Run it via Docker:

    docker run -p 8080:80 \
        -v $PWD/data:/data \
        ghcr.io/huggingface/text-embeddings-inference:latest \
        --model-id BAAI/bge-large-en-v1.5

Then set:
    MAXQ_TEI_URL=http://localhost:8080

Supports:
- Any sentence-transformers model
- Any model with pooling config
- GPU acceleration
- Batching and caching
"""

import os
from typing import Optional

import httpx

from maxq.core.embedding import Embedder


DEFAULT_TEI_URL = "http://localhost:8080"


class TEIEmbedder(Embedder):
    """Embedder using HuggingFace Text Embeddings Inference server.

    TEI can run ANY HuggingFace embedding model efficiently via Docker.
    No Python ML dependencies required - just HTTP calls.

    Usage:
        # Start TEI server with any model:
        docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \\
            --model-id sentence-transformers/all-mpnet-base-v2

        # Use in MaxQ:
        embedder = TEIEmbedder()  # Uses localhost:8080
        embedder = TEIEmbedder("http://tei:8080")  # Custom URL

        vectors = embedder.embed(["text 1", "text 2"])
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize TEI embedder.

        Args:
            url: TEI server URL (default: MAXQ_TEI_URL or localhost:8080)
            timeout: Request timeout in seconds
        """
        self.url = url or os.getenv("MAXQ_TEI_URL", DEFAULT_TEI_URL)
        self.timeout = timeout
        self._dimension: Optional[int] = None
        self._client = httpx.Client(timeout=timeout)

    def _get_info(self) -> dict:
        """Get model info from TEI server."""
        response = self._client.get(f"{self.url}/info")
        response.raise_for_status()
        return response.json()

    @property
    def dimension(self) -> int:
        """Get embedding dimension from server."""
        if self._dimension is None:
            try:
                info = self._get_info()
                # TEI returns max_input_length, we need to get dim from a test embed
                test_result = self.embed(["test"])
                self._dimension = len(test_result[0])
            except Exception:
                self._dimension = 768  # Fallback
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via TEI server.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self._client.post(
            f"{self.url}/embed",
            json={"inputs": texts},
        )
        response.raise_for_status()
        return response.json()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]

    def health_check(self) -> bool:
        """Check if TEI server is healthy."""
        try:
            response = self._client.get(f"{self.url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        try:
            return self._get_info()
        except Exception as e:
            return {"error": str(e)}


class AsyncTEIEmbedder:
    """Async version of TEI embedder for high-throughput scenarios."""

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.url = url or os.getenv("MAXQ_TEI_URL", DEFAULT_TEI_URL)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously."""
        await self._ensure_client()
        response = await self._client.post(
            f"{self.url}/embed",
            json={"inputs": texts},
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the async client."""
        if self._client:
            await self._client.aclose()
            self._client = None
