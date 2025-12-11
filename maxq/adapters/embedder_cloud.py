"""Cloud Inference embedder using Qdrant Cloud."""

from maxq.core.config import settings
from maxq.core.embedding import Embedder


class CloudEmbedder(Embedder):
    """Embedder that uses Qdrant Cloud Inference.

    Note: This embedder doesn't actually return vectors - it returns None.
    The actual embedding happens server-side when using models.Document()
    in upsert operations.

    This class is mainly for compatibility with the Embedder interface
    and for getting dimension information.
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.dense_model
        self._dimension = settings.get_model_dimension(self.model)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Cloud inference doesn't pre-embed - returns empty vectors.

        Use upsert_with_inference() instead for actual embedding.
        """
        raise NotImplementedError(
            "CloudEmbedder uses server-side inference. "
            "Use upsert_with_inference() for indexing and "
            "search_with_inference() for querying."
        )

    def embed_single(self, text: str) -> list[float]:
        """Cloud inference doesn't pre-embed.

        Use search_with_inference() instead.
        """
        raise NotImplementedError(
            "CloudEmbedder uses server-side inference. "
            "Use search_with_inference() for querying."
        )
