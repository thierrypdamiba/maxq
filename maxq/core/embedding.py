"""Embedding interface."""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract embedder interface."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
