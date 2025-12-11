"""Stub embedder for testing."""

import hashlib

from maxq.core.embedding import Embedder


class StubEmbedder(Embedder):
    """Deterministic stub embedder for testing.

    Generates consistent pseudo-vectors based on text hash.
    This allows tests to be deterministic without a real embedding model.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-embeddings."""
        return [self._hash_to_vector(text) for text in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to deterministic vector via hashing."""
        # Create a seed from the text
        text_hash = hashlib.sha256(text.encode()).digest()

        # Generate deterministic floats from hash bytes
        vector = []
        for i in range(self._dimension):
            # Cycle through hash bytes
            byte_val = text_hash[i % len(text_hash)]
            # Normalize to [-1, 1]
            normalized = (byte_val / 127.5) - 1.0
            vector.append(normalized)

        # Normalize to unit length
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector
