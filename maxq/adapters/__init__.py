"""MaxQ adapters for external integrations."""

from maxq.adapters.qdrant_sink import get_client, ensure_collection, upsert_chunks
from maxq.adapters.qdrant_retriever import search
from maxq.adapters.embedder_local import LocalEmbedder, LocalSparseEmbedder, get_embedder
from maxq.adapters.embedder_cloud import CloudEmbedder
from maxq.adapters.embedder_tei import TEIEmbedder, AsyncTEIEmbedder

__all__ = [
    # Qdrant
    "get_client",
    "ensure_collection",
    "upsert_chunks",
    "search",
    # Embedders
    "LocalEmbedder",
    "LocalSparseEmbedder",
    "CloudEmbedder",
    "TEIEmbedder",
    "AsyncTEIEmbedder",
    "get_embedder",
]
