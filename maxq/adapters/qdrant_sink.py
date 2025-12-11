"""Qdrant write operations (Cloud, Local Docker, or In-Memory)."""

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from maxq.core.config import settings
from maxq.core.types import Chunk


def get_client() -> QdrantClient:
    """Get Qdrant client (Cloud, Local Docker, or In-Memory).

    Priority:
    1. In-Memory: Set MAXQ_QDRANT_MODE=memory (for testing)
    2. Cloud: Set MAXQ_QDRANT_URL and MAXQ_QDRANT_API_KEY
    3. Local Docker: Default (uses MAXQ_QDRANT_HOST:MAXQ_QDRANT_PORT)
    """
    if settings.qdrant_mode == "memory":
        # In-memory mode (for testing)
        return QdrantClient(":memory:")
    elif settings.qdrant_url and settings.qdrant_api_key:
        # Qdrant Cloud
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    else:
        # Local Docker instance
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )


def ensure_collection(
    client: QdrantClient,
    collection: str,
    dimension: int,
) -> None:
    """Create collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == collection for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            },
        )


def ensure_hybrid_collection(
    client: QdrantClient,
    collection: str,
    dense_dimension: int,
) -> None:
    """Create collection for hybrid search (dense + sparse)."""
    collections = client.get_collections().collections
    exists = any(c.name == collection for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=dense_dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    batch_size: int = 100,
) -> int:
    """Upsert chunks with embeddings to Qdrant.

    Returns number of points upserted.
    """
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point = PointStruct(
            id=chunk.chunk_id,
            vector={"dense": embedding},
            payload={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                **chunk.metadata,
            },
        )
        points.append(point)

    # Upsert in batches
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    return total


def upsert_with_inference(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_model: str,
    batch_size: int = 100,
) -> int:
    """Upsert chunks using Qdrant Cloud Inference.

    Uses models.Document() for server-side embedding.
    Returns number of points upserted.
    """
    points = []
    for chunk in chunks:
        point = models.PointStruct(
            id=chunk.chunk_id,
            vector={
                "dense": models.Document(
                    text=chunk.text,
                    model=dense_model,
                )
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                **chunk.metadata,
            },
        )
        points.append(point)

    # Upsert in batches
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    return total
