"""Qdrant search operations (Cloud, Local Docker, or In-Memory)."""

from qdrant_client import QdrantClient, models

from maxq.core.config import settings
from maxq.core.types import SearchResult


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


def search(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int = 20,
) -> list[SearchResult]:
    """Search for similar vectors.

    Returns list of SearchResult with scores.
    """
    results = client.search(
        collection_name=collection,
        query_vector=("dense", query_vector),
        limit=top_k,
        with_payload=True,
    )

    search_results = []
    for hit in results:
        result = SearchResult(
            id=hit.id,
            score=hit.score,
            doc_id=hit.payload.get("doc_id", ""),
            text=hit.payload.get("text", ""),
        )
        search_results.append(result)

    return search_results


def search_with_inference(
    client: QdrantClient,
    collection: str,
    query_text: str,
    dense_model: str,
    top_k: int = 20,
) -> list[SearchResult]:
    """Search using Qdrant Cloud Inference.

    Uses models.Document() for server-side embedding of query.
    """
    results = client.query_points(
        collection_name=collection,
        query=models.Document(
            text=query_text,
            model=dense_model,
        ),
        using="dense",
        limit=top_k,
        with_payload=True,
    )

    search_results = []
    for hit in results.points:
        result = SearchResult(
            id=hit.id,
            score=hit.score,
            doc_id=hit.payload.get("doc_id", ""),
            text=hit.payload.get("text", ""),
        )
        search_results.append(result)

    return search_results


def hybrid_search_with_inference(
    client: QdrantClient,
    collection: str,
    query_text: str,
    dense_model: str,
    sparse_model: str = "Qdrant/bm25",
    top_k: int = 20,
) -> list[SearchResult]:
    """Hybrid search using Qdrant Cloud Inference (dense + sparse + RRF).

    Uses both dense and sparse vectors with Reciprocal Rank Fusion.
    """
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model=dense_model,
                ),
                using="dense",
                limit=top_k * 2,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model=sparse_model,
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    search_results = []
    for hit in results.points:
        result = SearchResult(
            id=hit.id,
            score=hit.score,
            doc_id=hit.payload.get("doc_id", ""),
            text=hit.payload.get("text", ""),
        )
        search_results.append(result)

    return search_results
