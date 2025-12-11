"""
Simple search router - query indexed data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal
import os

from qdrant_client import QdrantClient, models
from qdrant_client.models import Document

from maxq.server.database import ProjectStore

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


class SearchRequest(BaseModel):
    project_id: str
    query: str
    strategy: Literal["hybrid", "dense", "sparse"] = "hybrid"
    limit: int = 10
    model_name: Optional[str] = None  # Use default if not specified


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
    model_used: str
    collection_name: str
    strategy: str


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client from environment."""
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url or not key:
        raise HTTPException(status_code=500, detail="QDRANT_URL and QDRANT_API_KEY must be set")
    return QdrantClient(url=url, api_key=key, timeout=60, cloud_inference=True)


@router.post("/query", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search indexed data using hybrid, dense, or sparse search.
    """
    # Get project and validate
    project = ProjectStore.get(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.indexed_models:
        raise HTTPException(
            status_code=400, detail="No data indexed for this project. Index data first."
        )

    # Find the indexed model to use
    if request.model_name:
        indexed_model = next(
            (m for m in project.indexed_models if m.model_name == request.model_name), None
        )
        if not indexed_model:
            available = [m.model_name for m in project.indexed_models]
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' not indexed. Available: {available}",
            )
    else:
        # Use the first one (most recently indexed)
        indexed_model = project.indexed_models[0]

    # Check if collection has data
    if indexed_model.point_count == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{indexed_model.collection_name}' has no data. Re-index required.",
        )

    try:
        client = get_qdrant_client()

        # Verify collection exists
        if not client.collection_exists(indexed_model.collection_name):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{indexed_model.collection_name}' not found in Qdrant",
            )

        # Execute search based on strategy
        if request.strategy == "hybrid":
            # Hybrid search with RRF fusion
            points = client.query_points(
                collection_name=indexed_model.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=Document(text=request.query, model=indexed_model.model_name),
                        using="dense",
                        limit=request.limit,
                    ),
                    models.Prefetch(
                        query=Document(text=request.query, model="Qdrant/bm25"),
                        using="bm25",  # Must match the sparse vector name from indexing
                        limit=request.limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=request.limit,
                with_payload=True,
            ).points

        elif request.strategy == "dense":
            points = client.query_points(
                collection_name=indexed_model.collection_name,
                query=Document(text=request.query, model=indexed_model.model_name),
                using="dense",
                limit=request.limit,
                with_payload=True,
            ).points

        elif request.strategy == "sparse":
            points = client.query_points(
                collection_name=indexed_model.collection_name,
                query=Document(text=request.query, model="Qdrant/bm25"),
                using="bm25",  # Must match the sparse vector name from indexing
                limit=request.limit,
                with_payload=True,
            ).points

        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        # Format results
        results = []
        for point in points:
            text = point.payload.get("_text", "") if point.payload else ""
            metadata = {k: v for k, v in (point.payload or {}).items() if k != "_text"}

            results.append(
                SearchResult(
                    id=str(point.id), score=point.score or 0.0, text=text, metadata=metadata
                )
            )

        return SearchResponse(
            results=results,
            model_used=indexed_model.model_name,
            collection_name=indexed_model.collection_name,
            strategy=request.strategy,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/collections/{project_id}")
async def list_searchable_collections(project_id: str):
    """List all searchable collections for a project."""
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id": project_id,
        "collections": [
            {
                "model_name": m.model_name,
                "collection_name": m.collection_name,
                "point_count": m.point_count,
                "searchable": m.point_count > 0,
            }
            for m in project.indexed_models
        ],
    }
