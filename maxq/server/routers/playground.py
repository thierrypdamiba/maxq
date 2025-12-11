from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from maxq.server.dependencies import get_engine
from maxq.search_engine import MaxQEngine, SearchRequest, CollectionStrategy
from maxq.server.database import ProjectStore

router = APIRouter(
    prefix="/playground",
    tags=["playground"],
    responses={404: {"description": "Not found"}},
)

class SearchQuery(BaseModel):
    project_id: str
    query: str
    strategy: str = "hybrid"
    embedding_model: Optional[str] = None  # Optional: specify which indexed model to use


@router.post("/search")
async def search(
    query: SearchQuery,
    engine: MaxQEngine = Depends(get_engine)
):
    try:
        # Retrieve project
        project = ProjectStore.get(query.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Check if any data has been indexed
        if not project.indexed_models:
            raise HTTPException(
                status_code=400,
                detail="No data has been indexed for this project. Please index data first on the Indexing page."
            )

        # Determine which embedding model to use
        if query.embedding_model:
            # User specified a model - validate it's indexed
            indexed_model = next(
                (m for m in project.indexed_models if m.model_name == query.embedding_model),
                None
            )
            if not indexed_model:
                available = [m.model_name for m in project.indexed_models]
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{query.embedding_model}' not indexed. Available models: {available}"
                )
            model_name = indexed_model.model_name
            collection_name = indexed_model.collection_name
        else:
            # Use the first (most recently indexed) model
            indexed_model = project.indexed_models[0]
            model_name = indexed_model.model_name
            collection_name = indexed_model.collection_name

        config = CollectionStrategy(
            collection_name=collection_name,
            dense_model_name=model_name,
            sparse_model_name="Qdrant/bm25"
        )

        req = SearchRequest(
            query=query.query,
            strategy=query.strategy,
            limit=10
        )

        print(f"[DEBUG] Searching collection: {collection_name}, model: {model_name}, query: {query.query}")

        points = engine.query(config, req)

        print(f"[DEBUG] Found {len(points)} results")

        results = []
        for point in points:
            results.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("_text", "") if point.payload else "",
                "metadata": {k:v for k,v in point.payload.items() if k != "_text"} if point.payload else {}
            })

        return {
            "results": results,
            "model_used": model_name,
            "collection_name": collection_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
