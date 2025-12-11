from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from datetime import datetime
from maxq.server.dependencies import get_engine
from maxq.search_engine import MaxQEngine, CollectionStrategy
from maxq.server.database import ProjectStore
from maxq.server.models import IndexedModelInfo

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
)

class IngestRequest(BaseModel):
    project_id: str
    dataset_name: str
    embedding_model: str
    quantization: str
    sample_limit: int = 1000  # Number of samples to ingest


def _run_ingestion(engine: MaxQEngine, request: IngestRequest, config: CollectionStrategy):
    """Background task that runs ingestion and saves indexed model info"""
    try:
        count = engine.ingest_hf(
            dataset_name=request.dataset_name,
            config=config,
            limit=request.sample_limit
        )

        # Save indexed model info to database
        model_info = IndexedModelInfo(
            model_name=request.embedding_model,
            collection_name=config.collection_name,
            indexed_at=datetime.now(),
            point_count=count if isinstance(count, int) else 0
        )
        ProjectStore.add_indexed_model(request.project_id, model_info)
    except Exception as e:
        print(f"Ingestion failed: {e}")


@router.post("/ingest")
async def ingest_dataset(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    engine: MaxQEngine = Depends(get_engine)
):
    try:
        # Update project metadata with selected model
        project = ProjectStore.get(request.project_id)
        if project:
            project.embedding_model = request.embedding_model
            ProjectStore.update(project)

        # Map request to CollectionStrategy
        collection_name = MaxQEngine.get_collection_name(request.project_id, request.embedding_model)
        config = CollectionStrategy(
            collection_name=collection_name,
            dense_model_name=request.embedding_model,
            use_quantization=(request.quantization == "Int8")
        )

        # Run ingestion in background with callback to save model info
        background_tasks.add_task(_run_ingestion, engine, request, config)

        return {"status": "started", "message": f"Ingestion started for {request.dataset_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_datasets():
    return []

@router.get("/embedding-models")
async def list_embedding_models():
    """Return Qdrant Cloud Inference supported embedding models"""
    return [
        {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
            "description": "Fast, general-purpose model (recommended)"
        },
        {
            "model": "mixedbread-ai/mxbai-embed-large-v1",
            "dim": 1024,
            "description": "High-accuracy model for quality-focused use cases"
        }
    ]
