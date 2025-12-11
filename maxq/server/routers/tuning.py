from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import uuid
import json
import os
from ..models import Experiment
from maxq.config import MAXQ_APP_DIR

router = APIRouter(
    prefix="/tuning",
    tags=["tuning"],
    responses={404: {"description": "Not found"}},
)

DATA_FILE = MAXQ_APP_DIR / "experiments.json"

def save_db():
    with open(DATA_FILE, "w") as f:
        data = [e.model_dump(mode='json') for e in experiments_db]
        json.dump(data, f, indent=2)

def load_db() -> List[Experiment]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return [Experiment(**item) for item in data]
    except Exception:
        return []

experiments_db: List[Experiment] = load_db()

@router.get("/", response_model=List[Experiment])
async def list_experiments(project_id: Optional[str] = None):
    if project_id:
        return [e for e in experiments_db if e.project_id == project_id]
    return experiments_db

@router.get("/available-models")
async def get_available_models(project_id: str):
    """List which embedding models have ingested data for this project"""
    import os
    from qdrant_client import QdrantClient
    from maxq.search_engine import MaxQEngine
    from .index import DENSE_MODELS, SPARSE_MODELS

    try:
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            return {"available_models": []}

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=True)
        all_collections = [c.name for c in client.get_collections().collections]

        # Get list of all embedding models (dense only for experiments)
        all_models = DENSE_MODELS

        # Check which ones have collections for this project
        available = []
        for model in all_models:
            collection_name = MaxQEngine.get_collection_name(project_id, model["model"])
            if collection_name in all_collections:
                available.append(model["model"])

        return {"available_models": available}
    except Exception as e:
        return {"available_models": [], "error": str(e)}


@router.get("/collection-info/{collection_name}")
async def get_collection_info(collection_name: str):
    """Get detailed collection configuration from Qdrant."""
    import os
    from qdrant_client import QdrantClient

    try:
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise HTTPException(status_code=500, detail="Qdrant not configured")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=True)

        if not client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")

        info = client.get_collection(collection_name)

        # Build vectors config dict from the collection info
        vectors_config = {}
        if hasattr(info.config.params, 'vectors') and info.config.params.vectors:
            vectors = info.config.params.vectors
            if hasattr(vectors, '__iter__') and not isinstance(vectors, dict):
                # It's a mapping of vector names to params
                for name, params in vectors.items():
                    vectors_config[name] = {
                        "size": getattr(params, 'size', None),
                        "distance": getattr(params, 'distance', None),
                        "hnsw_config": {
                            "m": getattr(params.hnsw_config, 'm', 16) if params.hnsw_config else 16,
                            "ef_construct": getattr(params.hnsw_config, 'ef_construct', 100) if params.hnsw_config else 100,
                        } if hasattr(params, 'hnsw_config') else None,
                        "quantization_config": params.quantization_config if hasattr(params, 'quantization_config') else None,
                        "on_disk": getattr(params, 'on_disk', False),
                    }
            elif isinstance(vectors, dict):
                vectors_config = vectors

        # Safely get optimizer status
        opt_status = "ok"
        if info.optimizer_status:
            if hasattr(info.optimizer_status, 'status'):
                opt_status = str(info.optimizer_status.status)
            elif hasattr(info.optimizer_status, 'value'):
                opt_status = info.optimizer_status.value
            else:
                opt_status = str(info.optimizer_status)

        return {
            "name": collection_name,
            "points_count": info.points_count,
            "segments_count": getattr(info, 'segments_count', 0),
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
            "optimizer_status": opt_status,
            "config": {
                "params": {
                    "vectors": vectors_config,
                    "shard_number": info.config.params.shard_number if info.config else 1,
                    "replication_factor": info.config.params.replication_factor if info.config else 1,
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Experiment)
async def create_experiment(
    project_id: str,
    name: str,
    embedding_model: str,
    search_strategy: str = "hybrid",
):
    """Create an experiment instantly. User can run eval separately."""
    exp = Experiment(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=name,
        created_at=datetime.now(),
        embedding_model=embedding_model,
        search_strategy=search_strategy,
        status="pending",
        metrics={
            "ndcg": {"candidate": 0.0, "baseline": 0.0, "delta": "+0.0%"},
            "latency": {"candidate": "0ms", "baseline": "0ms", "delta": "-0%"}
        }
    )
    experiments_db.append(exp)
    save_db()
    return exp
