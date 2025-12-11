"""
Indexing router with presets, AI config, and advanced options.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
import uuid
import os

from qdrant_client import QdrantClient, models
from qdrant_client.models import Document
from datasets import load_dataset

from maxq.server.database import ProjectStore
from maxq.server.models import IndexedModelInfo
from maxq.autoconfig import (
    PRESETS,
    CloudConfig,
    HnswConfig,
    QuantizationConfig,
    list_presets,
    create_config,
    estimate_shards,
)

router = APIRouter(
    prefix="/index",
    tags=["index"],
)

# Supported models for Qdrant Cloud Inference
DENSE_MODELS = [
    {"model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "description": "Fast, general-purpose"},
    {"model": "mixedbread-ai/mxbai-embed-large-v1", "dim": 1024, "description": "High accuracy"},
]

SPARSE_MODELS = [
    {"model": "Qdrant/bm25", "description": "Classic BM25 keyword search (free, unlimited)"},
    {"model": "prithivida/Splade_PP_en_v1", "description": "SPLADE learned sparse embeddings"},
]

MODEL_DIMS = {m["model"]: m["dim"] for m in DENSE_MODELS}


class AdvancedConfig(BaseModel):
    """Advanced Qdrant configuration options."""
    hnsw_m: int = Field(default=32, ge=4, le=128, description="HNSW connections per node")
    hnsw_ef_construct: int = Field(default=200, ge=50, le=500, description="HNSW build-time search width")
    quantization: Optional[Literal["scalar", "binary", "none"]] = Field(default="scalar", description="Vector quantization")
    on_disk: bool = Field(default=True, description="Store vectors on disk")
    shard_number: Optional[int] = Field(default=None, ge=1, le=16, description="Number of shards (auto if None)")
    replication_factor: int = Field(default=1, ge=1, le=3, description="Replication factor")


class IndexRequest(BaseModel):
    project_id: str
    dataset_name: str  # HuggingFace dataset name
    dense_models: List[str] = ["sentence-transformers/all-MiniLM-L6-v2"]  # Can select multiple or empty
    sparse_models: List[str] = ["Qdrant/bm25"]  # Can select multiple or empty
    text_field: Optional[str] = None  # Auto-detect if None
    limit: int = 500  # Keep it small for fast indexing
    # New: config mode
    config_mode: Literal["quick", "preset", "advanced"] = "quick"
    preset_name: Optional[str] = None  # For preset mode
    advanced: Optional[AdvancedConfig] = None  # For advanced mode


class IndexedCollection(BaseModel):
    collection_name: str
    model: str
    points_indexed: int

class IndexResponse(BaseModel):
    success: bool
    collections: List[IndexedCollection]
    total_points: int
    message: str


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client from environment."""
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url or not key:
        raise HTTPException(status_code=500, detail="QDRANT_URL and QDRANT_API_KEY must be set")
    return QdrantClient(url=url, api_key=key, timeout=120, cloud_inference=True)


def make_collection_name(project_id: str, model_name: str) -> str:
    """Generate consistent collection name."""
    safe_model = model_name.replace("/", "_").replace("-", "_").lower()
    return f"{project_id}_{safe_model}"


def find_text_field(sample: dict) -> Optional[str]:
    """Auto-detect the best text field from a sample."""
    candidates = ["text", "content", "prompt", "question", "description", "body", "message"]

    # First try known names
    for name in candidates:
        if name in sample and isinstance(sample[name], str) and len(sample[name]) > 20:
            return name

    # Then find any string field > 20 chars
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 20:
            return key

    return None


@router.get("/models")
async def list_models():
    """List supported embedding models."""
    return {
        "dense": DENSE_MODELS,
        "sparse": SPARSE_MODELS
    }


@router.get("/presets")
async def get_presets():
    """List available configuration presets."""
    presets_list = []
    for key, config in PRESETS.items():
        presets_list.append({
            "key": key,
            "name": config.name,
            "description": config.description,
            "use_case": config.use_case,
            "dense_model": config.dense_model,
            "sparse_model": config.sparse_model,
            "hnsw_m": config.hnsw.m,
            "hnsw_ef_construct": config.hnsw.ef_construct,
            "quantization": config.quantization.type if config.quantization else None,
            "on_disk": config.on_disk,
            "shard_number": config.shard_number,
            "replication_factor": config.replication_factor,
            "multi_tenant": config.multi_tenant,
        })
    return {"presets": presets_list}


@router.post("/start", response_model=IndexResponse)
async def start_indexing(request: IndexRequest):
    """
    Index a HuggingFace dataset into Qdrant with multiple embedding models.
    Each dense model gets its own collection with all selected sparse models.
    """
    # Validate project exists
    project = ProjectStore.get(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Must have at least one model
    if not request.dense_models and not request.sparse_models:
        raise HTTPException(status_code=400, detail="Select at least one dense or sparse model")

    # Validate dense models
    for model in request.dense_models:
        if model not in MODEL_DIMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported dense model: {model}. Use one of: {list(MODEL_DIMS.keys())}"
            )

    # Validate sparse models
    valid_sparse = {m["model"] for m in SPARSE_MODELS}
    for model in request.sparse_models:
        if model not in valid_sparse:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported sparse model: {model}. Use one of: {list(valid_sparse)}"
            )

    try:
        client = get_qdrant_client()

        # 1. Load dataset once
        try:
            dataset = load_dataset(
                request.dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

        # 2. Get samples and find text field
        samples = []
        for i, row in enumerate(dataset):
            samples.append(dict(row))
            if i >= 2:
                break

        if not samples:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        text_field = request.text_field or find_text_field(samples[0])
        if not text_field:
            raise HTTPException(
                status_code=400,
                detail=f"Could not find text field. Available fields: {list(samples[0].keys())}"
            )

        # 3. Load all texts and payloads into memory
        dataset = load_dataset(
            request.dataset_name,
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        all_texts = []
        all_payloads = []
        for i, row in enumerate(dataset):
            if i >= request.limit:
                break
            text = str(row.get(text_field, ""))
            if len(text) < 10:
                continue
            all_texts.append(text)
            payload = {"_text": text}
            for k, v in row.items():
                if k != text_field and not k.startswith("_"):
                    payload[k] = v
            all_payloads.append(payload)

        # 4. Index with each dense model (each gets its own collection with all sparse models)
        indexed_collections = []
        total_points = 0

        for dense_model in request.dense_models:
            collection_name = make_collection_name(request.project_id, dense_model)
            vec_dim = MODEL_DIMS[dense_model]

            # Build sparse vectors config for all selected sparse models
            sparse_config = {}
            for sparse_model in request.sparse_models:
                sparse_name = sparse_model.split("/")[-1].lower().replace("-", "_")
                sparse_config[sparse_name] = models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )

            # Resolve config: preset, advanced, or default
            hnsw_m = 32
            hnsw_ef = 200
            quant_config = None
            on_disk = True
            shard_num = estimate_shards(request.limit)
            repl_factor = 1

            if request.config_mode == "preset" and request.preset_name:
                if request.preset_name in PRESETS:
                    preset = PRESETS[request.preset_name]
                    hnsw_m = preset.hnsw.m
                    hnsw_ef = preset.hnsw.ef_construct
                    on_disk = preset.on_disk
                    shard_num = preset.shard_number
                    repl_factor = preset.replication_factor
                    if preset.quantization:
                        quant_config = models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                always_ram=True
                            )
                        )
            elif request.config_mode == "advanced" and request.advanced:
                adv = request.advanced
                hnsw_m = adv.hnsw_m
                hnsw_ef = adv.hnsw_ef_construct
                on_disk = adv.on_disk
                shard_num = adv.shard_number or estimate_shards(request.limit)
                repl_factor = adv.replication_factor
                if adv.quantization and adv.quantization != "none":
                    quant_config = models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True
                        )
                    )

            # Build HNSW config
            hnsw_config = models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef)

            # Create collection with full config
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vec_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=hnsw_config,
                        quantization_config=quant_config,
                        on_disk=on_disk,
                    )
                },
                sparse_vectors_config=sparse_config if sparse_config else None,
                shard_number=shard_num,
                replication_factor=repl_factor,
            )

            # Upload in batches
            batch_size = 50
            for batch_start in range(0, len(all_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(all_texts))
                points = []
                for j in range(batch_start, batch_end):
                    # Build vector dict with dense + all sparse
                    vector_dict = {
                        "dense": Document(text=all_texts[j], model=dense_model),
                    }
                    for sparse_model in request.sparse_models:
                        sparse_name = sparse_model.split("/")[-1].lower().replace("-", "_")
                        vector_dict[sparse_name] = Document(text=all_texts[j], model=sparse_model)

                    points.append(models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=vector_dict,
                        payload=all_payloads[j]
                    ))
                client.upload_points(
                    collection_name=collection_name,
                    points=points,
                    batch_size=10,
                    wait=True
                )

            # Verify and save
            info = client.get_collection(collection_name)
            actual_points = info.points_count
            total_points += actual_points

            indexed_collections.append(IndexedCollection(
                collection_name=collection_name,
                model=dense_model,
                points_indexed=actual_points
            ))

            # Save to database with dataset and sparse model info
            model_info = IndexedModelInfo(
                model_name=dense_model,
                collection_name=collection_name,
                indexed_at=datetime.now(),
                point_count=actual_points,
                dataset_name=request.dataset_name,
                sparse_models=request.sparse_models,
            )
            ProjectStore.add_indexed_model(request.project_id, model_info)

        # Update project's default model (first dense one)
        if request.dense_models:
            project.embedding_model = request.dense_models[0]
            ProjectStore.update(project)

        dense_names = ", ".join([m.split("/")[-1] for m in request.dense_models]) or "none"
        sparse_names = ", ".join([m.split("/")[-1] for m in request.sparse_models]) or "none"
        return IndexResponse(
            success=True,
            collections=indexed_collections,
            total_points=total_points,
            message=f"Indexed {len(all_texts)} docs. Dense: {dense_names}. Sparse: {sparse_names}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/status/{project_id}")
async def get_index_status(project_id: str):
    """Get indexing status for a project."""
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id": project_id,
        "indexed_models": [
            {
                "model_name": m.model_name,
                "collection_name": m.collection_name,
                "point_count": m.point_count,
                "indexed_at": m.indexed_at.isoformat() if m.indexed_at else None,
                "dataset_name": getattr(m, "dataset_name", None),
                "sparse_models": getattr(m, "sparse_models", []),
            }
            for m in project.indexed_models
        ],
        "has_data": len(project.indexed_models) > 0 and any(m.point_count > 0 for m in project.indexed_models)
    }


@router.delete("/{project_id}/{model_name}")
async def delete_index(project_id: str, model_name: str):
    """Delete an indexed model's collection."""
    project = ProjectStore.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    collection_name = make_collection_name(project_id, model_name)

    try:
        client = get_qdrant_client()
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        # Note: We'd need to add a method to remove from indexed_models in DB
        # For now, it stays in DB but collection is deleted

        return {"success": True, "message": f"Deleted collection {collection_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
