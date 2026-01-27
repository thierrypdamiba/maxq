"""
Backfill router - re-embed, fill gaps, add vector spaces.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from maxq.backfill import BackfillAgent, BackfillConfig, BackfillMode

router = APIRouter(
    prefix="/backfill",
    tags=["backfill"],
)


@router.get("/analyze/{collection_name}")
async def analyze_collection(collection_name: str):
    """Analyze a collection's vector config and detect gaps."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = BackfillAgent(engine)
    result = agent.analyze(collection_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


class BackfillRequest(BaseModel):
    collection_name: str
    mode: str = "reembed"  # reembed, fill_gaps, add_vector
    dense_model: Optional[str] = None
    sparse_model: Optional[str] = None
    new_vector_name: Optional[str] = None
    new_vector_size: Optional[int] = None
    batch_size: int = 50
    text_field: str = "_text"


@router.post("/run")
async def run_backfill(body: BackfillRequest):
    """Run a backfill operation on a collection."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    try:
        mode = BackfillMode(body.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {body.mode}. Use reembed, fill_gaps, or add_vector.",
        )

    if mode == BackfillMode.REEMBED and not body.dense_model:
        raise HTTPException(status_code=400, detail="dense_model required for reembed mode")

    if mode == BackfillMode.ADD_VECTOR and not body.new_vector_name:
        raise HTTPException(status_code=400, detail="new_vector_name required for add_vector mode")

    config = BackfillConfig(
        collection_name=body.collection_name,
        mode=mode,
        dense_model=body.dense_model,
        sparse_model=body.sparse_model,
        new_vector_name=body.new_vector_name,
        new_vector_size=body.new_vector_size,
        batch_size=body.batch_size,
        text_field=body.text_field,
    )

    agent = BackfillAgent(engine)
    progress = agent.run(config)

    return {
        "status": progress.status,
        "total_points": progress.total_points,
        "processed": progress.processed,
        "skipped": progress.skipped,
        "failed": progress.failed,
    }
