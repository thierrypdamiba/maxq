"""
Cleanup router - analyze and clean up Qdrant clusters.
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from maxq.cleanup import CleanupAgent, CleanupAction, CleanupReport

router = APIRouter(
    prefix="/cleanup",
    tags=["cleanup"],
)


@router.get("/analyze", response_model=CleanupReport)
async def analyze(collection: Optional[str] = None):
    """Analyze cluster or specific collection for cleanup opportunities."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = CleanupAgent(engine)
    report = agent.analyze(collection)

    # Add LLM summary if available
    try:
        report.llm_summary = agent.summarize_with_llm(report)
    except Exception:
        pass

    return report


@router.get("/duplicates/{collection_name}")
async def find_duplicates(collection_name: str, sample_size: int = 500):
    """Find duplicate points in a collection."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = CleanupAgent(engine)

    if not engine.client.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    duplicates = agent.find_duplicates(collection_name, sample_size=sample_size)
    return {
        "collection": collection_name,
        "duplicate_groups": len(duplicates),
        "total_duplicate_points": sum(len(d.point_ids) - 1 for d in duplicates),
        "groups": [d.model_dump() for d in duplicates],
    }


class ExecuteRequest(BaseModel):
    actions: List[CleanupAction]
    dry_run: bool = True


@router.post("/execute")
async def execute_cleanup(body: ExecuteRequest):
    """Execute cleanup actions. Use dry_run=true to preview."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = CleanupAgent(engine)
    results = agent.execute(body.actions, dry_run=body.dry_run)
    return {"dry_run": body.dry_run, "results": results}
