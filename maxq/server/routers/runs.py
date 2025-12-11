"""Run API routes."""

import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from maxq.core import runs
from maxq.core.config import settings
from maxq.core.types import JobType, Metrics, RunConfig, RunStatus
from maxq.db import sqlite as db


router = APIRouter()


# ============ Request/Response Models ============

class CreateIndexRunRequest(BaseModel):
    """Request to create an index run."""
    dataset_path: str = Field(..., description="Path to dataset JSONL file or hf://dataset_name")
    collection: str = Field(..., description="Qdrant collection name")
    chunk_size: int = Field(800, description="Chunk size in characters")
    chunk_overlap: int = Field(120, description="Chunk overlap in characters")
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model to use"
    )
    use_cloud_inference: bool = Field(True, description="Use Qdrant Cloud Inference")


class CreateEvalRequest(BaseModel):
    """Request to create an eval job."""
    queries_path: str = Field("data/queries.jsonl", description="Path to queries JSONL")
    top_k: int = Field(20, description="Number of results to retrieve")
    use_cloud_inference: bool = Field(True, description="Use Qdrant Cloud Inference")


class CreateReportRequest(BaseModel):
    """Request to create a report job."""
    pass  # No additional options for now


class JobResponse(BaseModel):
    """Job information."""
    job_id: str
    run_id: str
    job_type: str
    status: str
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RunResponse(BaseModel):
    """Run information."""
    run_id: str
    status: str
    collection: str
    dataset_path: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    error: Optional[str] = None
    created_at: str
    updated_at: str
    jobs: list[JobResponse] = Field(default_factory=list)
    metrics: Optional[Metrics] = None


class RunListResponse(BaseModel):
    """List of runs."""
    runs: list[RunResponse]
    total: int


class CreateRunResponse(BaseModel):
    """Response after creating a run."""
    run_id: str
    job_id: str
    message: str


# ============ Routes ============

@router.post("/index", response_model=CreateRunResponse)
def create_index_run(request: CreateIndexRunRequest):
    """Create a new index run and queue the job."""
    # Input validation
    if not request.dataset_path.startswith("hf://"):
        if not Path(request.dataset_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset file not found: {request.dataset_path}"
            )

    if request.chunk_size <= request.chunk_overlap:
        raise HTTPException(
            status_code=400,
            detail=f"chunk_size ({request.chunk_size}) must be greater than chunk_overlap ({request.chunk_overlap})"
        )

    if request.chunk_size < 100:
        raise HTTPException(
            status_code=400,
            detail="chunk_size must be at least 100 characters"
        )

    # Create config
    config = RunConfig(
        collection=request.collection,
        dataset_path=request.dataset_path,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        embedding_model=request.embedding_model,
    )

    # Create run (filesystem)
    run = runs.create_run(config)

    # Create run record (database)
    db.create_run_record(run.run_id, config)

    # Create job with payload
    payload = {"use_cloud_inference": request.use_cloud_inference}
    job_id = db.create_job(run.run_id, JobType.INDEX, payload)

    return CreateRunResponse(
        run_id=run.run_id,
        job_id=job_id,
        message="Index run created and queued",
    )


@router.post("/{run_id}/eval", response_model=CreateRunResponse)
def create_eval_job(run_id: str, request: CreateEvalRequest):
    """Create an eval job for an existing run."""
    # Check run exists
    run_record = db.get_run(run_id)
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Check run is indexed
    valid_statuses = [RunStatus.INDEXED.value, RunStatus.EVALUATED.value, RunStatus.DONE.value]
    if run_record["status"] not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Run must be indexed first (current status: {run_record['status']})",
        )

    # Create job with payload
    payload = {
        "queries_path": request.queries_path,
        "top_k": request.top_k,
        "use_cloud_inference": request.use_cloud_inference,
    }
    job_id = db.create_job(run_id, JobType.EVAL, payload)

    return CreateRunResponse(
        run_id=run_id,
        job_id=job_id,
        message="Eval job created and queued",
    )


@router.post("/{run_id}/report", response_model=CreateRunResponse)
def create_report_job(run_id: str, request: CreateReportRequest = None):
    """Create a report job for an existing run."""
    # Check run exists
    run_record = db.get_run(run_id)
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Check run is evaluated
    valid_statuses = [RunStatus.EVALUATED.value, RunStatus.DONE.value]
    if run_record["status"] not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Run must be evaluated first (current status: {run_record['status']})",
        )

    # Create job
    job_id = db.create_job(run_id, JobType.REPORT)

    return CreateRunResponse(
        run_id=run_id,
        job_id=job_id,
        message="Report job created and queued",
    )


@router.get("", response_model=RunListResponse)
def list_runs(limit: int = 50, offset: int = 0):
    """List all runs."""
    run_records = db.list_runs(limit=limit, offset=offset)

    runs_response = []
    for record in run_records:
        jobs = db.get_jobs_for_run(record["run_id"])
        jobs_response = [
            JobResponse(
                job_id=j["job_id"],
                run_id=j["run_id"],
                job_type=j["job_type"],
                status=j["status"],
                error=j["error"],
                created_at=j["created_at"],
                started_at=j["started_at"],
                completed_at=j["completed_at"],
            )
            for j in jobs
        ]

        # Try to load metrics if available
        metrics = None
        metrics_path = Path(settings.runs_dir) / record["run_id"] / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = Metrics(**json.load(f))

        runs_response.append(RunResponse(
            run_id=record["run_id"],
            status=record["status"],
            collection=record["collection"],
            dataset_path=record["dataset_path"],
            chunk_size=record["chunk_size"],
            chunk_overlap=record["chunk_overlap"],
            embedding_model=record["embedding_model"],
            error=record["error"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            jobs=jobs_response,
            metrics=metrics,
        ))

    return RunListResponse(runs=runs_response, total=len(runs_response))


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str):
    """Get run details."""
    run_record = db.get_run(run_id)
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    jobs = db.get_jobs_for_run(run_id)
    jobs_response = [
        JobResponse(
            job_id=j["job_id"],
            run_id=j["run_id"],
            job_type=j["job_type"],
            status=j["status"],
            error=j["error"],
            created_at=j["created_at"],
            started_at=j["started_at"],
            completed_at=j["completed_at"],
        )
        for j in jobs
    ]

    # Try to load metrics
    metrics = None
    metrics_path = Path(settings.runs_dir) / run_id / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = Metrics(**json.load(f))

    return RunResponse(
        run_id=run_record["run_id"],
        status=run_record["status"],
        collection=run_record["collection"],
        dataset_path=run_record["dataset_path"],
        chunk_size=run_record["chunk_size"],
        chunk_overlap=run_record["chunk_overlap"],
        embedding_model=run_record["embedding_model"],
        error=run_record["error"],
        created_at=run_record["created_at"],
        updated_at=run_record["updated_at"],
        jobs=jobs_response,
        metrics=metrics,
    )


@router.delete("/{run_id}")
def delete_run(run_id: str):
    """Delete a run and its artifacts."""
    # Check run exists
    run_record = db.get_run(run_id)
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Delete from filesystem
    run_dir = Path(settings.runs_dir) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    # Delete from database (jobs first due to foreign key)
    db.delete_jobs_for_run(run_id)
    db.delete_run(run_id)

    return {"message": f"Run {run_id} deleted", "run_id": run_id}


@router.get("/{run_id}/artifact/{name}")
def get_artifact(run_id: str, name: str):
    """Download a run artifact."""
    # Validate run exists
    run_record = db.get_run(run_id)
    if not run_record:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Validate artifact name (prevent path traversal)
    artifact_path = (Path(settings.runs_dir) / run_id / name).resolve()
    run_dir = (Path(settings.runs_dir) / run_id).resolve()
    if not artifact_path.is_relative_to(run_dir):
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact {name} not found")

    # Determine content type
    if name.endswith(".json"):
        media_type = "application/json"
    elif name.endswith(".jsonl"):
        media_type = "application/x-ndjson"
    elif name.endswith(".md"):
        media_type = "text/markdown"
    elif name.endswith(".html"):
        media_type = "text/html"
    else:
        media_type = "application/octet-stream"

    return FileResponse(artifact_path, media_type=media_type, filename=name)
