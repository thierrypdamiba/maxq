"""Job API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from maxq.core.types import JobStatus
from maxq.db import sqlite as db


router = APIRouter()


# ============ Response Models ============

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


# ============ Routes ============

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Get job details."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(
        job_id=job["job_id"],
        run_id=job["run_id"],
        job_type=job["job_type"],
        status=job["status"],
        error=job["error"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
    )


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a queued job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.QUEUED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Can only cancel queued jobs (current status: {job['status']})"
        )

    db.cancel_job(job_id)

    return {"message": f"Job {job_id} cancelled", "job_id": job_id}
