"""SQLite database operations."""

import json
import uuid
from datetime import datetime
from typing import Optional

from maxq.core.config import settings
from maxq.core.types import JobStatus, JobType, RunConfig, RunStatus
from maxq.db.migrations import get_connection


# ============ Run Operations ============

def create_run_record(run_id: str, config: RunConfig) -> None:
    """Insert a new run record."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO runs (
            run_id, status, collection, dataset_path,
            chunk_size, chunk_overlap, embedding_model,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        RunStatus.QUEUED.value,
        config.collection,
        config.dataset_path,
        config.chunk_size,
        config.chunk_overlap,
        config.embedding_model,
        now,
        now,
    ))

    conn.commit()
    conn.close()


def get_run(run_id: str) -> Optional[dict]:
    """Get a run by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return dict(row)
    return None


def list_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    """List runs, newest first."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM runs
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def update_run_status(run_id: str, status: RunStatus, error: str = None) -> None:
    """Update run status."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    if error:
        cursor.execute("""
            UPDATE runs
            SET status = ?, error = ?, updated_at = ?
            WHERE run_id = ?
        """, (status.value, error, now, run_id))
    else:
        cursor.execute("""
            UPDATE runs
            SET status = ?, updated_at = ?
            WHERE run_id = ?
        """, (status.value, now, run_id))

    conn.commit()
    conn.close()


def delete_run(run_id: str) -> None:
    """Delete a run record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    conn.commit()
    conn.close()


# ============ Job Operations ============

def create_job(
    run_id: str,
    job_type: JobType,
    payload: dict = None,
) -> str:
    """Create a new job and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()
    payload_json = json.dumps(payload) if payload else None

    cursor.execute("""
        INSERT INTO jobs (job_id, run_id, job_type, status, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (job_id, run_id, job_type.value, JobStatus.QUEUED.value, payload_json, now))

    conn.commit()
    conn.close()

    return job_id


def claim_next_job() -> Optional[dict]:
    """Claim the next queued job atomically.

    Uses a transaction to ensure only one worker claims a job.
    Returns the job dict or None if no jobs available.
    """
    conn = get_connection()
    conn.isolation_level = "EXCLUSIVE"
    cursor = conn.cursor()

    try:
        # Start transaction
        cursor.execute("BEGIN EXCLUSIVE")

        # Find next queued job
        cursor.execute("""
            SELECT * FROM jobs
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT 1
        """, (JobStatus.QUEUED.value,))

        row = cursor.fetchone()

        if not row:
            conn.rollback()
            conn.close()
            return None

        job = dict(row)
        now = datetime.now().isoformat()

        # Mark as running
        cursor.execute("""
            UPDATE jobs
            SET status = ?, started_at = ?
            WHERE job_id = ?
        """, (JobStatus.RUNNING.value, now, job["job_id"]))

        conn.commit()
        conn.close()

        return job

    except Exception:
        conn.rollback()
        conn.close()
        raise


def complete_job(job_id: str, error: str = None) -> None:
    """Mark a job as completed (done or failed)."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    status = JobStatus.FAILED.value if error else JobStatus.DONE.value

    cursor.execute("""
        UPDATE jobs
        SET status = ?, error = ?, completed_at = ?
        WHERE job_id = ?
    """, (status, error, now, job_id))

    conn.commit()
    conn.close()


def get_jobs_for_run(run_id: str) -> list[dict]:
    """Get all jobs for a run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM jobs
        WHERE run_id = ?
        ORDER BY created_at ASC
    """, (run_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return dict(row)
    return None


def cancel_job(job_id: str) -> None:
    """Cancel a queued job."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        UPDATE jobs
        SET status = ?, completed_at = ?
        WHERE job_id = ?
    """, (JobStatus.CANCELLED.value, now, job_id))

    conn.commit()
    conn.close()


def delete_jobs_for_run(run_id: str) -> None:
    """Delete all jobs for a run."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))

    conn.commit()
    conn.close()
