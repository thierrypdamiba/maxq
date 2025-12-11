"""Background job worker with graceful shutdown.

Implements the harness pattern for long-running agents:
- Graceful shutdown with signal handlers
- Progress logging to claude-progress.txt
- Git integration for checkpoint commits
"""

import json
import signal
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

from rich.console import Console

from maxq.core.config import settings
from maxq.core.types import JobType, RunStatus
from maxq.db import sqlite as db
from maxq.db.migrations import run_migrations

console = Console()

# Graceful shutdown flag
_shutdown_requested = False


# ============================================
# Progress Logging (claude-progress.txt)
# ============================================

def log_progress(run_id: str, message: str, level: str = "INFO"):
    """
    Log progress to claude-progress.txt for human-readable session tracking.

    This follows the Anthropic harness pattern for long-running agents,
    providing a persistent log that survives context window resets.

    Args:
        run_id: The run identifier
        message: Human-readable progress message
        level: Log level (INFO, WARN, ERROR, SUCCESS)
    """
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] [{level}] {message}\n"

    # Write to run-specific progress file
    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_file = run_dir / "claude-progress.txt"

    with open(progress_file, "a") as f:
        f.write(log_entry)

    # Also log to console
    level_colors = {
        "INFO": "cyan",
        "WARN": "yellow",
        "ERROR": "red",
        "SUCCESS": "green",
    }
    color = level_colors.get(level, "white")
    console.print(f"[{color}]{log_entry.strip()}[/{color}]")


def log_worker_progress(message: str, level: str = "INFO"):
    """Log progress to global worker progress file."""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] [{level}] {message}\n"

    # Write to global worker progress file
    worker_progress = Path(settings.runs_dir) / "worker-progress.txt"
    worker_progress.parent.mkdir(parents=True, exist_ok=True)

    with open(worker_progress, "a") as f:
        f.write(log_entry)


# ============================================
# Git Integration for Progress Tracking
# ============================================

def git_commit_progress(run_id: str, message: str):
    """
    Create a git commit to checkpoint progress.

    This follows the Anthropic harness pattern where git commits
    serve as durable checkpoints that survive agent restarts.

    Args:
        run_id: The run identifier
        message: Commit message describing the progress
    """
    run_dir = Path(settings.runs_dir) / run_id

    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            cwd=settings.runs_dir.parent if settings.runs_dir else "."
        )

        if result.returncode != 0:
            # Not a git repo, skip
            return

        # Stage the run directory
        subprocess.run(
            ["git", "add", str(run_dir)],
            capture_output=True,
            cwd=settings.runs_dir.parent if settings.runs_dir else "."
        )

        # Create commit
        commit_message = f"[maxq] {run_id}: {message}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_message, "--allow-empty"],
            capture_output=True,
            text=True,
            cwd=settings.runs_dir.parent if settings.runs_dir else "."
        )

        if result.returncode == 0:
            log_progress(run_id, f"Git checkpoint: {message}", "SUCCESS")
        else:
            # No changes to commit or other issue - not an error
            pass

    except FileNotFoundError:
        # Git not installed, skip silently
        pass
    except Exception as e:
        # Don't fail the job for git issues
        log_progress(run_id, f"Git commit skipped: {e}", "WARN")


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]Shutdown requested, finishing current job...[/yellow]")


def run_worker():
    """Main worker loop with graceful shutdown."""
    global _shutdown_requested

    console.print("[bold blue]MaxQ Worker starting...[/bold blue]")
    log_worker_progress("Worker started", "INFO")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Ensure database exists
    run_migrations()

    console.print(f"Polling for jobs every {settings.worker_poll_interval}s")
    console.print("Press Ctrl+C to stop gracefully\n")

    while not _shutdown_requested:
        try:
            # Try to claim a job
            job = db.claim_next_job()

            if job:
                process_job(job)
            else:
                # No jobs, sleep and retry
                time.sleep(settings.worker_poll_interval)

        except Exception as e:
            console.print(f"[red]Worker error: {e}[/red]")
            log_worker_progress(f"Worker error: {e}", "ERROR")
            traceback.print_exc()
            time.sleep(settings.worker_poll_interval)

    log_worker_progress("Worker stopped gracefully", "INFO")
    console.print("[green]Worker stopped gracefully[/green]")


def process_job(job: dict):
    """Process a single job."""
    job_id = job["job_id"]
    job_type = job["job_type"]
    run_id = job["run_id"]
    payload = json.loads(job["payload_json"]) if job["payload_json"] else {}

    console.print(f"[cyan]Processing {job_type} job {job_id} for run {run_id}[/cyan]")
    log_progress(run_id, f"Starting {job_type} job (job_id: {job_id})", "INFO")

    try:
        if job_type == JobType.INDEX.value:
            execute_index_job(run_id, payload)
        elif job_type == JobType.EVAL.value:
            execute_eval_job(run_id, payload)
        elif job_type == JobType.REPORT.value:
            execute_report_job(run_id, payload)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        db.complete_job(job_id)
        log_progress(run_id, f"Completed {job_type} job successfully", "SUCCESS")
        git_commit_progress(run_id, f"Completed {job_type} job")
        console.print(f"[green]Job {job_id} completed[/green]")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        db.complete_job(job_id, error=error_msg)
        db.update_run_status(run_id, RunStatus.FAILED, error=error_msg)
        log_progress(run_id, f"Job failed: {error_msg}", "ERROR")
        console.print(f"[red]Job {job_id} failed: {error_msg}[/red]")
        traceback.print_exc()


def execute_index_job(run_id: str, payload: dict):
    """Execute an index job.

    Supports:
    - Local JSONL files
    - HuggingFace datasets
    - Cloud Inference (server-side embedding)
    """
    from maxq.adapters import embedder_stub, qdrant_sink
    from maxq.core import chunking, dataset, runs
    from maxq.core.types import RunStatus

    # Get run config from DB
    run_record = db.get_run(run_id)
    if not run_record:
        raise ValueError(f"Run {run_id} not found")

    db.update_run_status(run_id, RunStatus.INDEXING)
    log_progress(run_id, "Index job started - loading configuration", "INFO")

    dataset_path = run_record["dataset_path"]
    collection = run_record["collection"]
    chunk_size = run_record["chunk_size"]
    chunk_overlap = run_record["chunk_overlap"]
    embedding_model = run_record["embedding_model"]

    log_progress(run_id, f"Config: dataset={dataset_path}, collection={collection}, model={embedding_model}", "INFO")

    # Load documents based on source type
    log_progress(run_id, f"Loading documents from {dataset_path}", "INFO")
    if dataset_path.startswith("hf://"):
        # HuggingFace dataset
        from maxq.adapters.huggingface import load_dataset_from_huggingface
        hf_name = dataset_path[5:]  # Remove "hf://" prefix
        docs = load_dataset_from_huggingface(hf_name, limit=payload.get("limit"))
    else:
        # Local JSONL file
        docs = dataset.load_documents(dataset_path)

    log_progress(run_id, f"Loaded {len(docs)} documents", "SUCCESS")
    console.print(f"  Loaded {len(docs)} documents")

    # Chunk documents
    log_progress(run_id, f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})", "INFO")
    chunks = chunking.chunk_documents(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    log_progress(run_id, f"Created {len(chunks)} chunks", "SUCCESS")
    console.print(f"  Created {len(chunks)} chunks")

    # Write chunks artifact
    runs.write_jsonl_artifact(
        run_id,
        "chunks.jsonl",
        [c.model_dump() for c in chunks],
    )
    log_progress(run_id, "Wrote chunks.jsonl artifact", "INFO")
    git_commit_progress(run_id, "Chunking complete")

    # Setup Qdrant client
    client = qdrant_sink.get_client()

    # Decide embedding strategy
    use_cloud_inference = payload.get("use_cloud_inference", True)

    log_progress(run_id, f"Upserting to Qdrant (cloud_inference={use_cloud_inference})", "INFO")
    if use_cloud_inference and settings.qdrant_url and settings.qdrant_api_key:
        # Use Qdrant Cloud Inference
        dimension = settings.get_model_dimension(embedding_model)
        qdrant_sink.ensure_collection(client, collection, dimension)
        count = qdrant_sink.upsert_with_inference(
            client, collection, chunks, embedding_model
        )
        log_progress(run_id, f"Upserted {count} points with Cloud Inference", "SUCCESS")
        console.print(f"  Upserted {count} points with Cloud Inference")
    else:
        # Use local embedder (stub for testing)
        embedder = embedder_stub.StubEmbedder()
        qdrant_sink.ensure_collection(client, collection, embedder.dimension)

        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        count = qdrant_sink.upsert_chunks(client, collection, chunks, embeddings)
        log_progress(run_id, f"Upserted {count} points with local embedder", "SUCCESS")
        console.print(f"  Upserted {count} points with local embedder")

    # Write artifacts
    runs.write_artifact(run_id, "upsert_summary.json", {
        "collection": collection,
        "documents": len(docs),
        "chunks": len(chunks),
        "points_upserted": count,
        "embedding_model": embedding_model,
        "cloud_inference": use_cloud_inference,
    })

    runs.write_artifact(run_id, "dataset_manifest.json", {
        "path": dataset_path,
        "documents": len(docs),
        "doc_ids": [d.id for d in docs],
    })

    log_progress(run_id, "Index job complete - status: INDEXED", "SUCCESS")
    db.update_run_status(run_id, RunStatus.INDEXED)


def execute_eval_job(run_id: str, payload: dict):
    """Execute an eval job."""
    from maxq.adapters import embedder_stub, qdrant_retriever
    from maxq.core import dataset, eval as eval_module, runs
    from maxq.core.types import QueryResult, RunStatus

    queries_path = payload.get("queries_path", "data/queries.jsonl")
    top_k = payload.get("top_k", 20)

    # Get run config from DB
    run_record = db.get_run(run_id)
    if not run_record:
        raise ValueError(f"Run {run_id} not found")

    db.update_run_status(run_id, RunStatus.EVALUATING)
    log_progress(run_id, "Eval job started - loading configuration", "INFO")

    collection = run_record["collection"]
    embedding_model = run_record["embedding_model"]

    log_progress(run_id, f"Config: collection={collection}, queries={queries_path}, top_k={top_k}", "INFO")

    # Load queries
    log_progress(run_id, f"Loading queries from {queries_path}", "INFO")
    queries = dataset.load_queries(queries_path)
    log_progress(run_id, f"Loaded {len(queries)} queries", "SUCCESS")
    console.print(f"  Loaded {len(queries)} queries")

    # Get Qdrant client
    client = qdrant_retriever.get_client()

    # Decide search strategy
    use_cloud_inference = payload.get("use_cloud_inference", True)

    log_progress(run_id, f"Running {len(queries)} search queries (cloud_inference={use_cloud_inference})", "INFO")
    query_results = []
    for i, q in enumerate(queries):
        if use_cloud_inference and settings.qdrant_url and settings.qdrant_api_key:
            # Cloud Inference search
            results = qdrant_retriever.search_with_inference(
                client, collection, q.query, embedding_model, top_k=top_k
            )
        else:
            # Local embedder search
            embedder = embedder_stub.StubEmbedder()
            query_vector = embedder.embed_single(q.query)
            results = qdrant_retriever.search(
                client, collection, query_vector, top_k=top_k
            )

        qr = QueryResult(
            query_id=q.id,
            query=q.query,
            results=results,
            relevant_doc_ids=q.relevant_doc_ids,
            relevant_ids=q.relevant_ids,
        )
        query_results.append(qr)

        # Log progress every 10 queries
        if (i + 1) % 10 == 0:
            log_progress(run_id, f"Processed {i + 1}/{len(queries)} queries", "INFO")

    log_progress(run_id, f"Completed all {len(queries)} search queries", "SUCCESS")

    # Compute metrics
    log_progress(run_id, "Computing IR metrics (Recall, MRR, NDCG @ K=5,10,20)", "INFO")
    metrics = eval_module.compute_metrics(query_results, k_values=[5, 10, 20])
    log_progress(run_id, f"Metrics: Recall@10={metrics.recall_at_k.get(10, 0):.4f}, MRR@10={metrics.mrr_at_k.get(10, 0):.4f}", "SUCCESS")
    console.print(f"  Recall@10: {metrics.recall_at_k.get(10, 0):.4f}")

    # Write results
    runs.write_jsonl_artifact(
        run_id,
        "results.jsonl",
        [qr.model_dump() for qr in query_results],
    )
    log_progress(run_id, "Wrote results.jsonl artifact", "INFO")

    runs.write_artifact(run_id, "metrics.json", metrics.model_dump())
    log_progress(run_id, "Wrote metrics.json artifact", "INFO")
    git_commit_progress(run_id, "Evaluation complete")

    # Update run file
    run = runs.read_run_json(run_id)
    run.metrics = metrics
    runs.write_run_json(run)

    log_progress(run_id, "Eval job complete - status: EVALUATED", "SUCCESS")
    db.update_run_status(run_id, RunStatus.EVALUATED)


def execute_report_job(run_id: str, payload: dict):
    """Execute a report job."""
    from maxq.core import report as report_module, runs
    from maxq.core.types import Metrics, QueryResult, RunStatus

    db.update_run_status(run_id, RunStatus.REPORTING)
    log_progress(run_id, "Report job started - loading artifacts", "INFO")

    # Load run
    run = runs.read_run_json(run_id)

    # Load metrics
    log_progress(run_id, "Loading metrics.json", "INFO")
    metrics_path = Path(settings.runs_dir) / run_id / "metrics.json"
    with open(metrics_path) as f:
        metrics_data = json.load(f)
    metrics = Metrics(**metrics_data)

    # Load results
    log_progress(run_id, "Loading results.jsonl", "INFO")
    results_path = Path(settings.runs_dir) / run_id / "results.jsonl"
    query_results = []
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            qr = QueryResult(**data)
            query_results.append(qr)

    log_progress(run_id, f"Loaded {len(query_results)} query results", "SUCCESS")

    # Generate report
    log_progress(run_id, "Generating markdown report with failure analysis", "INFO")
    report_md = report_module.generate_report(
        run_id=run_id,
        metrics=metrics,
        query_results=query_results,
        config=run.config.model_dump(),
    )

    # Write report
    report_path = Path(settings.runs_dir) / run_id / "report.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    log_progress(run_id, f"Report written to {report_path}", "SUCCESS")
    console.print(f"  Report written to {report_path}")

    # Final git commit for the complete run
    git_commit_progress(run_id, "Report generated - run complete")

    log_progress(run_id, "Report job complete - status: DONE", "SUCCESS")
    log_progress(run_id, "=" * 50, "INFO")
    log_progress(run_id, "RUN COMPLETE", "SUCCESS")
    log_progress(run_id, "=" * 50, "INFO")

    db.update_run_status(run_id, RunStatus.DONE)


if __name__ == "__main__":
    run_worker()
