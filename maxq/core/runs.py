"""Run management utilities."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from maxq.core.config import settings
from maxq.core.types import Run, RunConfig, RunStatus

# Export for easy access
RUNS_DIR = Path(settings.runs_dir)


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{short_uuid}"


def get_run_dir(run_id: str) -> Path:
    """Get the directory path for a run (doesn't create it)."""
    return RUNS_DIR / run_id


def create_run_dir(run_id: str) -> Path:
    """Create a run directory and return its path."""
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_run(config: RunConfig) -> Run:
    """Create a new run with initial state."""
    now = datetime.now()
    run = Run(
        run_id=generate_run_id(),
        status=RunStatus.QUEUED,
        config=config,
        created_at=now,
        updated_at=now,
    )

    # Create run directory
    create_run_dir(run.run_id)

    # Write initial run.json
    write_run_json(run)

    return run


def write_run_json(run: Run) -> None:
    """Write run state to run.json."""
    run_dir = Path(settings.runs_dir) / run.run_id
    run_file = run_dir / "run.json"

    with open(run_file, "w") as f:
        json.dump(run.model_dump(mode="json"), f, indent=2, default=str)


def read_run_json(run_id: str) -> Run:
    """Read run state from run.json."""
    run_file = Path(settings.runs_dir) / run_id / "run.json"

    with open(run_file) as f:
        data = json.load(f)

    return Run(**data)


def update_run_status(run_id: str, status: RunStatus, error: str | None = None) -> Run:
    """Update run status and write to disk."""
    run = read_run_json(run_id)
    run.status = status
    run.updated_at = datetime.now()
    if error:
        run.error = error
    write_run_json(run)
    return run


def write_artifact(run_id: str, filename: str, data: dict | list) -> Path:
    """Write a JSON artifact to the run directory."""
    run_dir = Path(settings.runs_dir) / run_id
    artifact_path = run_dir / filename

    with open(artifact_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return artifact_path


def write_jsonl_artifact(run_id: str, filename: str, records: list[dict]) -> Path:
    """Write a JSONL artifact to the run directory."""
    run_dir = Path(settings.runs_dir) / run_id
    artifact_path = run_dir / filename

    with open(artifact_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")

    return artifact_path
