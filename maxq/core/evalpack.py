"""
Evaluation Pack (EvalPack) management for MaxQ.

An EvalPack contains:
- queries: list of test queries
- qrels: relevance judgments (query_id -> list of relevant doc_ids)
- metadata: name, description, created_at, etc.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from maxq.core.config import settings
from maxq.core.types import EvalQuery


class EvalPackMetadata(BaseModel):
    """Metadata for an evaluation pack."""

    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    num_queries: int = 0
    source: str = ""  # file path or URL it was loaded from
    tags: list[str] = Field(default_factory=list)


class EvalPack(BaseModel):
    """An evaluation pack with queries and relevance judgments."""

    metadata: EvalPackMetadata
    queries: list[EvalQuery] = Field(default_factory=list)


class RunMetadata(BaseModel):
    """Reproducibility metadata for a run."""

    maxq_version: str = "0.1.0"
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    qdrant_url_hash: Optional[str] = None  # hash of URL for privacy
    embedding_mode: str = "cloud_inference"
    created_at: datetime = Field(default_factory=datetime.now)
    hostname: Optional[str] = None
    python_version: Optional[str] = None


def get_evalpacks_dir() -> Path:
    """Get the evalpacks directory."""
    evalpacks_dir = settings.app_dir / "evalpacks"
    evalpacks_dir.mkdir(parents=True, exist_ok=True)
    return evalpacks_dir


def list_evalpacks() -> list[EvalPackMetadata]:
    """List all stored eval packs."""
    evalpacks_dir = get_evalpacks_dir()
    packs = []

    for pack_file in evalpacks_dir.glob("*.json"):
        try:
            with open(pack_file) as f:
                data = json.load(f)
            if "metadata" in data:
                packs.append(EvalPackMetadata(**data["metadata"]))
        except Exception:
            continue

    return sorted(packs, key=lambda p: p.created_at, reverse=True)


def get_evalpack(name: str) -> Optional[EvalPack]:
    """Get an eval pack by name."""
    pack_file = get_evalpacks_dir() / f"{name}.json"
    if not pack_file.exists():
        return None

    with open(pack_file) as f:
        data = json.load(f)
    return EvalPack(**data)


def save_evalpack(pack: EvalPack) -> Path:
    """Save an eval pack."""
    pack_file = get_evalpacks_dir() / f"{pack.metadata.name}.json"
    with open(pack_file, "w") as f:
        json.dump(pack.model_dump(mode="json"), f, indent=2, default=str)
    return pack_file


def delete_evalpack(name: str) -> bool:
    """Delete an eval pack."""
    pack_file = get_evalpacks_dir() / f"{name}.json"
    if pack_file.exists():
        pack_file.unlink()
        return True
    return False


def load_evalpack_from_file(filepath: str, name: str) -> EvalPack:
    """
    Load an eval pack from a JSON or JSONL file.

    Expected format (JSON):
    {
        "queries": [
            {"id": "q1", "query": "what is ML?", "relevant_doc_ids": ["doc1", "doc2"]},
            ...
        ]
    }

    Or JSONL (one query per line):
    {"id": "q1", "query": "what is ML?", "relevant_doc_ids": ["doc1", "doc2"]}
    {"id": "q2", "query": "how does qdrant work?", "relevant_doc_ids": ["doc3"]}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    queries = []

    if path.suffix == ".jsonl":
        # JSONL format
        with open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    queries.append(
                        EvalQuery(
                            id=data.get("id", f"q{i}"),
                            query=data["query"],
                            relevant_doc_ids=data.get("relevant_doc_ids", []),
                            relevant_ids=data.get("relevant_ids", []),
                        )
                    )
    else:
        # JSON format
        with open(path) as f:
            data = json.load(f)

        # Handle both formats: {"queries": [...]} or [...]
        query_list = data.get("queries", data) if isinstance(data, dict) else data

        for i, q in enumerate(query_list):
            queries.append(
                EvalQuery(
                    id=q.get("id", f"q{i}"),
                    query=q["query"],
                    relevant_doc_ids=q.get("relevant_doc_ids", []),
                    relevant_ids=q.get("relevant_ids", []),
                )
            )

    metadata = EvalPackMetadata(
        name=name,
        num_queries=len(queries),
        source=str(path.absolute()),
    )

    return EvalPack(metadata=metadata, queries=queries)


def get_reproducibility_metadata() -> RunMetadata:
    """Collect reproducibility metadata for the current environment."""
    import platform
    import hashlib
    import sys

    # Try to get git info
    git_sha = None
    git_branch = None
    git_dirty = False

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:12]
        )

        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Check for uncommitted changes
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        git_dirty = len(status) > 0
    except Exception:
        pass

    # Hash the Qdrant URL for privacy
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_url_hash = hashlib.sha256(qdrant_url.encode()).hexdigest()[:12] if qdrant_url else None

    return RunMetadata(
        maxq_version="0.1.0",
        git_sha=git_sha,
        git_branch=git_branch,
        git_dirty=git_dirty,
        qdrant_url_hash=qdrant_url_hash,
        hostname=platform.node(),
        python_version=sys.version.split()[0],
    )


def create_sample_evalpack() -> EvalPack:
    """Create a sample eval pack for demos."""
    queries = [
        EvalQuery(
            id="q1",
            query="What is machine learning?",
            relevant_doc_ids=["ml_intro", "ml_basics"],
        ),
        EvalQuery(
            id="q2",
            query="How do neural networks work?",
            relevant_doc_ids=["nn_overview", "deep_learning"],
        ),
        EvalQuery(
            id="q3",
            query="What is vector search?",
            relevant_doc_ids=["vector_db", "similarity_search"],
        ),
        EvalQuery(
            id="q4",
            query="How to use Qdrant?",
            relevant_doc_ids=["qdrant_quickstart", "qdrant_tutorial"],
        ),
        EvalQuery(
            id="q5",
            query="What is RAG?",
            relevant_doc_ids=["rag_intro", "retrieval_augmented"],
        ),
    ]

    metadata = EvalPackMetadata(
        name="sample",
        description="Sample eval pack for testing MaxQ",
        num_queries=len(queries),
        source="built-in",
        tags=["sample", "demo"],
    )

    return EvalPack(metadata=metadata, queries=queries)
