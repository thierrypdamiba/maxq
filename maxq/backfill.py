"""
MaxQ Embedding Backfill - Re-embed, fill gaps, and add new vector spaces.

Three modes:
1. Re-embed: Replace existing vectors with a new model
2. Fill gaps: Detect points missing dense or sparse vectors, fill them
3. Add vector: Add a new named vector space to existing points

All modes use scroll + upsert to preserve point IDs and payloads.
"""

import os
import logging
from enum import Enum
from typing import Optional, Any, Callable

from pydantic import BaseModel
from qdrant_client import models
from qdrant_client.models import Document

logger = logging.getLogger("maxq.backfill")


class BackfillMode(str, Enum):
    REEMBED = "reembed"      # Replace dense vectors with a new model
    FILL_GAPS = "fill_gaps"  # Fill missing dense or sparse vectors
    ADD_VECTOR = "add_vector"  # Add a new named vector to all points


class BackfillConfig(BaseModel):
    collection_name: str
    mode: BackfillMode = BackfillMode.REEMBED
    # For reembed / add_vector
    dense_model: Optional[str] = None
    sparse_model: Optional[str] = None
    # For add_vector: name of the new vector space
    new_vector_name: Optional[str] = None
    new_vector_size: Optional[int] = None
    # Processing
    batch_size: int = 50
    text_field: str = "_text"  # payload field containing the source text


class BackfillProgress(BaseModel):
    total_points: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    status: str = "pending"  # pending, running, completed, failed


class BackfillAgent:
    """Handles embedding backfill operations on existing collections."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.client = engine.client

    def analyze(self, collection_name: str) -> dict:
        """Analyze a collection to understand its vector configuration and gaps."""
        if not self.client.collection_exists(collection_name):
            return {"error": f"Collection '{collection_name}' not found"}

        info = self.client.get_collection(collection_name)
        points_count = info.points_count or 0

        # Get vector config
        vectors_config = {}
        if hasattr(info.config.params, "vectors") and info.config.params.vectors:
            vc = info.config.params.vectors
            if isinstance(vc, dict):
                for name, params in vc.items():
                    vectors_config[name] = {
                        "size": getattr(params, "size", None),
                        "distance": str(getattr(params, "distance", "")),
                    }

        sparse_config = {}
        if hasattr(info.config.params, "sparse_vectors") and info.config.params.sparse_vectors:
            sc = info.config.params.sparse_vectors
            if isinstance(sc, dict):
                for name, params in sc.items():
                    sparse_config[name] = {"modifier": str(getattr(params, "modifier", ""))}

        # Sample points to detect gaps
        gaps = {"missing_text": 0, "missing_dense": 0, "missing_sparse": 0}
        sample_size = min(100, points_count)
        if sample_size > 0:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                if not payload.get("_text") and not payload.get("text"):
                    gaps["missing_text"] += 1

                vectors = point.vector or {}
                if isinstance(vectors, dict):
                    if "dense" not in vectors or vectors.get("dense") is None:
                        gaps["missing_dense"] += 1
                    for sparse_name in sparse_config:
                        if sparse_name not in vectors or vectors.get(sparse_name) is None:
                            gaps["missing_sparse"] += 1

        return {
            "collection_name": collection_name,
            "points_count": points_count,
            "vectors_count": info.vectors_count,
            "dense_vectors": vectors_config,
            "sparse_vectors": sparse_config,
            "sample_size": sample_size,
            "gaps": gaps,
            "status": str(info.status),
        }

    def run(
        self,
        config: BackfillConfig,
        callback: Optional[Callable[[BackfillProgress], None]] = None,
    ) -> BackfillProgress:
        """Run the backfill operation."""
        progress = BackfillProgress(status="running")

        if not self.client.collection_exists(config.collection_name):
            progress.status = "failed"
            return progress

        info = self.client.get_collection(config.collection_name)
        progress.total_points = info.points_count or 0

        if progress.total_points == 0:
            progress.status = "completed"
            return progress

        # If adding a new vector space, create it first
        if config.mode == BackfillMode.ADD_VECTOR:
            self._ensure_vector_space(config)

        # Scroll through all points and process
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=config.collection_name,
                limit=config.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            try:
                self._process_batch(config, points, progress)
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                progress.failed += len(points)

            if callback:
                callback(progress)

            if offset is None:
                break

        progress.status = "completed"
        if callback:
            callback(progress)

        return progress

    def _process_batch(
        self,
        config: BackfillConfig,
        points: list,
        progress: BackfillProgress,
    ):
        """Process a batch of points based on the backfill mode."""
        updated_points = []

        for point in points:
            payload = point.payload or {}
            text = payload.get(config.text_field, payload.get("text", ""))

            if not text:
                progress.skipped += 1
                continue

            vectors = point.vector if isinstance(point.vector, dict) else {}

            if config.mode == BackfillMode.REEMBED:
                new_vectors = self._reembed_point(config, text, vectors)
            elif config.mode == BackfillMode.FILL_GAPS:
                new_vectors = self._fill_gaps_point(config, text, vectors)
            elif config.mode == BackfillMode.ADD_VECTOR:
                new_vectors = self._add_vector_point(config, text, vectors)
            else:
                progress.skipped += 1
                continue

            if new_vectors is None:
                progress.skipped += 1
                continue

            # Build updated point preserving ID and payload
            updated_points.append(
                models.PointStruct(
                    id=point.id,
                    vector=new_vectors,
                    payload=payload,
                )
            )
            progress.processed += 1

        if updated_points:
            self.client.upsert(
                collection_name=config.collection_name,
                points=updated_points,
                wait=True,
            )

    def _reembed_point(self, config: BackfillConfig, text: str, existing_vectors: dict) -> dict:
        """Replace dense (and optionally sparse) vectors with new model."""
        new_vectors = dict(existing_vectors)  # Preserve any vectors we're not replacing

        if config.dense_model:
            new_vectors["dense"] = Document(text=text, model=config.dense_model)

        if config.sparse_model:
            sparse_name = config.sparse_model.split("/")[-1].lower().replace("-", "_")
            new_vectors[sparse_name] = Document(text=text, model=config.sparse_model)

        return new_vectors

    def _fill_gaps_point(self, config: BackfillConfig, text: str, existing_vectors: dict) -> Optional[dict]:
        """Fill in missing vectors only. Returns None if no gaps found."""
        new_vectors = dict(existing_vectors)
        changed = False

        # Fill missing dense
        dense_model = config.dense_model or os.getenv(
            "MAXQ_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        if "dense" not in existing_vectors or existing_vectors.get("dense") is None:
            new_vectors["dense"] = Document(text=text, model=dense_model)
            changed = True

        # Fill missing sparse
        sparse_model = config.sparse_model or os.getenv("MAXQ_SPARSE_MODEL", "Qdrant/bm25")
        sparse_name = sparse_model.split("/")[-1].lower().replace("-", "_")
        if sparse_name not in existing_vectors or existing_vectors.get(sparse_name) is None:
            new_vectors[sparse_name] = Document(text=text, model=sparse_model)
            changed = True

        return new_vectors if changed else None

    def _add_vector_point(self, config: BackfillConfig, text: str, existing_vectors: dict) -> dict:
        """Add a new named vector to the point."""
        new_vectors = dict(existing_vectors)

        if config.new_vector_name and config.dense_model:
            new_vectors[config.new_vector_name] = Document(text=text, model=config.dense_model)
        elif config.sparse_model and config.new_vector_name:
            new_vectors[config.new_vector_name] = Document(text=text, model=config.sparse_model)

        return new_vectors

    def _ensure_vector_space(self, config: BackfillConfig):
        """Add a new named vector configuration to the collection."""
        if not config.new_vector_name:
            return

        # For dense vectors, we need the size
        if config.dense_model and config.new_vector_size:
            try:
                self.client.update_collection(
                    collection_name=config.collection_name,
                    vectors_config={
                        config.new_vector_name: models.VectorParams(
                            size=config.new_vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            except Exception as e:
                logger.warning(f"Could not add vector space '{config.new_vector_name}': {e}")
