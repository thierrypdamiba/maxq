"""
MaxQ Cluster Cleanup Agent - Analyze and clean up Qdrant clusters.

Provides:
- Collection stats and health overview
- Stale/empty collection detection
- Duplicate point detection within collections
- LLM-powered cleanup recommendations
- Safe execution with dry-run support
"""

import hashlib
import json
from typing import List, Optional, Any

from pydantic import BaseModel


class CollectionStats(BaseModel):
    name: str
    points_count: int = 0
    vectors_count: int = 0
    segments_count: int = 0
    status: str = "unknown"
    on_disk_payload_size: Optional[int] = None
    error: Optional[str] = None


class CleanupAction(BaseModel):
    action: str  # "delete_collection", "delete_points", "optimize"
    target: str  # collection name
    reason: str
    estimated_points: int = 0


class DuplicateGroup(BaseModel):
    collection: str
    hash: str
    point_ids: List[str]
    sample_text: str


class CleanupReport(BaseModel):
    total_collections: int = 0
    total_points: int = 0
    collections: List[CollectionStats] = []
    empty_collections: List[str] = []
    stale_collections: List[str] = []
    duplicate_groups: List[DuplicateGroup] = []
    suggested_actions: List[CleanupAction] = []
    reclaimable_points: int = 0
    llm_summary: Optional[str] = None


class CleanupAgent:
    """Analyzes and cleans up Qdrant clusters."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.client = engine.client
        self.llm = engine.llm_client

    def analyze(self, collection_name: Optional[str] = None) -> CleanupReport:
        """Analyze cluster or specific collection and generate cleanup report."""
        report = CleanupReport()

        if collection_name:
            collections_to_check = [collection_name]
        else:
            result = self.client.get_collections()
            collections_to_check = [c.name for c in result.collections]

        report.total_collections = len(collections_to_check)

        for name in collections_to_check:
            stats = self._get_collection_stats(name)
            report.collections.append(stats)
            report.total_points += stats.points_count

            if stats.points_count == 0:
                report.empty_collections.append(name)
                report.suggested_actions.append(CleanupAction(
                    action="delete_collection",
                    target=name,
                    reason="Collection is empty (0 points)",
                    estimated_points=0,
                ))

        # Detect stale collections (naming heuristics)
        for stats in report.collections:
            name = stats.name.lower()
            if any(tag in name for tag in ["test", "tmp", "temp", "old", "backup", "copy"]):
                if stats.name not in report.empty_collections:
                    report.stale_collections.append(stats.name)
                    report.suggested_actions.append(CleanupAction(
                        action="delete_collection",
                        target=stats.name,
                        reason=f"Collection name suggests it is temporary/stale: '{stats.name}'",
                        estimated_points=stats.points_count,
                    ))

        report.reclaimable_points = sum(a.estimated_points for a in report.suggested_actions)
        return report

    def find_duplicates(
        self, collection_name: str, sample_size: int = 500, threshold: float = 0.98
    ) -> List[DuplicateGroup]:
        """Find duplicate points in a collection by text content hashing."""
        if not self.client.collection_exists(collection_name):
            return []

        # Scroll through sample of points
        points, _ = self.client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )

        # Group by content hash
        hash_groups: dict[str, list] = {}
        for point in points:
            payload = point.payload or {}
            text = payload.get("_text", payload.get("text", ""))
            if not text:
                continue
            content_hash = hashlib.md5(str(text).strip().lower().encode()).hexdigest()
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append({
                "id": str(point.id),
                "text": str(text)[:200],
            })

        # Return only groups with duplicates
        duplicates = []
        for h, group in hash_groups.items():
            if len(group) > 1:
                duplicates.append(DuplicateGroup(
                    collection=collection_name,
                    hash=h,
                    point_ids=[g["id"] for g in group],
                    sample_text=group[0]["text"],
                ))

        return duplicates

    def execute(self, actions: List[CleanupAction], dry_run: bool = True) -> List[dict]:
        """Execute cleanup actions. Returns results for each action."""
        results = []
        for action in actions:
            result = {"action": action.action, "target": action.target, "dry_run": dry_run}

            if dry_run:
                result["status"] = "skipped"
                result["message"] = f"Dry run: would {action.action} on '{action.target}'"
                results.append(result)
                continue

            try:
                if action.action == "delete_collection":
                    self.client.delete_collection(action.target)
                    result["status"] = "completed"
                    result["message"] = f"Deleted collection '{action.target}'"

                elif action.action == "delete_points":
                    # Expects target format: "collection:filter_field:filter_value"
                    parts = action.target.split(":", 2)
                    if len(parts) == 3:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        self.client.delete(
                            collection_name=parts[0],
                            points_selector=Filter(
                                must=[FieldCondition(key=parts[1], match=MatchValue(value=parts[2]))]
                            ),
                        )
                        result["status"] = "completed"
                        result["message"] = f"Deleted points matching {parts[1]}={parts[2]} from '{parts[0]}'"
                    else:
                        result["status"] = "failed"
                        result["message"] = "Invalid target format for delete_points"

                elif action.action == "optimize":
                    # Trigger optimization
                    self.client.update_collection(
                        collection_name=action.target,
                        optimizer_config={"indexing_threshold": 0},
                    )
                    result["status"] = "completed"
                    result["message"] = f"Triggered optimization for '{action.target}'"

                else:
                    result["status"] = "failed"
                    result["message"] = f"Unknown action: {action.action}"

            except Exception as e:
                result["status"] = "failed"
                result["message"] = str(e)

            results.append(result)

        return results

    def summarize_with_llm(self, report: CleanupReport) -> str:
        """Generate LLM-powered natural language summary of cleanup report."""
        if not self.llm:
            return "OpenAI API key not configured. Set OPENAI_API_KEY for LLM summaries."

        report_data = report.model_dump()
        # Truncate for context window
        report_json = json.dumps(report_data, default=str)[:4000]

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MaxQ, a vector database assistant. "
                        "Analyze this Qdrant cluster cleanup report and provide a concise summary "
                        "with actionable recommendations. Be direct and specific."
                    ),
                },
                {"role": "user", "content": f"Cleanup report:\n{report_json}"},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content or "No summary generated."

    def _get_collection_stats(self, name: str) -> CollectionStats:
        """Get stats for a single collection."""
        try:
            info = self.client.get_collection(name)
            return CollectionStats(
                name=name,
                points_count=info.points_count or 0,
                vectors_count=info.vectors_count or 0,
                segments_count=info.segments_count or 0,
                status=str(info.status),
            )
        except Exception as e:
            return CollectionStats(name=name, error=str(e))
