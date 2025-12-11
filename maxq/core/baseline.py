"""
Baseline and CI functionality for MaxQ.

This module provides regression testing capabilities:
- Store blessed run artifacts as baselines
- Compare new runs against baselines
- CI integration with exit codes and reports
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field

from maxq.core.config import settings
from maxq.core.types import Metrics, QueryResult


class BaselineMetadata(BaseModel):
    """Metadata for a stored baseline."""

    name: str
    run_id: str
    collection: str
    created_at: datetime
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: Metrics
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class QueryDelta(BaseModel):
    """Per-query comparison between two runs."""

    query_id: str
    query: str

    # Metric deltas (new - baseline)
    ndcg_delta: float = 0.0
    recall_delta: float = 0.0
    mrr_delta: float = 0.0

    # Result changes
    results_added: list[str] = Field(default_factory=list)  # doc_ids that appeared
    results_removed: list[str] = Field(default_factory=list)  # doc_ids that disappeared
    rank_changes: dict[str, int] = Field(default_factory=dict)  # doc_id -> rank change

    # Classification
    is_regression: bool = False
    is_improvement: bool = False


class DiffResult(BaseModel):
    """Result of comparing two runs."""

    baseline_run_id: str
    compare_run_id: str
    baseline_name: Optional[str] = None

    # Aggregate metric deltas
    ndcg_10_delta: float = 0.0
    recall_10_delta: float = 0.0
    mrr_10_delta: float = 0.0

    # Per-query analysis
    query_deltas: list[QueryDelta] = Field(default_factory=list)

    # Summary counts
    total_queries: int = 0
    regressions: int = 0
    improvements: int = 0
    unchanged: int = 0

    # Worst regressions (sorted by ndcg_delta ascending)
    worst_regressions: list[QueryDelta] = Field(default_factory=list)
    best_improvements: list[QueryDelta] = Field(default_factory=list)


class CIResult(BaseModel):
    """Result of a CI check."""

    passed: bool
    baseline_name: str
    run_id: str

    # Metric checks
    checks: list[dict[str, Any]] = Field(default_factory=list)
    failed_checks: list[dict[str, Any]] = Field(default_factory=list)

    # Diff summary
    diff: Optional[DiffResult] = None

    # Report
    report_path: Optional[str] = None
    markdown_report: str = ""


def get_baselines_dir() -> Path:
    """Get the baselines directory."""
    baselines_dir = settings.app_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    return baselines_dir


def list_baselines() -> list[BaselineMetadata]:
    """List all stored baselines."""
    baselines_dir = get_baselines_dir()
    baselines = []

    for baseline_file in baselines_dir.glob("*.json"):
        try:
            with open(baseline_file) as f:
                data = json.load(f)
            baselines.append(BaselineMetadata(**data))
        except Exception:
            continue

    return sorted(baselines, key=lambda b: b.created_at, reverse=True)


def get_baseline(name: str) -> Optional[BaselineMetadata]:
    """Get a baseline by name."""
    baseline_file = get_baselines_dir() / f"{name}.json"
    if not baseline_file.exists():
        return None

    with open(baseline_file) as f:
        data = json.load(f)
    return BaselineMetadata(**data)


def save_baseline(
    name: str,
    run_id: str,
    collection: str,
    metrics: Metrics,
    config: dict[str, Any],
    description: str = "",
    tags: Optional[list[str]] = None,
) -> BaselineMetadata:
    """Save a run as a baseline."""
    baseline = BaselineMetadata(
        name=name,
        run_id=run_id,
        collection=collection,
        created_at=datetime.now(),
        config=config,
        metrics=metrics,
        description=description,
        tags=tags if tags is not None else [],
    )

    baseline_file = get_baselines_dir() / f"{name}.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline.model_dump(mode="json"), f, indent=2, default=str)

    # Also copy the query results if they exist
    run_dir = Path(settings.runs_dir) / run_id
    query_results_file = run_dir / "query_results.jsonl"
    if query_results_file.exists():
        baseline_queries_file = get_baselines_dir() / f"{name}_queries.jsonl"
        import shutil

        shutil.copy(query_results_file, baseline_queries_file)

    return baseline


def delete_baseline(name: str) -> bool:
    """Delete a baseline."""
    baseline_file = get_baselines_dir() / f"{name}.json"
    queries_file = get_baselines_dir() / f"{name}_queries.jsonl"

    deleted = False
    if baseline_file.exists():
        baseline_file.unlink()
        deleted = True
    if queries_file.exists():
        queries_file.unlink()

    return deleted


def load_query_results(run_id: str) -> list[QueryResult]:
    """Load query results from a run."""
    run_dir = Path(settings.runs_dir) / run_id
    query_results_file = run_dir / "query_results.jsonl"

    if not query_results_file.exists():
        return []

    results = []
    with open(query_results_file) as f:
        for line in f:
            if line.strip():
                results.append(QueryResult(**json.loads(line)))

    return results


def load_baseline_query_results(name: str) -> list[QueryResult]:
    """Load query results from a baseline."""
    queries_file = get_baselines_dir() / f"{name}_queries.jsonl"

    if not queries_file.exists():
        return []

    results = []
    with open(queries_file) as f:
        for line in f:
            if line.strip():
                results.append(QueryResult(**json.loads(line)))

    return results


def compute_query_delta(
    baseline_qr: QueryResult,
    compare_qr: QueryResult,
    k: int = 10,
) -> QueryDelta:
    """Compute delta between two query results."""
    from maxq.core.eval import ndcg_at_k, recall_at_k, mrr_at_k

    relevant = set(baseline_qr.relevant_doc_ids)

    baseline_ids = [r.doc_id for r in baseline_qr.results]
    compare_ids = [r.doc_id for r in compare_qr.results]

    # Compute metric deltas
    baseline_ndcg = ndcg_at_k(baseline_ids, relevant, k)
    compare_ndcg = ndcg_at_k(compare_ids, relevant, k)

    baseline_recall = recall_at_k(baseline_ids, relevant, k)
    compare_recall = recall_at_k(compare_ids, relevant, k)

    baseline_mrr = mrr_at_k(baseline_ids, relevant, k)
    compare_mrr = mrr_at_k(compare_ids, relevant, k)

    ndcg_delta = compare_ndcg - baseline_ndcg
    recall_delta = compare_recall - baseline_recall
    mrr_delta = compare_mrr - baseline_mrr

    # Compute result changes
    baseline_set = set(baseline_ids[:k])
    compare_set = set(compare_ids[:k])

    results_added = list(compare_set - baseline_set)
    results_removed = list(baseline_set - compare_set)

    # Compute rank changes for docs in both
    rank_changes = {}
    baseline_ranks = {doc_id: i for i, doc_id in enumerate(baseline_ids[:k])}
    compare_ranks = {doc_id: i for i, doc_id in enumerate(compare_ids[:k])}

    for doc_id in baseline_set & compare_set:
        rank_change = baseline_ranks[doc_id] - compare_ranks[doc_id]  # positive = improved
        if rank_change != 0:
            rank_changes[doc_id] = rank_change

    # Classify
    is_regression = ndcg_delta < -0.01 or recall_delta < -0.05
    is_improvement = ndcg_delta > 0.01 or recall_delta > 0.05

    return QueryDelta(
        query_id=baseline_qr.query_id,
        query=baseline_qr.query,
        ndcg_delta=ndcg_delta,
        recall_delta=recall_delta,
        mrr_delta=mrr_delta,
        results_added=results_added,
        results_removed=results_removed,
        rank_changes=rank_changes,
        is_regression=is_regression,
        is_improvement=is_improvement,
    )


def diff_runs(
    baseline_run_id: str,
    compare_run_id: str,
    baseline_name: Optional[str] = None,
    k: int = 10,
    worst_n: int = 20,
) -> DiffResult:
    """Compare two runs and compute detailed diff."""
    # Load query results
    if baseline_name:
        baseline_results = load_baseline_query_results(baseline_name)
        baseline = get_baseline(baseline_name)
        if baseline:
            baseline_run_id = baseline.run_id
    else:
        baseline_results = load_query_results(baseline_run_id)

    compare_results = load_query_results(compare_run_id)

    # Index by query_id
    baseline_by_id = {qr.query_id: qr for qr in baseline_results}
    compare_by_id = {qr.query_id: qr for qr in compare_results}

    # Compute per-query deltas
    query_deltas = []
    for query_id in baseline_by_id:
        if query_id in compare_by_id:
            delta = compute_query_delta(
                baseline_by_id[query_id],
                compare_by_id[query_id],
                k=k,
            )
            query_deltas.append(delta)

    # Aggregate stats
    total_queries = len(query_deltas)
    regressions = sum(1 for d in query_deltas if d.is_regression)
    improvements = sum(1 for d in query_deltas if d.is_improvement)
    unchanged = total_queries - regressions - improvements

    # Compute aggregate metric deltas
    if query_deltas:
        ndcg_10_delta = sum(d.ndcg_delta for d in query_deltas) / len(query_deltas)
        recall_10_delta = sum(d.recall_delta for d in query_deltas) / len(query_deltas)
        mrr_10_delta = sum(d.mrr_delta for d in query_deltas) / len(query_deltas)
    else:
        ndcg_10_delta = recall_10_delta = mrr_10_delta = 0.0

    # Sort for worst regressions and best improvements
    sorted_by_ndcg = sorted(query_deltas, key=lambda d: d.ndcg_delta)
    worst_regressions = [d for d in sorted_by_ndcg[:worst_n] if d.is_regression]
    best_improvements = [d for d in reversed(sorted_by_ndcg[-worst_n:]) if d.is_improvement]

    return DiffResult(
        baseline_run_id=baseline_run_id,
        compare_run_id=compare_run_id,
        baseline_name=baseline_name,
        ndcg_10_delta=ndcg_10_delta,
        recall_10_delta=recall_10_delta,
        mrr_10_delta=mrr_10_delta,
        query_deltas=query_deltas,
        total_queries=total_queries,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        worst_regressions=worst_regressions,
        best_improvements=best_improvements,
    )


def run_ci_check(
    run_id: str,
    baseline_name: str,
    min_ndcg_10: Optional[float] = None,
    max_ndcg_drop: Optional[float] = None,
    min_recall_10: Optional[float] = None,
    max_recall_drop: Optional[float] = None,
    max_p95_ms: Optional[float] = None,
) -> CIResult:
    """Run CI checks against a baseline."""
    baseline = get_baseline(baseline_name)
    if not baseline:
        return CIResult(
            passed=False,
            baseline_name=baseline_name,
            run_id=run_id,
            checks=[],
            failed_checks=[
                {"check": "baseline_exists", "error": f"Baseline '{baseline_name}' not found"}
            ],
        )

    # Load run metrics
    from maxq.core.runs import read_run_json

    try:
        run = read_run_json(run_id)
    except FileNotFoundError:
        return CIResult(
            passed=False,
            baseline_name=baseline_name,
            run_id=run_id,
            checks=[],
            failed_checks=[{"check": "run_exists", "error": f"Run '{run_id}' not found"}],
        )

    if not run.metrics:
        return CIResult(
            passed=False,
            baseline_name=baseline_name,
            run_id=run_id,
            checks=[],
            failed_checks=[
                {"check": "has_metrics", "error": "Run has no metrics (not evaluated yet?)"}
            ],
        )

    # Compute diff
    diff = diff_runs(baseline.run_id, run_id, baseline_name=baseline_name)

    # Compute metric deltas from run metrics (more reliable than query-level diff)
    baseline_ndcg_10 = baseline.metrics.ndcg_at_k.get(10, 0.0)
    baseline_recall_10 = baseline.metrics.recall_at_k.get(10, 0.0)
    current_ndcg_10 = run.metrics.ndcg_at_k.get(10, 0.0)
    current_recall_10 = run.metrics.recall_at_k.get(10, 0.0)

    ndcg_delta = current_ndcg_10 - baseline_ndcg_10
    recall_delta = current_recall_10 - baseline_recall_10

    # Run checks
    checks = []
    failed_checks = []

    # NDCG@10 absolute minimum
    if min_ndcg_10 is not None:
        check = {
            "check": "min_ndcg_10",
            "threshold": min_ndcg_10,
            "actual": current_ndcg_10,
            "passed": current_ndcg_10 >= min_ndcg_10,
        }
        checks.append(check)
        if not check["passed"]:
            failed_checks.append(check)

    # NDCG@10 max drop from baseline
    if max_ndcg_drop is not None:
        check = {
            "check": "max_ndcg_drop",
            "threshold": max_ndcg_drop,
            "actual": ndcg_delta,
            "passed": ndcg_delta >= max_ndcg_drop,  # delta should be >= threshold (less negative)
        }
        checks.append(check)
        if not check["passed"]:
            failed_checks.append(check)

    # Recall@10 absolute minimum
    if min_recall_10 is not None:
        check = {
            "check": "min_recall_10",
            "threshold": min_recall_10,
            "actual": current_recall_10,
            "passed": current_recall_10 >= min_recall_10,
        }
        checks.append(check)
        if not check["passed"]:
            failed_checks.append(check)

    # Recall@10 max drop from baseline
    if max_recall_drop is not None:
        check = {
            "check": "max_recall_drop",
            "threshold": max_recall_drop,
            "actual": recall_delta,
            "passed": diff.recall_10_delta >= max_recall_drop,
        }
        checks.append(check)
        if not check["passed"]:
            failed_checks.append(check)

    # P95 latency budget
    if max_p95_ms is not None:
        current_p95 = run.metrics.latency_p95_ms if run.metrics.latency_p95_ms else 0.0
        check = {
            "check": "max_p95_latency_ms",
            "threshold": max_p95_ms,
            "actual": current_p95,
            "passed": current_p95 <= max_p95_ms,
        }
        checks.append(check)
        if not check["passed"]:
            failed_checks.append(check)

    passed = len(failed_checks) == 0

    # Generate markdown report
    markdown_report = generate_ci_report(
        baseline_name=baseline_name,
        baseline=baseline,
        run_id=run_id,
        run=run,
        diff=diff,
        checks=checks,
        failed_checks=failed_checks,
        passed=passed,
    )

    return CIResult(
        passed=passed,
        baseline_name=baseline_name,
        run_id=run_id,
        checks=checks,
        failed_checks=failed_checks,
        diff=diff,
        markdown_report=markdown_report,
    )


def generate_ci_report(
    baseline_name: str,
    baseline: BaselineMetadata,
    run_id: str,
    run: Any,  # Run type
    diff: DiffResult,
    checks: list[dict],
    failed_checks: list[dict],
    passed: bool,
) -> str:
    """Generate a markdown CI report."""
    status_emoji = "" if passed else ""
    status_text = "PASSED" if passed else "FAILED"

    lines = [
        f"# MaxQ CI Report {status_emoji}",
        "",
        f"**Status:** {status_text}",
        f"**Baseline:** `{baseline_name}` (run: `{baseline.run_id}`)",
        f"**Compare:** `{run_id}`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
    ]

    # Add metric rows
    for k in [5, 10, 20]:
        baseline_ndcg = baseline.metrics.ndcg_at_k.get(k, 0.0)
        current_ndcg = run.metrics.ndcg_at_k.get(k, 0.0) if run.metrics else 0.0
        delta = current_ndcg - baseline_ndcg
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"| NDCG@{k} | {baseline_ndcg:.4f} | {current_ndcg:.4f} | {delta_str} |")

    for k in [5, 10, 20]:
        baseline_recall = baseline.metrics.recall_at_k.get(k, 0.0)
        current_recall = run.metrics.recall_at_k.get(k, 0.0) if run.metrics else 0.0
        delta = current_recall - baseline_recall
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"| Recall@{k} | {baseline_recall:.4f} | {current_recall:.4f} | {delta_str} |")

    # Checks section
    if checks:
        lines.extend(
            [
                "",
                "## CI Checks",
                "",
            ]
        )
        for check in checks:
            emoji = "" if check["passed"] else ""
            lines.append(
                f"- {emoji} **{check['check']}**: {check['actual']:.4f} (threshold: {check['threshold']})"
            )

    # Regressions section
    if diff.worst_regressions:
        lines.extend(
            [
                "",
                f"## Top {len(diff.worst_regressions)} Regressions",
                "",
                "| Query | NDCG Delta | Recall Delta |",
                "|-------|------------|--------------|",
            ]
        )
        for qd in diff.worst_regressions[:10]:
            query_short = qd.query[:50] + "..." if len(qd.query) > 50 else qd.query
            lines.append(f"| {query_short} | {qd.ndcg_delta:+.4f} | {qd.recall_delta:+.4f} |")

    # Summary
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- **Total queries:** {diff.total_queries}",
            f"- **Regressions:** {diff.regressions}",
            f"- **Improvements:** {diff.improvements}",
            f"- **Unchanged:** {diff.unchanged}",
            "",
            "---",
            f"*Generated by MaxQ at {datetime.now().isoformat()}*",
        ]
    )

    return "\n".join(lines)


def write_ci_report(run_id: str, report: str) -> Path:
    """Write CI report to the run directory."""
    run_dir = Path(settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "maxq_ci_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report_path
