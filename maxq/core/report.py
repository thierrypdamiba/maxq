"""Report generation."""

from datetime import datetime

from maxq.core.types import Metrics, QueryResult


def generate_report(
    run_id: str,
    metrics: Metrics,
    query_results: list[QueryResult],
    config: dict,
    max_failures: int = 5,
) -> str:
    """Generate markdown report.

    Args:
        run_id: Run identifier
        metrics: Computed metrics
        query_results: All query results
        config: Run configuration
        max_failures: Max failure examples to show
    """
    lines = []

    # Header
    lines.append("# MaxQ Evaluation Report")
    lines.append("")
    lines.append(f"**Run ID:** {run_id}")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # Metrics summary
    lines.append("## Metrics Summary")
    lines.append("")
    lines.append(f"- **Total Queries:** {metrics.total_queries}")
    lines.append(f"- **Queries with Hits:** {metrics.queries_with_hits}")
    lines.append("")

    # Metrics table
    lines.append("### Recall@K")
    lines.append("")
    lines.append("| K | Recall |")
    lines.append("|---|--------|")
    for k, v in sorted(metrics.recall_at_k.items()):
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    lines.append("### MRR@K")
    lines.append("")
    lines.append("| K | MRR |")
    lines.append("|---|-----|")
    for k, v in sorted(metrics.mrr_at_k.items()):
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    lines.append("### NDCG@K")
    lines.append("")
    lines.append("| K | NDCG |")
    lines.append("|---|------|")
    for k, v in sorted(metrics.ndcg_at_k.items()):
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    # Failure analysis
    failures = find_failures(query_results)
    if failures:
        lines.append("## Failure Analysis")
        lines.append("")
        lines.append("Queries with no relevant results in top-20:")
        lines.append("")

        for qr in failures[:max_failures]:
            lines.append(f"### Query: {qr.query_id}")
            lines.append("")
            lines.append(f"**Query text:** {qr.query}")
            lines.append("")
            lines.append(f"**Expected:** {', '.join(qr.relevant_doc_ids)}")
            lines.append("")
            lines.append("**Top 5 results:**")
            lines.append("")
            for i, r in enumerate(qr.results[:5]):
                lines.append(f"{i+1}. `{r.doc_id}` (score: {r.score:.4f})")
            lines.append("")

        if len(failures) > max_failures:
            lines.append(f"*... and {len(failures) - max_failures} more failures*")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    avg_recall = sum(metrics.recall_at_k.values()) / len(metrics.recall_at_k) if metrics.recall_at_k else 0

    if avg_recall < 0.5:
        lines.append(f"- Consider adjusting chunk size (current: {config.get('chunk_size', 'N/A')})")
        lines.append("- Try a different embedding model")
        lines.append("- Review relevance judgments for accuracy")
    elif avg_recall < 0.8:
        lines.append("- Fine-tune chunk overlap for better coverage")
        lines.append("- Consider hybrid search (dense + sparse)")
    else:
        lines.append("- Results look good! Consider testing with more queries.")

    return "\n".join(lines)


def find_failures(query_results: list[QueryResult], top_k: int = 20) -> list[QueryResult]:
    """Find queries with no relevant results in top-k."""
    failures = []

    for qr in query_results:
        relevant = set(qr.relevant_doc_ids)
        result_doc_ids = [r.doc_id for r in qr.results[:top_k]]

        if not (set(result_doc_ids) & relevant):
            failures.append(qr)

    return failures
