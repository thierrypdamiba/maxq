"""IR evaluation metrics."""

import math
from collections import defaultdict

from maxq.core.types import Metrics, QueryResult


def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Recall@K.

    What fraction of relevant items appear in top-k results?
    """
    if not relevant:
        return 0.0

    top_k_results = set(results[:k])
    hits = len(top_k_results & relevant)
    return hits / len(relevant)


def mrr_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank @ K.

    What is 1/rank of the first relevant result?
    """
    for i, result in enumerate(results[:k]):
        if result in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Discounted Cumulative Gain @ K."""
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        if result in relevant:
            # Binary relevance: 1 if relevant, 0 if not
            dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed
    return dcg


def ndcg_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Normalized DCG @ K.

    DCG / Ideal DCG (if all relevant items were ranked first).
    """
    if not relevant:
        return 0.0

    dcg = dcg_at_k(results, relevant, k)

    # Ideal DCG: all relevant items ranked first
    ideal_results = list(relevant)[:k]
    idcg = dcg_at_k(ideal_results, relevant, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_metrics(
    query_results: list[QueryResult],
    k_values: list[int] | None = None,
    use_chunk_ids: bool = False,
) -> Metrics:
    """Compute aggregate metrics across all queries.

    Args:
        query_results: List of query results with ground truth
        k_values: K values to compute metrics at (default: [5, 10, 20])
        use_chunk_ids: If True, use chunk IDs for relevance; else use doc IDs
    """
    if k_values is None:
        k_values = [5, 10, 20]

    recall_sums: dict[int, float] = defaultdict(float)
    mrr_sums: dict[int, float] = defaultdict(float)
    ndcg_sums: dict[int, float] = defaultdict(float)

    queries_with_hits = 0

    for qr in query_results:
        # Get relevant set
        if use_chunk_ids and qr.relevant_ids:
            relevant = set(qr.relevant_ids)
            result_ids = [r.id for r in qr.results]
        else:
            relevant = set(qr.relevant_doc_ids)
            result_ids = [r.doc_id for r in qr.results]

        # Check if any hits
        if set(result_ids) & relevant:
            queries_with_hits += 1

        for k in k_values:
            recall_sums[k] += recall_at_k(result_ids, relevant, k)
            mrr_sums[k] += mrr_at_k(result_ids, relevant, k)
            ndcg_sums[k] += ndcg_at_k(result_ids, relevant, k)

    n = len(query_results)
    if n == 0:
        return Metrics()

    return Metrics(
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        mrr_at_k={k: mrr_sums[k] / n for k in k_values},
        ndcg_at_k={k: ndcg_sums[k] / n for k in k_values},
        total_queries=n,
        queries_with_hits=queries_with_hits,
    )
