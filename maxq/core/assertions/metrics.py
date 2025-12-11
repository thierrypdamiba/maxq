"""IR Metric assertions: ndcg, mrr, recall, precision, hit-rate."""

import math

from maxq.core.assertions.base import Assertion, AssertionResult, EvalContext, SearchResult
from maxq.core.assertions.registry import register_assertion


def _dcg(relevances: list[float], k: int | None = None) -> float:
    """Calculate Discounted Cumulative Gain."""
    if k is not None:
        relevances = relevances[:k]
    return sum(
        rel / math.log2(i + 2)  # +2 because log2(1) = 0
        for i, rel in enumerate(relevances)
    )


def _ndcg(result_ids: list[str], ground_truth: list[str], k: int | None = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    Args:
        result_ids: List of returned document IDs (in order)
        ground_truth: List of relevant document IDs
        k: Cutoff (None = use all results)

    Returns:
        NDCG score between 0 and 1
    """
    if not ground_truth:
        return 0.0

    gt_set = set(ground_truth)

    # Calculate relevance for each result (binary: 1 if in ground truth, 0 otherwise)
    relevances = [1.0 if rid in gt_set else 0.0 for rid in result_ids]

    # Ideal relevances (all relevant docs at top)
    ideal_relevances = [1.0] * min(len(ground_truth), len(result_ids))

    dcg = _dcg(relevances, k)
    idcg = _dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def _mrr(result_ids: list[str], ground_truth: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Returns the reciprocal of the rank of the first relevant result.
    """
    gt_set = set(ground_truth)
    for i, rid in enumerate(result_ids):
        if rid in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def _recall(result_ids: list[str], ground_truth: list[str], k: int | None = None) -> float:
    """
    Calculate Recall@K.

    Args:
        result_ids: List of returned document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff (None = use all results)

    Returns:
        Recall score between 0 and 1
    """
    if not ground_truth:
        return 0.0

    if k is not None:
        result_ids = result_ids[:k]

    gt_set = set(ground_truth)
    found = sum(1 for rid in result_ids if rid in gt_set)
    return found / len(ground_truth)


def _precision(result_ids: list[str], ground_truth: list[str], k: int | None = None) -> float:
    """
    Calculate Precision@K.

    Args:
        result_ids: List of returned document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff (None = use all results)

    Returns:
        Precision score between 0 and 1
    """
    if k is not None:
        result_ids = result_ids[:k]

    if not result_ids:
        return 0.0

    gt_set = set(ground_truth)
    found = sum(1 for rid in result_ids if rid in gt_set)
    return found / len(result_ids)


def _hit_rate(result_ids: list[str], ground_truth: list[str], k: int | None = None) -> float:
    """
    Calculate Hit Rate (Hit@K).

    Returns 1.0 if any relevant document is in top-k, 0.0 otherwise.
    """
    if k is not None:
        result_ids = result_ids[:k]

    gt_set = set(ground_truth)
    return 1.0 if any(rid in gt_set for rid in result_ids) else 0.0


@register_assertion("ndcg")
class NDCGAssertion(Assertion):
    """Assert NDCG@K meets threshold."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.8)
        k = self.config.get("k")

        if not context.ground_truth:
            return self._make_result(
                passed=False,
                message="Cannot compute NDCG without ground_truth",
                expected=f"NDCG >= {threshold}",
                actual="N/A",
            )

        result_ids = [r.id for r in results]
        score = _ndcg(result_ids, context.ground_truth, k)
        passed = score >= threshold

        k_str = f"@{k}" if k else ""
        return self._make_result(
            passed=passed,
            message=f"NDCG{k_str}: {score:.3f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
        )


@register_assertion("mrr")
class MRRAssertion(Assertion):
    """Assert MRR meets threshold."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.5)

        if not context.ground_truth:
            return self._make_result(
                passed=False,
                message="Cannot compute MRR without ground_truth",
                expected=f"MRR >= {threshold}",
                actual="N/A",
            )

        result_ids = [r.id for r in results]
        score = _mrr(result_ids, context.ground_truth)
        passed = score >= threshold

        return self._make_result(
            passed=passed,
            message=f"MRR: {score:.3f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
        )


@register_assertion("recall")
class RecallAssertion(Assertion):
    """Assert Recall@K meets threshold."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.8)
        k = self.config.get("k")

        if not context.ground_truth:
            return self._make_result(
                passed=False,
                message="Cannot compute Recall without ground_truth",
                expected=f"Recall >= {threshold}",
                actual="N/A",
            )

        result_ids = [r.id for r in results]
        score = _recall(result_ids, context.ground_truth, k)
        passed = score >= threshold

        k_str = f"@{k}" if k else ""
        return self._make_result(
            passed=passed,
            message=f"Recall{k_str}: {score:.3f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
        )


@register_assertion("precision")
class PrecisionAssertion(Assertion):
    """Assert Precision@K meets threshold."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.8)
        k = self.config.get("k")

        if not context.ground_truth:
            return self._make_result(
                passed=False,
                message="Cannot compute Precision without ground_truth",
                expected=f"Precision >= {threshold}",
                actual="N/A",
            )

        result_ids = [r.id for r in results]
        score = _precision(result_ids, context.ground_truth, k)
        passed = score >= threshold

        k_str = f"@{k}" if k else ""
        return self._make_result(
            passed=passed,
            message=f"Precision{k_str}: {score:.3f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
        )


@register_assertion("hit-rate")
class HitRateAssertion(Assertion):
    """Assert Hit@K (any relevant doc in top-k)."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 1.0)
        k = self.config.get("k")

        if not context.ground_truth:
            return self._make_result(
                passed=False,
                message="Cannot compute Hit Rate without ground_truth",
                expected=f"Hit Rate >= {threshold}",
                actual="N/A",
            )

        result_ids = [r.id for r in results]
        score = _hit_rate(result_ids, context.ground_truth, k)
        passed = score >= threshold

        k_str = f"@{k}" if k else ""
        return self._make_result(
            passed=passed,
            message=f"Hit{k_str}: {'Yes' if score == 1.0 else 'No'}",
            expected="Hit" if threshold == 1.0 else f">= {threshold}",
            actual="Hit" if score == 1.0 else "Miss",
        )
