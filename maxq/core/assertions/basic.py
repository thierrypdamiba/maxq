"""Basic assertions: not-empty, count, latency, contains-id."""

from maxq.core.assertions.base import Assertion, AssertionResult, EvalContext, SearchResult
from maxq.core.assertions.registry import register_assertion


@register_assertion("not-empty")
class NotEmptyAssertion(Assertion):
    """Assert that results are not empty."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        count = len(results)
        passed = count > 0
        return self._make_result(
            passed=passed,
            message=f"Got {count} results" if passed else "No results returned",
            expected=">0 results",
            actual=count,
        )


@register_assertion("count")
class CountAssertion(Assertion):
    """Assert exact result count."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        expected_count = self.config.get("value", 0)
        actual_count = len(results)
        passed = actual_count == expected_count
        return self._make_result(
            passed=passed,
            message=f"Expected {expected_count} results, got {actual_count}",
            expected=expected_count,
            actual=actual_count,
        )


@register_assertion("latency")
class LatencyAssertion(Assertion):
    """Assert response latency is within bounds."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        max_ms = self.config.get("max_ms")
        min_ms = self.config.get("min_ms", 0)
        actual_ms = context.latency_ms

        passed = True
        message = f"Latency: {actual_ms:.1f}ms"

        if max_ms is not None and actual_ms > max_ms:
            passed = False
            message = f"Latency {actual_ms:.1f}ms exceeds max {max_ms}ms"
        elif min_ms and actual_ms < min_ms:
            passed = False
            message = f"Latency {actual_ms:.1f}ms below min {min_ms}ms"

        return self._make_result(
            passed=passed,
            message=message,
            expected=f"<= {max_ms}ms" if max_ms else f">= {min_ms}ms",
            actual=f"{actual_ms:.1f}ms",
        )


@register_assertion("contains-id")
class ContainsIdAssertion(Assertion):
    """Assert that specific document IDs are in the results."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        expected_ids = self.config.get("value", [])
        if isinstance(expected_ids, str):
            expected_ids = [expected_ids]

        result_ids = {r.id for r in results}
        found = [id for id in expected_ids if id in result_ids]
        missing = [id for id in expected_ids if id not in result_ids]

        passed = len(missing) == 0
        if passed:
            message = f"Found all {len(expected_ids)} expected IDs"
        else:
            message = f"Missing IDs: {missing}"

        return self._make_result(
            passed=passed,
            message=message,
            expected=expected_ids,
            actual=list(result_ids),
            details={"found": found, "missing": missing},
        )
