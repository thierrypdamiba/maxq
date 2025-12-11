"""Base assertion interface and result types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Unified search result format."""

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AssertionResult:
    """Result of evaluating an assertion."""

    passed: bool
    assertion_type: str
    message: str
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} [{self.assertion_type}]: {self.message}"


@dataclass
class EvalContext:
    """Context passed to assertions during evaluation."""

    query: str
    ground_truth: list[str] | None = None
    latency_ms: float = 0.0
    top_k: int = 10


class Assertion(ABC):
    """Base class for all assertions."""

    assertion_type: str = "base"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the assertion with configuration.

        Args:
            config: Assertion configuration from YAML
        """
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        """
        Evaluate the assertion against search results.

        Args:
            results: Search results to evaluate
            context: Evaluation context with query, ground truth, etc.

        Returns:
            AssertionResult indicating pass/fail and details
        """
        pass

    def _make_result(
        self,
        passed: bool,
        message: str,
        expected: Any = None,
        actual: Any = None,
        details: dict[str, Any] | None = None,
    ) -> AssertionResult:
        """Helper to create an AssertionResult."""
        return AssertionResult(
            passed=passed,
            assertion_type=self.assertion_type,
            message=message,
            expected=expected,
            actual=actual,
            details=details,
        )
