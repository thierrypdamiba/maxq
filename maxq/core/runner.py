"""Test runner for maxq declarative tests."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from maxq.core.assertions import get_assertion, list_assertions
from maxq.core.assertions.base import AssertionResult, EvalContext, SearchResult
from maxq.core.testconfig import MaxQConfig, TestCase, load_config


@dataclass
class TestResult:
    """Result of running a single test case."""

    test: TestCase
    query: str
    passed: bool
    results: list[SearchResult]
    assertion_results: list[AssertionResult]
    latency_ms: float
    error: str | None = None

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        return [a for a in self.assertion_results if not a.passed]


@dataclass
class RunResult:
    """Result of a complete test run."""

    run_id: str
    config_path: str
    started_at: datetime
    completed_at: datetime
    tests: list[TestResult]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tests)

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed_tests(self) -> int:
        return sum(1 for t in self.tests if not t.passed)

    @property
    def total_assertions(self) -> int:
        return sum(len(t.assertion_results) for t in self.tests)

    @property
    def passed_assertions(self) -> int:
        return sum(
            sum(1 for a in t.assertion_results if a.passed)
            for t in self.tests
        )


class TestRunner:
    """Runner for maxq declarative tests."""

    def __init__(
        self,
        config: MaxQConfig,
        qdrant_client=None,
    ):
        """
        Initialize the test runner.

        Args:
            config: Parsed MaxQConfig
            qdrant_client: Optional QdrantClient (will create one if not provided)
        """
        self.config = config
        self._client = qdrant_client

    @property
    def client(self):
        """Lazy-load Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            import os

            url = self.config.provider.url or os.getenv("QDRANT_URL")
            api_key = self.config.provider.api_key or os.getenv("QDRANT_API_KEY")

            if url:
                self._client = QdrantClient(url=url, api_key=api_key)
            else:
                # Try local connection
                self._client = QdrantClient(host="localhost", port=6333)

        return self._client

    def _search(self, query: str, top_k: int, filters: dict | None = None) -> tuple[list[SearchResult], float]:
        """
        Execute a search query.

        Supports both cloud mode (Qdrant Cloud Inference) and local mode (FastEmbed).

        Returns:
            Tuple of (results, latency_ms)
        """
        from maxq.core.config import settings

        start = time.perf_counter()

        collection = self.config.provider.collection
        model = self.config.provider.model

        if settings.is_cloud_mode():
            # Cloud mode: use Qdrant Cloud Inference (server-side embedding)
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query,
                    using=model,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points if hasattr(response, 'points') else []
            except Exception:
                # Fallback: try query without model (pre-embedded)
                response = self.client.query_points(
                    collection_name=collection,
                    query=query,
                    limit=top_k,
                    with_payload=True,
                )
                points = response.points if hasattr(response, 'points') else []
        elif settings.is_tei_mode():
            # TEI mode: embed via Text Embeddings Inference server (any HF model)
            from maxq.adapters.embedder_tei import TEIEmbedder

            embedder = TEIEmbedder(url=settings.tei_url)
            query_vector = embedder.embed_single(query)

            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = response.points if hasattr(response, 'points') else []
        else:
            # Local mode: embed query locally with FastEmbed
            from maxq.adapters.embedder_local import LocalEmbedder

            embedder = LocalEmbedder(model=model)
            query_vector = embedder.embed_single(query)

            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = response.points if hasattr(response, 'points') else []

        latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for point in points:
            payload = point.payload or {}
            text = (
                payload.get("_text")
                or payload.get("text")
                or payload.get("content")
                or ""
            )
            results.append(SearchResult(
                id=str(point.id),
                score=getattr(point, 'score', 0.0) or 0.0,
                text=text,
                metadata=payload,
            ))

        return results, latency_ms

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        top_k = test.top_k or self.config.defaults.top_k
        query = test.query

        try:
            # Execute search
            results, latency_ms = self._search(query, top_k, test.filters)

            # Build evaluation context
            ground_truth = test.ground_truth
            if test.ground_truth_file:
                gt_path = Path(test.ground_truth_file)
                if gt_path.exists():
                    with open(gt_path) as f:
                        ground_truth = json.load(f)

            context = EvalContext(
                query=query,
                ground_truth=ground_truth,
                latency_ms=latency_ms,
                top_k=top_k,
            )

            # Run assertions
            assertion_results = []
            for assertion_config in test.assertions:
                config_dict = assertion_config.model_dump(exclude_none=True)
                assertion_type = config_dict.pop("type")
                assertion = get_assertion(assertion_type, config_dict)
                result = assertion.evaluate(results, context)
                assertion_results.append(result)

            # Determine if test passed (all assertions pass)
            passed = all(a.passed for a in assertion_results)

            return TestResult(
                test=test,
                query=query,
                passed=passed,
                results=results,
                assertion_results=assertion_results,
                latency_ms=latency_ms,
            )

        except Exception as e:
            return TestResult(
                test=test,
                query=query,
                passed=False,
                results=[],
                assertion_results=[],
                latency_ms=0.0,
                error=str(e),
            )

    def run(
        self,
        filter_tags: list[str] | None = None,
        filter_query: str | None = None,
    ) -> RunResult:
        """
        Run all tests in the configuration.

        Args:
            filter_tags: Only run tests with these tags
            filter_query: Only run tests whose query contains this string

        Returns:
            RunResult with all test results
        """
        from maxq.core.ids import generate_run_id

        run_id = generate_run_id()
        started_at = datetime.now()

        tests_to_run = self.config.tests

        # Apply filters
        if filter_tags:
            tag_set = set(filter_tags)
            tests_to_run = [
                t for t in tests_to_run
                if any(tag in tag_set for tag in t.tags)
            ]

        if filter_query:
            tests_to_run = [
                t for t in tests_to_run
                if filter_query.lower() in t.query.lower()
            ]

        # Run tests
        test_results = []
        for test in tests_to_run:
            result = self.run_test(test)
            test_results.append(result)

        completed_at = datetime.now()

        run_result = RunResult(
            run_id=run_id,
            config_path=str(getattr(self.config, '_source_path', 'maxq.yaml')),
            started_at=started_at,
            completed_at=completed_at,
            tests=test_results,
        )

        # Build summary
        run_result.summary = {
            "total_tests": run_result.total_tests,
            "passed_tests": run_result.passed_tests,
            "failed_tests": run_result.failed_tests,
            "total_assertions": run_result.total_assertions,
            "passed_assertions": run_result.passed_assertions,
            "pass_rate": run_result.passed_tests / run_result.total_tests if run_result.total_tests > 0 else 0,
            "duration_ms": (completed_at - started_at).total_seconds() * 1000,
        }

        return run_result


def run_tests(
    config_path: str | Path = "maxq.yaml",
    filter_tags: list[str] | None = None,
    filter_query: str | None = None,
) -> RunResult:
    """
    Convenience function to run tests from a config file.

    Args:
        config_path: Path to maxq.yaml config
        filter_tags: Only run tests with these tags
        filter_query: Only run tests whose query contains this string

    Returns:
        RunResult with all test results
    """
    config = load_config(config_path)
    runner = TestRunner(config)
    return runner.run(filter_tags=filter_tags, filter_query=filter_query)
