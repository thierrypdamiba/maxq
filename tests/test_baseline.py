"""
Tests for baseline, CI, and diff functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from maxq.core.types import Metrics, QueryResult, SearchResult, Run, RunConfig, RunStatus
from maxq.core.baseline import (
    BaselineMetadata,
    QueryDelta,
    DiffResult,
    CIResult,
    get_baselines_dir,
    list_baselines,
    get_baseline,
    save_baseline,
    delete_baseline,
    compute_query_delta,
    diff_runs,
    run_ci_check,
    generate_ci_report,
)


class TestBaselineMetadata:
    """Tests for BaselineMetadata model."""

    def test_create_baseline_metadata(self):
        """Test creating baseline metadata."""
        metrics = Metrics(
            ndcg_at_k={5: 0.8, 10: 0.75, 20: 0.7},
            recall_at_k={5: 0.6, 10: 0.7, 20: 0.8},
            mrr_at_k={5: 0.9, 10: 0.85, 20: 0.8},
            total_queries=100,
            queries_with_hits=95,
        )

        baseline = BaselineMetadata(
            name="main",
            run_id="run_20251229_123456_abc123",
            collection="test_collection",
            created_at=datetime.now(),
            config={"chunk_size": 800},
            metrics=metrics,
            description="Test baseline",
            tags=["production", "v1"],
        )

        assert baseline.name == "main"
        assert baseline.metrics.ndcg_at_k[10] == 0.75
        assert "production" in baseline.tags


class TestQueryDelta:
    """Tests for QueryDelta computation."""

    def test_compute_query_delta_regression(self):
        """Test computing delta for a regressed query."""
        # Baseline query result - good results
        baseline_qr = QueryResult(
            query_id="q1",
            query="test query",
            results=[
                SearchResult(id="1", score=0.9, doc_id="doc1", text="text1"),
                SearchResult(id="2", score=0.8, doc_id="doc2", text="text2"),
                SearchResult(id="3", score=0.7, doc_id="doc3", text="text3"),
            ],
            relevant_doc_ids=["doc1", "doc2"],
            relevant_ids=[],
        )

        # Compare query result - worse results (doc1 dropped)
        compare_qr = QueryResult(
            query_id="q1",
            query="test query",
            results=[
                SearchResult(id="4", score=0.85, doc_id="doc4", text="text4"),
                SearchResult(id="2", score=0.75, doc_id="doc2", text="text2"),
                SearchResult(id="5", score=0.65, doc_id="doc5", text="text5"),
            ],
            relevant_doc_ids=["doc1", "doc2"],
            relevant_ids=[],
        )

        delta = compute_query_delta(baseline_qr, compare_qr, k=10)

        assert delta.query_id == "q1"
        assert delta.ndcg_delta < 0  # Should be negative (regression)
        assert delta.recall_delta < 0  # Lost doc1
        assert "doc1" in delta.results_removed
        assert "doc4" in delta.results_added

    def test_compute_query_delta_improvement(self):
        """Test computing delta for an improved query."""
        # Baseline - only one relevant doc
        baseline_qr = QueryResult(
            query_id="q1",
            query="test query",
            results=[
                SearchResult(id="1", score=0.9, doc_id="doc1", text="text1"),
                SearchResult(id="2", score=0.8, doc_id="doc3", text="text3"),
            ],
            relevant_doc_ids=["doc1", "doc2"],
            relevant_ids=[],
        )

        # Compare - both relevant docs found
        compare_qr = QueryResult(
            query_id="q1",
            query="test query",
            results=[
                SearchResult(id="1", score=0.95, doc_id="doc1", text="text1"),
                SearchResult(id="2", score=0.9, doc_id="doc2", text="text2"),
            ],
            relevant_doc_ids=["doc1", "doc2"],
            relevant_ids=[],
        )

        delta = compute_query_delta(baseline_qr, compare_qr, k=10)

        assert delta.recall_delta > 0  # Found doc2
        assert "doc2" in delta.results_added


class TestBaselineStorage:
    """Tests for baseline storage operations."""

    @pytest.fixture
    def temp_app_dir(self, tmp_path):
        """Create a temporary app directory."""
        with patch("maxq.core.baseline.settings") as mock_settings:
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")
            yield tmp_path

    def test_save_and_get_baseline(self, temp_app_dir):
        """Test saving and retrieving a baseline."""
        metrics = Metrics(
            ndcg_at_k={10: 0.75},
            recall_at_k={10: 0.7},
            mrr_at_k={10: 0.8},
        )

        saved = save_baseline(
            name="test_baseline",
            run_id="run_123",
            collection="test_col",
            metrics=metrics,
            config={"model": "test"},
            description="Test",
        )

        assert saved.name == "test_baseline"

        # Retrieve it
        retrieved = get_baseline("test_baseline")
        assert retrieved is not None
        assert retrieved.name == "test_baseline"
        assert retrieved.metrics.ndcg_at_k[10] == 0.75

    def test_list_baselines(self, temp_app_dir):
        """Test listing all baselines."""
        metrics = Metrics(ndcg_at_k={10: 0.75})

        save_baseline("baseline1", "run_1", "col1", metrics, {})
        save_baseline("baseline2", "run_2", "col2", metrics, {})

        baselines = list_baselines()
        assert len(baselines) == 2
        names = [b.name for b in baselines]
        assert "baseline1" in names
        assert "baseline2" in names

    def test_delete_baseline(self, temp_app_dir):
        """Test deleting a baseline."""
        metrics = Metrics(ndcg_at_k={10: 0.75})
        save_baseline("to_delete", "run_1", "col1", metrics, {})

        assert get_baseline("to_delete") is not None

        result = delete_baseline("to_delete")
        assert result is True
        assert get_baseline("to_delete") is None

    def test_get_nonexistent_baseline(self, temp_app_dir):
        """Test getting a baseline that doesn't exist."""
        result = get_baseline("nonexistent")
        assert result is None


class TestDiffRuns:
    """Tests for run diffing functionality."""

    @pytest.fixture
    def mock_query_results(self):
        """Create mock query results for two runs."""
        baseline_results = [
            QueryResult(
                query_id="q1",
                query="machine learning",
                results=[
                    SearchResult(id="1", score=0.9, doc_id="doc1", text="ML intro"),
                    SearchResult(id="2", score=0.8, doc_id="doc2", text="Deep learning"),
                ],
                relevant_doc_ids=["doc1", "doc2"],
                relevant_ids=[],
            ),
            QueryResult(
                query_id="q2",
                query="vector database",
                results=[
                    SearchResult(id="3", score=0.85, doc_id="doc3", text="Qdrant"),
                ],
                relevant_doc_ids=["doc3", "doc4"],
                relevant_ids=[],
            ),
        ]

        compare_results = [
            QueryResult(
                query_id="q1",
                query="machine learning",
                results=[
                    SearchResult(id="1", score=0.95, doc_id="doc1", text="ML intro"),
                    SearchResult(id="4", score=0.7, doc_id="doc5", text="Other"),  # doc2 dropped
                ],
                relevant_doc_ids=["doc1", "doc2"],
                relevant_ids=[],
            ),
            QueryResult(
                query_id="q2",
                query="vector database",
                results=[
                    SearchResult(id="3", score=0.9, doc_id="doc3", text="Qdrant"),
                    SearchResult(id="5", score=0.85, doc_id="doc4", text="Pinecone"),  # doc4 added
                ],
                relevant_doc_ids=["doc3", "doc4"],
                relevant_ids=[],
            ),
        ]

        return baseline_results, compare_results

    def test_diff_runs_with_mock_data(self, mock_query_results, tmp_path):
        """Test diffing two runs."""
        baseline_results, compare_results = mock_query_results

        with patch("maxq.core.baseline.settings") as mock_settings:
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")

            # Create run directories with query results
            (tmp_path / "runs" / "run_baseline").mkdir(parents=True)
            (tmp_path / "runs" / "run_compare").mkdir(parents=True)

            with open(tmp_path / "runs" / "run_baseline" / "query_results.jsonl", "w") as f:
                for qr in baseline_results:
                    f.write(qr.model_dump_json() + "\n")

            with open(tmp_path / "runs" / "run_compare" / "query_results.jsonl", "w") as f:
                for qr in compare_results:
                    f.write(qr.model_dump_json() + "\n")

            diff = diff_runs("run_baseline", "run_compare")

            assert diff.total_queries == 2
            assert diff.regressions >= 0
            assert diff.improvements >= 0
            assert len(diff.query_deltas) == 2


class TestCICheck:
    """Tests for CI check functionality."""

    def test_ci_check_passes_within_threshold(self, tmp_path):
        """Test CI check passes when within threshold."""
        with (
            patch("maxq.core.baseline.settings") as mock_settings,
            patch("maxq.core.runs.settings") as mock_runs_settings,
        ):
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")
            mock_runs_settings.runs_dir = str(tmp_path / "runs")

            # Create baseline
            baseline_metrics = Metrics(
                ndcg_at_k={5: 0.8, 10: 0.75, 20: 0.7},
                recall_at_k={5: 0.6, 10: 0.7, 20: 0.8},
                mrr_at_k={5: 0.9, 10: 0.85, 20: 0.8},
                total_queries=10,
                queries_with_hits=9,
            )

            save_baseline(
                name="main",
                run_id="run_baseline",
                collection="test_col",
                metrics=baseline_metrics,
                config={"model": "test"},
            )

            # Create compare run
            compare_metrics = Metrics(
                ndcg_at_k={5: 0.78, 10: 0.73, 20: 0.68},
                recall_at_k={5: 0.58, 10: 0.68, 20: 0.78},
                mrr_at_k={5: 0.88, 10: 0.83, 20: 0.78},
                total_queries=10,
                queries_with_hits=8,
            )

            run_dir = tmp_path / "runs" / "run_compare"
            run_dir.mkdir(parents=True)

            run = Run(
                run_id="run_compare",
                status=RunStatus.DONE,
                config=RunConfig(
                    collection="test_col",
                    dataset_path="test/dataset",
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics=compare_metrics,
            )

            with open(run_dir / "run.json", "w") as f:
                json.dump(run.model_dump(mode="json"), f, default=str)

            # Create empty query results files
            (tmp_path / "baselines").mkdir(exist_ok=True)
            with open(tmp_path / "baselines" / "main_queries.jsonl", "w") as f:
                pass
            with open(run_dir / "query_results.jsonl", "w") as f:
                pass

            result = run_ci_check(
                run_id="run_compare",
                baseline_name="main",
                max_ndcg_drop=-0.05,  # Allow 5% drop
                max_recall_drop=-0.05,
            )

            assert result.passed is True
            assert result.baseline_name == "main"
            assert len(result.failed_checks) == 0

    def test_ci_check_fails_on_regression(self, tmp_path):
        """Test CI check fails when regression exceeds threshold."""
        with (
            patch("maxq.core.baseline.settings") as mock_settings,
            patch("maxq.core.runs.settings") as mock_runs_settings,
        ):
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")
            mock_runs_settings.runs_dir = str(tmp_path / "runs")

            baseline_metrics = Metrics(
                ndcg_at_k={10: 0.75},
                recall_at_k={10: 0.7},
            )

            save_baseline("main", "run_baseline", "test_col", baseline_metrics, {})

            compare_metrics = Metrics(
                ndcg_at_k={10: 0.70},  # 5% drop
                recall_at_k={10: 0.65},
            )

            run_dir = tmp_path / "runs" / "run_compare"
            run_dir.mkdir(parents=True)

            run = Run(
                run_id="run_compare",
                status=RunStatus.DONE,
                config=RunConfig(collection="test_col", dataset_path="test/dataset"),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics=compare_metrics,
            )

            with open(run_dir / "run.json", "w") as f:
                json.dump(run.model_dump(mode="json"), f, default=str)

            (tmp_path / "baselines").mkdir(exist_ok=True)
            with open(tmp_path / "baselines" / "main_queries.jsonl", "w") as f:
                pass
            with open(run_dir / "query_results.jsonl", "w") as f:
                pass

            result = run_ci_check(
                run_id="run_compare",
                baseline_name="main",
                max_ndcg_drop=-0.01,  # Only allow 1% drop (will fail)
            )

            assert result.passed is False
            assert len(result.failed_checks) > 0
            assert any(c["check"] == "max_ndcg_drop" for c in result.failed_checks)

    def test_ci_check_with_min_threshold(self, tmp_path):
        """Test CI check with minimum absolute threshold."""
        with (
            patch("maxq.core.baseline.settings") as mock_settings,
            patch("maxq.core.runs.settings") as mock_runs_settings,
        ):
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")
            mock_runs_settings.runs_dir = str(tmp_path / "runs")

            baseline_metrics = Metrics(ndcg_at_k={10: 0.75})
            save_baseline("main", "run_baseline", "test_col", baseline_metrics, {})

            compare_metrics = Metrics(ndcg_at_k={10: 0.73})

            run_dir = tmp_path / "runs" / "run_compare"
            run_dir.mkdir(parents=True)

            run = Run(
                run_id="run_compare",
                status=RunStatus.DONE,
                config=RunConfig(collection="test_col", dataset_path="test/dataset"),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics=compare_metrics,
            )

            with open(run_dir / "run.json", "w") as f:
                json.dump(run.model_dump(mode="json"), f, default=str)

            (tmp_path / "baselines").mkdir(exist_ok=True)
            with open(tmp_path / "baselines" / "main_queries.jsonl", "w") as f:
                pass
            with open(run_dir / "query_results.jsonl", "w") as f:
                pass

            result = run_ci_check(
                run_id="run_compare",
                baseline_name="main",
                min_ndcg_10=0.80,  # Require 80% (will fail since we have 73%)
            )

            assert result.passed is False
            assert any(c["check"] == "min_ndcg_10" for c in result.failed_checks)

    def test_ci_check_baseline_not_found(self, tmp_path):
        """Test CI check when baseline doesn't exist."""
        with patch("maxq.core.baseline.settings") as mock_settings:
            mock_settings.app_dir = tmp_path
            mock_settings.runs_dir = str(tmp_path / "runs")

            result = run_ci_check(
                run_id="run_compare",
                baseline_name="nonexistent",
            )

            assert result.passed is False
            assert any("not found" in str(c) for c in result.failed_checks)


class TestCIReport:
    """Tests for CI report generation."""

    def test_generate_ci_report(self):
        """Test generating a markdown CI report."""
        baseline = BaselineMetadata(
            name="main",
            run_id="run_baseline",
            collection="test_col",
            created_at=datetime.now(),
            config={},
            metrics=Metrics(
                ndcg_at_k={5: 0.8, 10: 0.75, 20: 0.7},
                recall_at_k={5: 0.6, 10: 0.7, 20: 0.8},
            ),
        )

        run = MagicMock()
        run.metrics = Metrics(
            ndcg_at_k={5: 0.78, 10: 0.73, 20: 0.68},
            recall_at_k={5: 0.58, 10: 0.68, 20: 0.78},
        )

        diff = DiffResult(
            baseline_run_id="run_baseline",
            compare_run_id="run_compare",
            ndcg_10_delta=-0.02,
            recall_10_delta=-0.02,
            total_queries=10,
            regressions=2,
            improvements=1,
            unchanged=7,
        )

        checks = [
            {"check": "max_ndcg_drop", "threshold": -0.05, "actual": -0.02, "passed": True},
        ]

        report = generate_ci_report(
            baseline_name="main",
            baseline=baseline,
            run_id="run_compare",
            run=run,
            diff=diff,
            checks=checks,
            failed_checks=[],
            passed=True,
        )

        assert "MaxQ CI Report" in report
        assert "PASSED" in report
        assert "NDCG@10" in report
        assert "main" in report


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_baseline_list_command(self):
        """Test baseline list command."""
        from typer.testing import CliRunner
        from maxq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["baseline", "list"])

        # Should not crash, may show "No baselines" or list
        assert result.exit_code == 0

    def test_ci_command_missing_baseline(self):
        """Test ci command with missing baseline."""
        from typer.testing import CliRunner
        from maxq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ci", "fake_run", "--against", "nonexistent_baseline"])

        # Should fail gracefully
        assert result.exit_code == 1 or "not found" in result.output.lower()

    def test_diff_command_help(self):
        """Test diff command shows help."""
        from typer.testing import CliRunner
        from maxq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["diff", "--help"])

        assert result.exit_code == 0
        assert "Compare two runs" in result.output

    def test_pick_command_help(self):
        """Test pick command shows help."""
        from typer.testing import CliRunner
        from maxq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["pick", "--help"])

        assert result.exit_code == 0
        assert "Pick the best run" in result.output

    def test_eval_list_command(self):
        """Test eval list command."""
        from typer.testing import CliRunner
        from maxq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "list"])

        # Should not crash
        assert result.exit_code == 0
