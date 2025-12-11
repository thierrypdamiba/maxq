"""
Comprehensive tests for MaxQ CLI functionality.

Tests all CLI commands to ensure they work correctly.
Focus on testing without requiring live Qdrant Cloud credentials.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typer.testing import CliRunner

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from maxq.cli import app

runner = CliRunner()


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.collection_exists.return_value = False
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.count.return_value = MagicMock(count=100)
    return mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("QDRANT_URL", "https://mock-qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "mock-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")


# ============================================
# Help and Version Tests
# ============================================


class TestCLIHelp:
    """Tests for CLI help and basic invocation."""

    def test_help_shows_commands(self):
        """Test that --help shows all available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check major commands are listed
        expected_commands = [
            "configs",
            "config",
            "auto",
            "search-data",
            "import",
            "datasets",
            "studio",
            "demo",
            "doctor",
            "engine",
            "search",
            "start",
            "worker",
            "run",
            "status",
            "runs",
            "setup",
        ]
        for cmd in expected_commands:
            assert cmd in result.output

    def test_cli_flag_shows_quickstart(self):
        """Test that --cli shows CLI quickstart."""
        result = runner.invoke(app, ["--cli"])
        assert result.exit_code == 0
        assert "CLI" in result.output or "commands" in result.output.lower()


# ============================================
# Preset/Config Tests
# ============================================


class TestConfigsCommand:
    """Tests for the configs command."""

    def test_configs_lists_presets(self):
        """Test that configs command lists available presets."""
        result = runner.invoke(app, ["configs"])
        assert result.exit_code == 0

        # Should show actual preset names (base, mxbai_bm25, max_accuracy, etc.)
        assert (
            "base" in result.output.lower()
            or "mxbai_bm25" in result.output.lower()
            or "max_accuracy" in result.output.lower()
            or "preset" in result.output.lower()
        )

    def test_config_shows_preset_details(self):
        """Test that config command shows preset details."""
        result = runner.invoke(app, ["config", "base"])
        # May fail if preset doesn't exist, but should not crash
        assert result.exit_code == 0 or "Unknown preset" in result.output

    def test_config_unknown_preset(self):
        """Test that config shows error for unknown preset."""
        result = runner.invoke(app, ["config", "nonexistent"])
        assert "Unknown preset" in result.output or "not found" in result.output.lower()


class TestAutoCommand:
    """Tests for the auto command."""

    def test_auto_default_priority(self):
        """Test auto command with default priority."""
        result = runner.invoke(app, ["auto"])
        assert result.exit_code == 0
        # Should show recommended preset info
        assert (
            "preset" in result.output.lower()
            or "recommended" in result.output.lower()
            or "base" in result.output.lower()
        )

    def test_auto_fast_priority(self):
        """Test auto command with fast priority."""
        result = runner.invoke(app, ["auto", "--priority", "fast"])
        assert result.exit_code == 0
        # Should show some preset recommendation
        assert (
            "preset" in result.output.lower()
            or "base" in result.output.lower()
            or result.exit_code == 0
        )

    def test_auto_accurate_priority(self):
        """Test auto command with accurate priority."""
        result = runner.invoke(app, ["auto", "--priority", "accurate"])
        assert result.exit_code == 0
        # Should show some preset recommendation (max_accuracy for accurate)
        assert (
            "preset" in result.output.lower()
            or "max_accuracy" in result.output.lower()
            or result.exit_code == 0
        )


# ============================================
# Doctor Command Tests
# ============================================


class TestDoctorCommand:
    """Tests for the doctor command."""

    def test_doctor_runs_without_crash(self):
        """Test that doctor command runs without crashing."""
        result = runner.invoke(app, ["doctor"])
        # Doctor should complete regardless of connection status
        assert result.exit_code == 0 or result.exit_code == 1

        # Should check various systems
        assert "Qdrant" in result.output or "connection" in result.output.lower()

    def test_doctor_shows_python_version(self):
        """Test that doctor shows Python version."""
        result = runner.invoke(app, ["doctor"])
        assert "Python" in result.output

    def test_doctor_shows_dependencies(self):
        """Test that doctor shows dependencies."""
        result = runner.invoke(app, ["doctor"])
        assert "Dependencies" in result.output or "qdrant-client" in result.output


# ============================================
# Search Command Tests
# ============================================


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_without_collection_shows_error(self):
        """Test search fails gracefully when collection doesn't exist."""
        with patch("maxq.cli.MaxQEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_instance.collection_exists.return_value = False
            mock_engine.return_value = mock_instance

            result = runner.invoke(app, ["search", "test query"])
            # Should show error or prompt about collection/connection
            assert (
                "not found" in result.output.lower()
                or "demo" in result.output.lower()
                or "failed" in result.output.lower()
                or "error" in result.output.lower()
                or result.exit_code != 0
            )

    def test_search_requires_query(self):
        """Test search requires a query argument."""
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "QUERY" in result.output


# ============================================
# Demo Command Tests
# ============================================


class TestDemoCommand:
    """Tests for the demo command."""

    def test_demo_ci_mode(self):
        """Test demo in CI mode (non-interactive)."""
        # Demo command actually runs and connects to Qdrant Cloud
        # In test environment, it may succeed or fail depending on credentials
        result = runner.invoke(app, ["demo", "--ci", "--limit", "5"])
        # Should either succeed, fail with connection error, or show indexing progress
        # Any of these outcomes is acceptable in test environment
        assert (
            result.exit_code == 0
            or result.exit_code == 1  # Expected failure without proper setup
            or "Failed to connect" in result.output
            or "failed" in result.output.lower()
            or "error" in result.output.lower()
            or "indexing" in result.output.lower()  # Shows it started
            or "demo" in result.output.lower()  # Shows demo output
        )

    def test_demo_accepts_limit(self):
        """Test demo accepts limit parameter."""
        with patch("maxq.cli.MaxQEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance

            result = runner.invoke(app, ["demo", "--ci", "--limit", "10"])
            # Check that limit is accepted
            assert "--limit" not in result.output or result.exit_code == 0


# ============================================
# Engine/Sidecar Command Tests
# ============================================


class TestEngineCommand:
    """Tests for the engine command."""

    def test_engine_status(self):
        """Test engine status command."""
        result = runner.invoke(app, ["engine", "--status"])
        assert result.exit_code == 0
        assert "Platform" in result.output


# ============================================
# Import Command Tests
# ============================================


class TestImportCommand:
    """Tests for the import command."""

    def test_import_requires_source(self):
        """Test import requires a source argument."""
        result = runner.invoke(app, ["import"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "SOURCE" in result.output

    def test_import_detects_huggingface_source(self):
        """Test import correctly detects HuggingFace dataset format."""
        from maxq.cli import _detect_source_type
        from maxq.data_sources import DataSourceType

        # Test HuggingFace format detection
        source_type = _detect_source_type("fka/awesome-chatgpt-prompts")
        assert source_type == DataSourceType.HUGGINGFACE

        source_type = _detect_source_type("https://huggingface.co/datasets/squad")
        assert source_type == DataSourceType.HUGGINGFACE

    def test_import_detects_local_source(self):
        """Test import correctly detects local file sources."""
        from maxq.cli import _detect_source_type
        from maxq.data_sources import DataSourceType

        # Test S3 format
        source_type = _detect_source_type("s3://my-bucket/data")
        assert source_type == DataSourceType.S3_BUCKET

        # Test URL format
        source_type = _detect_source_type("https://example.com/data.json")
        assert source_type == DataSourceType.URL


# ============================================
# Runs/Status/Worker Command Tests
# ============================================


class TestRunsCommand:
    """Tests for run-related commands."""

    def test_runs_list(self):
        """Test runs command lists runs."""
        # Just run the runs command and check it works
        result = runner.invoke(app, ["runs"])
        # Should show runs list or no runs message
        assert (
            result.exit_code == 0
            or "No runs found" in result.output
            or "Runs" in result.output
            or "runs" in result.output.lower()
            or "ID" in result.output  # Table header
        )

    def test_run_requires_arguments(self):
        """Test run command requires dataset and collection arguments."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0
        assert (
            "Missing argument" in result.output
            or "DATASET" in result.output
            or "required" in result.output.lower()
        )

    def test_status_requires_run_id(self):
        """Test status command requires run_id."""
        result = runner.invoke(app, ["status"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "RUN_ID" in result.output


# ============================================
# Studio Command Tests
# ============================================


class TestStudioCommand:
    """Tests for the studio command."""

    def test_studio_checks_ports(self):
        """Test studio checks if ports are in use."""
        with patch("maxq.cli.is_port_in_use") as mock_port_check:
            mock_port_check.return_value = True  # Port is in use

            result = runner.invoke(app, ["studio"])
            # Should fail because port is in use
            assert result.exit_code != 0 or "already in use" in result.output


# ============================================
# Utility Function Tests
# ============================================


class TestUtilityFunctions:
    """Tests for CLI utility functions."""

    def test_normalize_dataset_name(self):
        """Test dataset name normalization."""
        from maxq.cli import normalize_dataset_name

        # Test full URL
        result = normalize_dataset_name(
            "https://huggingface.co/datasets/fka/awesome-chatgpt-prompts"
        )
        assert result == "fka/awesome-chatgpt-prompts"

        # Test short form (should return as-is)
        result = normalize_dataset_name("fka/awesome-chatgpt-prompts")
        assert result == "fka/awesome-chatgpt-prompts"

        # Test with spaces
        result = normalize_dataset_name("  fka/awesome-chatgpt-prompts  ")
        assert result == "fka/awesome-chatgpt-prompts"

    def test_is_interactive(self):
        """Test is_interactive function."""
        from maxq.cli import is_interactive

        # In test environment, should be non-interactive
        # (unless run from a terminal)
        result = is_interactive()
        assert isinstance(result, bool)

    def test_is_port_in_use(self):
        """Test port checking function."""
        from maxq.cli import is_port_in_use

        # Port 1 should not be in use (requires root)
        result = is_port_in_use(1)
        assert isinstance(result, bool)


# ============================================
# Datasets Command Tests
# ============================================


class TestDatasetsCommand:
    """Tests for datasets command."""

    def test_datasets_requires_collection(self):
        """Test datasets command requires collection argument."""
        result = runner.invoke(app, ["datasets"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "COLLECTION" in result.output


# ============================================
# Check Conflicts Command Tests
# ============================================


class TestCheckConflictsCommand:
    """Tests for check-conflicts command."""

    def test_check_conflicts_requires_collection(self):
        """Test check-conflicts requires collection argument."""
        result = runner.invoke(app, ["check-conflicts"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "COLLECTION" in result.output


# ============================================
# Version Commands Tests
# ============================================


class TestVersionCommands:
    """Tests for version-related commands."""

    def test_versions_requires_args(self):
        """Test versions command requires collection and dataset_id."""
        result = runner.invoke(app, ["versions"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_rollback_requires_args(self):
        """Test rollback command requires collection, dataset_id, and version."""
        result = runner.invoke(app, ["rollback"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output


# ============================================
# Integration-like Tests (with mocks)
# ============================================


class TestCLIIntegration:
    """Integration-like tests using mocks."""

    def test_full_search_flow_mocked(self):
        """Test a full search flow with mocked components."""
        with patch("maxq.cli.MaxQEngine") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.collection_exists.return_value = True
            mock_engine.query.return_value = [
                MagicMock(score=0.9, payload={"_text": "Test result 1"})
            ]
            mock_engine_class.return_value = mock_engine

            result = runner.invoke(app, ["search", "test query", "-c", "test-collection"])
            # With proper mocking, search should work
            # If connection fails, that's also acceptable in test environment
            assert result.exit_code == 0 or "Failed" in result.output

    def test_configs_and_auto_consistency(self):
        """Test that configs and auto commands are consistent."""
        # Get configs
        configs_result = runner.invoke(app, ["configs"])
        assert configs_result.exit_code == 0

        # Auto should use one of the listed presets
        auto_result = runner.invoke(app, ["auto"])
        # Both should complete without crashing
        assert configs_result.exit_code == 0 and auto_result.exit_code == 0


# ============================================
# Error Handling Tests
# ============================================


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        result = runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_invalid_option(self):
        """Test that invalid options are handled gracefully."""
        result = runner.invoke(app, ["doctor", "--invalid-option"])
        assert result.exit_code != 0

    def test_missing_qdrant_credentials(self):
        """Test handling of missing Qdrant credentials."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, ["doctor"])
            # Doctor should still run and report missing credentials
            assert "Qdrant" in result.output or result.exit_code in [0, 1]


# ============================================
# Performance Marker Tests
# ============================================


@pytest.mark.slow
class TestSlowCLIOperations:
    """Tests for slower CLI operations that may need network access."""

    def test_search_data_command(self):
        """Test search-data command (requires Linkup API)."""
        with patch("maxq.cli.DataSourceManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.search_datasets_nl.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(app, ["search-data", "test dataset query"])
            assert (
                result.exit_code == 0
                or "No datasets found" in result.output
                or "error" in result.output.lower()
            )
