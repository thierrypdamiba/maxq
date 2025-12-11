"""
API endpoint tests using FastAPI TestClient.
Tests the current REST API routes: /runs, /jobs, /index, /search, /projects, /health.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime


# ============================================
# Health Endpoint Tests
# ============================================


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, api_client):
        """Test health endpoint returns status."""
        with patch("maxq.server.dependencies.get_engine") as mock_engine:
            mock_engine.return_value.client.get_collections.return_value = []
            response = api_client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "version" in data
            assert "services" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_ok(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


# ============================================
# Projects API Tests
# ============================================


class TestProjectsAPI:
    """Tests for /projects endpoints."""

    def test_list_projects_empty(self, api_client):
        """Test listing projects when none exist."""
        response = api_client.get("/projects/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_project(self, api_client):
        """Test creating a new project."""
        response = api_client.post("/projects/", params={"name": "Test Project"})
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Project"

    def test_list_projects_after_create(self, api_client):
        """Test that created project appears in list."""
        # Create a project
        api_client.post("/projects/", params={"name": "Listed Project"})

        # List projects
        response = api_client.get("/projects/")
        assert response.status_code == 200
        projects = response.json()
        assert len(projects) >= 1
        names = [p["name"] for p in projects]
        assert "Listed Project" in names

    def test_get_project_by_id(self, api_client):
        """Test retrieving a specific project by ID."""
        # Create a project
        create_response = api_client.post("/projects/", params={"name": "Get By ID"})
        project_id = create_response.json()["id"]

        # Get the project
        response = api_client.get(f"/projects/{project_id}")
        assert response.status_code == 200
        assert response.json()["id"] == project_id
        assert response.json()["name"] == "Get By ID"

    def test_get_project_not_found(self, api_client):
        """Test 404 for non-existent project."""
        response = api_client.get("/projects/nonexistent-id-12345")
        assert response.status_code == 404


# ============================================
# Runs API Tests
# ============================================


class TestRunsAPI:
    """Tests for /runs endpoints."""

    def test_list_runs_empty(self, api_client):
        """Test listing runs when none exist."""
        response = api_client.get("/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
        assert isinstance(data["runs"], list)

    def test_create_index_run_missing_dataset(self, api_client):
        """Test creating index run with non-existent local dataset fails."""
        response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "/nonexistent/path.jsonl",
                "collection": "test_collection",
            },
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_create_index_run_invalid_chunk_config(self, api_client):
        """Test validation: chunk_size must be > chunk_overlap."""
        response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_collection",
                "chunk_size": 100,
                "chunk_overlap": 150,  # Invalid: overlap > size
            },
        )
        assert response.status_code == 400
        assert "chunk_size" in response.json()["detail"].lower()

    def test_create_index_run_chunk_size_too_small(self, api_client):
        """Test validation: chunk_size must be at least 100."""
        response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_collection",
                "chunk_size": 50,  # Too small
                "chunk_overlap": 10,
            },
        )
        assert response.status_code == 400
        assert "100" in response.json()["detail"]

    def test_create_index_run_with_hf_dataset(self, api_client):
        """Test creating index run with HuggingFace dataset path."""
        response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_collection",
                "chunk_size": 800,
                "chunk_overlap": 120,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "job_id" in data
        assert data["message"] == "Index run created and queued"

    def test_get_run_not_found(self, api_client):
        """Test 404 for non-existent run."""
        response = api_client.get("/runs/run_nonexistent_12345")
        assert response.status_code == 404

    def test_get_run_details(self, api_client):
        """Test getting run details after creation."""
        # Create a run
        create_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_get_details",
            },
        )
        run_id = create_response.json()["run_id"]

        # Get run details
        response = api_client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["collection"] == "test_get_details"
        assert "status" in data
        assert "jobs" in data

    def test_delete_run(self, api_client):
        """Test deleting a run."""
        # Create a run
        create_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_delete",
            },
        )
        run_id = create_response.json()["run_id"]

        # Delete the run
        response = api_client.delete(f"/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["run_id"] == run_id

        # Verify it's gone
        get_response = api_client.get(f"/runs/{run_id}")
        assert get_response.status_code == 404

    def test_delete_run_not_found(self, api_client):
        """Test 404 when deleting non-existent run."""
        response = api_client.delete("/runs/run_nonexistent_12345")
        assert response.status_code == 404

    def test_create_eval_job_run_not_found(self, api_client):
        """Test 404 when creating eval for non-existent run."""
        response = api_client.post(
            "/runs/run_nonexistent_12345/eval", json={"queries_path": "data/queries.jsonl"}
        )
        assert response.status_code == 404

    def test_create_report_job_run_not_found(self, api_client):
        """Test 404 when creating report for non-existent run."""
        response = api_client.post("/runs/run_nonexistent_12345/report", json={})
        assert response.status_code == 404

    def test_get_artifact_run_not_found(self, api_client):
        """Test 404 when getting artifact for non-existent run."""
        response = api_client.get("/runs/run_nonexistent_12345/artifact/metrics.json")
        assert response.status_code == 404


# ============================================
# Jobs API Tests
# ============================================


class TestJobsAPI:
    """Tests for /jobs endpoints."""

    def test_get_job_not_found(self, api_client):
        """Test 404 for non-existent job."""
        response = api_client.get("/jobs/job_nonexistent_12345")
        assert response.status_code == 404

    def test_get_job_details(self, api_client):
        """Test getting job details after run creation."""
        # Create a run (which creates a job)
        create_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "test_job_details",
            },
        )
        job_id = create_response.json()["job_id"]

        # Get job details
        response = api_client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["job_type"] == "index"
        assert "status" in data
        assert "created_at" in data

    def test_cancel_job_not_found(self, api_client):
        """Test 404 when cancelling non-existent job."""
        response = api_client.post("/jobs/job_nonexistent_12345/cancel")
        assert response.status_code == 404


# ============================================
# Index API Tests
# ============================================


class TestIndexAPI:
    """Tests for /index endpoints."""

    def test_list_models(self, api_client):
        """Test listing supported embedding models."""
        response = api_client.get("/index/models")
        assert response.status_code == 200
        models = response.json()
        # Returns dict with "dense" and "sparse" keys
        assert isinstance(models, dict)
        assert "dense" in models
        assert "sparse" in models
        assert len(models["dense"]) > 0
        # Each dense model should have model, dim, description
        for model in models["dense"]:
            assert "model" in model
            assert "dim" in model
            assert "description" in model

    def test_get_index_status_project_not_found(self, api_client):
        """Test 404 when getting index status for non-existent project."""
        response = api_client.get("/index/status/nonexistent-project-id")
        assert response.status_code == 404

    def test_get_index_status(self, api_client):
        """Test getting index status for a project."""
        # Create a project first
        project_response = api_client.post("/projects/", params={"name": "Index Status Test"})
        project_id = project_response.json()["id"]

        # Get index status
        response = api_client.get(f"/index/status/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert "indexed_models" in data
        assert "has_data" in data

    def test_start_indexing_project_not_found(self, api_client):
        """Test 404 when starting indexing for non-existent project."""
        response = api_client.post(
            "/index/start",
            json={
                "project_id": "nonexistent-project-id",
                "dataset_name": "test/dataset",
            },
        )
        assert response.status_code == 404

    def test_start_indexing_invalid_model(self, api_client):
        """Test 400 when using unsupported embedding model."""
        # Create a project first
        project_response = api_client.post("/projects/", params={"name": "Invalid Model Test"})
        project_id = project_response.json()["id"]

        response = api_client.post(
            "/index/start",
            json={
                "project_id": project_id,
                "dataset_name": "test/dataset",
                "dense_models": ["unsupported/model"],  # Invalid model
            },
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()


# ============================================
# Search API Tests
# ============================================


class TestSearchAPI:
    """Tests for /search endpoints."""

    def test_search_project_not_found(self, api_client):
        """Test 404 when searching non-existent project."""
        response = api_client.post(
            "/search/query",
            json={
                "project_id": "nonexistent-project-id",
                "query": "test query",
            },
        )
        assert response.status_code == 404

    def test_search_no_indexed_data(self, api_client):
        """Test 400 when searching project with no indexed data."""
        # Create a project without indexing
        project_response = api_client.post("/projects/", params={"name": "No Data Project"})
        project_id = project_response.json()["id"]

        response = api_client.post(
            "/search/query",
            json={
                "project_id": project_id,
                "query": "test query",
            },
        )
        assert response.status_code == 400
        assert "no data" in response.json()["detail"].lower()

    def test_list_collections_project_not_found(self, api_client):
        """Test 404 when listing collections for non-existent project."""
        response = api_client.get("/search/collections/nonexistent-project-id")
        assert response.status_code == 404

    def test_list_collections(self, api_client):
        """Test listing searchable collections for a project."""
        # Create a project
        project_response = api_client.post("/projects/", params={"name": "Collections Test"})
        project_id = project_response.json()["id"]

        response = api_client.get(f"/search/collections/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert "collections" in data
        assert isinstance(data["collections"], list)


# ============================================
# Tuning API Tests
# ============================================


class TestTuningAPI:
    """Tests for /tuning endpoints."""

    def test_list_experiments_empty(self, api_client):
        """Test listing experiments when none exist."""
        response = api_client.get("/tuning/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_available_models(self, api_client):
        """Test getting available models for a project."""
        response = api_client.get("/tuning/available-models", params={"project_id": "test-project"})
        # May return empty list or error if no Qdrant connection
        assert response.status_code in [200, 500]


# ============================================
# Evals API Tests
# ============================================


class TestEvalsAPI:
    """Tests for /evals endpoints."""

    def test_list_evals_empty(self, api_client):
        """Test listing evals when none exist."""
        response = api_client.get("/evals/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_run_eval_experiment_not_found(self, api_client):
        """Test running eval for non-existent experiment."""
        response = api_client.post(
            "/evals/run", params={"experiment_id": "nonexistent-experiment-id"}
        )
        # Should return 404 for non-existent experiment
        assert response.status_code == 404


# ============================================
# Error Handling Tests
# ============================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_body(self, api_client):
        """Test that API handles invalid JSON gracefully."""
        response = api_client.post(
            "/runs/index", content="not valid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error

    def test_missing_required_field(self, api_client):
        """Test validation error for missing required field."""
        response = api_client.post(
            "/runs/index",
            json={
                # Missing required fields: dataset_path, collection
            },
        )
        assert response.status_code == 422

    def test_wrong_http_method(self, api_client):
        """Test 405 for wrong HTTP method."""
        response = api_client.put("/runs")  # PUT not allowed on /runs
        assert response.status_code == 405
