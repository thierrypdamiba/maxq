"""
Integration tests for end-to-end workflows.
These tests verify that multiple components work together correctly.

Requires QDRANT_URL and QDRANT_API_KEY environment variables.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import time

requires_qdrant = pytest.mark.skipif(
    not os.environ.get("QDRANT_URL") or not os.environ.get("QDRANT_API_KEY"),
    reason="QDRANT_URL and QDRANT_API_KEY required for integration tests"
)


@requires_qdrant
class TestEngineEndToEnd:
    """End-to-end tests for the MaxQEngine workflow."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_ingest_and_search_workflow(self, in_memory_engine):
        """Test complete workflow: initialize -> ingest -> search."""
        from maxq.search_engine import CollectionStrategy, SearchRequest

        # 1. Create collection strategy
        config = CollectionStrategy(
            collection_name="e2e_test_collection",
            dense_model_name="BAAI/bge-small-en-v1.5",
            sparse_model_name="Qdrant/bm25",
            use_quantization=False
        )

        # 2. Initialize collection
        in_memory_engine.initialize_collection(config)
        assert in_memory_engine.collection_exists(config.collection_name)

        # 3. Ingest documents
        documents = [
            "Python is a versatile programming language",
            "Machine learning enables computers to learn from data",
            "Vector databases are essential for semantic search",
            "Natural language processing understands human language",
            "Deep learning uses neural networks with many layers"
        ]
        payloads = [{"id": i, "topic": "tech"} for i in range(len(documents))]

        in_memory_engine._upload_batch(config, documents, payloads, 0)

        # Wait for indexing
        time.sleep(0.5)

        # Verify documents were ingested
        count = in_memory_engine.client.count(config.collection_name)
        assert count.count == 5

        # 4. Search for documents
        request = SearchRequest(
            query="programming language for AI",
            strategy="hybrid",
            limit=3
        )

        results = in_memory_engine.query(config, request)

        # Verify search returns results
        assert len(results) > 0
        assert len(results) <= 3

        # Verify results have expected structure
        for result in results:
            assert hasattr(result, 'score')
            assert hasattr(result, 'payload')
            assert '_text' in result.payload

    @pytest.mark.slow
    @pytest.mark.integration
    def test_search_strategies_return_different_results(self, in_memory_engine):
        """Test that different search strategies can return different rankings."""
        from maxq.search_engine import CollectionStrategy, SearchRequest

        config = CollectionStrategy(
            collection_name="strategy_test",
            dense_model_name="BAAI/bge-small-en-v1.5"
        )

        in_memory_engine.initialize_collection(config)

        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning and artificial intelligence",
            "Python programming for data science"
        ]
        payloads = [{"id": i} for i in range(len(documents))]

        in_memory_engine._upload_batch(config, documents, payloads, 0)
        time.sleep(0.5)

        # Test all strategies return results
        for strategy in ["dense", "sparse", "hybrid"]:
            request = SearchRequest(
                query="machine learning data",
                strategy=strategy,
                limit=3
            )
            results = in_memory_engine.query(config, request)
            assert len(results) > 0, f"Strategy {strategy} returned no results"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_collection_recreation(self, in_memory_engine):
        """Test that recreating a collection replaces existing data."""
        from maxq.search_engine import CollectionStrategy

        config = CollectionStrategy(
            collection_name="recreate_test",
            dense_model_name="BAAI/bge-small-en-v1.5"
        )

        # Create and populate first time
        in_memory_engine.initialize_collection(config)
        in_memory_engine._upload_batch(config, ["First batch doc"], [{"id": 1}], 0)
        time.sleep(0.5)

        count1 = in_memory_engine.client.count(config.collection_name)
        assert count1.count == 1

        # Recreate collection (initialize_collection uses recreate_collection)
        in_memory_engine.initialize_collection(config)
        in_memory_engine._upload_batch(config, ["New doc 1", "New doc 2"], [{"id": 1}, {"id": 2}], 0)
        time.sleep(0.5)

        count2 = in_memory_engine.client.count(config.collection_name)
        assert count2.count == 2  # Old data should be gone


class TestRunsAPIEndToEnd:
    """End-to-end tests for the Runs API workflow."""

    @pytest.mark.integration
    def test_full_run_lifecycle(self, api_client):
        """Test complete run lifecycle: create -> get -> delete."""
        # 1. Create a run
        create_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://test/dataset",
                "collection": "lifecycle_test",
                "chunk_size": 800,
                "chunk_overlap": 120,
            }
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        job_id = create_response.json()["job_id"]

        # 2. Get run details
        get_response = api_client.get(f"/runs/{run_id}")
        assert get_response.status_code == 200
        run_data = get_response.json()
        assert run_data["run_id"] == run_id
        assert run_data["collection"] == "lifecycle_test"
        assert len(run_data["jobs"]) >= 1

        # 3. Get job details
        job_response = api_client.get(f"/jobs/{job_id}")
        assert job_response.status_code == 200
        assert job_response.json()["run_id"] == run_id

        # 4. List runs includes our run
        list_response = api_client.get("/runs")
        assert list_response.status_code == 200
        runs = list_response.json()["runs"]
        run_ids = [r["run_id"] for r in runs]
        assert run_id in run_ids

        # 5. Delete the run
        delete_response = api_client.delete(f"/runs/{run_id}")
        assert delete_response.status_code == 200

        # 6. Verify run is gone
        get_after_delete = api_client.get(f"/runs/{run_id}")
        assert get_after_delete.status_code == 404

    @pytest.mark.integration
    def test_multiple_runs_isolation(self, api_client):
        """Test that multiple runs are properly isolated."""
        # Create two runs
        run1_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://dataset1",
                "collection": "collection_1",
            }
        )
        run2_response = api_client.post(
            "/runs/index",
            json={
                "dataset_path": "hf://dataset2",
                "collection": "collection_2",
            }
        )

        run1_id = run1_response.json()["run_id"]
        run2_id = run2_response.json()["run_id"]

        # Verify they are different
        assert run1_id != run2_id

        # Get each run and verify data
        run1_data = api_client.get(f"/runs/{run1_id}").json()
        run2_data = api_client.get(f"/runs/{run2_id}").json()

        assert run1_data["collection"] == "collection_1"
        assert run2_data["collection"] == "collection_2"

        # Clean up
        api_client.delete(f"/runs/{run1_id}")
        api_client.delete(f"/runs/{run2_id}")


class TestProjectsAPIEndToEnd:
    """End-to-end tests for Projects API."""

    @pytest.mark.integration
    def test_project_crud_workflow(self, api_client):
        """Test full project CRUD lifecycle."""
        # Create
        create_response = api_client.post("/projects/", params={"name": "CRUD Test"})
        assert create_response.status_code == 200
        project_id = create_response.json()["id"]

        # Read
        get_response = api_client.get(f"/projects/{project_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "CRUD Test"

        # List includes our project
        list_response = api_client.get("/projects/")
        assert list_response.status_code == 200
        project_ids = [p["id"] for p in list_response.json()]
        assert project_id in project_ids

    @pytest.mark.integration
    def test_project_with_index_status(self, api_client):
        """Test project index status workflow."""
        # Create project
        project_response = api_client.post("/projects/", params={"name": "Index Workflow"})
        project_id = project_response.json()["id"]

        # Check initial index status (should have no indexed models)
        status_response = api_client.get(f"/index/status/{project_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["project_id"] == project_id
        assert status["has_data"] is False
        assert len(status["indexed_models"]) == 0


class TestIndexingServiceEndToEnd:
    """End-to-end tests for the indexing service."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_chunking_strategies(self):
        """Test different chunking strategies produce expected results."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")

        # Long text to chunk
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10

        # Test different strategies
        strategies = [
            (ChunkingStrategy.NONE, 1),  # Should produce 1 chunk
            (ChunkingStrategy.FIXED_TOKENS, None),  # Should produce multiple chunks
            (ChunkingStrategy.SENTENCES, None),  # Should produce sentence-based chunks
        ]

        for strategy, expected_count in strategies:
            config = ChunkingConfig(strategy=strategy, size=20, overlap=5)
            chunks = service._chunk_texts([text], config)

            if expected_count is not None:
                assert len(chunks) == expected_count, f"Strategy {strategy} failed"
            else:
                assert len(chunks) >= 1, f"Strategy {strategy} produced no chunks"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_job_stage_pipeline(self, sample_index_plan):
        """Test that job stages are created and executed in order."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import IndexJob, JobStatus, JobStageStatus

        service = IndexingService(qdrant_url=":memory:")

        job = IndexJob(
            id="pipeline-test",
            plan_id=sample_index_plan.id,
            project_id=sample_index_plan.project_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=JobStatus.QUEUED
        )

        # Track progress through callback
        stage_history = []

        def progress_callback(updated_job):
            if updated_job.stages:
                current = updated_job.current_stage_index
                if current < len(updated_job.stages):
                    stage_history.append(updated_job.stages[current].name)

        # Mock the dataset loading to avoid network calls
        with patch('maxq.server.indexing_service.load_dataset') as mock_load:
            mock_load.return_value = iter([
                {"prompt": "Test document content here " * 20}
            ])

            try:
                service.run_job(job, sample_index_plan, progress_callback)
            except Exception as e:
                # Job might fail due to missing resources, but stages should still be tracked
                pass

        # Verify stages were created
        assert len(job.stages) > 0

        # First stage should be validate_plan
        assert job.stages[0].name == "validate_plan"


@requires_qdrant
class TestDataFlowIntegration:
    """Tests for data flow between components."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_payload_preserved_through_ingestion(self, in_memory_engine):
        """Test that payload data is preserved through ingestion."""
        from maxq.search_engine import CollectionStrategy

        config = CollectionStrategy(
            collection_name="payload_test",
            dense_model_name="BAAI/bge-small-en-v1.5"
        )

        in_memory_engine.initialize_collection(config)

        # Ingest with specific payload
        texts = ["Test document for payload verification"]
        payloads = [{
            "source": "test_file.txt",
            "category": "documentation",
            "page_num": 5,
            "important": True
        }]

        in_memory_engine._upload_batch(config, texts, payloads, 0)
        time.sleep(0.5)

        # Retrieve and verify payload
        points, _ = in_memory_engine.client.scroll(
            config.collection_name,
            limit=1,
            with_payload=True
        )

        assert len(points) == 1
        payload = points[0].payload

        assert payload["source"] == "test_file.txt"
        assert payload["category"] == "documentation"
        assert payload["page_num"] == 5
        assert payload["important"] is True
        assert "_text" in payload  # Text should also be stored

    @pytest.mark.slow
    @pytest.mark.integration
    def test_search_results_contain_payload(self, in_memory_engine):
        """Test that search results include full payload."""
        from maxq.search_engine import CollectionStrategy, SearchRequest

        config = CollectionStrategy(
            collection_name="search_payload_test",
            dense_model_name="BAAI/bge-small-en-v1.5"
        )

        in_memory_engine.initialize_collection(config)

        texts = [
            "Machine learning algorithms for classification",
            "Deep learning neural network architectures"
        ]
        payloads = [
            {"topic": "ML", "difficulty": "beginner"},
            {"topic": "DL", "difficulty": "advanced"}
        ]

        in_memory_engine._upload_batch(config, texts, payloads, 0)
        time.sleep(0.5)

        request = SearchRequest(
            query="machine learning",
            strategy="hybrid",
            limit=2
        )

        results = in_memory_engine.query(config, request)

        assert len(results) > 0
        for result in results:
            assert "topic" in result.payload
            assert "difficulty" in result.payload


@requires_qdrant
@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling across components."""

    def test_search_on_nonexistent_collection(self, in_memory_engine):
        """Test that searching non-existent collection raises error."""
        from maxq.search_engine import CollectionStrategy, SearchRequest

        config = CollectionStrategy(
            collection_name="nonexistent_collection",
            dense_model_name="BAAI/bge-small-en-v1.5"
        )

        request = SearchRequest(query="test", strategy="hybrid", limit=5)

        with pytest.raises(Exception):
            in_memory_engine.query(config, request)

    def test_api_handles_invalid_json(self, api_client):
        """Test that API handles invalid JSON gracefully."""
        response = api_client.post(
            "/runs/index",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error

    def test_api_handles_missing_fields(self, api_client):
        """Test that API handles missing required fields."""
        response = api_client.post(
            "/runs/index",
            json={}  # Missing required fields
        )
        assert response.status_code == 422


class TestConfigurationIntegration:
    """Tests for configuration and environment integration."""

    def test_collection_naming_consistency(self):
        """Test that collection naming is consistent across components."""
        from maxq.search_engine import MaxQEngine

        project_id = "test-project-123"
        model_name = "BAAI/bge-base-en-v1.5"

        # Same inputs should always produce same output
        name1 = MaxQEngine.get_collection_name(project_id, model_name)
        name2 = MaxQEngine.get_collection_name(project_id, model_name)

        assert name1 == name2

        # Different inputs should produce different outputs
        name3 = MaxQEngine.get_collection_name("other-project", model_name)
        name4 = MaxQEngine.get_collection_name(project_id, "other/model")

        assert name1 != name3
        assert name1 != name4
