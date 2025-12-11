"""
Unit tests for Pydantic data models.
Tests validation, serialization, and default values.
"""
import pytest
from datetime import datetime


class TestProjectModel:
    """Tests for Project model."""

    def test_project_creation(self):
        """Test basic project creation."""
        from maxq.server.models import Project, ProjectStatus

        project = Project(
            id="test-id",
            name="Test Project",
            created_at=datetime.now()
        )
        assert project.id == "test-id"
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.ACTIVE

    def test_project_defaults(self):
        """Test project default values."""
        from maxq.server.models import Project

        project = Project(
            id="test-id",
            name="Test",
            created_at=datetime.now()
        )
        assert project.description is None
        assert project.task_type is None
        assert project.embedding_model == "BAAI/bge-base-en-v1.5"
        assert project.last_accessed is None

    def test_project_status_enum(self):
        """Test ProjectStatus enum values."""
        from maxq.server.models import ProjectStatus

        assert ProjectStatus.ACTIVE == "active"
        assert ProjectStatus.ARCHIVED == "archived"

    def test_project_with_all_fields(self):
        """Test project with all fields populated."""
        from maxq.server.models import Project, ProjectStatus

        now = datetime.now()
        project = Project(
            id="test-id",
            name="Full Project",
            description="A complete project",
            created_at=now,
            status=ProjectStatus.ACTIVE,
            task_type="legal",
            embedding_model="BAAI/bge-small-en-v1.5",
            last_accessed=now
        )
        assert project.description == "A complete project"
        assert project.task_type == "legal"
        assert project.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_project_json_serialization(self, sample_project_model):
        """Test project can be serialized to JSON."""
        json_data = sample_project_model.model_dump(mode='json')
        assert isinstance(json_data, dict)
        assert json_data["id"] == "test-id"
        assert json_data["name"] == "Test Project"


class TestDatasetModel:
    """Tests for Dataset model."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        from maxq.server.models import Dataset

        dataset = Dataset(
            id="ds-id",
            project_id="proj-id",
            name="Test Dataset",
            source_type="huggingface",
            status="pending"
        )
        assert dataset.id == "ds-id"
        assert dataset.project_id == "proj-id"
        assert dataset.doc_count == 0

    def test_dataset_with_doc_count(self):
        """Test dataset with document count."""
        from maxq.server.models import Dataset

        dataset = Dataset(
            id="ds-id",
            project_id="proj-id",
            name="Test Dataset",
            source_type="upload",
            status="completed",
            doc_count=1000
        )
        assert dataset.doc_count == 1000


class TestExperimentModel:
    """Tests for Experiment model."""

    def test_experiment_creation(self):
        """Test basic experiment creation."""
        from maxq.server.models import Experiment

        exp = Experiment(
            id="exp-id",
            project_id="proj-id",
            name="Test Experiment",
            created_at=datetime.now(),
            embedding_model="BAAI/bge-base-en-v1.5"
        )
        assert exp.id == "exp-id"
        assert exp.status == "pending"
        assert exp.chunk_size == 512
        assert exp.search_strategy == "hybrid"

    def test_experiment_defaults(self):
        """Test experiment default values."""
        from maxq.server.models import Experiment

        exp = Experiment(
            id="exp-id",
            project_id="proj-id",
            name="Test",
            created_at=datetime.now(),
            embedding_model="BAAI/bge-base-en-v1.5"
        )
        assert exp.progress_current == 0
        assert exp.progress_total == 0
        assert exp.progress_message is None
        assert exp.started_at is None
        assert exp.metrics is None

    def test_experiment_with_metrics(self):
        """Test experiment with metrics."""
        from maxq.server.models import Experiment

        metrics = {
            "ndcg": {"candidate": 0.85, "baseline": 0.75, "delta": "+13.3%"},
            "latency": {"candidate": "100ms", "baseline": "150ms", "delta": "-33%"}
        }
        exp = Experiment(
            id="exp-id",
            project_id="proj-id",
            name="Test",
            created_at=datetime.now(),
            embedding_model="BAAI/bge-base-en-v1.5",
            metrics=metrics
        )
        assert exp.metrics["ndcg"]["candidate"] == 0.85


class TestEvalResultModel:
    """Tests for EvalResult model."""

    def test_eval_result_creation(self):
        """Test basic eval result creation."""
        from maxq.server.models import EvalResult

        result = EvalResult(
            id="eval-id",
            experiment_id="exp-id",
            metrics={"context_recall": 0.85, "context_precision": 0.9}
        )
        assert result.id == "eval-id"
        assert result.metrics["context_recall"] == 0.85

    def test_eval_result_with_details(self):
        """Test eval result with detailed output."""
        from maxq.server.models import EvalResult

        result = EvalResult(
            id="eval-id",
            experiment_id="exp-id",
            metrics={"score": 0.9},
            details={"per_question": [{"q": "Question 1", "score": 0.9}]}
        )
        assert result.details is not None
        assert len(result.details["per_question"]) == 1


class TestIndexingModels:
    """Tests for indexing-related models."""

    def test_job_stage_status_enum(self):
        """Test JobStageStatus enum values."""
        from maxq.server.indexing_models import JobStageStatus

        assert JobStageStatus.QUEUED == "queued"
        assert JobStageStatus.RUNNING == "running"
        assert JobStageStatus.SUCCESS == "success"
        assert JobStageStatus.FAILED == "failed"
        assert JobStageStatus.SKIPPED == "skipped"

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        from maxq.server.indexing_models import JobStatus

        assert JobStatus.QUEUED == "queued"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"
        assert JobStatus.PAUSED == "paused"

    def test_data_source_type_enum(self):
        """Test DataSourceType enum values."""
        from maxq.server.indexing_models import DataSourceType

        assert DataSourceType.HUGGINGFACE == "huggingface"
        assert DataSourceType.UPLOAD == "upload"
        assert DataSourceType.S3 == "s3"
        assert DataSourceType.URL == "url"

    def test_vectorization_strategy_enum(self):
        """Test VectorizationStrategy enum values."""
        from maxq.server.indexing_models import VectorizationStrategy

        assert VectorizationStrategy.SINGLE_FIELD == "single_field"
        assert VectorizationStrategy.COMBINE_FIELDS == "combine_fields"
        assert VectorizationStrategy.LLM_ENRICH == "llm_enrich"

    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum values."""
        from maxq.server.indexing_models import ChunkingStrategy

        assert ChunkingStrategy.NONE == "none"
        assert ChunkingStrategy.FIXED_TOKENS == "fixed_tokens"
        assert ChunkingStrategy.SENTENCES == "sentences"

    def test_job_stage_creation(self):
        """Test JobStage creation."""
        from maxq.server.indexing_models import JobStage, JobStageStatus

        stage = JobStage(
            name="ingest",
            active_form="Ingesting data"
        )
        assert stage.name == "ingest"
        assert stage.status == JobStageStatus.QUEUED
        assert stage.docs_processed == 0
        assert stage.logs == []

    def test_data_source_config_defaults(self, sample_data_source_config):
        """Test DataSourceConfig has correct defaults."""
        assert sample_data_source_config.split == "train"
        assert sample_data_source_config.streaming is True

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig has correct defaults."""
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        config = ChunkingConfig()
        assert config.strategy == ChunkingStrategy.FIXED_TOKENS
        assert config.size == 512
        assert config.overlap == 50
        assert config.keep_parent_id is True

    def test_dense_vector_config(self):
        """Test DenseVectorConfig creation."""
        from maxq.server.indexing_models import DenseVectorConfig, VectorProvider, VectorDistance

        config = DenseVectorConfig(
            name="dense",
            provider=VectorProvider.LOCAL,
            model_name="BAAI/bge-base-en-v1.5",
            distance=VectorDistance.COSINE
        )
        assert config.batch_size == 32
        assert config.concurrency == 1

    def test_sparse_vector_config(self):
        """Test SparseVectorConfig creation."""
        from maxq.server.indexing_models import SparseVectorConfig

        config = SparseVectorConfig()
        assert config.name == "sparse"
        assert config.enabled is True
        assert config.generator == "bm25"  # Default changed to bm25 for Qdrant Cloud

    def test_performance_storage_config_defaults(self):
        """Test PerformanceStorageConfig defaults."""
        from maxq.server.indexing_models import PerformanceStorageConfig, PresetType

        config = PerformanceStorageConfig()
        assert config.preset == PresetType.DEV
        assert config.shard_number == 1
        assert config.replication_factor == 1
        assert config.on_disk_payload is False

    def test_run_config_defaults(self):
        """Test RunConfig defaults."""
        from maxq.server.indexing_models import RunConfig

        config = RunConfig()
        assert config.dry_run is False
        assert config.build_new_collection is False
        assert config.run_verification is True

    def test_index_plan_serialization(self, sample_index_plan):
        """Test IndexPlan can be serialized."""
        json_data = sample_index_plan.model_dump(mode='json')
        assert isinstance(json_data, dict)
        assert json_data["id"] == "test-plan-id"
        assert json_data["project_id"] == "test-project-id"

    def test_index_job_creation(self, sample_index_job):
        """Test IndexJob has correct defaults."""
        from maxq.server.indexing_models import JobStatus

        assert sample_index_job.status == JobStatus.QUEUED
        assert sample_index_job.total_docs == 0
        assert sample_index_job.stages == []

    def test_dry_run_estimate_creation(self):
        """Test DryRunEstimate creation."""
        from maxq.server.indexing_models import DryRunEstimate

        estimate = DryRunEstimate(
            estimated_docs=1000,
            estimated_chunks=5000,
            estimated_points=5000,
            estimated_storage_gb=0.5,
            estimated_embed_time_minutes=5.0
        )
        assert estimate.estimated_docs == 1000
        assert estimate.sample_chunks == []

    def test_job_progress_creation(self):
        """Test JobProgress creation."""
        from maxq.server.indexing_models import JobProgress, JobStatus

        progress = JobProgress(
            job_id="job-123",
            status=JobStatus.RUNNING,
            progress_percent=50.5
        )
        assert progress.job_id == "job-123"
        assert progress.current_stage is None
        assert progress.recent_logs == []

    def test_create_plan_request(self, sample_data_source_config, sample_vector_spaces_config):
        """Test CreatePlanRequest creation."""
        from maxq.server.indexing_models import CreatePlanRequest

        request = CreatePlanRequest(
            project_id="proj-123",
            name="Test Plan",
            data_source=sample_data_source_config,
            vector_spaces=sample_vector_spaces_config
        )
        assert request.project_id == "proj-123"
        assert request.name == "Test Plan"

    def test_start_job_request(self):
        """Test StartJobRequest creation."""
        from maxq.server.indexing_models import StartJobRequest

        request = StartJobRequest(
            plan_id="plan-123",
            dry_run=True
        )
        assert request.plan_id == "plan-123"
        assert request.dry_run is True


class TestVectorSpacesConfig:
    """Tests for VectorSpacesConfig model."""

    def test_vector_spaces_with_dense_only(self):
        """Test VectorSpacesConfig with only dense vector."""
        from maxq.server.indexing_models import (
            VectorSpacesConfig, DenseVectorConfig, VectorProvider, VectorDistance
        )

        config = VectorSpacesConfig(
            dense=DenseVectorConfig(
                name="dense",
                provider=VectorProvider.LOCAL,
                model_name="BAAI/bge-base-en-v1.5",
                distance=VectorDistance.COSINE
            )
        )
        assert config.dense is not None
        assert config.sparse is None
        assert config.late is None

    def test_vector_spaces_with_sparse(self, sample_vector_spaces_config):
        """Test VectorSpacesConfig with sparse enabled."""
        assert sample_vector_spaces_config.sparse is not None
        assert sample_vector_spaces_config.sparse.enabled is True

    def test_vector_provider_enum(self):
        """Test VectorProvider enum values."""
        from maxq.server.indexing_models import VectorProvider

        assert VectorProvider.LOCAL == "local"
        assert VectorProvider.OPENAI == "openai"
        assert VectorProvider.QCI == "qci"

    def test_vector_distance_enum(self):
        """Test VectorDistance enum values."""
        from maxq.server.indexing_models import VectorDistance

        assert VectorDistance.COSINE == "cosine"
        assert VectorDistance.DOT == "dot"
        assert VectorDistance.EUCLIDEAN == "euclidean"


class TestPayloadSchema:
    """Tests for PayloadSchema models."""

    def test_payload_field_creation(self):
        """Test PayloadField creation."""
        from maxq.server.indexing_models import PayloadField, PayloadFieldType

        field = PayloadField(
            field_name="category",
            field_type=PayloadFieldType.KEYWORD,
            indexed=True
        )
        assert field.field_name == "category"
        assert field.indexed is True
        assert field.full_text_indexed is False

    def test_payload_schema_config_defaults(self):
        """Test PayloadSchemaConfig defaults."""
        from maxq.server.indexing_models import PayloadSchemaConfig

        config = PayloadSchemaConfig()
        assert config.fields == []
        assert config.auto_index_keywords is True

    def test_payload_field_type_enum(self):
        """Test PayloadFieldType enum values."""
        from maxq.server.indexing_models import PayloadFieldType

        assert PayloadFieldType.KEYWORD == "keyword"
        assert PayloadFieldType.TEXT == "text"
        assert PayloadFieldType.INTEGER == "integer"
        assert PayloadFieldType.FLOAT == "float"
        assert PayloadFieldType.DATETIME == "datetime"
        assert PayloadFieldType.BOOLEAN == "boolean"


class TestHNSWAndQuantization:
    """Tests for HNSW and Quantization config models."""

    def test_hnsw_config_defaults(self):
        """Test HNSWConfig defaults."""
        from maxq.server.indexing_models import HNSWConfig

        config = HNSWConfig()
        assert config.m == 16
        assert config.ef_construct == 100
        assert config.full_scan_threshold == 10000

    def test_quantization_config_defaults(self):
        """Test QuantizationConfig defaults."""
        from maxq.server.indexing_models import QuantizationConfig

        config = QuantizationConfig()
        assert config.enabled is False
        assert config.type == "int8"
        assert config.always_ram is True

    def test_optimizers_config_defaults(self):
        """Test OptimizersConfig defaults."""
        from maxq.server.indexing_models import OptimizersConfig

        config = OptimizersConfig()
        assert config.indexing_threshold == 20000
        assert config.memmap_threshold == 50000
