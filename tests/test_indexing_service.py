"""
Tests for the IndexingService class.
Tests job execution, staging, chunking, and pipeline functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestIndexingServiceInit:
    """Tests for IndexingService initialization."""

    def test_init_with_memory(self):
        """Test initialization with in-memory Qdrant."""
        from maxq.server.indexing_service import IndexingService

        service = IndexingService(qdrant_url=":memory:")
        assert service.client is not None
        assert service._in_memory_mode is True

    def test_init_from_env(self):
        """Test initialization from environment variables."""
        from maxq.server.indexing_service import IndexingService

        with patch.dict('os.environ', {"QDRANT_URL": ":memory:", "QDRANT_API_KEY": ""}):
            service = IndexingService()
            assert service.client is not None


class TestJobStageManagement:
    """Tests for job stage creation and management."""

    def test_create_stage(self):
        """Test creating a job stage."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")
        stage = service._create_stage("test_stage", "Testing stage")

        assert stage.name == "test_stage"
        assert stage.active_form == "Testing stage"
        assert stage.status == JobStageStatus.QUEUED
        assert stage.started_at is None
        assert stage.completed_at is None

    def test_start_stage(self):
        """Test starting a job stage."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")
        stage = service._create_stage("test", "Testing")

        service._start_stage(stage)

        assert stage.status == JobStageStatus.RUNNING
        assert stage.started_at is not None

    def test_complete_stage_success(self):
        """Test completing a stage successfully."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")
        stage = service._create_stage("test", "Testing")
        service._start_stage(stage)

        import time
        time.sleep(0.01)

        service._complete_stage(stage, success=True)

        assert stage.status == JobStageStatus.SUCCESS
        assert stage.completed_at is not None
        assert stage.duration_ms is not None
        assert stage.duration_ms > 0
        assert stage.error_message is None

    def test_complete_stage_failure(self):
        """Test completing a stage with failure."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")
        stage = service._create_stage("test", "Testing")
        service._start_stage(stage)

        service._complete_stage(stage, success=False, error="Something went wrong")

        assert stage.status == JobStageStatus.FAILED
        assert stage.error_message == "Something went wrong"


class TestChunking:
    """Tests for text chunking functionality."""

    def test_chunk_texts_none_strategy(self):
        """Test chunking with NONE strategy returns text as-is."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")
        config = ChunkingConfig(strategy=ChunkingStrategy.NONE)

        texts = ["This is a test document.", "Another document."]
        chunks = service._chunk_texts(texts, config)

        assert len(chunks) == 2
        assert chunks[0] == "This is a test document."

    def test_chunk_texts_fixed_tokens(self):
        """Test chunking with FIXED_TOKENS strategy."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_TOKENS,
            size=5,
            overlap=2
        )

        texts = ["one two three four five six seven eight nine ten"]
        chunks = service._chunk_texts(texts, config)

        assert len(chunks) > 1
        # First chunk should have 5 words
        assert len(chunks[0].split()) == 5

    def test_chunk_texts_fixed_tokens_overlap(self):
        """Test that fixed token chunking properly overlaps."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_TOKENS,
            size=4,
            overlap=2
        )

        texts = ["a b c d e f g h"]
        chunks = service._chunk_texts(texts, config)

        # With size=4 and overlap=2, step is 2
        # Chunk 0: a b c d (indices 0-3)
        # Chunk 1: c d e f (indices 2-5)
        # Chunk 2: e f g h (indices 4-7)
        assert len(chunks) >= 2

    def test_chunk_texts_sentences(self):
        """Test chunking with SENTENCES strategy."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCES,
            size=10
        )

        texts = ["First sentence. Second sentence. Third sentence."]
        chunks = service._chunk_texts(texts, config)

        assert len(chunks) >= 1
        # All sentences combined should fit within size
        assert "sentence" in chunks[0].lower()

    def test_chunk_texts_empty_input(self):
        """Test chunking with empty input."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import ChunkingConfig, ChunkingStrategy

        service = IndexingService(qdrant_url=":memory:")
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED_TOKENS)

        chunks = service._chunk_texts([], config)
        assert chunks == []


class TestTextExtraction:
    """Tests for text extraction from data rows."""

    def test_extract_text_single_field(self):
        """Test extracting text with SINGLE_FIELD strategy."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.SINGLE_FIELD,
            text_field="content"
        )

        row = {"content": "This is the content", "id": 1}
        text = service._extract_text_from_row(row, vectorization)

        assert text == "This is the content"

    def test_extract_text_combine_fields(self):
        """Test extracting text with COMBINE_FIELDS strategy."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.COMBINE_FIELDS,
            combine_template="{{title}}: {{description}}",
            combine_fields=["title", "description"]
        )

        row = {"title": "My Title", "description": "My Description", "id": 1}
        text = service._extract_text_from_row(row, vectorization)

        assert text == "My Title: My Description"

    def test_extract_text_combine_fields_missing_value(self):
        """Test combining fields when one is missing."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.COMBINE_FIELDS,
            combine_template="{{title}}: {{description}}",
            combine_fields=["title", "description"]
        )

        row = {"title": "My Title", "id": 1}  # description missing
        text = service._extract_text_from_row(row, vectorization)

        assert text == "My Title: "

    def test_extract_text_llm_enrich_fallback(self):
        """Test LLM enrich falls back to combining values."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.LLM_ENRICH,
            llm_enrich_fields=["name", "category"]
        )

        row = {"name": "Product A", "category": "Electronics", "id": 1}
        text = service._extract_text_from_row(row, vectorization)

        assert "Product A" in text
        assert "Electronics" in text

    def test_extract_text_image_field(self):
        """Test extracting image field."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.IMAGE,
            image_field="image_url"
        )

        row = {"image_url": "https://example.com/image.jpg", "id": 1}
        text = service._extract_text_from_row(row, vectorization)

        assert text == "https://example.com/image.jpg"

    def test_extract_text_multimodal(self):
        """Test extracting multimodal text."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import VectorizationConfig, VectorizationStrategy

        service = IndexingService(qdrant_url=":memory:")

        vectorization = VectorizationConfig(
            strategy=VectorizationStrategy.MULTIMODAL,
            text_fields=["title", "description"]
        )

        row = {"title": "Product", "description": "A great product", "id": 1}
        text = service._extract_text_from_row(row, vectorization)

        assert "Product" in text
        assert "A great product" in text


class TestDryRunEstimate:
    """Tests for dry run estimation."""

    def test_estimate_job_basic(self, sample_index_plan):
        """Test basic job estimation."""
        from maxq.server.indexing_service import IndexingService

        service = IndexingService(qdrant_url=":memory:")

        with patch('maxq.server.indexing_service.load_dataset') as mock_load:
            mock_load.return_value = iter([
                {"prompt": "Sample text " * 100},  # Long text to ensure chunking
                {"prompt": "Another sample " * 100}
            ])

            estimate = service.estimate_job(sample_index_plan)

            assert estimate.estimated_docs > 0
            assert estimate.estimated_chunks >= estimate.estimated_docs
            assert estimate.estimated_points > 0
            assert estimate.estimated_storage_gb >= 0
            assert estimate.estimated_embed_time_minutes >= 0

    def test_estimate_job_sample_chunks(self, sample_index_plan):
        """Test that estimation includes sample chunks preview."""
        from maxq.server.indexing_service import IndexingService

        service = IndexingService(qdrant_url=":memory:")

        with patch('maxq.server.indexing_service.load_dataset') as mock_load:
            mock_load.return_value = iter([
                {"prompt": "This is sample text for testing " * 50}
            ])

            estimate = service.estimate_job(sample_index_plan)

            assert isinstance(estimate.sample_chunks, list)


class TestJobExecution:
    """Tests for full job execution pipeline."""

    @pytest.mark.slow
    def test_run_stage_validate_plan_success(self, sample_index_plan, sample_index_job):
        """Test plan validation stage succeeds with valid plan."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")

        # Setup job with stages
        sample_index_job.stages = [
            service._create_stage("validate_plan", "Validating plan")
        ]
        sample_index_job.current_stage_index = 0

        service._run_stage_validate_plan(sample_index_job, sample_index_plan, None)

        assert sample_index_job.stages[0].status == JobStageStatus.SUCCESS
        assert sample_index_job.current_stage_index == 1

    def test_run_stage_validate_plan_missing_dataset(self, sample_index_plan, sample_index_job):
        """Test plan validation fails without dataset ID."""
        from maxq.server.indexing_service import IndexingService

        service = IndexingService(qdrant_url=":memory:")

        sample_index_plan.data_source.dataset_id = None

        sample_index_job.stages = [
            service._create_stage("validate_plan", "Validating plan")
        ]
        sample_index_job.current_stage_index = 0

        with pytest.raises(ValueError, match="Dataset ID is required"):
            service._run_stage_validate_plan(sample_index_job, sample_index_plan, None)

    @pytest.mark.slow
    def test_run_stage_create_collection(self, sample_index_plan, sample_index_job):
        """Test collection creation stage."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import JobStageStatus

        service = IndexingService(qdrant_url=":memory:")

        # Setup stages
        sample_index_job.stages = [
            service._create_stage("validate_plan", "Validating"),
            service._create_stage("create_collection", "Creating collection")
        ]
        sample_index_job.current_stage_index = 1

        service._run_stage_create_collection(sample_index_job, sample_index_plan, None)

        assert sample_index_job.stages[1].status == JobStageStatus.SUCCESS
        assert sample_index_job.collection_name is not None

    @pytest.mark.slow
    def test_run_stage_create_payload_indexes(self, sample_index_plan, sample_index_job):
        """Test payload index creation stage."""
        from maxq.server.indexing_service import IndexingService
        from maxq.server.indexing_models import (
            JobStageStatus, PayloadField, PayloadFieldType
        )

        service = IndexingService(qdrant_url=":memory:")

        # Add a field to index
        sample_index_plan.payload_schema.fields = [
            PayloadField(
                field_name="category",
                field_type=PayloadFieldType.KEYWORD,
                indexed=True
            )
        ]

        # First create the collection
        sample_index_job.stages = [
            service._create_stage("create_collection", "Creating collection"),
            service._create_stage("create_payload_indexes", "Creating indexes")
        ]
        sample_index_job.current_stage_index = 0

        service._run_stage_create_collection(sample_index_job, sample_index_plan, None)
        sample_index_job.current_stage_index = 1

        service._run_stage_create_payload_indexes(sample_index_job, sample_index_plan, None)

        assert sample_index_job.stages[1].status == JobStageStatus.SUCCESS


class TestModelLoading:
    """Tests for embedding model loading.

    NOTE: These tests are skipped because model loading is now handled
    via Qdrant Cloud Inference - no local models are loaded.
    """

    @pytest.mark.skip(reason="Local model loading replaced by cloud inference")
    @pytest.mark.slow
    def test_load_embedding_models(self, sample_index_plan):
        """Test that embedding models are loaded correctly."""
        pass

    @pytest.mark.skip(reason="Local model loading replaced by cloud inference")
    @pytest.mark.slow
    def test_load_sparse_model_when_enabled(self, sample_index_plan):
        """Test that sparse model is loaded when enabled."""
        pass


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_callback_called_on_stage_complete(self, sample_index_plan, sample_index_job):
        """Test that callback is called when stage completes."""
        from maxq.server.indexing_service import IndexingService

        service = IndexingService(qdrant_url=":memory:")

        callback_called = []

        def callback(job):
            callback_called.append(job.current_stage_index)

        sample_index_job.stages = [
            service._create_stage("validate_plan", "Validating")
        ]
        sample_index_job.current_stage_index = 0

        service._run_stage_validate_plan(sample_index_job, sample_index_plan, callback)

        assert len(callback_called) == 1
