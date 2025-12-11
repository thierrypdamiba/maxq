"""
Pytest configuration and shared fixtures for MaxQ tests.

To run tests with Qdrant Cloud, set these environment variables:
    export QDRANT_URL="https://your-cluster.qdrant.io"
    export QDRANT_API_KEY="your-api-key"

If not set, tests will fall back to in-memory mode (some sparse tests may fail).
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use Qdrant Cloud if credentials are provided, otherwise fall back to in-memory
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


# --- Fixtures for Engine Testing ---


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"embedding_column": "text", "reason": "Main text field"}'
                )
            )
        ]
    )
    return mock


@pytest.fixture
def in_memory_engine():
    """Create a MaxQEngine for testing with mocked Qdrant client.

    NOTE: MaxQEngine requires Qdrant Cloud and doesn't support in-memory mode.
    We mock the QdrantClient for unit tests.
    """
    from maxq.search_engine import MaxQEngine

    with patch("maxq.search_engine.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        engine = MaxQEngine(qdrant_url="https://test.qdrant.io", qdrant_api_key="test-key")
        engine.client = mock_client
        return engine


@pytest.fixture
def engine_with_mock_llm(mock_openai_client):
    """Create an engine with mocked LLM client."""
    from maxq.search_engine import MaxQEngine

    with patch("maxq.search_engine.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_cls.return_value = mock_client

        with patch("maxq.search_engine.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = mock_openai_client
            engine = MaxQEngine(
                qdrant_url="https://test.qdrant.io",
                qdrant_api_key="test-key",
                openai_api_key="test-key",
            )
        engine.client = mock_client
        engine.llm_client = mock_openai_client
        return engine


@pytest.fixture
def sample_collection_strategy():
    """Create a sample CollectionStrategy for testing."""
    from maxq.search_engine import CollectionStrategy

    return CollectionStrategy(
        collection_name="test_collection",
        estimated_doc_count=1000,
        dense_model_name="BAAI/bge-small-en-v1.5",  # Use smaller model for faster tests
        sparse_model_name="prithivida/Splade_PP_en_v1",
        use_quantization=False,
    )


@pytest.fixture
def sample_search_request():
    """Create a sample SearchRequest for testing."""
    from maxq.search_engine import SearchRequest

    return SearchRequest(query="test query", limit=5, strategy="hybrid", score_threshold=0.0)


# --- Fixtures for API Testing ---


@pytest.fixture
def temp_maxq_dir():
    """Create a temporary directory for MaxQ data files."""
    temp_dir = tempfile.mkdtemp()
    original_dir = None

    # Patch the MAXQ_APP_DIR
    import maxq.config as config

    original_dir = config.MAXQ_APP_DIR
    config.MAXQ_APP_DIR = Path(temp_dir)

    yield Path(temp_dir)

    # Cleanup
    config.MAXQ_APP_DIR = original_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def api_client(temp_maxq_dir):
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient

    # Patch the MAXQ_APP_DIR before importing the app
    import maxq.config as config

    config.MAXQ_APP_DIR = temp_maxq_dir

    # Set up fresh SQLite database for projects
    from maxq.server import database

    database.DATABASE_PATH = temp_maxq_dir / "maxq.db"
    database.init_db()  # Create fresh tables

    # Reload routers to pick up new config
    from maxq.server import main
    from maxq.server.routers import datasets, playground, indexing, evals, tuning

    # Reset module-level state for non-SQLite routers
    indexing.plans_db = []
    indexing.jobs_db = []
    indexing.PLANS_FILE = temp_maxq_dir / "index_plans.json"
    indexing.JOBS_FILE = temp_maxq_dir / "index_jobs.json"

    tuning.experiments_db = []
    tuning.DATA_FILE = temp_maxq_dir / "experiments.json"

    evals.evals_db = []
    evals.DATA_FILE = temp_maxq_dir / "evals.json"

    return TestClient(main.app)


@pytest.fixture
def sample_project():
    """Create a sample project data."""
    return {
        "name": "Test Project",
        "description": "A test project for unit tests",
        "task_type": "general",
    }


@pytest.fixture
def sample_dataset_request():
    """Create a sample dataset ingest request."""
    return {
        "project_id": "test-project-id",
        "dataset_name": "fka/awesome-chatgpt-prompts",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "quantization": "None",
        "sample_limit": 10,
    }


# --- Fixtures for Indexing Models ---


@pytest.fixture
def sample_data_source_config():
    """Create a sample DataSourceConfig."""
    from maxq.server.indexing_models import (
        DataSourceConfig,
        DataSourceType,
        VectorizationConfig,
        VectorizationStrategy,
    )

    return DataSourceConfig(
        source_type=DataSourceType.HUGGINGFACE,
        dataset_id="fka/awesome-chatgpt-prompts",
        split="train",
        streaming=True,
        vectorization=VectorizationConfig(
            strategy=VectorizationStrategy.SINGLE_FIELD, text_field="prompt"
        ),
        sample_limit=10,
    )


@pytest.fixture
def sample_vector_spaces_config():
    """Create a sample VectorSpacesConfig."""
    from maxq.server.indexing_models import (
        VectorSpacesConfig,
        DenseVectorConfig,
        SparseVectorConfig,
        VectorProvider,
        VectorDistance,
    )

    return VectorSpacesConfig(
        dense=DenseVectorConfig(
            name="dense",
            provider=VectorProvider.LOCAL,
            model_name="BAAI/bge-small-en-v1.5",
            distance=VectorDistance.COSINE,
        ),
        sparse=SparseVectorConfig(
            name="sparse", enabled=True, generator="splade", model_name="prithivida/Splade_PP_en_v1"
        ),
    )


@pytest.fixture
def sample_index_plan(sample_data_source_config, sample_vector_spaces_config):
    """Create a sample IndexPlan."""
    from maxq.server.indexing_models import (
        IndexPlan,
        ChunkingConfig,
        PayloadSchemaConfig,
        PerformanceStorageConfig,
        RunConfig,
    )

    return IndexPlan(
        id="test-plan-id",
        project_id="test-project-id",
        name="Test Index Plan",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        data_source=sample_data_source_config,
        chunking=ChunkingConfig(strategy="fixed_tokens", size=256, overlap=25),
        payload_schema=PayloadSchemaConfig(),
        vector_spaces=sample_vector_spaces_config,
        performance=PerformanceStorageConfig(),
        run_config=RunConfig(dry_run=False),
    )


@pytest.fixture
def sample_index_job():
    """Create a sample IndexJob."""
    from maxq.server.indexing_models import IndexJob, JobStatus

    return IndexJob(
        id="test-job-id",
        plan_id="test-plan-id",
        project_id="test-project-id",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=JobStatus.QUEUED,
    )


# --- Fixtures for Pydantic Model Testing ---


@pytest.fixture
def sample_project_model():
    """Create a sample Project model."""
    from maxq.server.models import Project, ProjectStatus

    return Project(
        id="test-id",
        name="Test Project",
        description="Test description",
        created_at=datetime.now(),
        status=ProjectStatus.ACTIVE,
        task_type="general",
        embedding_model="BAAI/bge-base-en-v1.5",
    )


@pytest.fixture
def sample_experiment_model():
    """Create a sample Experiment model."""
    from maxq.server.models import Experiment

    return Experiment(
        id="test-exp-id",
        project_id="test-project-id",
        name="Test Experiment",
        status="pending",
        created_at=datetime.now(),
        embedding_model="BAAI/bge-base-en-v1.5",
        chunk_size=512,
        search_strategy="hybrid",
    )


# --- Mock Data Fixtures ---


@pytest.fixture
def sample_documents():
    """Sample documents for testing ingestion."""
    return [
        {"text": "The quick brown fox jumps over the lazy dog. " * 10, "id": 1},
        {"text": "Machine learning is a subset of artificial intelligence. " * 10, "id": 2},
        {"text": "Vector databases are optimized for similarity search. " * 10, "id": 3},
        {
            "text": "Natural language processing enables computers to understand text. " * 10,
            "id": 4,
        },
        {"text": "Deep learning uses neural networks with multiple layers. " * 10, "id": 5},
    ]


@pytest.fixture
def mock_hf_dataset(sample_documents):
    """Mock HuggingFace dataset for testing."""

    def _dataset_generator():
        for doc in sample_documents:
            yield doc

    return _dataset_generator


# --- Utility Fixtures ---


@pytest.fixture
def cleanup_collections(in_memory_engine):
    """Cleanup fixture to delete test collections after tests."""
    created_collections = []

    def track_collection(name):
        created_collections.append(name)
        return name

    yield track_collection

    # Cleanup
    for name in created_collections:
        try:
            if in_memory_engine.collection_exists(name):
                in_memory_engine.client.delete_collection(name)
        except:
            pass
