from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"

class IndexedModelInfo(BaseModel):
    """Info about an indexed collection with dataset and vectors"""
    model_name: str  # Dense model (primary identifier)
    collection_name: str
    indexed_at: datetime
    point_count: int = 0
    # New fields for better display
    dataset_name: Optional[str] = None  # HuggingFace dataset or local source
    sparse_models: List[str] = []  # List of sparse models in this collection

class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    status: ProjectStatus = ProjectStatus.ACTIVE
    # Metadata like task type (Legal, Code, Web)
    task_type: Optional[str] = None
    embedding_model: Optional[str] = "BAAI/bge-base-en-v1.5"
    last_accessed: Optional[datetime] = None
    # Track all embedding models that have been used to index data
    indexed_models: List[IndexedModelInfo] = []

class Dataset(BaseModel):
    id: str
    project_id: str
    name: str
    source_type: str  # upload, huggingface, etc.
    status: str
    doc_count: int = 0

# Add more models as needed

class Experiment(BaseModel):
    id: str
    project_id: str
    name: str
    status: str = "pending" # pending, running, completed, failed
    created_at: datetime
    # Configuration Snapshot
    embedding_model: str
    chunk_size: int = 512
    search_strategy: str = "hybrid"
    
    # Progress Tracking
    progress_current: int = 0
    progress_total: int = 0
    progress_message: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Results (Summary)
    metrics: Optional[Dict[str, Any]] = None

class EvalResult(BaseModel):
    id: str
    experiment_id: str
    metrics: Dict[str, float] # e.g. {"context_recall": 0.85, "context_precision": 0.9}
    details: Optional[Dict[str, Any]] = None # Detailed Ragas output
