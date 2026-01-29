# MaxQ Configuration
# This file contains service credentials and cloud inference settings

import os
from pathlib import Path

# Persistence Directory (~/.maxq)
MAXQ_APP_DIR = Path.home() / ".maxq"
MAXQ_APP_DIR.mkdir(exist_ok=True)

# =============================================================================
# Qdrant Cloud Inference Settings (Required)
# =============================================================================
# Set these environment variables:
#   export QDRANT_URL="https://your-cluster.cloud.qdrant.io"
#   export QDRANT_API_KEY="your-api-key"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Cloud Inference Model Defaults
# Supported dense models: sentence-transformers/all-MiniLM-L6-v2, mixedbread-ai/mxbai-embed-large-v1
# Supported sparse models: Qdrant/bm25, prithivida/Splade_PP_en_v1
DEFAULT_DENSE_MODEL = os.getenv("MAXQ_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SPARSE_MODEL = os.getenv("MAXQ_SPARSE_MODEL", "Qdrant/bm25")

# Model dimension mapping for cloud inference
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-minilm-l6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}

# =============================================================================
# Optional Services
# =============================================================================
# Linkup API Key (provided by MaxQ for natural language dataset search)
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")

# OpenAI API Key (optional, for LLM features like HyDE, RAG evaluation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

