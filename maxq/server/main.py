import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

# Load environment variables - try multiple locations
# 1. Current working directory
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
# 2. ~/.maxq/.env (where maxq stores credentials)
maxq_env = Path.home() / ".maxq" / ".env"
if maxq_env.exists():
    load_dotenv(maxq_env)
# 3. Parent directory .env
parent_env = Path.cwd().parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maxq.server")

# Version
MAXQ_VERSION = "0.0.1"

app = FastAPI(
    title="MaxQ Studio API",
    version=MAXQ_VERSION,
    description="Vector search tuning and evaluation platform"
)

# Configure CORS - explicitly allow X-API-Key header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3333", "http://127.0.0.1:3333", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    expose_headers=["X-API-Key"],
)

# ============================================
# Rate Limiting (Optional - using slowapi if available)
# ============================================
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMITING_ENABLED = True
    logger.info("Rate limiting enabled")
except ImportError:
    limiter = None
    RATE_LIMITING_ENABLED = False
    logger.info("Rate limiting disabled (slowapi not installed)")


# ============================================
# Error Handling Middleware
# ============================================
class ErrorResponse(BaseModel):
    error: str
    code: str
    retryable: bool = False
    detail: Optional[str] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for consistent error responses."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Don't modify HTTPExceptions
    if isinstance(exc, HTTPException):
        raise exc

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "internal_error",
            "retryable": True,
            "detail": str(exc) if os.getenv("DEBUG") else None
        }
    )


# ============================================
# Health Check Endpoint
# ============================================
@app.get("/health")
async def health():
    """
    System health check endpoint.

    Returns status of MaxQ and its dependencies.
    """
    from maxq.server.dependencies import get_engine as get_search_engine

    # Check Qdrant connection
    qdrant_status = "unknown"
    qdrant_url = os.getenv("QDRANT_URL", "Not configured")

    try:
        search_engine = get_search_engine()
        # Try to list collections as health check
        search_engine.client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)[:50]}"

    return {
        "status": "ok" if qdrant_status == "healthy" else "degraded",
        "version": MAXQ_VERSION,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "qdrant": {
                "status": qdrant_status,
                "url": qdrant_url,
                "mode": "cloud_inference"
            }
        },
        "rate_limiting": RATE_LIMITING_ENABLED
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MaxQ Studio API is running",
        "version": MAXQ_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# ============================================
# Include Routers
# ============================================
from maxq.server.routers import projects, settings, evals, tuning
from maxq.server.routers import index as simple_index
from maxq.server.routers import search as simple_search
from maxq.server.routers import runs, jobs
from maxq.server.routers import chat, cleanup

# Run database migrations on startup
from maxq.db.migrations import run_migrations
run_migrations()

# Core routers
app.include_router(settings.router)
app.include_router(projects.router)

# New run-based API (from maxq3)
app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

# Simple indexing and search (replaces complex datasets, playground, indexing)
app.include_router(simple_index.router)
app.include_router(simple_search.router)

# Legacy routers for tuning/evals pages
app.include_router(evals.router)
app.include_router(tuning.router)

# Chat and cleanup agents
app.include_router(chat.router)
app.include_router(cleanup.router)
