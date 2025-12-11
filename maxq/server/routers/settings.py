"""
Settings router for API key management.
"""

import os
import logging
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("maxq.server.settings")

router = APIRouter(prefix="/settings", tags=["settings"])

# Store keys in ~/.maxq/.env file
MAXQ_DIR = Path.home() / ".maxq"
ENV_FILE = MAXQ_DIR / ".env"


class APIKeys(BaseModel):
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    linkup_api_key: Optional[str] = None


class APIKeysStatus(BaseModel):
    qdrant_url: Optional[str] = None
    qdrant_configured: bool = False
    openai_configured: bool = False
    linkup_configured: bool = False


def _load_env_file() -> dict:
    """Load existing .env file into a dict."""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def _save_env_file(env_vars: dict):
    """Save dict to .env file."""
    MAXQ_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w") as f:
        for key, value in env_vars.items():
            if value:
                f.write(f'{key}="{value}"\n')


@router.get("/api-keys", response_model=APIKeysStatus)
async def get_api_keys_status():
    """
    Get the current API key configuration status.
    Returns whether keys are configured (not the actual keys for security).
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    linkup_api_key = os.getenv("LINKUP_API_KEY")

    return APIKeysStatus(
        qdrant_url=qdrant_url,
        qdrant_configured=bool(qdrant_url and qdrant_api_key),
        openai_configured=bool(openai_api_key),
        linkup_configured=bool(linkup_api_key),
    )


@router.post("/api-keys")
async def save_api_keys(keys: APIKeys):
    """
    Save API keys to environment and persist to ~/.maxq/.env file.
    Keys are applied immediately to the current process.
    """
    try:
        # Load existing env vars
        env_vars = _load_env_file()

        # Update with new values (only if provided)
        if keys.qdrant_url is not None:
            env_vars["QDRANT_URL"] = keys.qdrant_url
            os.environ["QDRANT_URL"] = keys.qdrant_url

        if keys.qdrant_api_key is not None:
            env_vars["QDRANT_API_KEY"] = keys.qdrant_api_key
            os.environ["QDRANT_API_KEY"] = keys.qdrant_api_key

        if keys.openai_api_key is not None:
            env_vars["OPENAI_API_KEY"] = keys.openai_api_key
            os.environ["OPENAI_API_KEY"] = keys.openai_api_key

        if keys.linkup_api_key is not None:
            env_vars["LINKUP_API_KEY"] = keys.linkup_api_key
            os.environ["LINKUP_API_KEY"] = keys.linkup_api_key

        # Save to file
        _save_env_file(env_vars)

        # Clear the cached engine so it picks up new keys
        from maxq.server.dependencies import get_engine

        get_engine.cache_clear()

        logger.info("API keys updated successfully")

        return {
            "status": "success",
            "message": "API keys saved successfully",
            "qdrant_configured": bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "linkup_configured": bool(os.getenv("LINKUP_API_KEY")),
        }

    except Exception as e:
        logger.error(f"Failed to save API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api-keys/test")
async def test_api_keys():
    """
    Test the current API key configuration by attempting to connect to services.
    """
    results = {
        "qdrant": {"status": "not_configured", "message": None},
        "openai": {"status": "not_configured", "message": None},
        "linkup": {"status": "not_configured", "message": None},
    }

    # Test Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10)
            client.get_collections()
            results["qdrant"] = {
                "status": "connected",
                "message": "Successfully connected to Qdrant",
            }
        except Exception as e:
            results["qdrant"] = {"status": "error", "message": str(e)[:100]}

    # Test OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        try:
            import openai

            client = openai.OpenAI(api_key=openai_api_key)
            # Just validate the key format, don't make an actual API call
            if openai_api_key.startswith("sk-"):
                results["openai"] = {"status": "configured", "message": "API key format valid"}
            else:
                results["openai"] = {
                    "status": "warning",
                    "message": "API key format may be invalid",
                }
        except Exception as e:
            results["openai"] = {"status": "error", "message": str(e)[:100]}

    # Test Linkup
    linkup_api_key = os.getenv("LINKUP_API_KEY")

    if linkup_api_key:
        # Just validate the key is present, don't make API call
        if len(linkup_api_key) > 10:
            results["linkup"] = {"status": "configured", "message": "API key configured"}
        else:
            results["linkup"] = {"status": "warning", "message": "API key may be too short"}

    return results
