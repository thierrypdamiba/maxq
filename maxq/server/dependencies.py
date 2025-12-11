import os
from functools import lru_cache
from maxq.search_engine import MaxQEngine

@lru_cache()
def get_engine():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    return MaxQEngine(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        openai_api_key=openai_api_key
    )
