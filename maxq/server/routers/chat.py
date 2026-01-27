"""
Chat router - conversational interface to Qdrant data.
Supports SSE streaming for real-time responses.
"""

import json
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from maxq.chat import ChatAgent, ChatRequest, ChatScope, ChatMessage

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


class ChatRequestBody(BaseModel):
    message: str
    scope: str = "cluster"
    collection_name: Optional[str] = None
    point_id: Optional[str] = None
    history: List[dict] = []


@router.post("/stream")
async def chat_stream(body: ChatRequestBody):
    """Stream a chat response via SSE."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = ChatAgent(engine)

    # Convert history dicts to ChatMessage objects
    history = []
    for msg in body.history:
        if "role" in msg and "content" in msg:
            history.append(ChatMessage(role=msg["role"], content=msg["content"]))

    try:
        scope = ChatScope(body.scope)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid scope: {body.scope}. Use cluster, collection, or point.")

    chat_req = ChatRequest(
        message=body.message,
        scope=scope,
        collection_name=body.collection_name,
        point_id=body.point_id,
        history=history,
    )

    def event_generator():
        try:
            for token in agent.chat(chat_req):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/message")
async def chat_message(body: ChatRequestBody):
    """Non-streaming chat — returns full response at once."""
    from maxq.server.dependencies import get_engine

    try:
        engine = get_engine()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine not available: {e}")

    agent = ChatAgent(engine)

    history = []
    for msg in body.history:
        if "role" in msg and "content" in msg:
            history.append(ChatMessage(role=msg["role"], content=msg["content"]))

    try:
        scope = ChatScope(body.scope)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid scope: {body.scope}")

    chat_req = ChatRequest(
        message=body.message,
        scope=scope,
        collection_name=body.collection_name,
        point_id=body.point_id,
        history=history,
    )

    tokens = []
    for token in agent.chat(chat_req):
        tokens.append(token)

    return {"response": "".join(tokens), "scope": body.scope}
