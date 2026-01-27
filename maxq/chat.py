"""
MaxQ Chat Agent - Chat with your Qdrant clusters, collections, and points.

Supports three scopes:
- Cluster: overview of all collections, health, stats
- Collection: RAG-powered chat with collection data
- Point: explain a specific point, find similar points
"""

import json
from enum import Enum
from typing import List, Optional, Generator, Any

from pydantic import BaseModel
from qdrant_client import models
from qdrant_client.models import Document


class ChatScope(str, Enum):
    CLUSTER = "cluster"
    COLLECTION = "collection"
    POINT = "point"


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    message: str
    scope: ChatScope = ChatScope.CLUSTER
    collection_name: Optional[str] = None
    point_id: Optional[str] = None
    history: List[ChatMessage] = []


class ChatAgent:
    """Chat agent that can converse about Qdrant clusters, collections, and points."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.client = engine.client
        self.llm = engine.llm_client

    def chat(self, request: ChatRequest) -> Generator[str, None, None]:
        """Route to the appropriate handler and yield streaming tokens."""
        if not self.llm:
            yield "Error: OpenAI API key not configured. Set OPENAI_API_KEY to use chat."
            return

        if request.scope == ChatScope.CLUSTER:
            yield from self._chat_cluster(request)
        elif request.scope == ChatScope.COLLECTION:
            if not request.collection_name:
                yield "Error: collection_name is required for collection chat."
                return
            yield from self._chat_collection(request)
        elif request.scope == ChatScope.POINT:
            if not request.collection_name or not request.point_id:
                yield "Error: collection_name and point_id are required for point chat."
                return
            yield from self._chat_point(request)

    def _chat_cluster(self, request: ChatRequest) -> Generator[str, None, None]:
        """Chat about the entire cluster — collections, stats, health."""
        collections = self.client.get_collections().collections
        cluster_info = []
        for col in collections:
            try:
                info = self.client.get_collection(col.name)
                cluster_info.append({
                    "name": col.name,
                    "points": info.points_count,
                    "vectors": info.vectors_count,
                    "status": str(info.status),
                    "segments": info.segments_count,
                })
            except Exception:
                cluster_info.append({"name": col.name, "error": "could not fetch info"})

        system_prompt = (
            "You are MaxQ, a vector search assistant. You have access to a Qdrant cluster. "
            "Answer questions about the cluster using the data below. Be concise and helpful.\n\n"
            f"Cluster has {len(collections)} collections:\n"
            f"{json.dumps(cluster_info, indent=2)}"
        )

        yield from self._stream_llm(system_prompt, request)

    def _chat_collection(self, request: ChatRequest) -> Generator[str, None, None]:
        """RAG chat — search the collection and use results as context."""
        collection_name = request.collection_name

        # Check collection exists
        if not self.client.collection_exists(collection_name):
            yield f"Error: Collection '{collection_name}' not found."
            return

        # Get collection info for context
        info = self.client.get_collection(collection_name)

        # Search for relevant documents using the user's message
        try:
            # Get vector config to find model names
            vector_config = info.config.params.vectors
            dense_model = None
            sparse_model = None

            # Try to find documents via hybrid search
            # Use default models since we can't always determine the model from config
            results = self.client.query_points(
                collection_name=collection_name,
                query=Document(text=request.message, model="sentence-transformers/all-MiniLM-L6-v2"),
                using="dense",
                limit=5,
                with_payload=True,
            ).points
        except Exception:
            # Fallback: just scroll some sample points
            results, _ = self.client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
            )

        # Build context from results
        context_docs = []
        for i, point in enumerate(results):
            payload = point.payload or {}
            text = payload.get("_text", payload.get("text", str(payload)[:500]))
            context_docs.append(f"Document {i + 1} (id={point.id}):\n{text}")

        context = "\n\n".join(context_docs)

        system_prompt = (
            "You are MaxQ, a vector search assistant. You are chatting about a Qdrant collection.\n\n"
            f"Collection: {collection_name}\n"
            f"Total points: {info.points_count}\n"
            f"Status: {info.status}\n\n"
            "Below are the most relevant documents from the collection for the user's query. "
            "Use them to answer. If the documents don't contain enough info, say so.\n\n"
            f"{context}"
        )

        yield from self._stream_llm(system_prompt, request)

    def _chat_point(self, request: ChatRequest) -> Generator[str, None, None]:
        """Chat about a specific point — explain it, find similar points."""
        collection_name = request.collection_name
        point_id = request.point_id

        if not self.client.collection_exists(collection_name):
            yield f"Error: Collection '{collection_name}' not found."
            return

        # Retrieve the specific point
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            # Try as integer ID
            try:
                points = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[int(point_id)],
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                yield f"Error: Could not retrieve point '{point_id}': {e}"
                return

        if not points:
            yield f"Error: Point '{point_id}' not found in '{collection_name}'."
            return

        point = points[0]
        payload = point.payload or {}
        point_text = payload.get("_text", payload.get("text", json.dumps(payload, default=str)[:1000]))

        # Find similar points using the point's text
        similar_docs = []
        try:
            similar = self.client.query_points(
                collection_name=collection_name,
                query=Document(text=point_text[:500], model="sentence-transformers/all-MiniLM-L6-v2"),
                using="dense",
                limit=5,
                with_payload=True,
            ).points
            for s in similar:
                if str(s.id) != str(point_id):
                    sp = s.payload or {}
                    similar_docs.append(f"- (id={s.id}, score={s.score:.3f}) {sp.get('_text', str(sp)[:200])[:200]}")
        except Exception:
            pass

        similar_text = "\n".join(similar_docs[:4]) if similar_docs else "No similar points found."

        system_prompt = (
            "You are MaxQ, a vector search assistant. The user is examining a specific point in a Qdrant collection.\n\n"
            f"Collection: {collection_name}\n"
            f"Point ID: {point_id}\n"
            f"Point payload:\n{json.dumps(payload, default=str, indent=2)[:2000]}\n\n"
            f"Similar points:\n{similar_text}\n\n"
            "Help the user understand this point. Explain what it contains, "
            "how it relates to similar points, and answer their questions."
        )

        yield from self._stream_llm(system_prompt, request)

    def _stream_llm(self, system_prompt: str, request: ChatRequest) -> Generator[str, None, None]:
        """Stream LLM response with conversation history."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in request.history[-10:]:  # Last 10 messages
            messages.append({"role": msg.role, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
