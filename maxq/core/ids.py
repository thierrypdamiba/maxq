"""Stable ID generation for chunks."""

import hashlib
import re


def normalize_text(text: str) -> str:
    """Normalize text for consistent hashing.

    - Strip leading/trailing whitespace
    - Collapse multiple whitespace to single space
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def hash8(text: str) -> str:
    """Generate 8-character hash of text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Generate stable chunk ID.

    Format: chunk:{doc_id}:{chunk_index}:{hash8(normalized_text)}

    This ensures:
    - Same text always produces same ID
    - Tiny whitespace differences produce same ID (after normalization)
    - IDs are human-readable and debuggable
    """
    text_hash = hash8(text)
    return f"chunk:{doc_id}:{chunk_index}:{text_hash}"
