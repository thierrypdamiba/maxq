"""Text chunking utilities."""

from maxq.core.ids import make_chunk_id
from maxq.core.types import Chunk, Document


def chunk_document(
    doc: Document,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk a document into overlapping segments.

    Uses character-based chunking with overlap.
    """
    text = doc.text
    chunks = []

    if len(text) <= chunk_size:
        # Single chunk for short documents
        chunk_id = make_chunk_id(doc.id, 0, text)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc.id,
            text=text,
            start=0,
            end=len(text),
            metadata=doc.metadata,
        ))
        return chunks

    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunk_id = make_chunk_id(doc.id, chunk_index, chunk_text)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc.id,
            text=chunk_text,
            start=start,
            end=end,
            metadata=doc.metadata,
        ))

        # Move start forward by (chunk_size - overlap)
        start += chunk_size - overlap
        chunk_index += 1

        # Avoid tiny final chunks
        if len(text) - start < overlap:
            break

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk multiple documents."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks
