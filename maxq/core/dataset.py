"""Dataset loading utilities."""

import json
from pathlib import Path

from maxq.core.types import Document, EvalQuery


def load_documents(path: str | Path) -> list[Document]:
    """Load documents from JSONL file.

    Expected format per line:
    {"id": "doc1", "text": "...", "source": "..."}
    """
    path = Path(path)
    documents = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc = Document(
                id=data["id"],
                text=data["text"],
                source=data.get("source", ""),
                metadata=data.get("metadata", {}),
            )
            documents.append(doc)

    return documents


def load_queries(path: str | Path) -> list[EvalQuery]:
    """Load evaluation queries from JSONL file.

    Expected format per line:
    {"id": "q1", "query": "...", "relevant_doc_ids": [...]}
    """
    path = Path(path)
    queries = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query = EvalQuery(
                id=data["id"],
                query=data["query"],
                relevant_doc_ids=data.get("relevant_doc_ids", []),
                relevant_ids=data.get("relevant_ids", []),
            )
            queries.append(query)

    return queries
