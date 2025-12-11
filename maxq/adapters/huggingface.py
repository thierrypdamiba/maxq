"""HuggingFace dataset loading adapter."""

from typing import Iterator

from maxq.core.types import Document


def load_dataset_from_huggingface(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    id_column: str | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Load documents from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "squad", "wiki_qa")
        split: Dataset split to load (default: "train")
        text_column: Column name containing text (default: "text")
        id_column: Column name for document ID (optional, auto-generated if None)
        limit: Maximum number of documents to load (optional)

    Returns:
        List of Document objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed. "
            "Install with: pip install datasets"
        )

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    documents = []
    for i, row in enumerate(dataset):
        if limit and i >= limit:
            break

        # Get document ID
        if id_column and id_column in row:
            doc_id = str(row[id_column])
        else:
            doc_id = f"{dataset_name}_{split}_{i}"

        # Get text
        if text_column not in row:
            # Try common column names
            for col in ["text", "content", "passage", "context", "document"]:
                if col in row:
                    text_column = col
                    break
            else:
                raise ValueError(
                    f"Could not find text column. Available: {list(row.keys())}"
                )

        text = str(row[text_column])

        # Build metadata from other columns
        metadata = {
            k: v for k, v in row.items()
            if k not in [text_column, id_column] and v is not None
        }

        doc = Document(
            id=doc_id,
            text=text,
            source=dataset_name,
            metadata=metadata,
        )
        documents.append(doc)

    return documents


def stream_dataset_from_huggingface(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    id_column: str | None = None,
) -> Iterator[Document]:
    """Stream documents from a HuggingFace dataset.

    Memory-efficient streaming for large datasets.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        text_column: Column name containing text
        id_column: Column name for document ID

    Yields:
        Document objects one at a time
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed. "
            "Install with: pip install datasets"
        )

    # Load dataset in streaming mode
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    for i, row in enumerate(dataset):
        # Get document ID
        if id_column and id_column in row:
            doc_id = str(row[id_column])
        else:
            doc_id = f"{dataset_name}_{split}_{i}"

        # Get text
        actual_text_col = text_column
        if text_column not in row:
            for col in ["text", "content", "passage", "context", "document"]:
                if col in row:
                    actual_text_col = col
                    break

        text = str(row.get(actual_text_col, ""))

        # Build metadata
        metadata = {
            k: v for k, v in row.items()
            if k not in [actual_text_col, id_column] and v is not None
        }

        yield Document(
            id=doc_id,
            text=text,
            source=dataset_name,
            metadata=metadata,
        )
