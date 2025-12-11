"""
MaxQ Data Sources Module

Handles multi-dataset collection management with:
- Conflict detection (embedding model mismatch, dimension conflicts)
- Multiple import sources (HuggingFace, Qdrant snapshots, S3, URLs, local files)
- Natural language dataset search via Linkup API

Supports adding multiple datasets to a single collection while
detecting and preventing configuration conflicts.
"""

import os
import json
import uuid
import tempfile
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# DATA SOURCE TYPES
# =============================================================================

class DataSourceType(str, Enum):
    """Supported data source types."""
    HUGGINGFACE = "huggingface"
    QDRANT_SNAPSHOT = "qdrant_snapshot"
    QDRANT_COLLECTION = "qdrant_collection"
    S3_BUCKET = "s3"
    URL = "url"
    LOCAL_FILE = "local_file"
    LOCAL_FOLDER = "local_folder"
    LINKUP_SEARCH = "linkup_search"


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""
    source_type: DataSourceType
    source_path: str  # URL, path, dataset name, etc.
    name: str = ""  # Display name
    embedding_column: Optional[str] = None
    metadata_columns: List[str] = Field(default_factory=list)
    limit: Optional[int] = None

    # S3 specific
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: Optional[str] = None

    # Qdrant specific
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None


class DatasetVersion(BaseModel):
    """Version information for a dataset."""
    version: str  # Semantic version (e.g., "1.0.0", "1.1.0")
    created_at: str
    document_count: int
    description: str = ""
    is_current: bool = True

    # Change tracking
    previous_version: Optional[str] = None
    change_type: Literal["initial", "update", "patch", "reindex"] = "initial"
    changes_summary: str = ""

    # Data integrity
    content_hash: str = ""  # Hash of document content for change detection


class DatasetInfo(BaseModel):
    """Metadata about a dataset in a collection."""
    dataset_id: str
    name: str
    source_type: DataSourceType
    source_path: str
    added_at: str
    document_count: int
    embedding_model: str
    embedding_dimensions: int
    embedding_column: Optional[str] = None

    # Versioning
    version: str = "1.0.0"
    version_history: List[DatasetVersion] = Field(default_factory=list)

    # For conflict detection
    dense_model_hash: str = ""
    sparse_model_hash: str = ""

    def get_next_version(self, change_type: str = "update") -> str:
        """Calculate the next version number based on change type."""
        parts = self.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if change_type == "major" or change_type == "reindex":
            return f"{major + 1}.0.0"
        elif change_type == "update" or change_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"

    def add_version(
        self,
        new_version: str,
        document_count: int,
        change_type: str = "update",
        description: str = "",
        content_hash: str = ""
    ) -> "DatasetInfo":
        """Add a new version to the history."""
        # Mark old versions as not current
        for v in self.version_history:
            v.is_current = False

        # Create new version entry
        new_version_entry = DatasetVersion(
            version=new_version,
            created_at=datetime.now().isoformat(),
            document_count=document_count,
            description=description,
            is_current=True,
            previous_version=self.version,
            change_type=change_type,
            changes_summary=f"{change_type}: {description}" if description else change_type,
            content_hash=content_hash
        )

        self.version_history.append(new_version_entry)
        self.version = new_version
        self.document_count = document_count

        return self


class ConflictType(str, Enum):
    """Types of conflicts that can occur."""
    EMBEDDING_MODEL_MISMATCH = "embedding_model_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    DISTANCE_METRIC_MISMATCH = "distance_metric_mismatch"
    SPARSE_MODEL_MISMATCH = "sparse_model_mismatch"
    QUANTIZATION_MISMATCH = "quantization_mismatch"


class ConflictReport(BaseModel):
    """Report of detected conflicts."""
    has_conflicts: bool
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    can_proceed: bool = True  # Some conflicts are warnings, not blockers
    resolution_options: List[str] = Field(default_factory=list)


class CollectionMetadata(BaseModel):
    """Metadata for a multi-dataset collection."""
    collection_name: str
    created_at: str
    datasets: List[DatasetInfo] = Field(default_factory=list)

    # Collection config (must match for all datasets)
    dense_model: str
    dense_dimensions: int
    sparse_model: str
    distance_metric: str = "Cosine"
    quantization: Optional[str] = None

    total_documents: int = 0


# =============================================================================
# CONFLICT DETECTION
# =============================================================================

class ConflictDetector:
    """Detects conflicts when adding datasets to a collection."""

    def __init__(self, qdrant_client=None):
        self.client = qdrant_client

    def check_collection_compatibility(
        self,
        collection_name: str,
        new_dataset_config: Dict[str, Any]
    ) -> ConflictReport:
        """
        Check if a new dataset is compatible with an existing collection.

        Args:
            collection_name: Name of the existing collection
            new_dataset_config: Config for the new dataset being added

        Returns:
            ConflictReport with any detected issues
        """
        conflicts = []
        warnings = []
        can_proceed = True
        resolution_options = []

        if not self.client:
            return ConflictReport(
                has_conflicts=False,
                can_proceed=True,
                warnings=["No Qdrant client - skipping compatibility check"]
            )

        try:
            # Get existing collection info
            if not self.client.collection_exists(collection_name):
                return ConflictReport(
                    has_conflicts=False,
                    can_proceed=True,
                    warnings=["Collection does not exist - will be created"]
                )

            collection_info = self.client.get_collection(collection_name)
            params = collection_info.config.params

            # Check dense vector dimensions
            existing_dense_size = None
            if isinstance(params.vectors, dict) and "dense" in params.vectors:
                existing_dense_size = params.vectors["dense"].size
            elif hasattr(params.vectors, "size"):
                existing_dense_size = params.vectors.size

            new_dense_size = new_dataset_config.get("dense_dimensions")

            if existing_dense_size and new_dense_size and existing_dense_size != new_dense_size:
                conflicts.append({
                    "type": ConflictType.DIMENSION_MISMATCH,
                    "message": f"Dimension mismatch: collection has {existing_dense_size}d, new dataset uses {new_dense_size}d",
                    "existing": existing_dense_size,
                    "new": new_dense_size,
                    "severity": "error"
                })
                can_proceed = False
                resolution_options.append(f"Use a model with {existing_dense_size} dimensions")
                resolution_options.append("Create a new collection for this dataset")

            # Check distance metric
            existing_distance = None
            if isinstance(params.vectors, dict) and "dense" in params.vectors:
                existing_distance = str(params.vectors["dense"].distance)
            elif hasattr(params.vectors, "distance"):
                existing_distance = str(params.vectors.distance)

            new_distance = new_dataset_config.get("distance_metric", "Cosine")

            if existing_distance and new_distance:
                # Normalize for comparison
                existing_norm = existing_distance.lower().replace("distance.", "")
                new_norm = new_distance.lower()

                if existing_norm != new_norm:
                    conflicts.append({
                        "type": ConflictType.DISTANCE_METRIC_MISMATCH,
                        "message": f"Distance metric mismatch: collection uses {existing_distance}, config uses {new_distance}",
                        "existing": existing_distance,
                        "new": new_distance,
                        "severity": "warning"
                    })
                    warnings.append("Distance metric mismatch may affect search quality")

            # Check embedding model compatibility (from payload metadata if stored)
            # This is a softer check - we warn but allow proceeding
            try:
                # Sample a point to check metadata
                sample = self.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True
                )[0]

                if sample:
                    existing_model = sample[0].payload.get("_embedding_model")
                    new_model = new_dataset_config.get("dense_model")

                    if existing_model and new_model and existing_model != new_model:
                        warnings.append(
                            f"Different embedding models: existing '{existing_model}', new '{new_model}'. "
                            "Search results may be inconsistent."
                        )
                        conflicts.append({
                            "type": ConflictType.EMBEDDING_MODEL_MISMATCH,
                            "message": f"Model mismatch: {existing_model} vs {new_model}",
                            "existing": existing_model,
                            "new": new_model,
                            "severity": "warning"
                        })
            except Exception:
                pass  # Metadata check is optional

        except Exception as e:
            warnings.append(f"Could not fully check compatibility: {e}")

        return ConflictReport(
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts,
            warnings=warnings,
            can_proceed=can_proceed,
            resolution_options=resolution_options
        )

    def check_eval_compatibility(
        self,
        collection_name: str
    ) -> ConflictReport:
        """
        Check if a multi-dataset collection is suitable for evaluation.

        Evaluations may be unreliable if:
        - Multiple embedding models were used
        - Documents have different schemas
        - Large variance in document lengths
        """
        warnings = []

        if not self.client or not self.client.collection_exists(collection_name):
            return ConflictReport(
                has_conflicts=False,
                can_proceed=True
            )

        try:
            # Sample documents to check consistency
            samples, _ = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True
            )

            if not samples:
                return ConflictReport(has_conflicts=False, can_proceed=True)

            # Check for multiple datasets
            dataset_ids = set()
            embedding_models = set()

            for point in samples:
                payload = point.payload or {}
                if "_dataset_id" in payload:
                    dataset_ids.add(payload["_dataset_id"])
                if "_embedding_model" in payload:
                    embedding_models.add(payload["_embedding_model"])

            if len(dataset_ids) > 1:
                warnings.append(
                    f"Collection contains {len(dataset_ids)} datasets. "
                    "Evaluation metrics may not be directly comparable across datasets."
                )

            if len(embedding_models) > 1:
                warnings.append(
                    f"Collection uses {len(embedding_models)} different embedding models: "
                    f"{', '.join(embedding_models)}. This may affect evaluation accuracy."
                )
                return ConflictReport(
                    has_conflicts=True,
                    conflicts=[{
                        "type": ConflictType.EMBEDDING_MODEL_MISMATCH,
                        "message": "Multiple embedding models detected",
                        "models": list(embedding_models),
                        "severity": "warning"
                    }],
                    warnings=warnings,
                    can_proceed=True  # Can still eval, just with caveats
                )

        except Exception as e:
            warnings.append(f"Could not check eval compatibility: {e}")

        return ConflictReport(
            has_conflicts=False,
            warnings=warnings,
            can_proceed=True
        )


# =============================================================================
# DATA SOURCE MANAGER
# =============================================================================

class DataSourceManager:
    """
    Manages multiple data sources for a collection.
    Handles importing from various sources and tracks dataset metadata.
    """

    def __init__(
        self,
        qdrant_client=None,
        openai_api_key: str = None,
        linkup_api_key: str = None
    ):
        self.client = qdrant_client
        self.openai_api_key = openai_api_key
        self.linkup_api_key = linkup_api_key
        self.conflict_detector = ConflictDetector(qdrant_client)

        # Collection metadata cache
        self._metadata_cache: Dict[str, CollectionMetadata] = {}

    def _generate_dataset_id(self, source_path: str) -> str:
        """Generate a unique dataset ID."""
        hash_input = f"{source_path}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def get_collection_metadata(self, collection_name: str) -> Optional[CollectionMetadata]:
        """Get metadata for a collection, including all datasets."""
        if collection_name in self._metadata_cache:
            return self._metadata_cache[collection_name]

        if not self.client or not self.client.collection_exists(collection_name):
            return None

        try:
            # Try to read metadata from collection
            collection_info = self.client.get_collection(collection_name)

            # Sample to get dataset info
            samples, _ = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True
            )

            # Extract dataset IDs
            dataset_ids = set()
            for point in samples:
                if point.payload and "_dataset_id" in point.payload:
                    dataset_ids.add(point.payload["_dataset_id"])

            # Get dimensions
            params = collection_info.config.params
            dense_size = 768  # default
            if isinstance(params.vectors, dict) and "dense" in params.vectors:
                dense_size = params.vectors["dense"].size
            elif hasattr(params.vectors, "size"):
                dense_size = params.vectors.size

            metadata = CollectionMetadata(
                collection_name=collection_name,
                created_at=datetime.now().isoformat(),
                dense_model="unknown",
                dense_dimensions=dense_size,
                sparse_model="unknown",
                total_documents=self.client.count(collection_name).count
            )

            self._metadata_cache[collection_name] = metadata
            return metadata

        except Exception as e:
            print(f"Error getting collection metadata: {e}")
            return None

    def list_datasets_in_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """List all datasets in a collection with version info."""
        if not self.client or not self.client.collection_exists(collection_name):
            return []

        try:
            samples, _ = self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True
            )

            datasets = {}
            for point in samples:
                payload = point.payload or {}
                dataset_id = payload.get("_dataset_id", "default")

                if dataset_id not in datasets:
                    datasets[dataset_id] = {
                        "dataset_id": dataset_id,
                        "name": payload.get("_dataset_name", dataset_id),
                        "source": payload.get("_source_path", "unknown"),
                        "embedding_model": payload.get("_embedding_model", "unknown"),
                        "version": payload.get("_dataset_version", "1.0.0"),
                        "added_at": payload.get("_added_at", "unknown"),
                        "count": 0
                    }
                datasets[dataset_id]["count"] += 1

            return list(datasets.values())

        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []

    def get_dataset_versions(self, collection_name: str, dataset_id: str) -> List[Dict[str, Any]]:
        """Get version history for a specific dataset."""
        if not self.client or not self.client.collection_exists(collection_name):
            return []

        try:
            # Get all unique versions for this dataset
            samples, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [{"key": "_dataset_id", "match": {"value": dataset_id}}]
                },
                limit=1000,
                with_payload=True
            )

            versions = {}
            for point in samples:
                payload = point.payload or {}
                version = payload.get("_dataset_version", "1.0.0")

                if version not in versions:
                    versions[version] = {
                        "version": version,
                        "added_at": payload.get("_added_at", "unknown"),
                        "document_count": 0,
                        "is_current": payload.get("_is_current_version", True)
                    }
                versions[version]["document_count"] += 1

            # Sort by version
            version_list = list(versions.values())
            version_list.sort(key=lambda x: [int(p) for p in x["version"].split(".")], reverse=True)
            return version_list

        except Exception as e:
            print(f"Error getting dataset versions: {e}")
            return []

    def _compute_content_hash(self, texts: List[str]) -> str:
        """Compute a hash of document content for change detection."""
        combined = "".join(sorted(texts[:100]))  # Sample first 100 for efficiency
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # =========================================================================
    # LINKUP API - NATURAL LANGUAGE DATASET SEARCH
    # =========================================================================

    def search_datasets_nl(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets using natural language via Linkup API.

        Args:
            query: Natural language query (e.g., "dataset for legal eval")
            max_results: Maximum number of results to return

        Returns:
            List of dataset candidates with URLs and descriptions
        """
        if not self.linkup_api_key:
            # Try to get from config
            try:
                from .config import LINKUP_API_KEY
                self.linkup_api_key = LINKUP_API_KEY
            except ImportError:
                pass

        if not self.linkup_api_key:
            return [{
                "error": "Linkup API key not configured",
                "suggestion": "Set LINKUP_API_KEY in your environment or config"
            }]

        try:
            from linkup import LinkupClient

            client = LinkupClient(api_key=self.linkup_api_key)

            # Search for datasets
            search_query = f"Hugging Face dataset for: {query}"
            response = client.search(
                query=search_query,
                depth="standard",
                output_type="searchResults"  # Use searchResults instead of structured
            )

            results = response.results if hasattr(response, 'results') else []

            candidates = []
            seen_urls = set()

            for r in results:
                url = getattr(r, 'url', '')
                if "huggingface.co/datasets/" in url and url not in seen_urls:
                    seen_urls.add(url)

                    # Extract dataset name from URL
                    name = url.split("datasets/")[-1].strip("/")

                    candidates.append({
                        "name": name,
                        "url": url,
                        "title": getattr(r, 'title', name),
                        "description": getattr(r, 'snippet', '')[:200],
                        "source_type": DataSourceType.HUGGINGFACE
                    })

                    if len(candidates) >= max_results:
                        break

            # If no HuggingFace results, try to find other data sources
            if not candidates:
                for r in results[:max_results]:
                    url = getattr(r, 'url', '')
                    if url:
                        candidates.append({
                            "name": getattr(r, 'title', url),
                            "url": url,
                            "description": getattr(r, 'snippet', '')[:200],
                            "source_type": DataSourceType.URL
                        })

            return candidates

        except Exception as e:
            return [{
                "error": f"Linkup search failed: {e}",
                "query": query
            }]

    # =========================================================================
    # IMPORT METHODS
    # =========================================================================

    def import_from_huggingface(
        self,
        dataset_name: str,
        collection_name: str,
        embedding_model: str,
        embedding_column: str = None,
        limit: int = None,
        split: str = "train",
        callback=None
    ) -> Tuple[int, DatasetInfo]:
        """
        Import a HuggingFace dataset into the collection.

        Returns:
            Tuple of (document_count, DatasetInfo)
        """
        from datasets import load_dataset

        dataset_id = self._generate_dataset_id(dataset_name)

        # Load dataset
        ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

        count = 0
        for row in ds:
            if limit and count >= limit:
                break
            count += 1
            if callback and count % 50 == 0:
                callback(50)

        # Create DatasetInfo
        info = DatasetInfo(
            dataset_id=dataset_id,
            name=dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name,
            source_type=DataSourceType.HUGGINGFACE,
            source_path=dataset_name,
            added_at=datetime.now().isoformat(),
            document_count=count,
            embedding_model=embedding_model,
            embedding_dimensions=768,  # Will be updated by actual model
            embedding_column=embedding_column
        )

        return count, info

    def import_from_qdrant_snapshot(
        self,
        snapshot_path: str,
        collection_name: str,
        source_url: str = None,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Import from a Qdrant snapshot file.

        Args:
            snapshot_path: Local path to snapshot file or URL
            collection_name: Target collection name
            source_url: If snapshot_path is a URL, download from here

        Returns:
            Tuple of (document_count, error_message)
        """
        if not self.client:
            return 0, "No Qdrant client configured"

        try:
            # If URL, download first
            if snapshot_path.startswith(("http://", "https://")):
                import requests

                if callback:
                    callback("Downloading snapshot...")

                response = requests.get(snapshot_path, stream=True)
                response.raise_for_status()

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".snapshot") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                    snapshot_path = f.name

            # Recover from snapshot
            if callback:
                callback("Recovering from snapshot...")

            self.client.recover_snapshot(
                collection_name=collection_name,
                location=snapshot_path
            )

            count = self.client.count(collection_name).count
            return count, None

        except Exception as e:
            return 0, str(e)

    def import_from_qdrant_collection(
        self,
        source_url: str,
        source_api_key: str,
        source_collection: str,
        target_collection: str,
        limit: int = None,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Import from another Qdrant collection (remote or local).

        Args:
            source_url: URL of source Qdrant instance
            source_api_key: API key for source
            source_collection: Source collection name
            target_collection: Target collection name
            limit: Max documents to import

        Returns:
            Tuple of (document_count, error_message)
        """
        from qdrant_client import QdrantClient

        try:
            # Connect to source
            source_client = QdrantClient(url=source_url, api_key=source_api_key)

            if not source_client.collection_exists(source_collection):
                return 0, f"Source collection '{source_collection}' not found"

            # Get source collection info
            source_info = source_client.get_collection(source_collection)

            # Scroll and copy
            offset = None
            count = 0
            batch_size = 100

            while True:
                points, offset = source_client.scroll(
                    collection_name=source_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )

                if not points:
                    break

                # Tag with source info
                for point in points:
                    if point.payload is None:
                        point.payload = {}
                    point.payload["_imported_from"] = source_url
                    point.payload["_source_collection"] = source_collection

                # Upsert to target
                self.client.upsert(
                    collection_name=target_collection,
                    points=points,
                    wait=False
                )

                count += len(points)
                if callback:
                    callback(len(points))

                if limit and count >= limit:
                    break

                if offset is None:
                    break

            return count, None

        except Exception as e:
            return 0, str(e)

    def import_from_s3(
        self,
        bucket: str,
        prefix: str,
        collection_name: str,
        embedding_model: str,
        access_key: str = None,
        secret_key: str = None,
        region: str = "us-east-1",
        file_types: List[str] = None,
        limit: int = None,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Import files from S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix (folder path)
            collection_name: Target collection
            embedding_model: Model to use for embedding
            access_key: AWS access key (or use env)
            secret_key: AWS secret key (or use env)
            region: AWS region
            file_types: List of file extensions to include (e.g., ['.txt', '.md'])
            limit: Max files to import

        Returns:
            Tuple of (document_count, error_message)
        """
        try:
            import boto3
        except ImportError:
            return 0, "boto3 not installed. Run: pip install boto3"

        try:
            # Create S3 client
            session_kwargs = {}
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key

            s3 = boto3.client("s3", region_name=region, **session_kwargs)

            # List objects
            paginator = s3.get_paginator("list_objects_v2")

            count = 0
            texts = []
            payloads = []

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]

                    # Filter by file type
                    if file_types:
                        if not any(key.endswith(ext) for ext in file_types):
                            continue

                    # Download and read
                    try:
                        response = s3.get_object(Bucket=bucket, Key=key)
                        content = response["Body"].read().decode("utf-8")

                        if len(content) < 10:
                            continue

                        texts.append(content)
                        payloads.append({
                            "source": f"s3://{bucket}/{key}",
                            "filename": key.split("/")[-1],
                            "_source_type": "s3"
                        })

                        count += 1
                        if callback and count % 10 == 0:
                            callback(10)

                        if limit and count >= limit:
                            break

                    except Exception as e:
                        print(f"Skipping {key}: {e}")

                if limit and count >= limit:
                    break

            return count, None

        except Exception as e:
            return 0, str(e)

    def import_from_url(
        self,
        url: str,
        collection_name: str,
        embedding_model: str,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Import content from a URL (web page, JSON, CSV, etc.)

        Args:
            url: URL to fetch
            collection_name: Target collection
            embedding_model: Model for embedding

        Returns:
            Tuple of (document_count, error_message)
        """
        import requests

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            content = response.text

            documents = []

            # Handle different content types
            if "application/json" in content_type:
                data = response.json()
                if isinstance(data, list):
                    documents = data
                elif isinstance(data, dict):
                    documents = [data]

            elif "text/csv" in content_type:
                import csv
                from io import StringIO
                reader = csv.DictReader(StringIO(content))
                documents = list(reader)

            else:
                # Plain text or HTML
                documents = [{"text": content, "source": url}]

            if callback:
                callback(len(documents))

            return len(documents), None

        except Exception as e:
            return 0, str(e)

    def import_from_local(
        self,
        path: str,
        collection_name: str,
        embedding_model: str,
        glob_pattern: str = "**/*.*",
        file_types: List[str] = None,
        limit: int = None,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Import from local file or folder.

        Args:
            path: File or folder path
            collection_name: Target collection
            embedding_model: Model for embedding
            glob_pattern: Pattern for matching files
            file_types: List of extensions to include
            limit: Max files to import

        Returns:
            Tuple of (document_count, error_message)
        """
        import glob as glob_module

        try:
            path = Path(path)

            if path.is_file():
                files = [path]
            else:
                files = list(path.glob(glob_pattern))

            count = 0

            for file_path in files:
                if not file_path.is_file():
                    continue

                if file_types:
                    if not any(str(file_path).endswith(ext) for ext in file_types):
                        continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    if len(content) < 10:
                        continue

                    count += 1
                    if callback and count % 10 == 0:
                        callback(10)

                    if limit and count >= limit:
                        break

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

            return count, None

        except Exception as e:
            return 0, str(e)


# =============================================================================
# MULTI-DATASET COLLECTION MANAGER
# =============================================================================

class MultiDatasetCollection:
    """
    Manages a collection with multiple datasets.
    Provides methods for adding datasets with conflict detection.
    """

    def __init__(
        self,
        collection_name: str,
        engine,  # MaxQEngine or MaxQAutoEngine
        linkup_api_key: str = None
    ):
        self.collection_name = collection_name
        self.engine = engine
        self.data_manager = DataSourceManager(
            qdrant_client=engine.client,
            openai_api_key=engine.openai_api_key,
            linkup_api_key=linkup_api_key
        )
        self._datasets: List[DatasetInfo] = []

    def check_compatibility(self, new_config: Dict[str, Any]) -> ConflictReport:
        """Check if a new dataset config is compatible with existing collection."""
        return self.data_manager.conflict_detector.check_collection_compatibility(
            self.collection_name,
            new_config
        )

    def check_eval_readiness(self) -> ConflictReport:
        """Check if collection is ready for evaluation."""
        return self.data_manager.conflict_detector.check_eval_compatibility(
            self.collection_name
        )

    def search_datasets(self, query: str) -> List[Dict[str, Any]]:
        """Search for datasets using natural language."""
        return self.data_manager.search_datasets_nl(query)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in this collection."""
        return self.data_manager.list_datasets_in_collection(self.collection_name)

    def add_dataset(
        self,
        source_config: DataSourceConfig,
        config,  # CollectionStrategy or HybridPipelineConfig
        force: bool = False,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Add a dataset to the collection.

        Args:
            source_config: Configuration for the data source
            config: Embedding configuration
            force: Skip conflict checks
            callback: Progress callback

        Returns:
            Tuple of (document_count, error_message)
        """
        # Check compatibility first
        if not force:
            new_config = {
                "dense_model": getattr(config, 'dense_model_name', None) or
                              getattr(getattr(config, 'dense_embedding', None), 'model_name', None),
                "dense_dimensions": getattr(getattr(config, 'dense_embedding', {}).get('params', {}), 'size', 768)
                                    if hasattr(config, 'dense_embedding') else 768,
                "distance_metric": "Cosine"
            }

            conflict_report = self.check_compatibility(new_config)

            if not conflict_report.can_proceed:
                error_msg = "Conflicts detected:\n"
                for conflict in conflict_report.conflicts:
                    error_msg += f"  - {conflict['message']}\n"
                if conflict_report.resolution_options:
                    error_msg += "Resolution options:\n"
                    for option in conflict_report.resolution_options:
                        error_msg += f"  - {option}\n"
                return 0, error_msg

        # Import based on source type
        source_type = source_config.source_type

        if source_type == DataSourceType.HUGGINGFACE:
            return self._add_huggingface(source_config, config, callback)
        elif source_type == DataSourceType.QDRANT_SNAPSHOT:
            return self.data_manager.import_from_qdrant_snapshot(
                source_config.source_path,
                self.collection_name,
                callback=callback
            )
        elif source_type == DataSourceType.QDRANT_COLLECTION:
            return self.data_manager.import_from_qdrant_collection(
                source_config.qdrant_url,
                source_config.qdrant_api_key,
                source_config.source_path,
                self.collection_name,
                limit=source_config.limit,
                callback=callback
            )
        elif source_type == DataSourceType.S3_BUCKET:
            # Parse s3://bucket/prefix
            parts = source_config.source_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            return self.data_manager.import_from_s3(
                bucket=bucket,
                prefix=prefix,
                collection_name=self.collection_name,
                embedding_model=getattr(config, 'dense_model_name', 'unknown'),
                access_key=source_config.s3_access_key,
                secret_key=source_config.s3_secret_key,
                region=source_config.s3_region,
                limit=source_config.limit,
                callback=callback
            )
        elif source_type == DataSourceType.URL:
            return self.data_manager.import_from_url(
                source_config.source_path,
                self.collection_name,
                getattr(config, 'dense_model_name', 'unknown'),
                callback=callback
            )
        elif source_type in [DataSourceType.LOCAL_FILE, DataSourceType.LOCAL_FOLDER]:
            return self.data_manager.import_from_local(
                source_config.source_path,
                self.collection_name,
                getattr(config, 'dense_model_name', 'unknown'),
                limit=source_config.limit,
                callback=callback
            )
        else:
            return 0, f"Unsupported source type: {source_type}"

    def _add_huggingface(
        self,
        source_config: DataSourceConfig,
        config,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """Add a HuggingFace dataset."""
        try:
            # Use engine's ingest method with dataset tagging
            from datasets import load_dataset
            from .search_engine import CollectionStrategy

            # Convert to CollectionStrategy if needed
            if hasattr(config, 'dense_embedding'):
                strategy = CollectionStrategy(
                    collection_name=self.collection_name,
                    dense_model_name=config.dense_embedding.model_name,
                    estimated_doc_count=source_config.limit or 100000
                )
            else:
                strategy = config
                strategy.collection_name = self.collection_name

            dataset_id = self.data_manager._generate_dataset_id(source_config.source_path)

            # Custom ingest with dataset tagging
            count = self._ingest_with_tagging(
                source_config.source_path,
                strategy,
                source_config.limit or 1000,
                dataset_id=dataset_id,
                dataset_name=source_config.name or source_config.source_path,
                embedding_column=source_config.embedding_column,
                callback=callback
            )

            return count, None

        except Exception as e:
            return 0, str(e)

    def _ingest_with_tagging(
        self,
        dataset_name: str,
        config,
        limit: int,
        dataset_id: str,
        dataset_name_display: str,
        embedding_column: str = None,
        version: str = "1.0.0",
        callback=None
    ) -> int:
        """Ingest with dataset tagging and versioning for multi-dataset support."""
        from datasets import load_dataset

        # Initialize collection if needed
        if not self.engine.collection_exists(config.collection_name):
            self.engine.initialize_collection(config)

        self.engine._load_models(config)

        dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        batch_text = []
        batch_payload = []
        count = 0
        added_at = datetime.now().isoformat()

        # Get current max ID
        try:
            existing_count = self.engine.client.count(config.collection_name).count
            start_id = existing_count
        except Exception:
            start_id = 0

        for row in dataset:
            if count >= limit:
                break

            text_val = None
            if embedding_column and embedding_column in row:
                text_val = str(row[embedding_column])
            else:
                text_val = next((str(v) for k, v in row.items() if isinstance(v, str) and len(str(v)) > 20), None)

            if not text_val:
                continue

            # Add dataset metadata with versioning
            payload = dict(row)
            payload["_dataset_id"] = dataset_id
            payload["_dataset_name"] = dataset_name_display
            payload["_source_path"] = dataset_name
            payload["_embedding_model"] = config.dense_model_name
            payload["_dataset_version"] = version
            payload["_added_at"] = added_at
            payload["_is_current_version"] = True

            batch_text.append(text_val)
            batch_payload.append(payload)
            count += 1

            if len(batch_text) >= 50:
                self.engine._upload_batch(config, batch_text, batch_payload, start_id + count - 50)
                if callback:
                    callback(50)
                batch_text, batch_payload = [], []

        if batch_text:
            self.engine._upload_batch(config, batch_text, batch_payload, start_id + count - len(batch_text))
            if callback:
                callback(len(batch_text))

        return count

    def get_dataset_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get version history for a specific dataset."""
        return self.data_manager.get_dataset_versions(self.collection_name, dataset_id)

    def update_dataset(
        self,
        dataset_id: str,
        source_config: DataSourceConfig,
        config,
        change_type: Literal["update", "patch", "reindex"] = "update",
        description: str = "",
        keep_old_versions: bool = True,
        callback=None
    ) -> Tuple[int, Optional[str]]:
        """
        Update an existing dataset with a new version.

        Args:
            dataset_id: ID of the dataset to update
            source_config: New data source configuration
            config: Embedding configuration
            change_type: Type of change (update=minor, patch=patch, reindex=major)
            description: Description of changes
            keep_old_versions: Whether to keep old version documents
            callback: Progress callback

        Returns:
            Tuple of (document_count, error_message)
        """
        # Get current dataset info
        datasets = self.list_datasets()
        current_dataset = None
        for ds in datasets:
            if ds.get("dataset_id") == dataset_id:
                current_dataset = ds
                break

        if not current_dataset:
            return 0, f"Dataset '{dataset_id}' not found in collection"

        # Calculate new version
        current_version = current_dataset.get("version", "1.0.0")
        parts = current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if change_type == "reindex":
            new_version = f"{major + 1}.0.0"
        elif change_type == "update":
            new_version = f"{major}.{minor + 1}.0"
        else:  # patch
            new_version = f"{major}.{minor}.{patch + 1}"

        # Mark old versions as not current if not keeping them
        if not keep_old_versions:
            self._delete_dataset_version(dataset_id, current_version)
        else:
            self._mark_version_not_current(dataset_id, current_version)

        # Import new version
        if source_config.source_type == DataSourceType.HUGGINGFACE:
            from .search_engine import CollectionStrategy

            if hasattr(config, 'dense_embedding'):
                strategy = CollectionStrategy(
                    collection_name=self.collection_name,
                    dense_model_name=config.dense_embedding.model_name,
                    estimated_doc_count=source_config.limit or 100000
                )
            else:
                strategy = config
                strategy.collection_name = self.collection_name

            count = self._ingest_with_tagging(
                source_config.source_path,
                strategy,
                source_config.limit or 1000,
                dataset_id=dataset_id,  # Keep same dataset ID
                dataset_name_display=source_config.name or current_dataset.get("name"),
                embedding_column=source_config.embedding_column,
                version=new_version,
                callback=callback
            )

            return count, None
        else:
            return 0, f"Update not yet supported for source type: {source_config.source_type}"

    def _mark_version_not_current(self, dataset_id: str, version: str):
        """Mark a specific version as not current."""
        if not self.engine.client:
            return

        try:
            from qdrant_client import models

            # Update payload for all points with this dataset_id and version
            self.engine.client.set_payload(
                collection_name=self.collection_name,
                payload={"_is_current_version": False},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="_dataset_id",
                            match=models.MatchValue(value=dataset_id)
                        ),
                        models.FieldCondition(
                            key="_dataset_version",
                            match=models.MatchValue(value=version)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Error marking version as not current: {e}")

    def _delete_dataset_version(self, dataset_id: str, version: str):
        """Delete all documents from a specific dataset version."""
        if not self.engine.client:
            return

        try:
            from qdrant_client import models

            self.engine.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="_dataset_id",
                                match=models.MatchValue(value=dataset_id)
                            ),
                            models.FieldCondition(
                                key="_dataset_version",
                                match=models.MatchValue(value=version)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            print(f"Error deleting dataset version: {e}")

    def rollback_dataset(
        self,
        dataset_id: str,
        target_version: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Rollback a dataset to a previous version.

        Args:
            dataset_id: ID of the dataset
            target_version: Version to rollback to

        Returns:
            Tuple of (success, error_message)
        """
        # Get all versions
        versions = self.get_dataset_versions(dataset_id)
        version_numbers = [v["version"] for v in versions]

        if target_version not in version_numbers:
            return False, f"Version {target_version} not found. Available: {version_numbers}"

        # Find current version
        current_version = None
        for v in versions:
            if v.get("is_current", False):
                current_version = v["version"]
                break

        if current_version == target_version:
            return True, None  # Already at target version

        # Delete versions newer than target
        try:
            for v in versions:
                v_parts = [int(p) for p in v["version"].split(".")]
                t_parts = [int(p) for p in target_version.split(".")]

                # Compare versions
                if v_parts > t_parts:
                    self._delete_dataset_version(dataset_id, v["version"])

            # Mark target version as current
            self._set_version_current(dataset_id, target_version)

            return True, None

        except Exception as e:
            return False, str(e)

    def _set_version_current(self, dataset_id: str, version: str):
        """Set a specific version as the current version."""
        if not self.engine.client:
            return

        try:
            from qdrant_client import models

            # First, mark all versions as not current
            self.engine.client.set_payload(
                collection_name=self.collection_name,
                payload={"_is_current_version": False},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="_dataset_id",
                            match=models.MatchValue(value=dataset_id)
                        )
                    ]
                )
            )

            # Then mark target version as current
            self.engine.client.set_payload(
                collection_name=self.collection_name,
                payload={"_is_current_version": True},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="_dataset_id",
                            match=models.MatchValue(value=dataset_id)
                        ),
                        models.FieldCondition(
                            key="_dataset_version",
                            match=models.MatchValue(value=version)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Error setting version as current: {e}")
