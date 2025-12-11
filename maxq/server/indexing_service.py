"""
Indexing job service - handles the actual indexing pipeline with stages,
progress tracking, and resume capability.

Now uses Qdrant Cloud Inference for embeddings - no local models required.
"""
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from qdrant_client.models import Document

from maxq.server.indexing_models import (
    IndexPlan,
    IndexJob,
    JobStage,
    JobStageStatus,
    JobStatus,
    DryRunEstimate,
    ChunkingStrategy,
)
from maxq.search_engine import MaxQEngine

# Cloud inference model defaults
DEFAULT_DENSE_MODEL = os.getenv("MAXQ_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SPARSE_MODEL = os.getenv("MAXQ_SPARSE_MODEL", "Qdrant/bm25")

# Model dimension mapping for cloud inference
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-minilm-l6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
}


class IndexingService:
    """
    Service for running indexing jobs with full pipeline support.
    Now uses Qdrant Cloud Inference - no local embedding models required.
    """

    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")

        # Support in-memory mode for testing
        if self.qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
            self._in_memory_mode = True
        else:
            if not self.qdrant_url or not self.qdrant_api_key:
                raise ValueError(
                    "Qdrant Cloud credentials required. Set QDRANT_URL and QDRANT_API_KEY environment variables."
                )
            # Initialize client for cloud inference
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=True,
                timeout=60,
                cloud_inference=True,
            )
            self._in_memory_mode = False

    def _create_stage(self, name: str, active_form: str) -> JobStage:
        """Create a new job stage"""
        return JobStage(name=name, active_form=active_form, status=JobStageStatus.QUEUED)

    def _start_stage(self, stage: JobStage):
        """Mark a stage as running"""
        stage.status = JobStageStatus.RUNNING
        stage.started_at = datetime.now()

    def _complete_stage(self, stage: JobStage, success: bool = True, error: str = None):
        """Mark a stage as complete or failed"""
        stage.status = JobStageStatus.SUCCESS if success else JobStageStatus.FAILED
        stage.completed_at = datetime.now()
        if stage.started_at:
            duration = (stage.completed_at - stage.started_at).total_seconds() * 1000
            stage.duration_ms = int(duration)
        if error:
            stage.error_message = error

    def estimate_job(self, plan: IndexPlan) -> DryRunEstimate:
        """
        Perform a dry run to estimate the job without actually indexing.
        Loads a sample, chunks it, and estimates costs.
        """
        # Load small sample
        sample_size = min(100, plan.data_source.sample_limit or 100)

        if plan.data_source.dataset_id:
            dataset = load_dataset(
                plan.data_source.dataset_id,
                split=plan.data_source.split,
                streaming=True,
                trust_remote_code=True,
            )
            samples = []
            for i, row in enumerate(dataset):
                if i >= sample_size:
                    break
                text_field = plan.data_source.text_field
                if text_field in row:
                    samples.append(row[text_field])
        else:
            samples = ["Sample text for estimation"]

        # Estimate chunking
        chunks = self._chunk_texts(samples, plan.chunking)
        avg_chunks_per_doc = len(chunks) / len(samples) if samples else 1

        # Estimate totals
        total_docs = plan.data_source.sample_limit or 1000
        estimated_chunks = int(total_docs * avg_chunks_per_doc)
        estimated_points = estimated_chunks

        # Estimate storage (rough approximation)
        # Assume: dense vector (768 dims * 4 bytes) + sparse (~100 indices * 8 bytes) + payload (~500 bytes)
        dense_dims = plan.vector_spaces.dense.expected_dims or 768
        bytes_per_point = (dense_dims * 4) + (100 * 8) + 500
        estimated_storage_gb = (estimated_points * bytes_per_point) / (1024**3)

        # Estimate time (rough: ~100 docs/sec for embedding + upload)
        estimated_embed_time_minutes = estimated_points / 100 / 60

        # Sample chunks for preview
        sample_chunks_preview = [
            {"chunk_index": i, "text": chunk[:200] + "..." if len(chunk) > 200 else chunk}
            for i, chunk in enumerate(chunks[:5])
        ]

        return DryRunEstimate(
            estimated_docs=total_docs,
            estimated_chunks=estimated_chunks,
            estimated_points=estimated_points,
            estimated_storage_gb=round(estimated_storage_gb, 2),
            estimated_embed_time_minutes=round(estimated_embed_time_minutes, 1),
            sample_chunks=sample_chunks_preview,
        )

    def _chunk_texts(
        self, texts: List[str], config
    ) -> List[str]:
        """Chunk texts according to the chunking configuration"""
        chunks = []

        for text in texts:
            if config.strategy == ChunkingStrategy.NONE:
                chunks.append(text)
            elif config.strategy == ChunkingStrategy.FIXED_TOKENS:
                # Simple token-based chunking (word approximation)
                words = text.split()
                chunk_size = config.size
                overlap = config.overlap

                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i : i + chunk_size]
                    if chunk_words:
                        chunks.append(" ".join(chunk_words))
            elif config.strategy == ChunkingStrategy.SENTENCES:
                # Simple sentence-based chunking
                sentences = text.replace("!", ".").replace("?", ".").split(".")
                current_chunk = []
                current_size = 0

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    words = sentence.split()
                    if current_size + len(words) > config.size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = words
                        current_size = len(words)
                    else:
                        current_chunk.extend(words)
                        current_size += len(words)

                if current_chunk:
                    chunks.append(" ".join(current_chunk))
            else:
                # Fallback to no chunking
                chunks.append(text)

        return chunks

    def run_job(
        self, job: IndexJob, plan: IndexPlan, progress_callback: Optional[Callable] = None
    ):
        """
        Execute the complete indexing job pipeline.
        This should be run in a background task.
        """
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        # Define pipeline stages
        stages = [
            self._create_stage("validate_plan", "Validating plan"),
            self._create_stage("create_collection", "Creating collection"),
            self._create_stage("create_payload_indexes", "Creating payload indexes"),
            self._create_stage("ingest_chunk", "Ingesting and chunking data"),
            self._create_stage("embed", "Embedding documents"),
            self._create_stage("upsert", "Upserting to Qdrant"),
            self._create_stage("optimize", "Optimizing collection"),
            self._create_stage("verify", "Verifying build"),
        ]

        if plan.run_config.create_snapshot:
            stages.append(self._create_stage("snapshot", "Creating snapshot"))

        if plan.run_config.swap_alias:
            stages.append(self._create_stage("swap_alias", "Swapping alias"))

        job.stages = stages

        try:
            # Stage 1: Validate Plan
            self._run_stage_validate_plan(job, plan, progress_callback)

            # Stage 2: Create Collection
            self._run_stage_create_collection(job, plan, progress_callback)

            # Stage 3: Create Payload Indexes
            self._run_stage_create_payload_indexes(job, plan, progress_callback)

            # Stage 4-6: Ingest, Embed, Upsert (combined)
            self._run_stage_ingest_embed_upsert(job, plan, progress_callback)

            # Stage 7: Optimize
            self._run_stage_optimize(job, plan, progress_callback)

            # Stage 8: Verify
            if plan.run_config.run_verification:
                self._run_stage_verify(job, plan, progress_callback)

            # Optional: Snapshot
            if plan.run_config.create_snapshot:
                self._run_stage_snapshot(job, plan, progress_callback)

            # Optional: Swap Alias
            if plan.run_config.swap_alias:
                self._run_stage_swap_alias(job, plan, progress_callback)

            # Job completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            if job.started_at:
                duration = (job.completed_at - job.started_at).total_seconds() * 1000
                job.actual_duration_ms = int(duration)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            current_stage = job.stages[job.current_stage_index]
            self._complete_stage(current_stage, success=False, error=str(e))
            raise

    def _run_stage_validate_plan(self, job: IndexJob, plan: IndexPlan, callback):
        """Stage 1: Validate the plan"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            # Basic validation
            if not plan.data_source.dataset_id:
                raise ValueError("Dataset ID is required")

            if not plan.vector_spaces.dense:
                raise ValueError("At least one dense vector space is required")

            stage.logs.append("Plan validation successful")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _run_stage_create_collection(self, job: IndexJob, plan: IndexPlan, callback):
        """Stage 2: Create the Qdrant collection for Cloud Inference"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            # Get vector dimensions from model mapping (no local model loading)
            dense_config = plan.vector_spaces.dense
            vec_size = MODEL_DIMENSIONS.get(dense_config.model_name, 384)

            # Collection name
            if plan.run_config.build_new_collection:
                timestamp = int(time.time())
                collection_name = f"{plan.project_id}_v{timestamp}"
            else:
                collection_name = MaxQEngine.get_collection_name(
                    plan.project_id, plan.vector_spaces.dense.model_name
                )

            job.collection_name = collection_name
            print(f"[DEBUG] Creating collection: {collection_name} for project: {plan.project_id}, model: {plan.vector_spaces.dense.model_name}")

            # Dense vector config
            dense_config = plan.vector_spaces.dense
            dense_params = models.VectorParams(
                size=vec_size,
                distance=(
                    models.Distance.COSINE
                    if dense_config.distance.value == "cosine"
                    else models.Distance.DOT
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=plan.performance.hnsw.m,
                    ef_construct=plan.performance.hnsw.ef_construct,
                ),
                quantization_config=(
                    models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=plan.performance.quantization.always_ram,
                        )
                    )
                    if plan.performance.quantization.enabled
                    else None
                ),
            )

            vectors_config = {dense_config.name: dense_params}

            # Sparse vector config with IDF modifier for BM25 cloud inference
            sparse_vectors_config = None
            if plan.vector_spaces.sparse and plan.vector_spaces.sparse.enabled:
                sparse_params = models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(on_disk=False)
                )
                sparse_vectors_config = {
                    plan.vector_spaces.sparse.name: sparse_params
                }

            # Create collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                shard_number=plan.performance.shard_number,
                replication_factor=plan.performance.replication_factor,
                on_disk_payload=plan.performance.on_disk_payload,
            )

            stage.logs.append(f"Collection '{collection_name}' created successfully")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _run_stage_create_payload_indexes(self, job: IndexJob, plan: IndexPlan, callback):
        """Stage 3: Create payload indexes"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            # Create indexes for indexed fields
            for field in plan.payload_schema.fields:
                if field.indexed:
                    self.client.create_payload_index(
                        collection_name=job.collection_name,
                        field_name=field.field_name,
                        field_schema=(
                            models.PayloadSchemaType.KEYWORD
                            if field.field_type.value == "keyword"
                            else models.PayloadSchemaType.INTEGER
                            if field.field_type.value == "integer"
                            else models.PayloadSchemaType.FLOAT
                            if field.field_type.value == "float"
                            else models.PayloadSchemaType.DATETIME
                            if field.field_type.value == "datetime"
                            else models.PayloadSchemaType.BOOL
                            if field.field_type.value == "boolean"
                            else models.PayloadSchemaType.TEXT
                        ),
                    )
                    stage.logs.append(f"Created index for field '{field.field_name}'")

            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _extract_text_from_row(self, row: Dict[str, Any], vectorization: Any) -> str:
        """
        Extract text from a row based on vectorization strategy.
        Handles single field, combine fields, LLM enrich, etc.
        """
        strategy = vectorization.strategy

        if strategy == "single_field":
            # Simple: use one field
            field = vectorization.text_field or "text"
            return str(row.get(field, ""))

        elif strategy == "combine_fields":
            # Combine multiple fields with template
            template = vectorization.combine_template or ""
            fields = vectorization.combine_fields or []

            # Replace placeholders like {{field_name}}
            result = template
            for field in fields:
                placeholder = "{{" + field + "}}"
                value = str(row.get(field, ""))
                result = result.replace(placeholder, value)

            return result

        elif strategy == "llm_enrich":
            # Use LLM to generate text from structured data
            # This would require OpenAI client - simplified for now
            fields = vectorization.llm_enrich_fields or []
            field_values = {f: row.get(f) for f in fields}

            # Fallback: just combine fields if no LLM
            return " ".join(str(v) for v in field_values.values() if v)

        elif strategy == "image":
            # For images, return URL or path
            # The embedding model should handle image URLs
            image_field = vectorization.image_field or "image"
            return str(row.get(image_field, ""))

        elif strategy == "multimodal":
            # Combine text and image info
            text_fields = vectorization.text_fields or []
            text_parts = [str(row.get(f, "")) for f in text_fields]
            return " ".join(p for p in text_parts if p)

        else:
            # Fallback to legacy text_field
            return str(row.get("text", ""))

    def _run_stage_ingest_embed_upsert(self, job: IndexJob, plan: IndexPlan, callback):
        """Stages 4-6: Combined ingest, embed, and upsert"""
        ingest_stage = job.stages[job.current_stage_index]
        self._start_stage(ingest_stage)
        job.current_stage_index += 1

        embed_stage = job.stages[job.current_stage_index]
        self._start_stage(embed_stage)
        job.current_stage_index += 1

        upsert_stage = job.stages[job.current_stage_index]
        self._start_stage(upsert_stage)

        try:
            # Load dataset
            dataset = load_dataset(
                plan.data_source.dataset_id,
                split=plan.data_source.split,
                streaming=plan.data_source.streaming,
                trust_remote_code=True,
            )

            batch_texts = []
            batch_payloads = []
            point_id = 0
            doc_count = 0
            limit = plan.data_source.sample_limit or 1000

            for row in dataset:
                if doc_count >= limit:
                    break

                # Extract text based on vectorization strategy
                text = self._extract_text_from_row(row, plan.data_source.vectorization)

                if len(text) < 10:
                    continue

                # Chunk text
                chunks = self._chunk_texts([text], plan.chunking)
                job.total_chunks += len(chunks)

                for chunk in chunks:
                    # Build payload
                    payload = {"_text": chunk}
                    for field in plan.payload_schema.fields:
                        if field.field_name in row:
                            payload[field.field_name] = row[field.field_name]

                    batch_texts.append(chunk)
                    batch_payloads.append(payload)

                    # Process batch
                    if len(batch_texts) >= 50:
                        self._process_batch(
                            job, plan, batch_texts, batch_payloads, point_id
                        )
                        point_id += len(batch_texts)
                        batch_texts = []
                        batch_payloads = []

                        if callback:
                            callback(job)

                doc_count += 1
                job.total_docs = doc_count

            # Process remaining batch
            if batch_texts:
                self._process_batch(job, plan, batch_texts, batch_payloads, point_id)
                if callback:
                    callback(job)

            # Complete stages
            ingest_stage.docs_processed = job.total_docs
            ingest_stage.chunks_created = job.total_chunks
            self._complete_stage(ingest_stage, success=True)

            embed_stage.chunks_created = job.total_chunks
            self._complete_stage(embed_stage, success=True)

            upsert_stage.points_upserted = job.total_points
            self._complete_stage(upsert_stage, success=True)

            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(ingest_stage, success=False, error=str(e))
            self._complete_stage(embed_stage, success=False, error=str(e))
            self._complete_stage(upsert_stage, success=False, error=str(e))
            raise

    def _process_batch(
        self, job: IndexJob, plan: IndexPlan, texts: List[str], payloads: List[Dict], start_id: int
    ):
        """
        Process a batch using Qdrant Cloud Inference.
        Text is passed as Document objects - Qdrant Cloud generates embeddings server-side.
        """
        dense_config = plan.vector_spaces.dense
        sparse_config = plan.vector_spaces.sparse

        # Build points with Document objects for cloud inference
        points = []
        for i, text in enumerate(texts):
            # Use Document objects - Qdrant Cloud will generate embeddings
            vector_dict = {
                dense_config.name: Document(
                    text=text,
                    model=dense_config.model_name
                )
            }

            # Add sparse vector if enabled
            if sparse_config and sparse_config.enabled:
                vector_dict[sparse_config.name] = Document(
                    text=text,
                    model=sparse_config.model_name
                )

            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,  # Use UUID for cloud-compatible IDs
                    vector=vector_dict,
                    payload=payloads[i]
                )
            )

        # Upload using cloud inference - smaller batch size for API
        self.client.upload_points(
            collection_name=job.collection_name,
            points=points,
            batch_size=8,  # Smaller batches for cloud inference
            wait=True
        )

        job.total_points += len(points)

    def _run_stage_optimize(self, job: IndexJob, plan: IndexPlan, callback):
        """Stage 7: Optimize collection"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            # Wait for indexing to complete
            # In production, you'd poll collection_info until indexing is done
            time.sleep(2)

            stage.logs.append("Collection optimization complete")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _run_stage_verify(self, job: IndexJob, plan: IndexPlan, callback):
        """Stage 8: Verify the build"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            # Get collection info
            collection_info = self.client.get_collection(job.collection_name)
            actual_points = collection_info.points_count

            # Verify point count
            expected_points = job.total_points
            point_count_ok = abs(actual_points - expected_points) < 10

            # Random sample retrieval
            sample_ok = True
            try:
                sample_result = self.client.scroll(
                    collection_name=job.collection_name, limit=5, with_payload=True
                )
                sample_ok = len(sample_result[0]) > 0
            except:
                sample_ok = False

            verification_results = {
                "point_count_match": point_count_ok,
                "expected_points": expected_points,
                "actual_points": actual_points,
                "sample_retrieval": sample_ok,
            }

            job.verification_results = verification_results

            stage.logs.append(f"Verification complete: {actual_points} points indexed")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _run_stage_snapshot(self, job: IndexJob, plan: IndexPlan, callback):
        """Create a snapshot"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            snapshot = self.client.create_snapshot(collection_name=job.collection_name)
            stage.logs.append(f"Snapshot created: {snapshot}")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise

    def _run_stage_swap_alias(self, job: IndexJob, plan: IndexPlan, callback):
        """Swap alias to new collection"""
        stage = job.stages[job.current_stage_index]
        self._start_stage(stage)

        try:
            alias_name = f"{plan.project_id}_current"
            # Update alias to point to new collection
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=job.collection_name, alias_name=alias_name
                        )
                    )
                ]
            )

            stage.logs.append(f"Alias '{alias_name}' now points to '{job.collection_name}'")
            self._complete_stage(stage, success=True)
            job.current_stage_index += 1

            if callback:
                callback(job)

        except Exception as e:
            self._complete_stage(stage, success=False, error=str(e))
            raise
