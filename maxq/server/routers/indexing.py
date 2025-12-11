"""
API router for the enhanced indexing system.
Provides endpoints for plans, jobs, and job monitoring.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime
import uuid
import json
import os
from pathlib import Path

from maxq.config import MAXQ_APP_DIR
from maxq.server.indexing_models import (
    IndexPlan,
    IndexJob,
    CreatePlanRequest,
    StartJobRequest,
    DryRunEstimate,
    JobProgress,
    JobStatus,
    DataSourceConfig,
    ChunkingConfig,
    PayloadSchemaConfig,
    VectorSpacesConfig,
    PerformanceStorageConfig,
    RunConfig,
    DenseVectorConfig,
    SparseVectorConfig,
    VectorProvider,
)
from maxq.server.indexing_service import IndexingService
from maxq.server.models import IndexedModelInfo

router = APIRouter(
    prefix="/indexing",
    tags=["indexing"],
    responses={404: {"description": "Not found"}},
)

# In-memory storage (replace with DB in production)
PLANS_FILE = MAXQ_APP_DIR / "index_plans.json"
JOBS_FILE = MAXQ_APP_DIR / "index_jobs.json"


def load_plans() -> List[IndexPlan]:
    """Load plans from file"""
    if not PLANS_FILE.exists():
        return []
    try:
        with open(PLANS_FILE, "r") as f:
            data = json.load(f)
            return [IndexPlan(**item) for item in data]
    except:
        return []


def save_plans(plans: List[IndexPlan]):
    """Save plans to file"""
    with open(PLANS_FILE, "w") as f:
        data = [p.model_dump(mode="json") for p in plans]
        json.dump(data, f, indent=2)


def load_jobs() -> List[IndexJob]:
    """Load jobs from file"""
    if not JOBS_FILE.exists():
        return []
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
            return [IndexJob(**item) for item in data]
    except:
        return []


def save_jobs(jobs: List[IndexJob]):
    """Save jobs to file"""
    with open(JOBS_FILE, "w") as f:
        data = [j.model_dump(mode="json") for j in jobs]
        json.dump(data, f, indent=2)


plans_db: List[IndexPlan] = load_plans()
jobs_db: List[IndexJob] = load_jobs()


# ================ PLANS ================


@router.post("/plans", response_model=IndexPlan)
async def create_plan(request: CreatePlanRequest):
    """Create a new index plan"""
    plan = IndexPlan(
        id=str(uuid.uuid4()),
        project_id=request.project_id,
        name=request.name,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        data_source=request.data_source,
        chunking=request.chunking or ChunkingConfig(),
        payload_schema=request.payload_schema or PayloadSchemaConfig(),
        vector_spaces=request.vector_spaces,
        performance=request.performance or PerformanceStorageConfig(),
        run_config=request.run_config or RunConfig(),
        description=request.description,
    )

    plans_db.append(plan)
    save_plans(plans_db)

    return plan


@router.get("/plans", response_model=List[IndexPlan])
async def list_plans(project_id: Optional[str] = None):
    """List all plans, optionally filtered by project"""
    if project_id:
        return [p for p in plans_db if p.project_id == project_id]
    return plans_db


@router.get("/plans/{plan_id}", response_model=IndexPlan)
async def get_plan(plan_id: str):
    """Get a specific plan"""
    plan = next((p for p in plans_db if p.id == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plan


@router.put("/plans/{plan_id}", response_model=IndexPlan)
async def update_plan(plan_id: str, updates: CreatePlanRequest):
    """Update an existing plan"""
    plan = next((p for p in plans_db if p.id == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Update fields
    plan.name = updates.name
    plan.data_source = updates.data_source
    plan.chunking = updates.chunking or plan.chunking
    plan.payload_schema = updates.payload_schema or plan.payload_schema
    plan.vector_spaces = updates.vector_spaces
    plan.performance = updates.performance or plan.performance
    plan.run_config = updates.run_config or plan.run_config
    plan.description = updates.description
    plan.updated_at = datetime.now()

    save_plans(plans_db)
    return plan


@router.delete("/plans/{plan_id}")
async def delete_plan(plan_id: str):
    """Delete a plan"""
    global plans_db
    plans_db = [p for p in plans_db if p.id != plan_id]
    save_plans(plans_db)
    return {"status": "deleted"}


# ================ DRY RUN ================


@router.post("/plans/{plan_id}/estimate", response_model=DryRunEstimate)
async def estimate_plan(plan_id: str):
    """Run a dry run estimate for a plan"""
    plan = next((p for p in plans_db if p.id == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    try:
        service = IndexingService()
        estimate = service.estimate_job(plan)
        return estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


# ================ JOBS ================


@router.post("/jobs", response_model=IndexJob)
async def create_job(request: StartJobRequest, background_tasks: BackgroundTasks):
    """Start a new indexing job"""
    plan = next((p for p in plans_db if p.id == request.plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Create job
    job = IndexJob(
        id=str(uuid.uuid4()),
        plan_id=plan.id,
        project_id=plan.project_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=JobStatus.QUEUED,
    )

    jobs_db.append(job)
    save_jobs(jobs_db)

    # Run in background
    def progress_callback(updated_job: IndexJob):
        """Callback to update job progress"""
        # Find and update job in DB
        for i, j in enumerate(jobs_db):
            if j.id == updated_job.id:
                jobs_db[i] = updated_job
                save_jobs(jobs_db)
                break

    def run_job_task():
        """Background task to run the job"""
        service = IndexingService()
        try:
            service.run_job(job, plan, progress_callback)

            # On successful completion, update project's indexed_models
            if job.status == JobStatus.COMPLETED:
                from maxq.server.database import ProjectStore

                project = ProjectStore.get(plan.project_id)
                if project:
                    model_name = plan.vector_spaces.dense.model_name
                    collection_name = job.collection_name

                    # Add or update indexed model
                    model_info = IndexedModelInfo(
                        model_name=model_name,
                        collection_name=collection_name,
                        indexed_at=datetime.now(),
                        point_count=job.total_points
                    )
                    print(f"[DEBUG] Saving indexed model: project={plan.project_id}, model={model_name}, collection={collection_name}, points={job.total_points}")
                    ProjectStore.add_indexed_model(plan.project_id, model_info)

                    # Set default embedding_model if not set
                    if not project.embedding_model or project.embedding_model == "BAAI/bge-base-en-v1.5":
                        project.embedding_model = model_name
                        ProjectStore.update(project)

                    print(f"Updated project {plan.project_id} with indexed model: {model_name}")

        except Exception as e:
            job.status = JobStatus.FAILED
            print(f"Job {job.id} failed: {e}")
        finally:
            save_jobs(jobs_db)

    background_tasks.add_task(run_job_task)

    return job


@router.get("/jobs", response_model=List[IndexJob])
async def list_jobs(project_id: Optional[str] = None, status: Optional[JobStatus] = None):
    """List all jobs, optionally filtered"""
    filtered_jobs = jobs_db

    if project_id:
        filtered_jobs = [j for j in filtered_jobs if j.project_id == project_id]

    if status:
        filtered_jobs = [j for j in filtered_jobs if j.status == status]

    return filtered_jobs


@router.get("/jobs/{job_id}", response_model=IndexJob)
async def get_job(job_id: str):
    """Get a specific job"""
    job = next((j for j in jobs_db if j.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(job_id: str):
    """Get real-time progress for a job"""
    job = next((j for j in jobs_db if j.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Calculate progress percentage
    total_stages = len(job.stages)
    completed_stages = sum(
        1 for s in job.stages if s.status.value in ["success", "failed", "skipped"]
    )
    progress_percent = (completed_stages / total_stages * 100) if total_stages > 0 else 0

    # Get current stage
    current_stage = None
    if job.current_stage_index < len(job.stages):
        current_stage = job.stages[job.current_stage_index]

    # Get recent logs
    recent_logs = []
    for stage in reversed(job.stages):
        recent_logs.extend(stage.logs[-3:])  # Last 3 logs per stage
        if len(recent_logs) >= 10:
            break

    return JobProgress(
        job_id=job.id,
        status=job.status,
        current_stage=current_stage,
        progress_percent=round(progress_percent, 1),
        estimated_time_remaining_ms=None,  # TODO: Calculate based on stage timing
        recent_logs=list(reversed(recent_logs[-10:])),
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    job = next((j for j in jobs_db if j.id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job is not running")

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now()
    save_jobs(jobs_db)

    return {"status": "cancelled"}


# ================ SAMPLE DATA ================


@router.post("/sample-data")
async def sample_dataset(data_source: DataSourceConfig):
    """
    Get a sample preview of the dataset (first 10 rows).
    Used for field mapping and chunking preview.
    """
    try:
        if data_source.source_type.value == "huggingface":
            from datasets import load_dataset

            dataset = load_dataset(
                data_source.dataset_id, split=data_source.split, streaming=True, trust_remote_code=True
            )

            samples = []
            for i, row in enumerate(dataset):
                if i >= 10:
                    break
                samples.append(dict(row))

            return {"samples": samples, "fields": list(samples[0].keys()) if samples else []}
        else:
            return {"samples": [], "fields": []}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sample data: {str(e)}")


@router.post("/recommend-columns-to-vectorize")
async def recommend_columns_to_vectorize(data_source: DataSourceConfig):
    """
    Use an LLM to recommend which column(s) to vectorize for semantic search.
    Uses a simple, fast model (gpt-4o-mini) for quick recommendations.

    This is smarter than just "text field" - it can recommend:
    - Single column (simple)
    - Multiple columns to combine
    - Columns that need enrichment
    - Image columns
    """
    try:
        import os
        from openai import OpenAI
        import json

        # Check if OpenAI API key is available
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            # Fallback: simple heuristic if no API key
            return {
                "recommended_columns": [],
                "confidence": "low",
                "reasoning": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "using_llm": False,
            }

        # Load sample data
        if data_source.source_type.value == "huggingface":
            from datasets import load_dataset

            dataset = load_dataset(
                data_source.dataset_id, split=data_source.split, streaming=True
            )

            samples = []
            for i, row in enumerate(dataset):
                if i >= 3:  # Just need 3 samples for LLM
                    break
                samples.append(dict(row))

            if not samples:
                raise HTTPException(status_code=400, detail="No samples found in dataset")

            # Use LLM to recommend columns
            client = OpenAI(api_key=openai_api_key)

            prompt = f"""Analyze these {len(samples)} sample records and recommend which column(s) to vectorize for semantic search.

Dataset samples:
{json.dumps(samples, indent=2)}

Instructions:
1. Identify columns with meaningful content for semantic search
2. Consider:
   - Single rich text columns (descriptions, content, etc.)
   - Multiple columns that should be combined (title + body)
   - Structured data that could be enriched
   - Image URLs/paths
3. Avoid: IDs, timestamps, pure metadata, very short text

Return ONLY a JSON object:
{{
    "recommended_columns": ["column1", "column2"],
    "primary_column": "column1" (the main one if single),
    "should_combine": true|false,
    "confidence": "high|medium|low",
    "reasoning": "brief explanation (1-2 sentences)",
    "alternative_columns": ["col3"] (if any)
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)
            result["using_llm"] = True

            return result
        else:
            return {
                "recommended_columns": [],
                "confidence": "low",
                "reasoning": "Source type not supported for recommendation",
                "using_llm": False,
            }

    except Exception as e:
        # Fallback to simple heuristic
        return {
            "recommended_columns": [],
            "confidence": "low",
            "reasoning": f"LLM recommendation failed: {str(e)}. Please select columns manually.",
            "using_llm": False,
        }


@router.post("/recommend-payload-fields")
async def recommend_payload_fields(data_source: DataSourceConfig):
    """
    Use an LLM to recommend which fields should be indexed in the payload.
    Analyzes data types and suggests which fields are good for filtering.
    """
    try:
        import os
        from openai import OpenAI
        import json

        # Check if OpenAI API key is available
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                "recommended_fields": [],
                "confidence": "low",
                "reasoning": "OpenAI API key not configured.",
                "using_llm": False,
            }

        # Load sample data
        if data_source.source_type.value == "huggingface":
            from datasets import load_dataset

            dataset = load_dataset(
                data_source.dataset_id, split=data_source.split, streaming=True, trust_remote_code=True
            )

            samples = []
            for i, row in enumerate(dataset):
                if i >= 3:
                    break
                samples.append(dict(row))

            if not samples:
                raise HTTPException(status_code=400, detail="No samples found in dataset")

            # Use LLM to recommend payload fields
            client = OpenAI(api_key=openai_api_key)

            prompt = f"""Analyze these sample records and recommend which fields should be indexed for filtering and metadata.

Dataset samples:
{json.dumps(samples, indent=2)}

Instructions:
1. Identify fields good for filtering (categories, IDs, dates, etc.)
2. Determine appropriate field types:
   - keyword: for exact match (IDs, categories, short strings)
   - text: for full-text search (longer text fields)
   - integer: for numeric IDs
   - float: for scores, ratings
   - datetime: for timestamps
3. Recommend indexing for fields commonly used in filters
4. Skip the main text field (that's for embedding)

Return ONLY a JSON object:
{{
    "recommended_fields": [
        {{"field_name": "category", "field_type": "keyword", "indexed": true, "reasoning": "why"}},
        {{"field_name": "created_at", "field_type": "datetime", "indexed": true, "reasoning": "why"}}
    ],
    "confidence": "high|medium|low",
    "reasoning": "overall explanation"
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)
            result["using_llm"] = True

            return result
        else:
            return {
                "recommended_fields": [],
                "confidence": "low",
                "reasoning": "Source type not supported",
                "using_llm": False,
            }

    except Exception as e:
        return {
            "recommended_fields": [],
            "confidence": "low",
            "reasoning": f"Failed: {str(e)}",
            "using_llm": False,
        }


@router.post("/recommend-vectorization-strategy")
async def recommend_vectorization_strategy(data_source: DataSourceConfig):
    """
    Use an LLM to recommend how to vectorize the data.
    Suggests: single field, combine fields, enrich with LLM, or handle images.
    """
    try:
        import os
        from openai import OpenAI
        import json

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                "strategy": "single_field",
                "recommended_field": None,
                "confidence": "low",
                "reasoning": "OpenAI API key not configured.",
                "using_llm": False,
            }

        if data_source.source_type.value == "huggingface":
            from datasets import load_dataset

            dataset = load_dataset(
                data_source.dataset_id, split=data_source.split, streaming=True, trust_remote_code=True
            )

            samples = []
            for i, row in enumerate(dataset):
                if i >= 3:
                    break
                samples.append(dict(row))

            if not samples:
                raise HTTPException(status_code=400, detail="No samples found")

            client = OpenAI(api_key=openai_api_key)

            prompt = f"""Analyze this dataset and recommend the best vectorization strategy.

Dataset samples:
{json.dumps(samples, indent=2)}

Possible strategies:
1. "single_field" - Use one text field as-is
2. "combine_fields" - Combine multiple fields with a template
3. "llm_enrich" - Use LLM to generate rich text from structured data
4. "image" - Dataset contains images
5. "multimodal" - Mix of text and images

For each strategy, provide:
- Which fields to use
- Template if combining (e.g., "{{title}}: {{description}}")
- LLM prompt if enriching

Return ONLY a JSON object:
{{
    "strategy": "single_field|combine_fields|llm_enrich|image|multimodal",
    "recommended_field": "field_name" (if single_field),
    "combine_template": "{{field1}} - {{field2}}" (if combine_fields),
    "combine_fields": ["field1", "field2"] (if combine_fields),
    "llm_prompt": "Generate a description from: {{field1}}, {{field2}}" (if llm_enrich),
    "image_field": "image_url" (if image/multimodal),
    "confidence": "high|medium|low",
    "reasoning": "why this strategy is best"
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)
            result["using_llm"] = True

            return result
        else:
            return {
                "strategy": "single_field",
                "confidence": "low",
                "reasoning": "Source type not supported",
                "using_llm": False,
            }

    except Exception as e:
        return {
            "strategy": "single_field",
            "confidence": "low",
            "reasoning": f"Failed: {str(e)}",
            "using_llm": False,
        }


# ================ PRESETS ================


@router.get("/presets/performance")
async def get_performance_presets():
    """Get available performance presets"""
    return {
        "dev": {
            "name": "Development",
            "description": "Fast setup for development and testing",
            "config": {
                "shard_number": 1,
                "replication_factor": 1,
                "on_disk_payload": False,
                "quantization": {"enabled": False},
                "hnsw": {"m": 16, "ef_construct": 100},
            },
        },
        "prod_small": {
            "name": "Production (Small)",
            "description": "For production with < 1M vectors",
            "config": {
                "shard_number": 3,
                "replication_factor": 2,
                "on_disk_payload": True,
                "quantization": {"enabled": True, "type": "int8"},
                "hnsw": {"m": 16, "ef_construct": 100},
            },
        },
        "prod_fast_writes": {
            "name": "Production (Fast Writes)",
            "description": "Optimized for high write throughput",
            "config": {
                "shard_number": 3,
                "replication_factor": 2,
                "on_disk_payload": True,
                "quantization": {"enabled": True, "type": "int8"},
                "hnsw": {"m": 16, "ef_construct": 100},
                "optimizers": {"indexing_threshold": 50000},
            },
        },
        "prod_fast_reads": {
            "name": "Production (Fast Reads)",
            "description": "Optimized for query performance",
            "config": {
                "shard_number": 3,
                "replication_factor": 2,
                "on_disk_payload": True,
                "quantization": {"enabled": True, "type": "int8"},
                "hnsw": {"m": 32, "ef_construct": 200},
            },
        },
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "plans_count": len(plans_db),
        "jobs_count": len(jobs_db),
        "active_jobs": len([j for j in jobs_db if j.status == JobStatus.RUNNING]),
    }
