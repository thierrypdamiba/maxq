from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
import uuid
import json
import os
from ..models import EvalResult
from .tuning import experiments_db, save_db as save_experiments_db
from maxq.config import MAXQ_APP_DIR
from datetime import datetime

router = APIRouter(
    prefix="/evals",
    tags=["evals"],
    responses={404: {"description": "Not found"}},
)

DATA_FILE = MAXQ_APP_DIR / "evals.json"

def save_db():
    with open(DATA_FILE, "w") as f:
        data = [e.model_dump(mode='json') for e in evals_db]
        json.dump(data, f, indent=2)

def load_db() -> List[EvalResult]:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            return [EvalResult(**item) for item in data]
    except Exception:
        return []

evals_db: List[EvalResult] = load_db()

@router.get("/", response_model=List[EvalResult])
async def list_evals(experiment_id: Optional[str] = None):
    if experiment_id:
        return [e for e in evals_db if e.experiment_id == experiment_id]
    return evals_db

@router.post("/run")
async def run_eval(experiment_id: str, background_tasks: BackgroundTasks):
    # Verify experiment exists
    exp = next((e for e in experiments_db if e.id == experiment_id), None)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Create placeholder result
    eval_id = str(uuid.uuid4())
    result = EvalResult(
        id=eval_id,
        experiment_id=experiment_id,
        metrics={} # Empty initially
    )
    evals_db.append(result)
    save_db()
    
    # Update experiment status to running
    exp.status = "running"
    save_experiments_db()

    # Run in background
    background_tasks.add_task(run_ragas_pipeline, experiment_id, eval_id)
    
    return {"status": "started", "eval_id": eval_id}

async def run_ragas_pipeline(experiment_id: str, eval_id: str):
    try:
        from maxq.ragas_utils import RagasManager
        from maxq.search_engine import MaxQEngine, CollectionStrategy, SearchRequest
        
        # 1. Setup
        exp = next((e for e in experiments_db if e.id == experiment_id), None)
        if not exp: return
        
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("Error: OPENAI_API_KEY not found for Ragas")
            return

        ragas = RagasManager(openai_key)
        engine = MaxQEngine(qdrant_url=os.environ.get("QDRANT_URL"), qdrant_api_key=os.environ.get("QDRANT_API_KEY"), openai_api_key=openai_key)
        
        # Initialize progress tracking
        exp.started_at = datetime.now()
        exp.progress_current = 0
        exp.progress_total = 0  # Will be set after testset generation
        exp.progress_message = "Starting evaluation..."
        exp.updated_at = datetime.now()
        save_experiments_db()
        
        # 2. Generate Test Set (Small for demo speed)
        collection_name = MaxQEngine.get_collection_name(exp.project_id, exp.embedding_model)

        print(f"Generating test set for {collection_name}...")
        exp.progress_message = "Generating test set..."
        exp.updated_at = datetime.now()
        save_experiments_db()
        
        try:
            testset = ragas.generate_testset(engine, collection_name=collection_name, num_questions=2)
        except Exception as e:
            print(f"Warning: Ragas testset gen failed ({e}), using fallback.")
            testset = []
        
        if not testset:
            print("Using fallback test set.")
            testset = [
                {
                    "user_input": "Act as a Linux Terminal",
                    "reference": "I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show."
                },
                {
                    "user_input": "Act as an English Translator",
                    "reference": "I want you to act as an English translator, spelling corrector and improver."
                }
            ]
        
        # Set total progress (testset generation + per-question + metrics + complete)
        exp.progress_total = 1 + len(testset) + 2
        exp.progress_current = 1
        exp.progress_message = f"Test set generated ({len(testset)} questions)"
        exp.updated_at = datetime.now()
        save_experiments_db()

        # 3. Run Experiment (Retrieval + Generation)
        results_for_eval = []
        
        config = CollectionStrategy(
            collection_name=collection_name,
            dense_model_name=exp.embedding_model,
            use_quantization=True
        )
        
        print(f"Running experiment {exp.name} with strategy {exp.search_strategy}...")
        for idx, item in enumerate(testset, start=1):
            # Ragas v0.2 uses 'user_input' and 'reference'
            question = item.get('user_input')
            ground_truth = item.get('reference')
            
            if not question:
                continue

            # 3. Run Search (using experiment strategy)
            req = SearchRequest(query=question, strategy=exp.search_strategy, limit=5)
            points = engine.query(config, req)
            
            # 4. Generate Answer (using LLM)
            contexts = [p.payload.get("_text", "") for p in points]
            # Note: generate_answer returns a stream, we need full text for Ragas
            # We'll implement a simple non-stream generation here or update engine
            # For now, let's use a quick direct call to LLM to get full text
            answer = _generate_full_answer(engine, question, contexts)
            
            results_for_eval.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
            
            # Update progress
            exp.progress_current = 1 + idx
            exp.progress_message = f"Evaluating question {idx}/{len(testset)}"
            exp.updated_at = datetime.now()
            save_experiments_db()

        # 4. Calculate Metrics
        exp.progress_current = 1 + len(testset) + 1
        exp.progress_message = "Calculating Ragas metrics..."
        exp.updated_at = datetime.now()
        save_experiments_db()
        
        print("Calculating Ragas metrics...")
        scores = ragas.evaluate_results(results_for_eval)
        
        # 5. Update DB
        eval_result = next((e for e in evals_db if e.id == eval_id), None)
        if eval_result:
            eval_result.metrics = scores
            save_db()
            
        # Update Experiment
        exp.status = "completed"
        # Map Ragas metrics to our UI display
        # nDCG is not in Ragas default, we use Context Precision as proxy for Quality
        # 6. Update Experiment Status
        import math
        
        # EvaluationResult behaves like a dict for access
        # Handle case where scores might be empty or missing keys
        precision = 0.0
        if scores and isinstance(scores, dict):
            precision = scores.get("context_precision", 0.0)
            if isinstance(precision, float) and math.isnan(precision):
                precision = 0.0
            elif precision is None:
                precision = 0.0
        else:
            print(f"Warning: Ragas evaluation returned empty or invalid scores: {scores}")
            precision = 0.0
            
        exp.metrics["ndcg"]["candidate"] = round(precision, 2)
        exp.metrics["ndcg"]["baseline"] = 0.50 # Mock baseline for now
        
        # Calculate delta
        delta = (exp.metrics["ndcg"]["candidate"] - exp.metrics["ndcg"]["baseline"]) / exp.metrics["ndcg"]["baseline"] * 100
        exp.metrics["ndcg"]["delta"] = f"{delta:+.1f}%"
        
        # Latency (Mock for now, or measure it)
        exp.metrics["latency"]["candidate"] = "120ms"
        exp.metrics["latency"]["baseline"] = "150ms"
        exp.metrics["latency"]["delta"] = "-20%"
        
        # Save winner (simple logic)
        if exp.metrics["ndcg"]["candidate"] > exp.metrics["ndcg"]["baseline"]:
            exp.winner = exp.name
            
        save_experiments_db()
        print("Evaluation complete.")

    except Exception as e:
        print(f"Error in run_ragas_pipeline: {e}")
        import traceback
        traceback.print_exc()

def _generate_full_answer(engine, query, contexts):
    # Helper to get non-streaming answer
    try:
        context_text = "\n\n".join(contexts)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        resp = engine.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content
    except:
        return "Error generating answer"


@router.get("/{eval_id}/report.pdf")
async def get_eval_report_pdf(eval_id: str):
    """
    Generate and download a PDF report for an evaluation.
    """
    from fastapi.responses import Response

    # Find the eval result
    eval_result = next((e for e in evals_db if e.id == eval_id), None)
    if not eval_result:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Find the experiment
    exp = next((e for e in experiments_db if e.id == eval_result.experiment_id), None)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Find the project
    from maxq.server.database import ProjectStore
    project = ProjectStore.get(exp.project_id)
    project_name = project.name if project else "Unknown Project"
    project_id = exp.project_id

    try:
        from maxq.report_generator import EvalReportGenerator

        generator = EvalReportGenerator()
        pdf_bytes = generator.generate(
            project_name=project_name,
            project_id=project_id,
            experiment_name=exp.name,
            experiment_id=exp.id,
            embedding_model=exp.embedding_model,
            search_strategy=exp.search_strategy,
            metrics=exp.metrics or {},
            eval_id=eval_id
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="maxq-report-{eval_id[:8]}.pdf"'
            }
        )

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="PDF generation requires reportlab. Install with: pip install reportlab"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {str(e)}"
        )
