"""
MaxQ Evaluation Module: ID-Matching (Default)
==============================================

This is the default evaluation module using deterministic ID-matching.
Generate questions from documents, then check if retrieval finds the source.

EVALUATION METHOD: ID-Matching
- Fast, deterministic, no LLM calls for judging
- Metrics: nDCG@K, Hit@K, MRR, Precision, Recall
- Ideal for comparing embedding models
- Inspired by smallevals approach

For LLM-based semantic evaluation, use eval_llm_judge.py which uses
Ragas framework for context_precision and context_recall metrics.

Features:
- nDCG@K metric (preferred ranking metric)
- Multi-model comparison
- AI-powered analysis explaining results (optional)
- Automatic test question generation

Usage:
    from maxq.eval import Evaluator, run_full_evaluation

    # Quick comparison
    evaluator = Evaluator(engine)
    results = evaluator.compare_models(
        models=['BAAI/bge-small-en-v1.5', 'BAAI/bge-base-en-v1.5'],
        testset=testset,
        project_id='my-project'
    )
    evaluator.print_comparison_report(results)

    # Or use the convenience function
    run_full_evaluation(engine, 'my-project', models, num_test_queries=20)
"""
import os
import json
import math
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
import numpy as np

console = Console()


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    hit_rate_1: float = 0.0
    hit_rate_3: float = 0.0
    hit_rate_5: float = 0.0
    mrr: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "hit_rate@1": self.hit_rate_1,
            "hit_rate@3": self.hit_rate_3,
            "hit_rate@5": self.hit_rate_5,
            "mrr": self.mrr,
            "ndcg@5": self.ndcg_5,
            "ndcg@10": self.ndcg_10,
            "precision@5": self.precision_5,
            "recall@5": self.recall_5,
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "latency_p99_ms": self.latency_p99,
        }


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query: str
    expected_id: Any
    expected_text: str = ""
    retrieved_ids: List[Any] = field(default_factory=list)
    retrieved_texts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    hit_at_1: bool = False
    hit_at_5: bool = False
    rank: int = -1  # -1 means not found


@dataclass
class ModelEvalResult:
    """Complete evaluation result for a single model."""
    model_name: str
    collection_name: str
    metrics: RetrievalMetrics
    query_results: List[QueryResult] = field(default_factory=list)
    index_time_seconds: float = 0.0
    total_documents: int = 0

    # Convenience accessors
    @property
    def ndcg_at_k(self) -> float:
        return self.metrics.ndcg_5

    @property
    def hit_at_k(self) -> float:
        return self.metrics.hit_rate_5

    @property
    def mrr(self) -> float:
        return self.metrics.mrr

    @property
    def latency_p50(self) -> float:
        return self.metrics.latency_p50


class Evaluator:
    """
    ID-Matching Evaluator for comparing embedding models.

    Inspired by smallevals approach:
    - Generate questions from documents
    - Test retrieval by checking if source document is found
    - Use nDCG as primary metric (position-aware ranking)

    This is faster and cheaper than LLM-as-Judge (Ragas) evaluation
    since it doesn't require LLM calls to judge relevance.
    """

    def __init__(self, engine, openai_api_key: Optional[str] = None):
        self.engine = engine
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None and self.openai_api_key:
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=self.openai_api_key)
        return self._llm_client

    # ========== Metric Calculations ==========

    @staticmethod
    def calculate_ndcg(retrieved_ids: List[Any], relevant_id: Any, k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        nDCG uses logarithmic position discounting to penalize relevant items
        that appear lower in the ranking.

        Formula: DCG@K / IDCG@K
        DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
        IDCG@K = 1.0 (since we have only 1 relevant item, best case is rank 1)
        """
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k]):
            if rid == relevant_id:
                # relevance = 1 for the matching document
                dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed and log2(1)=0
                break

        # IDCG is the ideal case: relevant item at position 1
        idcg = 1.0 / math.log2(2)  # = 1.0

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_mrr(retrieved_ids: List[Any], relevant_id: Any) -> float:
        """Calculate Mean Reciprocal Rank for a single query."""
        for i, rid in enumerate(retrieved_ids):
            if rid == relevant_id:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def calculate_hit_rate(retrieved_ids: List[Any], relevant_id: Any, k: int = 5) -> float:
        """Calculate Hit Rate at K (binary: 1 if found in top-k, else 0)."""
        return 1.0 if relevant_id in retrieved_ids[:k] else 0.0

    @staticmethod
    def calculate_precision(retrieved_ids: List[Any], relevant_id: Any, k: int = 5) -> float:
        """Calculate Precision at K."""
        hits = sum(1 for rid in retrieved_ids[:k] if rid == relevant_id)
        return hits / k

    # ========== Test Set Generation ==========

    def generate_testset_from_collection(
        self,
        collection_name: str,
        num_samples: int = 20,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a test set from documents in a collection.

        If use_llm=True, generates natural questions using GPT.
        Otherwise, uses document content as pseudo-query.
        """
        console.print(f"[bold]Generating test set from {collection_name}...[/bold]")

        # Fetch documents
        try:
            points, _ = self.engine.client.scroll(
                collection_name=collection_name,
                limit=min(100, num_samples * 2),
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            console.print(f"[red]Error fetching documents: {e}[/red]")
            return []

        if not points:
            console.print("[red]No documents found in collection.[/red]")
            return []

        # Sample points
        samples = random.sample(points, min(num_samples, len(points)))
        testset = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("[cyan]Generating questions...", total=len(samples))

            for point in samples:
                text = point.payload.get("_text", "")
                if not text or len(text) < 50:
                    progress.update(task, advance=1)
                    continue

                if use_llm and self.llm_client:
                    question = self._generate_question_with_llm(text)
                else:
                    # Use first sentence or key phrase as pseudo-query
                    question = self._extract_pseudo_query(text)

                if question:
                    testset.append({
                        "question": question,
                        "expected_id": point.id,
                        "expected_text": text[:500],
                        "full_text": text,
                        "payload": point.payload
                    })

                progress.update(task, advance=1)

        console.print(f"[green]Generated {len(testset)} test queries.[/green]")
        return testset

    def _generate_question_with_llm(self, text: str) -> Optional[str]:
        """Generate a natural question from document text using LLM."""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract ONE specific question that can be answered by this text. "
                                   "The question should be natural and specific enough to uniquely identify this document. "
                                   "Return ONLY the question, nothing else."
                    },
                    {"role": "user", "content": text[:2000]}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            console.print(f"[dim]LLM question generation failed: {e}[/dim]")
            return None

    def _extract_pseudo_query(self, text: str) -> str:
        """Extract a pseudo-query from text without LLM."""
        # Use first sentence as query
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        if sentences:
            first = sentences[0].strip()
            if len(first) > 20:
                return first[:200]
        return text[:100]

    def generate_testset_from_docs(
        self,
        docs: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: Optional[str] = None,
        num_samples: int = 20,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate a test set from a list of document dictionaries.

        Args:
            docs: List of document dictionaries
            text_field: The field containing the text to embed
            id_field: The field to use as document ID (or None to use index)
            num_samples: Number of test samples to generate
            use_llm: Whether to use LLM for question generation

        Returns:
            List of test cases with question, expected_id, expected_text
        """
        console.print(f"[bold]Generating test set from {len(docs)} documents...[/bold]")

        # Sample documents
        samples = random.sample(docs, min(num_samples, len(docs)))
        testset = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("[cyan]Generating questions...", total=len(samples))

            for i, doc in enumerate(samples):
                text = doc.get(text_field, "")
                if not text or len(str(text)) < 50:
                    progress.update(task, advance=1)
                    continue

                text = str(text)

                # Get document ID
                if id_field and id_field in doc:
                    doc_id = doc[id_field]
                else:
                    doc_id = docs.index(doc)  # Use original index as ID

                if use_llm and self.llm_client:
                    question = self._generate_question_with_llm(text)
                else:
                    question = self._extract_pseudo_query(text)

                if question:
                    testset.append({
                        "question": question,
                        "expected_id": doc_id,
                        "expected_text": text[:500],
                        "full_text": text
                    })

                progress.update(task, advance=1)

        console.print(f"[green]Generated {len(testset)} test queries.[/green]")
        return testset

    # ========== Evaluation ==========

    def evaluate_model(
        self,
        collection_name: str,
        model_name: str,
        testset: List[Dict[str, Any]],
        strategy: str = "hybrid",
        limit: int = 10
    ) -> ModelEvalResult:
        """
        Evaluate a single model/collection on the test set.
        """
        from .search_engine import CollectionStrategy, SearchRequest

        config = CollectionStrategy(
            collection_name=collection_name,
            dense_model_name=model_name,
            sparse_model_name="Qdrant/bm25"
        )

        query_results = []
        latencies = []
        hits_1, hits_3, hits_5 = [], [], []
        mrr_scores = []
        ndcg_5_scores, ndcg_10_scores = [], []
        precision_5_scores = []

        for test in testset:
            question = test["question"]
            expected_id = test["expected_id"]

            request = SearchRequest(query=question, strategy=strategy, limit=limit)

            start = time.time()
            try:
                results = self.engine.query(config, request)
            except Exception as e:
                console.print(f"[red]Query failed: {e}[/red]")
                results = []
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            retrieved_ids = [r.id for r in results]
            retrieved_texts = [r.payload.get("_text", "")[:200] for r in results]
            scores = [r.score for r in results]

            # Calculate metrics for this query
            hit_1 = self.calculate_hit_rate(retrieved_ids, expected_id, k=1)
            hit_3 = self.calculate_hit_rate(retrieved_ids, expected_id, k=3)
            hit_5 = self.calculate_hit_rate(retrieved_ids, expected_id, k=5)
            mrr = self.calculate_mrr(retrieved_ids, expected_id)
            ndcg_5 = self.calculate_ndcg(retrieved_ids, expected_id, k=5)
            ndcg_10 = self.calculate_ndcg(retrieved_ids, expected_id, k=10)
            precision_5 = self.calculate_precision(retrieved_ids, expected_id, k=5)

            hits_1.append(hit_1)
            hits_3.append(hit_3)
            hits_5.append(hit_5)
            mrr_scores.append(mrr)
            ndcg_5_scores.append(ndcg_5)
            ndcg_10_scores.append(ndcg_10)
            precision_5_scores.append(precision_5)

            # Find rank
            rank = -1
            if expected_id in retrieved_ids:
                rank = retrieved_ids.index(expected_id) + 1

            query_results.append(QueryResult(
                query=question,
                expected_id=expected_id,
                expected_text=test.get("expected_text", ""),
                retrieved_ids=retrieved_ids,
                retrieved_texts=retrieved_texts,
                scores=scores,
                latency_ms=latency,
                hit_at_1=bool(hit_1),
                hit_at_5=bool(hit_5),
                rank=rank
            ))

        metrics = RetrievalMetrics(
            hit_rate_1=np.mean(hits_1),
            hit_rate_3=np.mean(hits_3),
            hit_rate_5=np.mean(hits_5),
            mrr=np.mean(mrr_scores),
            ndcg_5=np.mean(ndcg_5_scores),
            ndcg_10=np.mean(ndcg_10_scores),
            precision_5=np.mean(precision_5_scores),
            recall_5=np.mean(hits_5),  # For single relevant doc, recall@5 == hit_rate@5
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
        )

        return ModelEvalResult(
            model_name=model_name,
            collection_name=collection_name,
            metrics=metrics,
            query_results=query_results
        )

    def compare_models(
        self,
        models: List[str],
        testset: List[Dict[str, Any]],
        collection_prefix: str = None,
        project_id: str = None,
        strategy: str = "hybrid",
        k: int = 10
    ) -> Dict[str, ModelEvalResult]:
        """
        Compare multiple models on the same test set.

        Args:
            models: List of model names (e.g., ['BAAI/bge-small-en-v1.5', 'BAAI/bge-base-en-v1.5'])
            testset: Test cases with question, expected_id
            collection_prefix: Prefix for collection names (e.g., 'enhanced_eval_test')
            project_id: Alternative to collection_prefix - uses get_collection_name()
            strategy: Search strategy ('hybrid', 'dense', 'sparse')
            k: Number of results to retrieve
        """
        from .search_engine import MaxQEngine

        results = {}

        console.print(f"\n[bold]Evaluating {len(models)} models on {len(testset)} queries...[/bold]\n")

        for model in models:
            model_short = model.split("/")[-1]

            # Determine collection name
            if collection_prefix:
                safe_name = model_short.replace("-", "_")
                collection_name = f"{collection_prefix}_{safe_name}"
            elif project_id:
                collection_name = MaxQEngine.get_collection_name(project_id, model)
            else:
                raise ValueError("Must provide either collection_prefix or project_id")

            console.print(f"[cyan]Evaluating {model_short}...[/cyan]")

            result = self.evaluate_model(
                collection_name=collection_name,
                model_name=model,
                testset=testset,
                strategy=strategy,
                limit=k
            )
            results[model_short] = result

            # Print quick summary
            m = result.metrics
            console.print(f"  Hit@1: {m.hit_rate_1:.1%} | nDCG@5: {m.ndcg_5:.4f} | MRR: {m.mrr:.4f}")

        return results

    # ========== AI Analysis ==========

    def generate_ai_analysis(
        self,
        results: Dict[str, ModelEvalResult],
        testset: List[Dict[str, Any]]
    ) -> str:
        """
        Generate AI-powered analysis explaining the evaluation results.
        """
        if not self.llm_client:
            return "AI analysis requires OpenAI API key."

        # Prepare data for analysis
        summary_data = self._prepare_analysis_data(results, testset)

        prompt = f"""You are an expert in information retrieval and embedding models. Analyze these evaluation results and explain:

1. **Overall Winner**: Which model performed best and why?
2. **Metric Breakdown**: Explain what each metric tells us
3. **Failure Analysis**: Why did certain queries fail for some models?
4. **Score Distribution**: What do the score gaps tell us about model behavior?
5. **Recommendations**: Which model should be used and when?

## Evaluation Data

{json.dumps(summary_data, indent=2)}

## Detailed Query Comparisons

{self._format_query_comparisons(results)}

Provide a clear, actionable analysis in markdown format. Focus on practical insights."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an IR evaluation expert. Provide clear, technical analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI analysis failed: {e}"

    def _prepare_analysis_data(
        self,
        results: Dict[str, ModelEvalResult],
        testset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare structured data for AI analysis."""
        data = {
            "num_queries": len(testset),
            "models": {}
        }

        for model_name, result in results.items():
            m = result.metrics

            # Find disagreements
            failures = [qr for qr in result.query_results if not qr.hit_at_1]

            data["models"][model_name] = {
                "metrics": m.to_dict(),
                "num_failures": len(failures),
                "failure_queries": [f.query[:50] for f in failures[:5]],
                "avg_top_score": np.mean([qr.scores[0] if qr.scores else 0 for qr in result.query_results]),
                "avg_score_gap": np.mean([
                    qr.scores[0] - qr.scores[1] if len(qr.scores) >= 2 else 0
                    for qr in result.query_results
                ])
            }

        return data

    def _format_query_comparisons(self, results: Dict[str, ModelEvalResult]) -> str:
        """Format query-level comparisons for AI analysis."""
        models = list(results.keys())
        if not models:
            return "No results"

        # Find queries where models disagree
        disagreements = []
        num_queries = len(results[models[0]].query_results)

        for i in range(min(num_queries, 10)):  # Check first 10 queries
            query_text = results[models[0]].query_results[i].query

            hits = {}
            scores = {}
            for model in models:
                qr = results[model].query_results[i]
                hits[model] = qr.hit_at_1
                scores[model] = qr.scores[:3] if qr.scores else []

            # Check if models disagree on hit@1
            if len(set(hits.values())) > 1:
                disagreements.append({
                    "query": query_text[:80],
                    "hits": hits,
                    "top_scores": {m: [f"{s:.3f}" for s in scores[m]] for m in models}
                })

        if not disagreements:
            return "All models agreed on all queries."

        lines = ["### Queries Where Models Disagreed\n"]
        for d in disagreements[:5]:
            lines.append(f"**Query**: {d['query']}")
            for model in models:
                hit = "✓" if d['hits'][model] else "✗"
                scores = d['top_scores'][model]
                lines.append(f"  - {model}: {hit} | Scores: {scores}")
            lines.append("")

        return "\n".join(lines)

    # ========== Reporting ==========

    def print_comparison_report(self, results: Dict[str, ModelEvalResult]):
        """Print a formatted comparison report."""
        console.print("\n")
        console.print(Panel("[bold]MODEL COMPARISON RESULTS[/bold]", style="blue"))

        # Metrics table
        table = Table(title="Retrieval Metrics", show_lines=True)
        table.add_column("Model", style="cyan")
        table.add_column("Hit@1", justify="right")
        table.add_column("Hit@5", justify="right")
        table.add_column("nDCG@5", justify="right", style="green")
        table.add_column("MRR", justify="right")
        table.add_column("P50 (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")

        best_ndcg = max(r.metrics.ndcg_5 for r in results.values())

        for model_name, result in results.items():
            m = result.metrics
            is_best = m.ndcg_5 == best_ndcg
            ndcg_str = f"[bold green]{m.ndcg_5:.4f}[/bold green]" if is_best else f"{m.ndcg_5:.4f}"

            table.add_row(
                model_name,
                f"{m.hit_rate_1:.1%}",
                f"{m.hit_rate_5:.1%}",
                ndcg_str,
                f"{m.mrr:.4f}",
                f"{m.latency_p50:.0f}",
                f"{m.latency_p95:.0f}"
            )

        console.print(table)

        # Winner
        best_model = max(results.keys(), key=lambda m: results[m].metrics.ndcg_5)
        console.print(f"\n[bold green]🏆 Best Model (by nDCG@5): {best_model}[/bold green]")

    def print_ai_analysis(self, analysis: str):
        """Print the AI analysis with formatting."""
        console.print("\n")
        console.print(Panel("[bold]AI-POWERED ANALYSIS[/bold]", style="magenta"))
        console.print(Markdown(analysis))


def run_full_evaluation(
    engine,
    project_id: str,
    models: List[str],
    num_test_queries: int = 20,
    strategy: str = "hybrid",
    generate_analysis: bool = True
) -> Dict[str, Any]:
    """
    Run a complete evaluation workflow.

    1. Generate test set from first model's collection
    2. Evaluate all models
    3. Generate AI analysis
    4. Print report
    """
    from .search_engine import MaxQEngine

    evaluator = EnhancedEvaluator(engine)

    # Generate test set from first model's collection
    first_model = models[0]
    collection_name = MaxQEngine.get_collection_name(project_id, first_model)

    testset = evaluator.generate_testset_from_collection(
        collection_name=collection_name,
        num_samples=num_test_queries,
        use_llm=True
    )

    if not testset:
        console.print("[red]Failed to generate test set.[/red]")
        return {}

    # Evaluate all models
    results = evaluator.compare_models(
        project_id=project_id,
        models=models,
        testset=testset,
        strategy=strategy
    )

    # Print comparison report
    evaluator.print_comparison_report(results)

    # Generate and print AI analysis
    if generate_analysis and evaluator.llm_client:
        console.print("\n[dim]Generating AI analysis...[/dim]")
        analysis = evaluator.generate_ai_analysis(results, testset)
        evaluator.print_ai_analysis(analysis)

    return {
        "testset": testset,
        "results": {k: v.metrics.to_dict() for k, v in results.items()},
        "detailed_results": results
    }


# Backward compatibility alias
EnhancedEvaluator = Evaluator

