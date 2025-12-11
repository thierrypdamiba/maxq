"""
MaxQ Evaluation Module: LLM-as-Judge (Ragas)
============================================

This module uses LLM-based evaluation via the Ragas framework.
An LLM judges whether retrieved contexts are relevant to the query.

EVALUATION METHOD: LLM-as-Judge
- Uses GPT to semantically judge context relevance
- Metrics: context_precision, context_recall (Ragas)
- More nuanced but slower and more expensive
- Good for: Deep qualitative analysis of retrieval quality

For faster, deterministic evaluation, use the default eval.py module
which uses ID-matching (did we retrieve the source document?).

Usage:
    from maxq.eval_llm_judge import LLMJudgeEvaluator

    evaluator = LLMJudgeEvaluator(engine, config)
    testset = evaluator.generate_testset(num_samples=10)
    evaluator.evaluate(testset)
"""

import os
import json
import random
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from openai import OpenAI
from datasets import Dataset

# Ragas Imports - may fail due to dependency issues in some environments
try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    RAGAS_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # TypeError can occur from metaclass conflicts in langchain/ragas
    RAGAS_AVAILABLE = False
    evaluate = None
    context_precision = None
    context_recall = None

console = Console()


class LLMJudgeEvaluator:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.client = OpenAI(api_key=engine.openai_api_key)
        
        # Ensure OpenAI key is available for Ragas (uses LangChain under the hood)
        if engine.openai_api_key:
            os.environ["OPENAI_API_KEY"] = engine.openai_api_key

    def generate_testset(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """
        Generates a synthetic testset (Question, Answer, GroundTruthContext) 
        from the existing collection.
        """
        console.print(f"[dim]Fetching {num_samples} documents for testset generation...[/dim]")
        
        try:
            # Get collection info to know count
            try:
                count = self.engine.client.count(self.config.collection_name).count
            except:
                count = 0
            
            if count == 0:
                console.print("[red]Collection is empty. Cannot generate testset.[/red]")
                return []

            limit = min(100, count)
            
            # Scroll
            points, _ = self.engine.client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                console.print("[red]No documents found in collection to generate testset.[/red]")
                return []
                
            # Sample random points
            samples = random.sample(points, min(num_samples, len(points)))
            
            testset = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]Generating Q&A pairs...[/cyan]", total=len(samples))
                
                for point in samples:
                    text = point.payload.get("_text", "")
                    if not text:
                        progress.update(task, advance=1)
                        continue
                        
                    # Truncate text for prompt
                    context = text[:2000]
                    
                    # Generate Q&A
                    prompt = f"""
                    You are an expert at creating evaluation datasets.
                    Given the following text, generate a specific question that can be answered by the text, and provide the answer.
                    
                    Text:
                    {context}
                    
                    Output format (JSON):
                    {{
                        "question": "The generated question",
                        "answer": "The concise answer based on the text"
                    }}
                    """
                    
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that generates Q&A pairs in JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            response_format={"type": "json_object"}
                        )
                        
                        content = response.choices[0].message.content
                        data = json.loads(content)
                        
                        testset.append({
                            "question": data["question"],
                            "ground_truth": data["answer"],
                            "context_id": point.id,
                            "context_text": text
                        })
                        
                    except Exception as e:
                        # console.print(f"[dim]Error generating for sample: {e}[/dim]")
                        pass
                    
                    progress.update(task, advance=1)
            
            return testset

        except Exception as e:
            console.print(f"[red]Error in testset generation: {e}[/red]")
            return []

    def evaluate(self, testset: List[Dict[str, str]]):
        """
        Runs the evaluation loop using Ragas and standard IR metrics.
        """
        if not testset:
            return

        from .search_engine import SearchRequest
        import time
        import numpy as np

        questions = []
        ground_truths = []
        contexts = []
        answers = [] # Dummy answers for Ragas if needed
        
        # Deterministic Metrics
        latencies = []
        hits = []
        reciprocal_ranks = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[magenta]Running Search & Preparing Ragas...[/magenta]", total=len(testset))
            
            for item in testset:
                question = item["question"]
                ground_truth = item["ground_truth"]
                target_id = item.get("context_id")
                
                questions.append(question)
                ground_truths.append(ground_truth)
                answers.append("N/A") # We are evaluating retrieval, not generation
                
                # Run Search
                req = SearchRequest(
                    query=question,
                    limit=5,
                    strategy="hybrid"
                )
                
                try:
                    start_time = time.time()
                    search_results = self.engine.query(self.config, req)
                    latencies.append(time.time() - start_time)
                    
                    # Extract text from results
                    retrieved_texts = [r.payload.get("_text", "") for r in search_results]
                    contexts.append(retrieved_texts)
                    
                    # Calculate Deterministic Metrics
                    if target_id:
                        found = False
                        for rank, result in enumerate(search_results):
                            if result.id == target_id:
                                hits.append(1)
                                reciprocal_ranks.append(1 / (rank + 1))
                                found = True
                                break
                        if not found:
                            hits.append(0)
                            reciprocal_ranks.append(0)
                            
                except Exception as e:
                    console.print(f"[red]Search failed for '{question}': {e}[/red]")
                    contexts.append([])
                    hits.append(0)
                    reciprocal_ranks.append(0)

                progress.update(task, advance=1)

        # Create Dataset for Ragas
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data_dict)

        console.print("[dim]Calculating Ragas Metrics (Context Recall, Context Precision)...[/dim]")
        
        ragas_results = {}
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            # Initialize models
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=self.engine.openai_api_key)
            embeddings = OpenAIEmbeddings(api_key=self.engine.openai_api_key)
            
            ragas_results = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    context_recall
                ],
                llm=llm,
                embeddings=embeddings
            )
            
        except Exception as e:
            console.print(f"[red]Ragas evaluation failed: {e}[/red]")

        # Prepare combined results
        results = {
            "ragas": ragas_results,
            "retrieval": {
                "hit_rate": np.mean(hits) if hits else 0,
                "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
                "latency_p50": np.percentile(latencies, 50) if latencies else 0,
                "latency_p95": np.percentile(latencies, 95) if latencies else 0,
                "latency_p99": np.percentile(latencies, 99) if latencies else 0,
            }
        }

        self._print_ragas_report(results)

    def _print_ragas_report(self, results):
        # Ragas results object is dict-like
        ragas = results.get("ragas", {})
        retrieval = results.get("retrieval", {})
        
        def get_mean(key):
            if not ragas: return 0.0
            val = ragas[key]
            if isinstance(val, list):
                return sum(val) / len(val) if val else 0.0
            return val
        
        console.print("\n")
        
        # Create a grid layout for metrics
        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="left", ratio=1)
        
        # Ragas Panel
        ragas_panel = Panel(
            f"Context Recall: [green]{get_mean('context_recall'):.4f}[/green]\n"
            f"Context Precision: [blue]{get_mean('context_precision'):.4f}[/blue]",
            title="[bold]Ragas (LLM)[/bold]", border_style="green"
        )
        
        # Retrieval Panel
        retrieval_panel = Panel(
            f"Hit Rate @ 5: [green]{retrieval.get('hit_rate', 0):.4f}[/green]\n"
            f"MRR @ 5: [blue]{retrieval.get('mrr', 0):.4f}[/blue]\n"
            f"Latency (p95): [yellow]{retrieval.get('latency_p95', 0):.3f}s[/yellow]",
            title="[bold]Ranking (Deterministic)[/bold]", border_style="cyan"
        )
        
        grid.add_row(ragas_panel, retrieval_panel)
        console.print(grid)
        
        # Detailed Table (if available in results, usually results is a Result object)
        try:
            if ragas:
                df = ragas.to_pandas()
                table = Table(title="Detailed Ragas Results", show_lines=True)
                table.add_column("Question", style="cyan", no_wrap=False)
                table.add_column("Context Recall", justify="center")
                table.add_column("Context Precision", justify="center")
                
                for index, row in df.iterrows():
                    table.add_row(
                        row['question'],
                        f"{row['context_recall']:.2f}",
                        f"{row['context_precision']:.2f}"
                    )
                console.print(table)
        except:
            pass


# Backward compatibility alias
MaxQEvaluator = LLMJudgeEvaluator
