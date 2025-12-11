"""
Ragas utilities for LLM-as-Judge evaluation.

This module wraps Ragas functionality for test set generation and evaluation.
It requires the ragas and langchain libraries to be properly installed.
"""
import os
from typing import List, Dict, Any

# Optional imports - Ragas may not work in all environments
RAGAS_AVAILABLE = False
try:
    from datasets import Dataset
    from langchain_core.documents import Document as LCDocument
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import default_query_distribution
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    RAGAS_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # May fail due to metaclass conflicts in langchain dependencies
    Dataset = None
    LCDocument = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    TestsetGenerator = None
    evaluate = None
    context_precision = None
    context_recall = None
    faithfulness = None
    answer_relevancy = None


class RagasManager:
    def __init__(self, openai_api_key: str):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas is not available in this environment. "
                "This may be due to dependency conflicts. "
                "Try: pip install ragas langchain-openai"
            )

        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize LangChain models for Ragas
        self.generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.critic_llm = ChatOpenAI(model="gpt-4o") # GPT-4 is better for criticism/evaluation
        self.embeddings = OpenAIEmbeddings()

    def generate_testset(self, engine, collection_name: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Generates a synthetic test set from the Qdrant collection.
        """
        # 1. Fetch documents from Qdrant (Scroll)
        # We fetch a bit more to ensure we have enough content
        points, _ = engine.client.scroll(
            collection_name=collection_name,
            limit=min(100, num_questions * 5), 
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            return []

        # 2. Convert to LangChain Documents
        docs = []
        for point in points:
            text = point.payload.get("_text", "")
            if len(text) > 100: # Filter short texts
                docs.append(LCDocument(page_content=text, metadata={"filename": point.payload.get("filename", "unknown")}))

        if not docs:
            return []

        # 3. Generate Test Set
        generator = TestsetGenerator.from_langchain(
            llm=self.generator_llm,
            embedding_model=self.embeddings,
        )
        
        # Use default distribution (Single Hop, Multi Hop, etc.)
        # We need to pass the generator's LLM to the distribution
        # Note: default_query_distribution requires a KnowledgeGraph if we want to filter, 
        # but here we just want defaults. However, from_langchain initializes an empty KG.
        # We can just let generate_with_langchain_docs handle the default distribution if we pass None.
        
        testset = generator.generate_with_langchain_docs(
            docs, 
            testset_size=num_questions,
            # query_distribution=None # Defaults to default_query_distribution
        )
        
        # Convert to list of dicts for easier handling
        return testset.to_pandas().to_dict(orient="records")

    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates the results using Ragas metrics.
        Input results should be a list of dicts with keys: 'question', 'answer', 'contexts', 'ground_truth'
        """
        if not results:
            return {}

        # Convert to HuggingFace Dataset
        ds = Dataset.from_list(results)
        
        # Run Evaluation
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
        
        scores = evaluate(
            ds,
            metrics=metrics,
        )
        
        return scores
