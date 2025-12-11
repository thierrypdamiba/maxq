"""Semantic assertions: semantic-similarity, semantic-diversity."""

import os
from typing import Any

from maxq.core.assertions.base import Assertion, AssertionResult, EvalContext, SearchResult
from maxq.core.assertions.registry import register_assertion


def _get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Get embeddings for texts using OpenAI."""
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package required for semantic assertions: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for semantic assertions")

    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _pairwise_diversity(embeddings: list[list[float]]) -> float:
    """
    Calculate average pairwise diversity (1 - similarity) between embeddings.

    Higher values = more diverse results.
    """
    if len(embeddings) < 2:
        return 0.0

    total_diversity = 0.0
    count = 0

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = _cosine_similarity(embeddings[i], embeddings[j])
            diversity = 1 - similarity
            total_diversity += diversity
            count += 1

    return total_diversity / count if count > 0 else 0.0


@register_assertion("semantic-similarity")
class SemanticSimilarityAssertion(Assertion):
    """Assert that results are semantically similar to expected text."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.7)
        expected_text = self.config.get("expected", "")
        model = self.config.get("model", "text-embedding-3-small")

        if not expected_text:
            return self._make_result(
                passed=False,
                message="semantic-similarity requires 'expected' text",
                expected="Expected text",
                actual="None",
            )

        if not results:
            return self._make_result(
                passed=False,
                message="No results to evaluate",
                expected=f"Similarity >= {threshold}",
                actual="N/A",
            )

        # Collect texts from results
        result_texts = []
        for r in results[:10]:  # Limit to top 10
            text = r.text or ""
            if text:
                result_texts.append(text[:1000])  # Truncate

        if not result_texts:
            return self._make_result(
                passed=False,
                message="No text content in results",
                expected=f"Similarity >= {threshold}",
                actual="N/A",
            )

        try:
            # Get embeddings for expected and results
            all_texts = [expected_text] + result_texts
            embeddings = _get_embeddings(all_texts, model=model)

            expected_embedding = embeddings[0]
            result_embeddings = embeddings[1:]

            # Calculate similarity to expected for each result
            similarities = [
                _cosine_similarity(expected_embedding, emb)
                for emb in result_embeddings
            ]

            # Use average similarity
            avg_similarity = sum(similarities) / len(similarities)

        except Exception as e:
            return self._make_result(
                passed=False,
                message=f"Semantic evaluation failed: {e}",
                expected=f"Similarity >= {threshold}",
                actual="Error",
            )

        passed = avg_similarity >= threshold
        return self._make_result(
            passed=passed,
            message=f"Semantic Similarity: {avg_similarity:.3f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=avg_similarity,
            details={
                "individual_similarities": similarities[:5],
                "model": model,
            },
        )


@register_assertion("semantic-diversity")
class SemanticDiversityAssertion(Assertion):
    """Assert that results are semantically diverse."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        min_diversity = self.config.get("min_diversity", 0.3)
        model = self.config.get("model", "text-embedding-3-small")

        if len(results) < 2:
            return self._make_result(
                passed=False,
                message="Need at least 2 results to measure diversity",
                expected=f"Diversity >= {min_diversity}",
                actual="N/A",
            )

        # Collect texts from results
        result_texts = []
        for r in results[:10]:
            text = r.text or ""
            if text:
                result_texts.append(text[:1000])

        if len(result_texts) < 2:
            return self._make_result(
                passed=False,
                message="Need at least 2 results with text content",
                expected=f"Diversity >= {min_diversity}",
                actual="N/A",
            )

        try:
            embeddings = _get_embeddings(result_texts, model=model)
            diversity = _pairwise_diversity(embeddings)
        except Exception as e:
            return self._make_result(
                passed=False,
                message=f"Diversity evaluation failed: {e}",
                expected=f"Diversity >= {min_diversity}",
                actual="Error",
            )

        passed = diversity >= min_diversity
        return self._make_result(
            passed=passed,
            message=f"Semantic Diversity: {diversity:.3f} {'≥' if passed else '<'} {min_diversity}",
            expected=f">= {min_diversity}",
            actual=diversity,
            details={"model": model, "num_results": len(result_texts)},
        )
