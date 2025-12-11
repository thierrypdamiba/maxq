"""LLM-powered assertions: llm-relevance, llm-rubric."""

import json
import os
from typing import Any

from maxq.core.assertions.base import Assertion, AssertionResult, EvalContext, SearchResult
from maxq.core.assertions.registry import register_assertion


def _call_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    """Call OpenAI LLM with the given prompt."""
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package required for LLM assertions: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required for LLM assertions")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


def _parse_score(response: str) -> float:
    """Parse a numeric score from LLM response."""
    # Try to find JSON first
    try:
        # Look for JSON object
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            if "score" in data:
                return float(data["score"])
            if "relevance" in data:
                return float(data["relevance"])
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract a decimal number
    import re
    numbers = re.findall(r'(?:^|[^\d.])(\d*\.?\d+)(?:$|[^\d.])', response)
    for num in numbers:
        try:
            score = float(num)
            if 0 <= score <= 1:
                return score
            if 0 <= score <= 10:
                return score / 10
            if 0 <= score <= 100:
                return score / 100
        except ValueError:
            continue

    return 0.0


@register_assertion("llm-relevance")
class LLMRelevanceAssertion(Assertion):
    """Assert that results are relevant to the query using LLM judgment."""

    PROMPT_TEMPLATE = """You are evaluating search result relevance.

Query: {query}

Search Results:
{results}

Rate the overall relevance of these search results to the query on a scale of 0.0 to 1.0:
- 1.0: All results are highly relevant and directly answer the query
- 0.7-0.9: Most results are relevant with minor irrelevant items
- 0.4-0.6: Mixed relevance, some useful results
- 0.1-0.3: Mostly irrelevant with a few useful results
- 0.0: Completely irrelevant

Respond with JSON: {{"score": <number>, "reasoning": "<brief explanation>"}}
"""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.7)
        model = self.config.get("model", "gpt-4o-mini")

        if not results:
            return self._make_result(
                passed=False,
                message="No results to evaluate",
                expected=f"Relevance >= {threshold}",
                actual="N/A",
            )

        # Format results for the prompt
        results_text = ""
        for i, r in enumerate(results[:10], 1):  # Limit to top 10
            text = (r.text or "")[:500]  # Truncate long texts
            results_text += f"{i}. [Score: {r.score:.3f}] {text}\n\n"

        prompt = self.PROMPT_TEMPLATE.format(
            query=context.query,
            results=results_text,
        )

        try:
            response = _call_llm(prompt, model=model)
            score = _parse_score(response)
        except Exception as e:
            return self._make_result(
                passed=False,
                message=f"LLM evaluation failed: {e}",
                expected=f"Relevance >= {threshold}",
                actual="Error",
            )

        passed = score >= threshold
        return self._make_result(
            passed=passed,
            message=f"LLM Relevance: {score:.2f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
            details={"model": model, "response": response[:200]},
        )


@register_assertion("llm-rubric")
class LLMRubricAssertion(Assertion):
    """Assert that results meet a custom rubric using LLM judgment."""

    PROMPT_TEMPLATE = """You are evaluating search results against specific criteria.

Query: {query}

Search Results:
{results}

Evaluation Criteria:
{rubric}

Evaluate how well the search results meet the criteria on a scale of 0.0 to 1.0:
- 1.0: Fully meets all criteria
- 0.7-0.9: Meets most criteria with minor gaps
- 0.4-0.6: Partially meets criteria
- 0.1-0.3: Barely meets criteria
- 0.0: Does not meet criteria at all

Respond with JSON: {{"score": <number>, "reasoning": "<brief explanation>"}}
"""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        threshold = self.config.get("threshold", 0.7)
        model = self.config.get("model", "gpt-4o-mini")
        rubric = self.config.get("rubric", "Results should be relevant and useful")

        if not results:
            return self._make_result(
                passed=False,
                message="No results to evaluate",
                expected=f"Rubric score >= {threshold}",
                actual="N/A",
            )

        # Format results for the prompt
        results_text = ""
        for i, r in enumerate(results[:10], 1):
            text = (r.text or "")[:500]
            results_text += f"{i}. [Score: {r.score:.3f}] {text}\n\n"

        prompt = self.PROMPT_TEMPLATE.format(
            query=context.query,
            results=results_text,
            rubric=rubric,
        )

        try:
            response = _call_llm(prompt, model=model)
            score = _parse_score(response)
        except Exception as e:
            return self._make_result(
                passed=False,
                message=f"LLM evaluation failed: {e}",
                expected=f"Rubric score >= {threshold}",
                actual="Error",
            )

        passed = score >= threshold
        return self._make_result(
            passed=passed,
            message=f"LLM Rubric: {score:.2f} {'≥' if passed else '<'} {threshold}",
            expected=f">= {threshold}",
            actual=score,
            details={"model": model, "rubric": rubric, "response": response[:200]},
        )
