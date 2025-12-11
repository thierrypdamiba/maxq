"""LLM-as-Judge for evaluating RAG answer quality."""

import json
import logging
import os
from typing import Optional

from pydantic import BaseModel, Field

from maxq.core.judge.prompts import (
    FAITHFULNESS_PROMPT,
    RELEVANCE_PROMPT,
    CORRECTNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    RATIONALE_QUALITY_PROMPT,
)

logger = logging.getLogger("maxq.judge")


class JudgeResult(BaseModel):
    """Result from an LLM judge evaluation."""

    metric: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    raw_response: str = ""
    error: Optional[str] = None


class LLMJudge:
    """LLM-as-Judge for evaluating RAG quality metrics."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the LLM Judge.

        Args:
            model: OpenAI model to use for judging
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for LLM judging")
        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, response: str, metric: str) -> JudgeResult:
        """Parse JSON response from LLM."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Handle markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            return JudgeResult(
                metric=metric,
                score=float(data.get("score", 0.0)),
                reasoning=data.get("reasoning", ""),
                raw_response=response,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return JudgeResult(
                metric=metric,
                score=0.0,
                reasoning="",
                raw_response=response,
                error=str(e),
            )

    def judge_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """
        Judge whether the answer is faithful to the provided context.

        Args:
            question: The original question
            answer: The generated answer
            context: The retrieved context/documents

        Returns:
            JudgeResult with faithfulness score (0-1)
        """
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            context=context,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "faithfulness")

    def judge_relevance(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """
        Judge whether the answer is relevant to the question.

        Args:
            question: The original question
            answer: The generated answer

        Returns:
            JudgeResult with relevance score (0-1)
        """
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "relevance")

    def judge_correctness(
        self,
        question: str,
        answer: str,
        expected_answer: str,
    ) -> JudgeResult:
        """
        Judge whether the answer is correct compared to expected answer.

        Args:
            question: The original question
            answer: The generated answer
            expected_answer: The ground truth answer

        Returns:
            JudgeResult with correctness score (0-1)
        """
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            answer=answer,
            expected_answer=expected_answer,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "correctness")

    def judge_context_precision(
        self,
        question: str,
        context: str,
        expected_answer: str,
    ) -> JudgeResult:
        """
        Judge the precision of retrieved context (relevance of docs).

        Args:
            question: The original question
            context: The retrieved context/documents
            expected_answer: The ground truth answer

        Returns:
            JudgeResult with context precision score (0-1)
        """
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context=context,
            expected_answer=expected_answer,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "context_precision")

    def judge_context_recall(
        self,
        question: str,
        context: str,
        expected_answer: str,
    ) -> JudgeResult:
        """
        Judge the recall of retrieved context (coverage of needed info).

        Args:
            question: The original question
            context: The retrieved context/documents
            expected_answer: The ground truth answer

        Returns:
            JudgeResult with context recall score (0-1)
        """
        prompt = CONTEXT_RECALL_PROMPT.format(
            question=question,
            context=context,
            expected_answer=expected_answer,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "context_recall")

    def judge_rationale_quality(
        self,
        question: str,
        context: str,
        rationale: str,
        answer: str,
    ) -> JudgeResult:
        """
        Judge the quality of a generated rationale (for Speculative RAG).

        Args:
            question: The original question
            context: The source documents
            rationale: The generated rationale explaining the answer
            answer: The generated answer

        Returns:
            JudgeResult with rationale quality score (0-1)
        """
        prompt = RATIONALE_QUALITY_PROMPT.format(
            question=question,
            context=context,
            rationale=rationale,
            answer=answer,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, "rationale_quality")

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        context: str,
        expected_answer: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> dict[str, JudgeResult]:
        """
        Run full RAG evaluation on a response.

        Args:
            question: The original question
            answer: The generated answer
            context: The retrieved context/documents
            expected_answer: Optional ground truth answer
            rationale: Optional rationale (for Speculative RAG)

        Returns:
            Dictionary of metric name -> JudgeResult
        """
        results = {}

        # Always run faithfulness and relevance
        results["faithfulness"] = self.judge_faithfulness(question, answer, context)
        results["relevance"] = self.judge_relevance(question, answer)

        # Run correctness if we have expected answer
        if expected_answer:
            results["correctness"] = self.judge_correctness(question, answer, expected_answer)
            results["context_precision"] = self.judge_context_precision(
                question, context, expected_answer
            )
            results["context_recall"] = self.judge_context_recall(
                question, context, expected_answer
            )

        # Run rationale quality if we have rationale
        if rationale:
            results["rationale_quality"] = self.judge_rationale_quality(
                question, context, rationale, answer
            )

        return results
