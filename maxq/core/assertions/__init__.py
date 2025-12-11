"""Assertion framework for maxq test evaluation."""

from maxq.core.assertions.base import Assertion, AssertionResult
from maxq.core.assertions.registry import get_assertion, register_assertion, list_assertions
from maxq.core.assertions.basic import (
    NotEmptyAssertion,
    CountAssertion,
    LatencyAssertion,
    ContainsIdAssertion,
)
from maxq.core.assertions.metrics import (
    NDCGAssertion,
    MRRAssertion,
    RecallAssertion,
    PrecisionAssertion,
    HitRateAssertion,
)
from maxq.core.assertions.content import (
    ContainsTextAssertion,
    RegexAssertion,
    FieldEqualsAssertion,
    FieldRangeAssertion,
)
from maxq.core.assertions.llm import (
    LLMRelevanceAssertion,
    LLMRubricAssertion,
)
from maxq.core.assertions.semantic import (
    SemanticSimilarityAssertion,
    SemanticDiversityAssertion,
)

__all__ = [
    "Assertion",
    "AssertionResult",
    "get_assertion",
    "register_assertion",
    "list_assertions",
    # Basic
    "NotEmptyAssertion",
    "CountAssertion",
    "LatencyAssertion",
    "ContainsIdAssertion",
    # Metrics
    "NDCGAssertion",
    "MRRAssertion",
    "RecallAssertion",
    "PrecisionAssertion",
    "HitRateAssertion",
    # Content
    "ContainsTextAssertion",
    "RegexAssertion",
    "FieldEqualsAssertion",
    "FieldRangeAssertion",
    # LLM
    "LLMRelevanceAssertion",
    "LLMRubricAssertion",
    # Semantic
    "SemanticSimilarityAssertion",
    "SemanticDiversityAssertion",
]
