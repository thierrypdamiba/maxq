"""LLM Judge for RAG evaluation."""

__all__ = [
    "LLMJudge",
    "JudgeResult",
    "FAITHFULNESS_PROMPT",
    "RELEVANCE_PROMPT",
    "CORRECTNESS_PROMPT",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "LLMJudge":
        from maxq.core.judge.llm_judge import LLMJudge

        return LLMJudge
    elif name == "JudgeResult":
        from maxq.core.judge.llm_judge import JudgeResult

        return JudgeResult
    elif name in ("FAITHFULNESS_PROMPT", "RELEVANCE_PROMPT", "CORRECTNESS_PROMPT"):
        from maxq.core.judge import prompts

        return getattr(prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
