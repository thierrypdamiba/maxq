"""Prompts for LLM-as-Judge evaluation."""

FAITHFULNESS_PROMPT = """You are an expert judge evaluating the faithfulness of a generated answer.

Faithfulness measures whether the answer is fully supported by the retrieved context.
An answer is faithful if every claim in it can be traced back to the provided context.

Context:
{context}

Question: {question}

Generated Answer: {answer}

Evaluate the faithfulness of the answer on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.75: Most claims are supported, with minor unsupported details
- 0.5: Some claims are supported, but significant parts are not grounded
- 0.25: Few claims are supported by the context
- 0.0: The answer contradicts the context or is completely unsupported

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""

RELEVANCE_PROMPT = """You are an expert judge evaluating the relevance of a generated answer.

Relevance measures whether the answer actually addresses the question asked.
An answer is relevant if it directly answers what was asked, without going off-topic.

Question: {question}

Generated Answer: {answer}

Evaluate the relevance of the answer on a scale of 0.0 to 1.0:
- 1.0: The answer directly and completely addresses the question
- 0.75: The answer mostly addresses the question with minor tangents
- 0.5: The answer partially addresses the question
- 0.25: The answer barely relates to the question
- 0.0: The answer does not address the question at all

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""

CORRECTNESS_PROMPT = """You are an expert judge evaluating the correctness of a generated answer.

You are given a question, the expected answer, and a generated answer.
Evaluate whether the generated answer is semantically equivalent to the expected answer.

Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {answer}

Evaluate the correctness on a scale of 0.0 to 1.0:
- 1.0: The generated answer is semantically equivalent to the expected answer
- 0.75: The generated answer is mostly correct with minor differences
- 0.5: The generated answer is partially correct
- 0.25: The generated answer has some correct elements but is mostly wrong
- 0.0: The generated answer is completely wrong or contradicts the expected answer

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""

CONTEXT_PRECISION_PROMPT = """You are an expert judge evaluating the precision of retrieved context.

Context Precision measures whether the retrieved documents are relevant to answering the question.

Question: {question}

Retrieved Documents:
{context}

Expected Answer: {expected_answer}

For each document, determine if it contains information relevant to answering the question.
Then calculate the precision as: (relevant documents) / (total documents).

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "relevant_docs": <number>, "total_docs": <number>, "reasoning": "<brief explanation>"}}
"""

CONTEXT_RECALL_PROMPT = """You are an expert judge evaluating the recall of retrieved context.

Context Recall measures whether all the information needed to answer the question was retrieved.

Question: {question}

Retrieved Documents:
{context}

Expected Answer: {expected_answer}

Determine what fraction of the information in the expected answer is present in the retrieved documents.

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""

RATIONALE_QUALITY_PROMPT = """You are an expert judge evaluating the quality of a rationale.

A rationale explains why a particular answer was chosen based on the evidence.
Good rationales are:
1. Faithful to the source documents
2. Logically coherent
3. Concise but complete

Source Documents:
{context}

Question: {question}

Generated Rationale: {rationale}

Generated Answer: {answer}

Evaluate the rationale quality on a scale of 0.0 to 1.0:
- 1.0: Rationale is faithful, logical, and well-explains the answer
- 0.75: Rationale is mostly good with minor issues
- 0.5: Rationale has some issues with faithfulness or logic
- 0.25: Rationale is weak or poorly supported
- 0.0: Rationale is unfaithful, illogical, or missing

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}
"""
