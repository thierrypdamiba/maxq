"""Standard RAG Pipeline - retrieve documents and generate answer."""

import os
import time
from typing import Optional

from maxq.core.rag.pipeline import RAGPipeline, RAGResult, RetrievedDocument


STANDARD_RAG_PROMPT = """Answer the question based on the provided context.
If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


class StandardRAG(RAGPipeline):
    """
    Standard RAG Pipeline.

    1. Retrieve top-k documents
    2. Concatenate into context
    3. Generate answer with LLM
    """

    def __init__(
        self,
        retriever,
        generator_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        collection_name: str = "",
        top_k: int = 10,
        prompt_template: str = STANDARD_RAG_PROMPT,
    ):
        super().__init__(
            retriever=retriever,
            generator_model=generator_model,
            api_key=api_key,
            collection_name=collection_name,
            top_k=top_k,
        )
        self.prompt_template = prompt_template
        self._openai_client = None

    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            import openai

            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def generate(self, query: str, context: str) -> tuple[str, float]:
        """
        Generate answer using LLM.

        Args:
            query: The user's question
            context: Formatted context from retrieved documents

        Returns:
            Tuple of (answer, latency_ms)
        """
        start = time.perf_counter()

        prompt = self.prompt_template.format(
            context=context,
            question=query,
        )

        response = self.openai_client.chat.completions.create(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content or ""
        latency_ms = (time.perf_counter() - start) * 1000

        return answer, latency_ms

    def run(self, query: str) -> RAGResult:
        """
        Run the Standard RAG pipeline.

        Args:
            query: The user's question

        Returns:
            RAGResult with answer and metadata
        """
        total_start = time.perf_counter()

        # Step 1: Retrieve documents
        docs, retrieval_latency = self.retrieve(query)

        # Step 2: Format context
        context = self.format_context(docs)

        # Step 3: Generate answer
        answer, generation_latency = self.generate(query, context)

        total_latency = (time.perf_counter() - total_start) * 1000

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_docs=docs,
            total_latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            pipeline_type="StandardRAG",
            model_used=self.generator_model,
            num_docs_retrieved=len(docs),
        )
