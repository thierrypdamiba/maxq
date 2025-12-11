"""
Speculative RAG Pipeline - Google's draft-then-verify approach.

Based on: "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting"
https://arxiv.org/abs/2407.08223

Key components:
1. Multi-perspective sampling: Cluster docs and sample diverse subsets
2. Parallel drafting: Small model generates drafts + rationales
3. Verification: Large model scores drafts by self-consistency
4. Selection: Pick highest-scoring draft
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from maxq.core.rag.pipeline import RAGPipeline, RAGResult, RetrievedDocument


class Draft(BaseModel):
    """A draft answer with rationale."""

    answer: str
    rationale: str
    doc_subset: list[str] = Field(default_factory=list)  # doc_ids used
    draft_score: float = 0.0  # P(answer, rationale | query, docs)
    self_consistency_score: float = 0.0  # P(answer, rationale | query)
    self_reflection_score: float = 0.0  # P("Yes" | query, answer, rationale, reflection)
    final_score: float = 0.0  # Combined score
    latency_ms: float = 0.0


DRAFTING_PROMPT = """You are a helpful assistant that answers questions based on provided evidence.

## Evidence:
{context}

## Question: {question}

First, provide a brief rationale explaining how the evidence supports your answer.
Then, provide your answer.

## Rationale: (explain why based on the evidence)
"""

VERIFICATION_PROMPT = """Given the following question, rationale, and answer, evaluate if the rationale supports the answer.

Question: {question}

Rationale: {rationale}

Answer: {answer}

Do you think the rationale supports the answer? (Yes or No)
"""


class SpeculativeRAG(RAGPipeline):
    """
    Speculative RAG Pipeline.

    Implements the draft-then-verify approach:
    1. Retrieve top-k documents
    2. Cluster documents by similarity
    3. Sample m diverse subsets (one doc from each cluster)
    4. Generate m drafts in parallel with small model
    5. Verify drafts with large model
    6. Select best draft
    """

    def __init__(
        self,
        retriever,
        drafter_model: str = "gpt-4o-mini",
        verifier_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        collection_name: str = "",
        top_k: int = 10,
        num_drafts: int = 5,
        docs_per_draft: int = 2,
        num_clusters: Optional[int] = None,
        parallel_drafts: bool = True,
    ):
        """
        Initialize Speculative RAG pipeline.

        Args:
            retriever: The retrieval client
            drafter_model: Small model for drafting (fast)
            verifier_model: Large model for verification (accurate)
            api_key: OpenAI API key
            collection_name: Collection to search
            top_k: Number of documents to retrieve
            num_drafts: Number of draft answers to generate (m)
            docs_per_draft: Documents per draft subset (k)
            num_clusters: Number of clusters (defaults to docs_per_draft)
            parallel_drafts: Whether to generate drafts in parallel
        """
        super().__init__(
            retriever=retriever,
            generator_model=drafter_model,
            api_key=api_key,
            collection_name=collection_name,
            top_k=top_k,
        )
        self.drafter_model = drafter_model
        self.verifier_model = verifier_model
        self.num_drafts = num_drafts
        self.docs_per_draft = docs_per_draft
        self.num_clusters = num_clusters or docs_per_draft
        self.parallel_drafts = parallel_drafts
        self._openai_client = None

    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            import openai

            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def cluster_documents(
        self,
        docs: list[RetrievedDocument],
        query: str,
    ) -> list[list[RetrievedDocument]]:
        """
        Cluster documents by content similarity.

        Uses simple text-based clustering. For production, use embeddings.

        Args:
            docs: Retrieved documents
            query: The query (for instruction-aware clustering)

        Returns:
            List of document clusters
        """
        if len(docs) <= self.num_clusters:
            # Not enough docs to cluster, return each as its own cluster
            return [[doc] for doc in docs]

        # Simple clustering by score bands (for demo)
        # In production, use k-means on embeddings
        sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)

        clusters: list[list[RetrievedDocument]] = [[] for _ in range(self.num_clusters)]
        for i, doc in enumerate(sorted_docs):
            cluster_idx = i % self.num_clusters
            clusters[cluster_idx].append(doc)

        return [c for c in clusters if c]  # Remove empty clusters

    def sample_subsets(
        self,
        clusters: list[list[RetrievedDocument]],
    ) -> list[list[RetrievedDocument]]:
        """
        Sample m diverse document subsets.

        Each subset contains one document from each cluster,
        ensuring diversity across perspectives.

        Args:
            clusters: Document clusters

        Returns:
            List of document subsets for drafting
        """
        subsets = []
        seen_combinations = set()

        max_attempts = self.num_drafts * 10
        attempts = 0

        while len(subsets) < self.num_drafts and attempts < max_attempts:
            attempts += 1

            # Sample one doc from each cluster
            subset = []
            doc_ids = []
            for cluster in clusters:
                if cluster:
                    doc = random.choice(cluster)
                    subset.append(doc)
                    doc_ids.append(doc.doc_id)

            # Check if this combination is unique
            combination = tuple(sorted(doc_ids))
            if combination not in seen_combinations and len(subset) >= self.docs_per_draft:
                seen_combinations.add(combination)
                subsets.append(subset[: self.docs_per_draft])

        return subsets

    def generate_draft(
        self,
        query: str,
        doc_subset: list[RetrievedDocument],
    ) -> Draft:
        """
        Generate a draft answer with rationale.

        Args:
            query: The user's question
            doc_subset: Documents to use as context

        Returns:
            Draft with answer and rationale
        """
        start = time.perf_counter()

        # Format context
        context_parts = []
        for i, doc in enumerate(doc_subset, 1):
            context_parts.append(f"[{i}] {doc.text}")
        context = "\n\n".join(context_parts)

        # Generate draft
        prompt = DRAFTING_PROMPT.format(context=context, question=query)

        response = self.openai_client.chat.completions.create(
            model=self.drafter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Some diversity in drafts
            max_tokens=500,
        )

        # Parse response
        content = response.choices[0].message.content or ""

        # Extract rationale and answer
        rationale = ""
        answer = content

        if "## Answer:" in content:
            parts = content.split("## Answer:")
            rationale = parts[0].replace("## Rationale:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else content
        elif "\n\n" in content:
            # Assume first part is rationale
            parts = content.split("\n\n", 1)
            rationale = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else content

        # Calculate draft generation probability (simplified)
        # In the paper, this is the actual log probability
        draft_score = 0.7  # Placeholder - would use logprobs in production

        latency = (time.perf_counter() - start) * 1000

        return Draft(
            answer=answer,
            rationale=rationale,
            doc_subset=[d.doc_id for d in doc_subset],
            draft_score=draft_score,
            latency_ms=latency,
        )

    def verify_draft(self, query: str, draft: Draft) -> Draft:
        """
        Verify a draft using the verifier model.

        Computes:
        1. Self-consistency score: P(answer, rationale | query)
        2. Self-reflection score: P("Yes" | query, answer, rationale, reflection)

        Args:
            query: The original question
            draft: Draft to verify

        Returns:
            Draft with verification scores
        """
        # Self-reflection verification
        verification_prompt = VERIFICATION_PROMPT.format(
            question=query,
            rationale=draft.rationale,
            answer=draft.answer,
        )

        response = self.openai_client.chat.completions.create(
            model=self.verifier_model,
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.0,
            max_tokens=10,
            logprobs=True,
            top_logprobs=5,
        )

        content = (response.choices[0].message.content or "").strip().lower()

        # Calculate self-reflection score
        if "yes" in content:
            self_reflection_score = 0.9
        elif "no" in content:
            self_reflection_score = 0.1
        else:
            self_reflection_score = 0.5

        # Self-consistency score (simplified)
        # In the paper, this is computed from logprobs
        self_consistency_score = 0.7 if draft.rationale else 0.3

        # Combined score
        final_score = draft.draft_score * self_consistency_score * self_reflection_score

        draft.self_consistency_score = self_consistency_score
        draft.self_reflection_score = self_reflection_score
        draft.final_score = final_score

        return draft

    def run(self, query: str) -> RAGResult:
        """
        Run the Speculative RAG pipeline.

        Args:
            query: The user's question

        Returns:
            RAGResult with answer and metadata
        """
        total_start = time.perf_counter()

        # Step 1: Retrieve documents
        docs, retrieval_latency = self.retrieve(query)

        # Step 2: Cluster documents
        clusters = self.cluster_documents(docs, query)

        # Step 3: Sample diverse subsets
        subsets = self.sample_subsets(clusters)

        # Step 4: Generate drafts (in parallel if enabled)
        drafting_start = time.perf_counter()
        drafts = []

        if self.parallel_drafts and len(subsets) > 1:
            with ThreadPoolExecutor(max_workers=min(len(subsets), 5)) as executor:
                futures = {
                    executor.submit(self.generate_draft, query, subset): i
                    for i, subset in enumerate(subsets)
                }
                for future in as_completed(futures):
                    try:
                        draft = future.result()
                        drafts.append(draft)
                    except Exception as e:
                        print(f"Draft generation failed: {e}")
        else:
            for subset in subsets:
                draft = self.generate_draft(query, subset)
                drafts.append(draft)

        drafting_latency = (time.perf_counter() - drafting_start) * 1000

        # Step 5: Verify drafts
        verification_start = time.perf_counter()
        for draft in drafts:
            self.verify_draft(query, draft)
        verification_latency = (time.perf_counter() - verification_start) * 1000

        # Step 6: Select best draft
        if drafts:
            best_draft = max(drafts, key=lambda d: d.final_score)
            answer = best_draft.answer
            rationale = best_draft.rationale
        else:
            answer = "I could not generate an answer."
            rationale = ""
            best_draft = None

        total_latency = (time.perf_counter() - total_start) * 1000

        # Prepare draft data for result
        draft_data = [
            {
                "answer": d.answer,
                "rationale": d.rationale,
                "doc_subset": d.doc_subset,
                "draft_score": d.draft_score,
                "self_consistency_score": d.self_consistency_score,
                "self_reflection_score": d.self_reflection_score,
                "final_score": d.final_score,
            }
            for d in drafts
        ]

        return RAGResult(
            query=query,
            answer=answer,
            rationale=rationale,
            retrieved_docs=docs,
            drafts=draft_data,
            total_latency_ms=total_latency,
            retrieval_latency_ms=retrieval_latency,
            drafting_latency_ms=drafting_latency,
            verification_latency_ms=verification_latency,
            pipeline_type="SpeculativeRAG",
            model_used=f"{self.drafter_model}/{self.verifier_model}",
            num_docs_retrieved=len(docs),
        )
