"""
BhojRAG RAG Generator
=======================
End-to-end Retrieval-Augmented Generation pipeline.
Retrieves relevant chunks, assembles context, generates grounded answers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.rag.llm_backends import BaseLLM, get_llm
from src.rag.prompts import PromptTemplates
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RAGResponse:
    """
    Complete RAG response with provenance tracking.
    
    Attributes:
        query: Original user query.
        answer: Generated answer text.
        sources: List of retrieved chunks used as context.
        prompt_used: Full prompt sent to the LLM.
        model_name: LLM model identifier.
        retriever_name: Retriever used for context retrieval.
        metadata: Additional metadata (timings, scores, etc.).
    """
    query: str
    answer: str
    sources: List[RetrievalResult]
    prompt_used: str
    model_name: str
    retriever_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON logging."""
        return {
            "query": self.query,
            "answer": self.answer,
            "model": self.model_name,
            "retriever": self.retriever_name,
            "num_sources": len(self.sources),
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "score": s.score,
                    "rank": s.rank,
                    "text_preview": s.text[:200],
                }
                for s in self.sources
            ],
            "metadata": self.metadata,
        }


class RAGGenerator:
    """
    RAG pipeline: retrieve → assemble context → generate answer.
    
    Supports multiple prompt strategies and LLM backends.
    Tracks provenance (which chunks were used, what prompt was sent).
    
    Usage:
        generator = RAGGenerator(
            retriever=hybrid_retriever,
            llm=api_llm,
            prompt_template="grounded_qa",
            top_k_context=5,
        )
        response = generator.generate("भोजपुरी के इतिहास का ह?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[BaseLLM] = None,
        prompt_template: str = "grounded_qa",
        top_k_context: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.top_k_context = top_k_context
        self._template_fn = PromptTemplates.get_template(prompt_template)

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        template_override: Optional[str] = None,
    ) -> RAGResponse:
        """
        Generate a grounded answer for a query.
        
        Steps:
            1. Retrieve top-k relevant chunks
            2. Assemble context from retrieved chunks
            3. Format prompt using selected template
            4. Call LLM to generate answer
            5. Return RAGResponse with full provenance
        
        Args:
            query: User query string.
            top_k: Override default top_k_context.
            template_override: Override default prompt template.
            
        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        if self.llm is None:
            raise RuntimeError(
                "No LLM backend configured. Pass llm= to constructor "
                "or call set_llm()."
            )

        k = top_k or self.top_k_context

        # Step 1: Retrieve
        logger.info(f"Retrieving top-{k} chunks for: {query[:80]}...")
        retrieved = self.retriever.retrieve(query, top_k=k)

        # Step 2: Assemble context
        context_texts = [r.text for r in retrieved]

        # Step 3: Format prompt
        template_fn = (
            PromptTemplates.get_template(template_override)
            if template_override
            else self._template_fn
        )
        prompt = template_fn(query, context_texts)

        # Step 4: Generate
        logger.info(f"Generating answer using {self.llm.get_model_name()}...")
        answer = self.llm.generate(prompt)

        # Step 5: Build response
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=retrieved,
            prompt_used=prompt,
            model_name=self.llm.get_model_name(),
            retriever_name=self.retriever.name,
            metadata={
                "top_k": k,
                "template": template_override or self.prompt_template,
                "num_chunks_retrieved": len(retrieved),
            },
        )

        logger.info(f"Generated answer ({len(answer)} chars)")
        return response

    def set_llm(self, llm: BaseLLM) -> None:
        """Replace the LLM backend."""
        self.llm = llm
        logger.info(f"LLM backend set to: {llm.get_model_name()}")

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """Replace the retriever."""
        self.retriever = retriever
        logger.info(f"Retriever set to: {retriever.name}")

    def batch_generate(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[RAGResponse]:
        """Generate answers for multiple queries."""
        return [self.generate(q, top_k=top_k) for q in queries]
