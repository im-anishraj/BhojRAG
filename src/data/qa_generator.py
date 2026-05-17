"""
BhojRAG Synthetic QA Generator
================================
Generates question-answer pairs from corpus chunks for:
  1. Dense retriever fine-tuning (contrastive learning)
  2. Evaluation set construction

Supports template-based and LLM-based generation.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.chunker import Chunk
from src.utils.io import save_jsonl
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Template-based QA generation
# ---------------------------------------------------------------------------
# Bhojpuri question templates — designed to produce natural questions
# that test retrieval over the source chunk.

BHOJPURI_TEMPLATES = [
    "ई बारे में बताईं कि {snippet}?",
    "{snippet} के बारे में का जानकारी बा?",
    "{snippet} का ह?",
    "ई के बारे में समझाईं: {snippet}",
    "{snippet} कइसे होला?",
    "{snippet} काहे जरूरी बा?",
]

HINDI_TEMPLATES = [
    "{snippet} क्या है?",
    "{snippet} के बारे में बताइए।",
    "{snippet} कैसे होता है?",
    "{snippet} क्यों महत्वपूर्ण है?",
]


class SyntheticQAGenerator:
    """
    Generate synthetic QA pairs from corpus chunks.

    Modes:
        template: Uses predefined question templates with extracted snippets.
        llm: Calls an LLM API to generate questions (requires API key).

    Usage:
        generator = SyntheticQAGenerator(method="template", num_per_chunk=2)
        qa_pairs = generator.generate(chunks)
        generator.save(qa_pairs, "data/synthetic/qa_pairs.jsonl")
    """

    def __init__(
        self,
        method: str = "template",
        num_per_chunk: int = 2,
        templates: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.method = method
        self.num_per_chunk = num_per_chunk
        self.templates = templates or (BHOJPURI_TEMPLATES + HINDI_TEMPLATES)
        self.rng = random.Random(seed)

    def generate(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of dicts: {question, answer, chunk_id, source}
        """
        if self.method == "template":
            return self._generate_template(chunks)
        elif self.method == "llm":
            return self._generate_llm(chunks)
        else:
            raise ValueError(f"Unknown QA generation method: {self.method}")

    def _generate_template(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Template-based QA: extract key phrase from chunk, fill template."""
        qa_pairs: List[Dict[str, Any]] = []

        for chunk in chunks:
            words = chunk.text.split()
            if len(words) < 5:
                continue

            for _ in range(self.num_per_chunk):
                # Extract a snippet (3-8 words) from the chunk
                snippet_len = min(self.rng.randint(3, 8), len(words))
                start = self.rng.randint(0, max(0, len(words) - snippet_len))
                snippet = " ".join(words[start : start + snippet_len])

                template = self.rng.choice(self.templates)
                question = template.format(snippet=snippet)

                qa_pairs.append(
                    {
                        "question": question,
                        "answer": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                    }
                )

        logger.info(
            f"Generated {len(qa_pairs)} template QA pairs from " f"{len(chunks)} chunks"
        )
        return qa_pairs

    def _generate_llm(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        LLM-based QA generation.
        Calls an API-compatible LLM to generate questions for each chunk.

        Requires OPENAI_API_KEY or GOOGLE_API_KEY in environment.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for LLM-based QA generation. "
                "Install with: pip install openai"
            ) from exc

        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        qa_pairs: List[Dict[str, Any]] = []
        system_prompt = (
            "You are a question generation system for Bhojpuri text. "
            "Given a passage, generate natural questions in Bhojpuri or Hindi "
            "that can be answered using the passage. Return only the questions, "
            "one per line."
        )

        for i, chunk in enumerate(chunks):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Passage:\n{chunk.text}\n\nGenerate {self.num_per_chunk} questions:",
                        },
                    ],
                    temperature=0.7,
                    max_tokens=256,
                )
                questions = response.choices[0].message.content.strip().split("\n")
                for q in questions[: self.num_per_chunk]:
                    q = q.strip().lstrip("0123456789.-) ")
                    if q:
                        qa_pairs.append(
                            {
                                "question": q,
                                "answer": chunk.text,
                                "chunk_id": chunk.chunk_id,
                                "source": chunk.source,
                            }
                        )
            except Exception as e:
                logger.warning(f"LLM QA generation failed for chunk {i}: {e}")

            if (i + 1) % 10 == 0:
                logger.info(f"LLM QA progress: {i+1}/{len(chunks)} chunks")

        logger.info(f"Generated {len(qa_pairs)} LLM QA pairs")
        return qa_pairs

    def save(self, qa_pairs: List[Dict[str, Any]], path: str | Path) -> None:
        """Save QA pairs to JSONL file."""
        save_jsonl(qa_pairs, path)
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {path}")
