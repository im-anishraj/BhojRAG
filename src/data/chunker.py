"""
BhojRAG Corpus Chunker
========================
Splits preprocessed documents into retrieval-ready chunks
with configurable size, overlap, and metadata preservation.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal

from src.data.ingest import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """A single text chunk derived from a document."""

    chunk_id: str
    text: str
    doc_id: str
    source: str
    chunk_index: int
    metadata: Dict[str, str] = field(default_factory=dict)


class CorpusChunker:
    """
    Chunk documents using fixed-window or sentence-based strategies.

    Usage:
        chunker = CorpusChunker(chunk_size=256, chunk_overlap=64)
        chunks = chunker.chunk_documents(documents)
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        method: Literal["fixed", "sentence"] = "fixed",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents and return flat list of Chunk objects."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            if self.method == "sentence":
                doc_chunks = self._chunk_by_sentence(doc)
            else:
                doc_chunks = self._chunk_by_fixed_window(doc)
            all_chunks.extend(doc_chunks)

        logger.info(
            f"Chunked {len(documents)} docs -> {len(all_chunks)} chunks "
            f"(method={self.method}, size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _chunk_by_fixed_window(self, doc: Document) -> List[Chunk]:
        """Split document into fixed-size windows with overlap (whitespace tokens)."""
        words = doc.text.split()
        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        if not words:
            return chunks

        i = 0
        chunk_idx = 0
        while i < len(words):
            window = words[i : i + self.chunk_size]
            chunk_text = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_c{chunk_idx:04d}",
                    text=chunk_text,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunk_index=chunk_idx,
                    metadata={**doc.metadata, "chunk_method": "fixed"},
                )
            )
            i += step
            chunk_idx += 1
            if i < len(words) and len(words) - i < self.chunk_overlap:
                break
        return chunks

    def _chunk_by_sentence(self, doc: Document) -> List[Chunk]:
        """Split on sentence boundaries (।  .  ?  !) then group into chunks."""
        sentences = re.split(r"(?<=[।.?!])\s+", doc.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[Chunk] = []
        current_words: List[str] = []
        chunk_idx = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if (
                current_words
                and len(current_words) + len(sentence_words) > self.chunk_size
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_c{chunk_idx:04d}",
                        text=" ".join(current_words),
                        doc_id=doc.doc_id,
                        source=doc.source,
                        chunk_index=chunk_idx,
                        metadata={**doc.metadata, "chunk_method": "sentence"},
                    )
                )
                chunk_idx += 1
                current_words = current_words[-self.chunk_overlap :]
            current_words.extend(sentence_words)

        if current_words:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_c{chunk_idx:04d}",
                    text=" ".join(current_words),
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunk_index=chunk_idx,
                    metadata={**doc.metadata, "chunk_method": "sentence"},
                )
            )
        return chunks
