"""
BhojRAG Word-Level BM25 Retriever
====================================
Baseline sparse retriever using standard word-level BM25.
This serves as the control against which character n-gram BM25 is compared.
"""

import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.data.chunker import Chunk
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WordBM25Retriever(BaseRetriever):
    """
    Standard word-level BM25 retriever using rank_bm25.

    Tokenization: whitespace split (appropriate for Devanagari
    which uses spaces between words).

    This is the baseline that character n-gram BM25 should outperform
    for orthographically inconsistent Bhojpuri text.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        name: str = "word_bm25",
    ):
        super().__init__(name=name)
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Okapi] = None

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from corpus chunks.

        Tokenizes each chunk by whitespace and builds the
        BM25Okapi index with configurable k1 and b parameters.
        """
        self._corpus = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._indexed = True
        logger.info(
            f"WordBM25: Indexed {len(chunks)} chunks " f"(k1={self.k1}, b={self.b})"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k chunks using word-level BM25 scoring."""
        if not self._indexed or self._bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self._corpus[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    rank=rank,
                    source=chunk.source,
                    metadata={**chunk.metadata, "retriever": self.name},
                )
            )
        return results

    def save_index(self, path: str) -> None:
        """Persist BM25 index and corpus to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self._bm25,
            "corpus": self._corpus,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved WordBM25 index to {path}")

    def load_index(self, path: str) -> None:
        """Load BM25 index and corpus from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._corpus = data["corpus"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self._indexed = True
        logger.info(f"Loaded WordBM25 index from {path}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization. Sufficient for Devanagari."""
        return text.split()
