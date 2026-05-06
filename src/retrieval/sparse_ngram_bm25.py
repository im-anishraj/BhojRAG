"""
BhojRAG Character N-gram BM25 Retriever
==========================================
Core sparse innovation: BM25 over character n-grams instead of words.

Motivation:
  Bhojpuri has high orthographic variation — the same word can be
  spelled multiple ways in Devanagari due to lack of standardization.
  Word-level BM25 fails because exact tokens rarely match.
  Character n-grams capture subword overlaps, making retrieval robust
  to spelling differences.

Example:
  "भोजपुरी" and "भोजपूरी" share many 3-grams:
    भोज, ोजप, जपु/जपू, पुर/पूर, ुरी/ूरी
  Word BM25 treats these as completely different tokens.
  Char n-gram BM25 captures their overlap.
"""

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.chunker import Chunk
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CharNgramBM25Retriever(BaseRetriever):
    """
    BM25 retriever using character n-gram tokenization.
    
    Instead of splitting text into words, each document is represented
    as a bag of character n-grams. This handles spelling variation
    naturally because similar words share n-gram overlap.
    
    Parameters:
        ngram_range: Tuple (min_n, max_n) for character n-gram extraction.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        k1: float = 1.5,
        b: float = 0.75,
        name: str = "char_ngram_bm25",
    ):
        super().__init__(name=name)
        self.ngram_range = ngram_range
        self.k1 = k1
        self.b = b

        # Index data structures
        self._doc_ngram_freqs: List[Counter] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._df: Counter = Counter()  # document frequency per n-gram
        self._n_docs: int = 0

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build character n-gram BM25 index.
        
        For each chunk:
          1. Extract all character n-grams in the configured range
          2. Count n-gram frequencies (term frequency)
          3. Track document frequency for IDF computation
        """
        self._corpus = chunks
        self._n_docs = len(chunks)
        self._doc_ngram_freqs = []
        self._doc_lengths = []
        self._df = Counter()

        for chunk in chunks:
            ngrams = self._extract_ngrams(chunk.text)
            freq = Counter(ngrams)
            self._doc_ngram_freqs.append(freq)
            self._doc_lengths.append(len(ngrams))

            # Update document frequency (count each n-gram once per doc)
            for ng in freq:
                self._df[ng] += 1

        self._avg_doc_length = (
            np.mean(self._doc_lengths) if self._doc_lengths else 1.0
        )
        self._indexed = True

        logger.info(
            f"CharNgramBM25: Indexed {self._n_docs} chunks "
            f"(ngram_range={self.ngram_range}, "
            f"vocab_size={len(self._df)}, "
            f"avg_doc_len={self._avg_doc_length:.1f})"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks using character n-gram BM25 scoring.
        
        BM25 formula per n-gram q in query:
          score(q, D) = IDF(q) * (tf(q,D) * (k1+1)) / 
                        (tf(q,D) + k1 * (1 - b + b * |D|/avgdl))
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_ngrams = self._extract_ngrams(query)
        query_freq = Counter(query_ngrams)

        scores = np.zeros(self._n_docs, dtype=np.float64)

        for ng, qf in query_freq.items():
            if ng not in self._df:
                continue

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self._df[ng]
            idf = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
            )

            for doc_idx in range(self._n_docs):
                tf = self._doc_ngram_freqs[doc_idx].get(ng, 0)
                if tf == 0:
                    continue

                doc_len = self._doc_lengths[doc_idx]
                # BM25 TF component
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (
                        1 - self.b + self.b * doc_len / self._avg_doc_length
                    )
                )
                scores[doc_idx] += idf * tf_norm

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                break
            chunk = self._corpus[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[idx]),
                rank=rank,
                source=chunk.source,
                metadata={
                    **chunk.metadata,
                    "retriever": self.name,
                    "ngram_range": str(self.ngram_range),
                },
            ))
        return results

    def _extract_ngrams(self, text: str) -> List[str]:
        """
        Extract character n-grams from text.
        
        Strips whitespace before n-gram extraction so that
        word boundaries don't fragment n-grams. Adds boundary
        markers ('^' and '$') to capture word-start/end patterns.
        """
        # Remove spaces for character-level n-gram extraction
        # but add word boundary markers
        words = text.split()
        ngrams: List[str] = []

        for word in words:
            # Add boundary markers
            marked = f"^{word}$"
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(marked) - n + 1):
                    ngrams.append(marked[i : i + n])

        return ngrams

    def save_index(self, path: str) -> None:
        """Persist n-gram BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self._corpus,
            "doc_ngram_freqs": self._doc_ngram_freqs,
            "doc_lengths": self._doc_lengths,
            "avg_doc_length": self._avg_doc_length,
            "df": self._df,
            "n_docs": self._n_docs,
            "ngram_range": self.ngram_range,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved CharNgramBM25 index to {path}")

    def load_index(self, path: str) -> None:
        """Load n-gram BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._doc_ngram_freqs = data["doc_ngram_freqs"]
        self._doc_lengths = data["doc_lengths"]
        self._avg_doc_length = data["avg_doc_length"]
        self._df = data["df"]
        self._n_docs = data["n_docs"]
        self.ngram_range = data["ngram_range"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self._indexed = True
        logger.info(f"Loaded CharNgramBM25 index from {path}")
