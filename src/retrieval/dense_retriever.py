"""
BhojRAG Dense Retriever
=========================
Dense embedding-based retriever using sentence-transformers + FAISS.
Supports zero-shot and fine-tuned multilingual encoders.

Default base model: google/muril-base-cased (optimized for Indic languages).
"""

import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch

from src.data.chunker import Chunk
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using a sentence-transformer model with FAISS index.
    
    Encodes corpus chunks into dense vectors, builds a FAISS index,
    and retrieves by cosine similarity.
    
    Supports:
        - Any sentence-transformers compatible model
        - Zero-shot (pretrained) or fine-tuned models
        - FAISS IndexFlatIP (exact search) or IVF (approximate)
        - Index save/load for persistence
    
    Usage:
        retriever = DenseRetriever(model_name="google/muril-base-cased")
        retriever.index(chunks)
        results = retriever.retrieve("भोजपुरी के इतिहास", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        max_seq_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        index_type: str = "flat",
        device: Optional[str] = None,
        name: str = "dense",
    ):
        super().__init__(name=name)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.index_type = index_type

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model = None
        self._faiss_index: Optional[faiss.Index] = None
        self._embedding_dim: int = 0

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(
            self.model_name, device=self.device
        )
        self._model.max_seq_length = self.max_seq_length
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded: dim={self._embedding_dim}, "
            f"max_seq_len={self.max_seq_length}"
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to dense vectors."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build FAISS index from corpus chunk embeddings.
        
        Args:
            chunks: List of Chunk objects to index.
        """
        self._corpus = chunks
        texts = [c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self._encode(texts)

        # Build FAISS index
        dim = embeddings.shape[1]
        self._embedding_dim = dim

        if self.index_type == "flat":
            self._faiss_index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            nlist = min(100, len(chunks) // 10 + 1)
            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._faiss_index.train(embeddings)
            self._faiss_index.nprobe = 10
        else:
            self._faiss_index = faiss.IndexFlatIP(dim)

        self._faiss_index.add(embeddings)
        self._indexed = True

        logger.info(
            f"DenseRetriever: Indexed {len(chunks)} chunks "
            f"(dim={dim}, type={self.index_type})"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k chunks by cosine similarity."""
        if not self._indexed or self._faiss_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_emb = self._encode([query])
        scores, indices = self._faiss_index.search(query_emb, top_k)

        results = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), 1
        ):
            if idx < 0:  # FAISS returns -1 for insufficient results
                break
            chunk = self._corpus[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                source=chunk.source,
                metadata={
                    **chunk.metadata,
                    "retriever": self.name,
                    "model": self.model_name,
                },
            ))
        return results

    def batch_retrieve(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[RetrievalResult]]:
        """Batch retrieve for multiple queries (more efficient than sequential)."""
        if not self._indexed or self._faiss_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_embs = self._encode(queries)
        scores, indices = self._faiss_index.search(query_embs, top_k)

        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank, (idx, score) in enumerate(
                zip(indices[q_idx], scores[q_idx]), 1
            ):
                if idx < 0:
                    break
                chunk = self._corpus[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    rank=rank,
                    source=chunk.source,
                    metadata={
                        **chunk.metadata,
                        "retriever": self.name,
                    },
                ))
            all_results.append(results)
        return all_results

    def save_index(self, path: str) -> None:
        """Save FAISS index, corpus metadata, and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._faiss_index, str(path / "index.faiss"))

        # Save corpus metadata (without full text to save space)
        corpus_meta = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "doc_id": c.doc_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self._corpus
        ]
        with open(path / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus_meta, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved DenseRetriever index to {path}")

    def load_index(self, path: str) -> None:
        """Load FAISS index, corpus, and config from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self._embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]

        # Load FAISS index
        self._faiss_index = faiss.read_index(str(path / "index.faiss"))

        # Load corpus
        with open(path / "corpus.json", encoding="utf-8") as f:
            corpus_meta = json.load(f)
        self._corpus = [
            Chunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                doc_id=c["doc_id"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                metadata=c["metadata"],
            )
            for c in corpus_meta
        ]
        self._indexed = True
        logger.info(f"Loaded DenseRetriever index from {path}")
