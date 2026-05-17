"""
BhojRAG Hybrid Retriever (Reciprocal Rank Fusion)
====================================================
Combines sparse and dense retriever results using RRF.

RRF is model-agnostic and does not require score calibration
between heterogeneous retrieval systems, making it ideal for
combining BM25 (sparse) and dense embedding retrieval.

Formula:
    RRF_score(d) = Σ_i  weight_i / (k + rank_i(d))

where i iterates over retrieval systems and k is a constant
(typically 60) that smooths the contribution of lower-ranked docs.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from src.data.chunker import Chunk
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining multiple systems via Reciprocal Rank Fusion.

    Supports:
        - RRF (default): rank-based fusion, no score calibration needed
        - Weighted: normalized score fusion with configurable weights

    Usage:
        hybrid = HybridRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            method="rrf",
            rrf_k=60,
        )
        hybrid.index(chunks)  # delegates to each sub-retriever
        results = hybrid.retrieve("query", top_k=10)
    """

    def __init__(
        self,
        retrievers: Optional[List[BaseRetriever]] = None,
        weights: Optional[List[float]] = None,
        method: str = "rrf",
        rrf_k: int = 60,
        top_k: int = 10,
        name: str = "hybrid",
    ):
        super().__init__(name=name)
        self.retrievers = retrievers or []
        self.method = method
        self.rrf_k = rrf_k
        self.default_top_k = top_k

        # Weights for each retriever (used in weighted fusion)
        if weights:
            if len(weights) != len(self.retrievers):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of retrievers ({len(self.retrievers)})"
                )
            self.weights = weights
        else:
            # Equal weights by default
            self.weights = [1.0] * len(self.retrievers)

    def add_retriever(
        self,
        retriever: BaseRetriever,
        weight: float = 1.0,
    ) -> None:
        """Add a retriever to the hybrid system."""
        self.retrievers.append(retriever)
        self.weights.append(weight)

    def index(self, chunks: List[Chunk]) -> None:
        """
        Build indices for all sub-retrievers.

        Each retriever builds its own index over the same corpus.
        """
        self._corpus = chunks
        for retriever in self.retrievers:
            if not retriever.is_indexed:
                logger.info(f"Indexing sub-retriever: {retriever.name}")
                retriever.index(chunks)
        self._indexed = True
        logger.info(
            f"HybridRetriever: All {len(self.retrievers)} " f"sub-retrievers indexed"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve and fuse results from all sub-retrievers.

        Each sub-retriever returns its own top-k results.
        Results are fused via RRF or weighted combination.
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        # Collect results from each retriever
        # Request more results from each to ensure good fusion coverage
        fetch_k = max(top_k * 2, 50)
        all_results: List[List[RetrievalResult]] = []
        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k=fetch_k)
            all_results.append(results)

        # Fuse results
        if self.method == "rrf":
            fused = self._reciprocal_rank_fusion(all_results)
        elif self.method == "weighted":
            fused = self._weighted_fusion(all_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        all_results: List[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion.

        RRF_score(d) = Σ_i  w_i / (k + rank_i(d))
        """
        # Accumulate RRF scores by chunk_id
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievalResult] = {}
        source_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)

        for sys_idx, results in enumerate(all_results):
            weight = self.weights[sys_idx]
            retriever_name = self.retrievers[sys_idx].name

            for result in results:
                rrf_scores[result.chunk_id] += weight / (self.rrf_k + result.rank)
                # Keep the first occurrence (highest-ranked) for text/metadata
                if result.chunk_id not in chunk_map:
                    chunk_map[result.chunk_id] = result
                # Track per-system ranks for error analysis
                source_ranks[result.chunk_id][retriever_name] = result.rank

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_results = []
        for rank, chunk_id in enumerate(sorted_ids, 1):
            original = chunk_map[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=original.text,
                    score=rrf_scores[chunk_id],
                    rank=rank,
                    source=original.source,
                    metadata={
                        **original.metadata,
                        "retriever": self.name,
                        "fusion_method": "rrf",
                        "source_ranks": str(source_ranks[chunk_id]),
                    },
                )
            )

        return fused_results

    def _weighted_fusion(
        self,
        all_results: List[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        """
        Weighted score fusion with min-max normalization.

        Normalizes scores from each system to [0, 1] range,
        then combines with configurable weights.
        """
        weighted_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievalResult] = {}

        for sys_idx, results in enumerate(all_results):
            if not results:
                continue

            weight = self.weights[sys_idx]

            # Min-max normalize scores
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            score_range = max_s - min_s if max_s != min_s else 1.0

            for result in results:
                norm_score = (result.score - min_s) / score_range
                weighted_scores[result.chunk_id] += weight * norm_score
                if result.chunk_id not in chunk_map:
                    chunk_map[result.chunk_id] = result

        sorted_ids = sorted(
            weighted_scores.keys(),
            key=lambda cid: weighted_scores[cid],
            reverse=True,
        )

        fused_results = []
        for rank, chunk_id in enumerate(sorted_ids, 1):
            original = chunk_map[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=original.text,
                    score=weighted_scores[chunk_id],
                    rank=rank,
                    source=original.source,
                    metadata={
                        **original.metadata,
                        "retriever": self.name,
                        "fusion_method": "weighted",
                    },
                )
            )

        return fused_results
