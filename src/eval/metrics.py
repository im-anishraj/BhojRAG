"""
BhojRAG Retrieval Metrics
===========================
Standard IR evaluation metrics for retrieval quality assessment.
Computes MRR, NDCG, Recall, Precision, and MAP at configurable k values.

All functions follow the convention:
    - results: dict mapping query_id -> list of retrieved chunk_ids (ranked)
    - gold: dict mapping query_id -> set of relevant chunk_ids
"""

import math
from typing import Dict, List, Set

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrievalMetrics:
    """
    Compute standard IR evaluation metrics.
    
    Supports:
        - MRR@k (Mean Reciprocal Rank)
        - Recall@k
        - Precision@k
        - NDCG@k (Normalized Discounted Cumulative Gain)
        - MAP (Mean Average Precision)
    
    Usage:
        metrics = RetrievalMetrics()
        results = {"q1": ["c1", "c2", "c3"], "q2": ["c4", "c1"]}
        gold = {"q1": {"c1"}, "q2": {"c4", "c5"}}
        scores = metrics.compute_all(results, gold, k=10)
    """

    @staticmethod
    def mrr_at_k(
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 10,
    ) -> float:
        """
        Mean Reciprocal Rank @ k.
        
        For each query, finds the rank of the first relevant result.
        MRR = (1/|Q|) * Σ (1 / rank_first_relevant)
        """
        rr_sum = 0.0
        n_queries = 0

        for qid, retrieved in results.items():
            if qid not in gold:
                continue
            relevant = gold[qid]
            n_queries += 1

            for rank, doc_id in enumerate(retrieved[:k], 1):
                if doc_id in relevant:
                    rr_sum += 1.0 / rank
                    break

        return rr_sum / n_queries if n_queries > 0 else 0.0

    @staticmethod
    def recall_at_k(
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 5,
    ) -> float:
        """
        Recall @ k.
        
        Fraction of relevant documents retrieved in the top-k.
        Recall@k = |retrieved_k ∩ relevant| / |relevant|
        """
        recall_sum = 0.0
        n_queries = 0

        for qid, retrieved in results.items():
            if qid not in gold:
                continue
            relevant = gold[qid]
            if not relevant:
                continue
            n_queries += 1

            retrieved_set = set(retrieved[:k])
            hits = len(retrieved_set & relevant)
            recall_sum += hits / len(relevant)

        return recall_sum / n_queries if n_queries > 0 else 0.0

    @staticmethod
    def precision_at_k(
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 5,
    ) -> float:
        """
        Precision @ k.
        
        Fraction of top-k results that are relevant.
        P@k = |retrieved_k ∩ relevant| / k
        """
        prec_sum = 0.0
        n_queries = 0

        for qid, retrieved in results.items():
            if qid not in gold:
                continue
            relevant = gold[qid]
            n_queries += 1

            retrieved_set = set(retrieved[:k])
            hits = len(retrieved_set & relevant)
            prec_sum += hits / k

        return prec_sum / n_queries if n_queries > 0 else 0.0

    @staticmethod
    def ndcg_at_k(
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 10,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain @ k.
        
        Uses binary relevance (1 if relevant, 0 otherwise).
        NDCG = DCG / IDCG
        """
        ndcg_sum = 0.0
        n_queries = 0

        for qid, retrieved in results.items():
            if qid not in gold:
                continue
            relevant = gold[qid]
            n_queries += 1

            # DCG
            dcg = 0.0
            for rank, doc_id in enumerate(retrieved[:k], 1):
                if doc_id in relevant:
                    dcg += 1.0 / math.log2(rank + 1)

            # IDCG (best possible ranking)
            n_rel = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_rel + 1))

            if idcg > 0:
                ndcg_sum += dcg / idcg

        return ndcg_sum / n_queries if n_queries > 0 else 0.0

    @staticmethod
    def mean_average_precision(
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
    ) -> float:
        """
        Mean Average Precision (MAP) over all queries.
        
        AP = (1/|relevant|) * Σ_k (P@k * rel(k))
        """
        ap_sum = 0.0
        n_queries = 0

        for qid, retrieved in results.items():
            if qid not in gold:
                continue
            relevant = gold[qid]
            if not relevant:
                continue
            n_queries += 1

            hits = 0
            ap = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    hits += 1
                    ap += hits / rank

            ap_sum += ap / len(relevant)

        return ap_sum / n_queries if n_queries > 0 else 0.0

    def compute_all(
        self,
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Returns dict with keys: mrr@k, recall@k, ndcg@k, precision@k, map
        """
        return {
            f"mrr@{k}": self.mrr_at_k(results, gold, k),
            f"recall@{k}": self.recall_at_k(results, gold, k),
            f"ndcg@{k}": self.ndcg_at_k(results, gold, k),
            f"precision@{k}": self.precision_at_k(results, gold, k),
            "map": self.mean_average_precision(results, gold),
        }

    def compute_per_query(
        self,
        results: Dict[str, List[str]],
        gold: Dict[str, Set[str]],
        k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per-query for detailed error analysis.
        
        Returns dict mapping query_id -> {metric_name: value}.
        """
        per_query: Dict[str, Dict[str, float]] = {}

        for qid in results:
            if qid not in gold:
                continue

            single_result = {qid: results[qid]}
            single_gold = {qid: gold[qid]}

            per_query[qid] = {
                f"mrr@{k}": self.mrr_at_k(single_result, single_gold, k),
                f"recall@{k}": self.recall_at_k(single_result, single_gold, k),
                f"ndcg@{k}": self.ndcg_at_k(single_result, single_gold, k),
                f"precision@{k}": self.precision_at_k(single_result, single_gold, k),
            }

        return per_query
