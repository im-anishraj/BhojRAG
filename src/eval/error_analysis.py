"""
BhojRAG Error Analysis
========================
Inspects retrieval failures and analyzes sparse vs. dense disagreements.
Essential for understanding where character n-gram BM25 helps and where
dense retrieval compensates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.retrieval.base import BaseRetriever
from src.utils.io import ensure_dir, save_jsonl
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ErrorAnalyzer:
    """
    Analyze retrieval failures and system disagreements.

    Key analyses:
        1. Failure cases: queries where no relevant doc is retrieved
        2. Sparse-only wins: relevant doc found by sparse but not dense
        3. Dense-only wins: relevant doc found by dense but not sparse
        4. Agreement analysis: how often systems agree on top-1
        5. Rank displacement: how much ranks differ between systems

    Usage:
        analyzer = ErrorAnalyzer(output_dir="outputs/error_analysis")
        analyzer.analyze(
            sparse_retriever, dense_retriever,
            queries, gold, top_k=10,
        )
    """

    def __init__(self, output_dir: str = "outputs/error_analysis"):
        self.output_dir = Path(output_dir)

    def analyze(
        self,
        sparse: BaseRetriever,
        dense: BaseRetriever,
        queries: List[Dict[str, Any]],
        gold: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Run full error analysis comparing sparse and dense retrievers.

        Args:
            sparse: Sparse retriever (e.g., CharNgramBM25).
            dense: Dense retriever.
            queries: List of dicts with "query_id" and "question".
            gold: Dict mapping query_id -> set of relevant chunk_ids.
            top_k: Number of results to analyze.

        Returns:
            Summary statistics dict.
        """
        ensure_dir(self.output_dir)

        failure_cases = []
        sparse_wins = []
        dense_wins = []
        both_succeed = []
        both_fail = []

        for record in queries:
            qid = record["query_id"]
            query = record["question"]
            relevant = gold.get(qid, set())
            if not relevant:
                continue

            sparse_results = sparse.retrieve(query, top_k)
            dense_results = dense.retrieve(query, top_k)

            sparse_ids = {r.chunk_id for r in sparse_results}
            dense_ids = {r.chunk_id for r in dense_results}

            sparse_hit = bool(sparse_ids & relevant)
            dense_hit = bool(dense_ids & relevant)

            case = {
                "query_id": qid,
                "query": query,
                "gold_chunks": list(relevant),
                "sparse_retrieved": [r.chunk_id for r in sparse_results[:5]],
                "dense_retrieved": [r.chunk_id for r in dense_results[:5]],
                "sparse_scores": [r.score for r in sparse_results[:5]],
                "dense_scores": [r.score for r in dense_results[:5]],
            }

            if sparse_hit and not dense_hit:
                sparse_wins.append(case)
            elif dense_hit and not sparse_hit:
                dense_wins.append(case)
            elif sparse_hit and dense_hit:
                both_succeed.append(case)
            else:
                both_fail.append(case)
                failure_cases.append(case)

        # Summary statistics
        total = len(queries)
        summary = {
            "total_queries": total,
            "both_succeed": len(both_succeed),
            "sparse_only_wins": len(sparse_wins),
            "dense_only_wins": len(dense_wins),
            "both_fail": len(both_fail),
            "sparse_recall": (len(both_succeed) + len(sparse_wins)) / max(total, 1),
            "dense_recall": (len(both_succeed) + len(dense_wins)) / max(total, 1),
            "agreement_rate": len(both_succeed) / max(total, 1),
        }

        # Save detailed analysis
        save_jsonl(sparse_wins, self.output_dir / "sparse_only_wins.jsonl")
        save_jsonl(dense_wins, self.output_dir / "dense_only_wins.jsonl")
        save_jsonl(both_fail, self.output_dir / "both_fail.jsonl")
        save_jsonl(both_succeed, self.output_dir / "both_succeed.jsonl")

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Error analysis summary: {summary}")
        logger.info(f"Results saved to {self.output_dir}")

        return summary

    def analyze_rank_displacement(
        self,
        sparse: BaseRetriever,
        dense: BaseRetriever,
        queries: List[Dict[str, Any]],
        gold: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Analyze how much the rank of relevant documents differs
        between sparse and dense retrievers.

        Useful for understanding RRF fusion behavior.
        """
        displacements = []

        for record in queries:
            qid = record["query_id"]
            query = record["question"]
            relevant = gold.get(qid, set())
            if not relevant:
                continue

            sparse_results = sparse.retrieve(query, top_k)
            dense_results = dense.retrieve(query, top_k)

            sparse_rank_map = {r.chunk_id: r.rank for r in sparse_results}
            dense_rank_map = {r.chunk_id: r.rank for r in dense_results}

            for rel_id in relevant:
                s_rank = sparse_rank_map.get(rel_id)
                d_rank = dense_rank_map.get(rel_id)
                displacements.append(
                    {
                        "query_id": qid,
                        "chunk_id": rel_id,
                        "sparse_rank": s_rank,
                        "dense_rank": d_rank,
                        "displacement": (
                            abs(s_rank - d_rank) if s_rank and d_rank else None
                        ),
                    }
                )

        save_jsonl(displacements, self.output_dir / "rank_displacement.jsonl")
        return displacements
