"""
BhojRAG Evaluator
===================
End-to-end evaluation runner that benchmarks all retriever variants
against a gold-standard QA evaluation set and logs results.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.chunker import Chunk
from src.eval.metrics import RetrievalMetrics
from src.retrieval.base import BaseRetriever
from src.utils.io import load_jsonl, ensure_dir
from src.utils.logger import setup_logger, ExperimentTracker

logger = setup_logger(__name__)


class RetrieverEvaluator:
    """
    Evaluate multiple retrievers against a gold QA dataset.

    Runs each retriever on the eval queries, computes IR metrics,
    and saves results for comparison and ablation analysis.

    Usage:
        evaluator = RetrieverEvaluator(
            eval_data_path="data/synthetic/qa_pairs.jsonl",
            results_dir="outputs/eval_results",
        )

        evaluator.evaluate_all(
            retrievers={
                "word_bm25": word_retriever,
                "hybrid": hybrid_retriever,
            },
            top_k=10,
        )
    """

    def __init__(
        self,
        eval_data_path: str,
        results_dir: str = "outputs/eval_results",
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.eval_data_path = eval_data_path
        self.results_dir = Path(results_dir)
        self.tracker = tracker
        self.metrics = RetrievalMetrics()

    def load_eval_data(
        self,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
        """
        Load evaluation data and build gold relevance judgments.

        Expected JSONL format:
            {"question": "...", "chunk_id": "...", "answer": "..."}

        Returns:
            (eval_records, gold_dict)
            where gold_dict maps query_id -> set of relevant chunk_ids
        """

        records = load_jsonl(self.eval_data_path)
        gold: Dict[str, Set[str]] = {}

        for i, record in enumerate(records):
            qid = f"q_{i:04d}"
            record["query_id"] = qid

            chunk_id = record.get("chunk_id", "")

            if chunk_id:
                if qid not in gold:
                    gold[qid] = set()

                gold[qid].add(chunk_id)

        logger.info(
            f"Loaded {len(records)} eval queries with "
            f"{len(gold)} gold judgments"
        )

        return records, gold

    def evaluate_retriever(
        self,
        retriever: BaseRetriever,
        eval_records: List[Dict[str, Any]],
        gold: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate a single retriever.

        Returns:
            Dict with aggregate metrics and per-query results.
        """

        results: Dict[str, List[str]] = {}
        per_query_details: List[Dict[str, Any]] = []

        for record in eval_records:
            qid = record["query_id"]
            query = record["question"]

            retrieved = retriever.retrieve(query, top_k=top_k)

            results[qid] = [r.chunk_id for r in retrieved]

            per_query_details.append(
                {
                    "query_id": qid,
                    "query": query,
                    "gold_chunks": list(gold.get(qid, set())),
                    "retrieved_chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.score,
                            "rank": r.rank,
                        }
                        for r in retrieved
                    ],
                }
            )

        # Compute aggregate metrics
        aggregate = self.metrics.compute_all(
            results,
            gold,
            k=top_k,
        )

        # Compute per-query metrics
        per_query_metrics = self.metrics.compute_per_query(
            results,
            gold,
            k=top_k,
        )

        return {
            "retriever": retriever.name,
            "top_k": top_k,
            "aggregate_metrics": aggregate,
            "per_query_metrics": per_query_metrics,
            "per_query_details": per_query_details,
        }

    def evaluate_all(
        self,
        retrievers: Dict[str, BaseRetriever],
        top_k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all retrievers and save comparison results.

        Args:
            retrievers:
                Dict mapping name -> BaseRetriever instance.

            top_k:
                Number of results to retrieve per query.

        Returns:
            Dict mapping retriever name -> aggregate metrics.
        """

        eval_records, gold = self.load_eval_data()

        ensure_dir(self.results_dir)

        all_aggregate: Dict[str, Dict[str, float]] = {}
        all_details: List[Dict[str, Any]] = []

        for name, retriever in retrievers.items():

            logger.info(f"Evaluating retriever: {name}")

            result = self.evaluate_retriever(
                retriever,
                eval_records,
                gold,
                top_k,
            )

            all_aggregate[name] = result["aggregate_metrics"]

            all_details.append(result)

            # Log metrics safely for MLflow
            if self.tracker:

                safe_metrics = {
                    f"{name}/{metric_name.replace('@', '_at_')}": value
                    for metric_name, value in result[
                        "aggregate_metrics"
                    ].items()
                }

                self.tracker.log_metrics(safe_metrics)

            logger.info(
                f"{name}: {result['aggregate_metrics']}"
            )

        # Save outputs
        self._save_aggregate_csv(all_aggregate)

        self._save_details_json(all_details)

        return all_aggregate

    def _save_aggregate_csv(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Save aggregate comparison table as CSV.
        """

        output_path = (
            self.results_dir / "aggregate_results.csv"
        )

        if not results:
            return

        metric_names = list(
            next(iter(results.values())).keys()
        )

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["retriever"] + metric_names
            )

            for name, metrics in results.items():

                row = [
                    name
                ] + [
                    f"{metrics[m]:.4f}"
                    for m in metric_names
                ]

                writer.writerow(row)

        logger.info(
            f"Saved aggregate results to {output_path}"
        )

    def _save_details_json(
        self,
        details: List[Dict[str, Any]],
    ) -> None:
        """
        Save detailed per-query results as JSON.
        """

        output_path = (
            self.results_dir / "detailed_results.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                details,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            f"Saved detailed results to {output_path}"
<<<<<<< HEAD
        )
=======
        )
>>>>>>> e5fcb8a8c5d18c5c4d50d879eff6cfc76d3d42ae
