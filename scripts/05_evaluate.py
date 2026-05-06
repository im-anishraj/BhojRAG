"""
Script 05: Full Evaluation & Ablation
========================================
Evaluates all retriever variants against the synthetic QA evaluation set.
Runs ablation studies on n-gram size and RRF k parameter.
Generates comparison tables and error analysis reports.

Usage:
    python scripts/05_evaluate.py --config configs/default.yaml
    python scripts/05_evaluate.py --config configs/default.yaml --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger, ExperimentTracker
from src.utils.io import load_jsonl, ensure_dir
from src.data.chunker import Chunk
from src.retrieval.sparse_bm25 import WordBM25Retriever
from src.retrieval.sparse_ngram_bm25 import CharNgramBM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.eval.evaluator import RetrieverEvaluator
from src.eval.error_analysis import ErrorAnalyzer

logger = setup_logger(__name__)


def load_chunks(processed_dir: str):
    """Load processed chunks."""
    records = load_jsonl(str(Path(processed_dir) / "chunks.jsonl"))
    return [
        Chunk(
            chunk_id=r["chunk_id"], text=r["text"], doc_id=r["doc_id"],
            source=r["source"], chunk_index=r["chunk_index"],
            metadata=r.get("metadata", {}),
        )
        for r in records
    ]


def main(config_path: str = "configs/default.yaml", dry_run: bool = False) -> None:
    """Run full evaluation and ablation studies."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — Full Evaluation & Ablation")
    logger.info("=" * 60)

    # Verify required files
    chunks_path = Path(config.data.processed_dir) / "chunks.jsonl"
    qa_path = Path(config.data.qa_pairs_path)
    for required in [chunks_path, qa_path]:
        if not required.exists():
            logger.error(f"Required file not found: {required}")
            logger.error("Run scripts/01_prepare_data.py and 02_generate_qa.py first.")
            sys.exit(1)

    chunks = load_chunks(config.data.processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    if dry_run:
        logger.info("DRY RUN — verifying pipeline setup only")
        logger.info("All required files found. Pipeline is ready.")
        return

    # Setup tracker
    tracker = ExperimentTracker(
        experiment_name=f"eval_{config.experiment.name}",
        tracking_mode=config.experiment.tracking,
        output_dir=config.evaluation.results_dir,
        mlflow_uri=config.experiment.mlflow_uri,
    )
    tracker.start_run(run_name="full_evaluation")

    # ---------------------------------------------------------------
    # Build all retrievers
    # ---------------------------------------------------------------
    retrievers: Dict[str, object] = {}

    # 1. Word BM25
    logger.info("Building Word BM25...")
    word_bm25 = WordBM25Retriever(k1=config.sparse.bm25_k1, b=config.sparse.bm25_b)
    word_bm25.index(chunks)
    retrievers["word_bm25"] = word_bm25

    # 2. Char n-gram BM25
    logger.info("Building Char N-gram BM25...")
    ngram_bm25 = CharNgramBM25Retriever(
        ngram_range=config.sparse.ngram_range,
        k1=config.sparse.bm25_k1, b=config.sparse.bm25_b,
    )
    ngram_bm25.index(chunks)
    retrievers["char_ngram_bm25"] = ngram_bm25

    # 3. Zero-shot dense
    logger.info("Building Zero-shot Dense...")
    zeroshot = DenseRetriever(
        model_name=config.dense.model_name,
        max_seq_length=config.dense.max_seq_length,
        batch_size=config.dense.batch_size,
        normalize_embeddings=config.dense.normalize_embeddings,
        name="dense_zeroshot",
    )
    zeroshot.index(chunks)
    retrievers["dense_zeroshot"] = zeroshot

    # 4. Fine-tuned dense (if available)
    ft_path = Path(config.training.output_dir)
    if ft_path.exists() and (ft_path / "config.json").exists():
        logger.info("Building Fine-tuned Dense...")
        finetuned = DenseRetriever(
            model_name=str(ft_path),
            max_seq_length=config.dense.max_seq_length,
            batch_size=config.dense.batch_size,
            normalize_embeddings=config.dense.normalize_embeddings,
            name="dense_finetuned",
        )
        finetuned.index(chunks)
        retrievers["dense_finetuned"] = finetuned

        # 5. Hybrid (char n-gram + fine-tuned dense)
        logger.info("Building Hybrid (ngram + finetuned)...")
        hybrid = HybridRetriever(
            retrievers=[ngram_bm25, finetuned],
            method=config.hybrid.method,
            rrf_k=config.hybrid.rrf_k,
            name="hybrid_rrf",
        )
        hybrid._indexed = True
        retrievers["hybrid_rrf"] = hybrid
    else:
        logger.warning(f"No fine-tuned model at {ft_path}, skipping finetuned & hybrid")

        # Hybrid with zero-shot as fallback
        logger.info("Building Hybrid (ngram + zeroshot)...")
        hybrid = HybridRetriever(
            retrievers=[ngram_bm25, zeroshot],
            method=config.hybrid.method,
            rrf_k=config.hybrid.rrf_k,
            name="hybrid_rrf_zeroshot",
        )
        hybrid._indexed = True
        retrievers["hybrid_rrf_zeroshot"] = hybrid

    # ---------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Running evaluation...")
    logger.info("=" * 60)

    evaluator = RetrieverEvaluator(
        eval_data_path=str(qa_path),
        results_dir=config.evaluation.results_dir,
        tracker=tracker,
    )

    results = evaluator.evaluate_all(retrievers, top_k=10)

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    header = f"{'Retriever':<25}"
    metric_names = list(next(iter(results.values())).keys())
    for m in metric_names:
        header += f" {m:>12}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<25}"
        for m in metric_names:
            row += f" {metrics[m]:>12.4f}"
        logger.info(row)

    # ---------------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------------
    if config.evaluation.error_analysis:
        logger.info("\nRunning error analysis...")
        analyzer = ErrorAnalyzer(
            output_dir=str(Path(config.evaluation.results_dir) / "error_analysis")
        )

        # Compare best sparse vs best dense
        sparse_key = "char_ngram_bm25"
        dense_key = (
            "dense_finetuned" if "dense_finetuned" in retrievers
            else "dense_zeroshot"
        )

        eval_records, gold = evaluator.load_eval_data()
        summary = analyzer.analyze(
            sparse=retrievers[sparse_key],
            dense=retrievers[dense_key],
            queries=eval_records,
            gold=gold,
            top_k=10,
        )
        logger.info(f"Error analysis summary: {json.dumps(summary, indent=2)}")

    # ---------------------------------------------------------------
    # N-gram ablation
    # ---------------------------------------------------------------
    if config.evaluation.ablation.ngram_sizes:
        logger.info("\n" + "=" * 60)
        logger.info("N-gram Size Ablation")
        logger.info("=" * 60)

        ablation_results = {}
        for ngram_range in config.evaluation.ablation.ngram_sizes:
            label = f"ngram_{ngram_range[0]}_{ngram_range[1]}"
            logger.info(f"  Testing n-gram range {ngram_range}...")

            ablation_retriever = CharNgramBM25Retriever(
                ngram_range=tuple(ngram_range),
                k1=config.sparse.bm25_k1,
                b=config.sparse.bm25_b,
                name=label,
            )
            ablation_retriever.index(chunks)

            ablation_eval = RetrieverEvaluator(
                eval_data_path=str(qa_path),
                results_dir=str(
                    Path(config.evaluation.results_dir) / "ablation_ngram"
                ),
            )
            eval_records, gold = ablation_eval.load_eval_data()
            result = ablation_eval.evaluate_retriever(
                ablation_retriever, eval_records, gold, top_k=10,
            )
            ablation_results[label] = result["aggregate_metrics"]
            logger.info(f"    {label}: {result['aggregate_metrics']}")

        # Save ablation results
        ablation_path = (
            Path(config.evaluation.results_dir) / "ablation_ngram_results.json"
        )
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        logger.info(f"  Ablation results saved to {ablation_path}")

    tracker.log_params({"retrievers_evaluated": list(retrievers.keys())})
    tracker.end_run()

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {config.evaluation.results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG Evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify setup without running evaluation",
    )
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
