"""
Script 04: Build Retrieval Indices
=====================================
Builds BM25 and FAISS indices for all retriever variants.

Creates indices for:
  1. Word-level BM25 (baseline)
  2. Character n-gram BM25 (core innovation)
  3. Zero-shot dense (pretrained MuRIL)
  4. Fine-tuned dense (if available)

Usage:
    python scripts/04_build_indices.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.chunker import Chunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_bm25 import WordBM25Retriever
from src.retrieval.sparse_ngram_bm25 import CharNgramBM25Retriever
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_jsonl
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def load_chunks(processed_dir: str):
    """Load processed chunks from JSONL."""
    chunks_path = Path(processed_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error(f"Chunks not found: {chunks_path}")
        logger.error("Run scripts/01_prepare_data.py first.")
        sys.exit(1)

    records = load_jsonl(str(chunks_path))
    return [
        Chunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            doc_id=r["doc_id"],
            source=r["source"],
            chunk_index=r["chunk_index"],
            metadata=r.get("metadata", {}),
        )
        for r in records
    ]


def main(config_path: str = "configs/default.yaml") -> None:
    """Build all retrieval indices."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — Build Retrieval Indices")
    logger.info("=" * 60)

    # Load chunks
    chunks = load_chunks(config.data.processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    index_dir = ensure_dir("models/indices")

    # ---------------------------------------------------------------
    # 1. Word-level BM25 (baseline)
    # ---------------------------------------------------------------
    logger.info("[1/4] Building Word BM25 index...")
    word_bm25 = WordBM25Retriever(
        k1=config.sparse.bm25_k1,
        b=config.sparse.bm25_b,
    )
    word_bm25.index(chunks)
    word_bm25.save_index(str(index_dir / "word_bm25.pkl"))

    # Quick sanity check
    test_results = word_bm25.retrieve("भोजपुरी भाषा", top_k=3)
    logger.info(
        f"  Sanity check — top result: {test_results[0].chunk_id} "
        f"(score={test_results[0].score:.3f})"
    )

    # ---------------------------------------------------------------
    # 2. Character n-gram BM25
    # ---------------------------------------------------------------
    logger.info("[2/4] Building Char N-gram BM25 index...")
    ngram_bm25 = CharNgramBM25Retriever(
        ngram_range=config.sparse.ngram_range,
        k1=config.sparse.bm25_k1,
        b=config.sparse.bm25_b,
    )
    ngram_bm25.index(chunks)
    ngram_bm25.save_index(str(index_dir / "ngram_bm25.pkl"))

    test_results = ngram_bm25.retrieve("भोजपूरी भासा", top_k=3)  # variant spelling
    logger.info(
        f"  Sanity check (variant) — top result: {test_results[0].chunk_id} "
        f"(score={test_results[0].score:.3f})"
    )

    # ---------------------------------------------------------------
    # 3. Zero-shot dense (pretrained MuRIL)
    # ---------------------------------------------------------------
    logger.info("[3/4] Building zero-shot dense index...")
    logger.info(f"  Model: {config.dense.model_name}")
    zeroshot_dense = DenseRetriever(
        model_name=config.dense.model_name,
        max_seq_length=config.dense.max_seq_length,
        batch_size=config.dense.batch_size,
        normalize_embeddings=config.dense.normalize_embeddings,
        index_type=config.dense.index_type,
        name="dense_zeroshot",
    )
    zeroshot_dense.index(chunks)
    zeroshot_dense.save_index(str(index_dir / "dense_zeroshot"))

    test_results = zeroshot_dense.retrieve("भोजपुरी के इतिहास", top_k=3)
    logger.info(
        f"  Sanity check — top result: {test_results[0].chunk_id} "
        f"(score={test_results[0].score:.3f})"
    )

    # ---------------------------------------------------------------
    # 4. Fine-tuned dense (if checkpoint exists)
    # ---------------------------------------------------------------
    finetuned_path = Path(config.training.output_dir)
    if finetuned_path.exists() and (finetuned_path / "config.json").exists():
        logger.info("[4/4] Building fine-tuned dense index...")
        finetuned_dense = DenseRetriever(
            model_name=str(finetuned_path),
            max_seq_length=config.dense.max_seq_length,
            batch_size=config.dense.batch_size,
            normalize_embeddings=config.dense.normalize_embeddings,
            index_type=config.dense.index_type,
            name="dense_finetuned",
        )
        finetuned_dense.index(chunks)
        finetuned_dense.save_index(str(index_dir / "dense_finetuned"))

        test_results = finetuned_dense.retrieve("भोजपुरी के इतिहास", top_k=3)
        logger.info(
            f"  Sanity check — top result: {test_results[0].chunk_id} "
            f"(score={test_results[0].score:.3f})"
        )
    else:
        logger.info(
            "[4/4] Skipping fine-tuned dense index "
            f"(no checkpoint at {finetuned_path})"
        )
        logger.info("  Run scripts/03_train_dense.py to create one.")

    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info(f"  Indices saved to: {index_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG Build Indices")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
