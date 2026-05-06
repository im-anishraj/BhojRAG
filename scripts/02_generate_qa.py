"""
Script 02: Synthetic QA Generation
=====================================
Generates synthetic question-answer pairs from processed corpus chunks
for dense retriever fine-tuning and evaluation.

Usage:
    python scripts/02_generate_qa.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.io import load_jsonl, ensure_dir
from src.data.chunker import Chunk
from src.data.qa_generator import SyntheticQAGenerator

logger = setup_logger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    """Generate synthetic QA pairs from processed chunks."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — Synthetic QA Generation")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Load processed chunks
    # ---------------------------------------------------------------
    chunks_path = Path(config.data.processed_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error(
            f"Chunks file not found: {chunks_path}\n"
            "Run scripts/01_prepare_data.py first."
        )
        sys.exit(1)

    logger.info(f"[Step 1] Loading chunks from {chunks_path}")
    chunk_records = load_jsonl(str(chunks_path))
    chunks = [
        Chunk(
            chunk_id=r["chunk_id"],
            text=r["text"],
            doc_id=r["doc_id"],
            source=r["source"],
            chunk_index=r["chunk_index"],
            metadata=r.get("metadata", {}),
        )
        for r in chunk_records
    ]
    logger.info(f"  Loaded {len(chunks)} chunks")

    # ---------------------------------------------------------------
    # Step 2: Generate QA pairs
    # ---------------------------------------------------------------
    logger.info(
        f"[Step 2] Generating QA pairs (method={config.qa_generation.method}, "
        f"n_per_chunk={config.qa_generation.num_questions_per_chunk})"
    )
    generator = SyntheticQAGenerator(
        method=config.qa_generation.method,
        num_per_chunk=config.qa_generation.num_questions_per_chunk,
        seed=config.experiment.seed,
    )
    qa_pairs = generator.generate(chunks)
    logger.info(f"  Generated {len(qa_pairs)} QA pairs")

    # ---------------------------------------------------------------
    # Step 3: Save
    # ---------------------------------------------------------------
    output_dir = ensure_dir(config.data.synthetic_dir)
    output_path = Path(config.data.qa_pairs_path)
    generator.save(qa_pairs, output_path)

    # Also save a human-readable preview
    preview_path = output_dir / "qa_preview.txt"
    with open(preview_path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(qa_pairs[:20]):
            f.write(f"--- QA Pair {i+1} ---\n")
            f.write(f"Q: {pair['question']}\n")
            f.write(f"A: {pair['answer'][:200]}...\n")
            f.write(f"Chunk: {pair['chunk_id']}\n\n")

    logger.info("=" * 60)
    logger.info("QA generation complete!")
    logger.info(f"  Total QA pairs: {len(qa_pairs)}")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  Preview: {preview_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG QA Generation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
