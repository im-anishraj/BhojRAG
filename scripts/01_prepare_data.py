"""
Script 01: Data Preparation
=============================
Loads raw corpus, preprocesses, chunks, and saves to data/processed/.

Usage:
    python scripts/01_prepare_data.py --config configs/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.io import save_jsonl, ensure_dir
from src.data.ingest import DocumentLoader
from src.data.preprocess import TextPreprocessor
from src.data.chunker import CorpusChunker

logger = setup_logger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    """Run the full data preparation pipeline."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — Data Preparation Pipeline")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Ingest raw documents
    # ---------------------------------------------------------------
    logger.info(f"[Step 1] Loading documents from: {config.data.corpus_path}")
    loader = DocumentLoader()
    documents = loader.load(config.data.corpus_path)
    logger.info(f"  Loaded {len(documents)} raw documents")

    # ---------------------------------------------------------------
    # Step 2: Preprocess
    # ---------------------------------------------------------------
    logger.info("[Step 2] Preprocessing documents...")
    preprocessor = TextPreprocessor(
        normalize_unicode=config.data.normalize_unicode,
        remove_urls=config.data.remove_urls,
        remove_emails=config.data.remove_emails,
        lowercase=config.data.lowercase,
        min_doc_length=config.data.min_doc_length,
        deduplicate=True,
        transliterate=config.data.transliteration.enabled,
    )
    cleaned_docs = preprocessor.process(documents)
    logger.info(f"  After preprocessing: {len(cleaned_docs)} documents")

    # ---------------------------------------------------------------
    # Step 3: Chunk
    # ---------------------------------------------------------------
    logger.info("[Step 3] Chunking documents...")
    chunker = CorpusChunker(
        chunk_size=config.data.chunk_size,
        chunk_overlap=config.data.chunk_overlap,
        method=config.data.chunking_method,
    )
    chunks = chunker.chunk_documents(cleaned_docs)
    logger.info(f"  Generated {len(chunks)} chunks")

    # ---------------------------------------------------------------
    # Step 4: Save processed data
    # ---------------------------------------------------------------
    output_dir = ensure_dir(config.data.processed_dir)

    # Save chunks as JSONL
    chunk_records = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "doc_id": c.doc_id,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    chunks_path = output_dir / "chunks.jsonl"
    save_jsonl(chunk_records, chunks_path)
    logger.info(f"  Saved chunks to {chunks_path}")

    # Save cleaned documents as JSONL
    doc_records = [
        {
            "doc_id": d.doc_id,
            "text": d.text,
            "source": d.source,
            "metadata": d.metadata,
        }
        for d in cleaned_docs
    ]
    docs_path = output_dir / "documents.jsonl"
    save_jsonl(doc_records, docs_path)
    logger.info(f"  Saved documents to {docs_path}")

    # Save corpus stats
    stats = {
        "raw_documents": len(documents),
        "cleaned_documents": len(cleaned_docs),
        "chunks": len(chunks),
        "chunk_size": config.data.chunk_size,
        "chunk_overlap": config.data.chunk_overlap,
        "chunking_method": config.data.chunking_method,
        "avg_chunk_length": sum(len(c.text) for c in chunks) / max(len(chunks), 1),
    }
    stats_path = output_dir / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"  Documents: {stats['raw_documents']} → {stats['cleaned_documents']}")
    logger.info(f"  Chunks: {stats['chunks']}")
    logger.info(f"  Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG Data Preparation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
