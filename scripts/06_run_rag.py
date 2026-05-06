"""
Script 06: RAG Generation Demo
=================================
Demonstrates end-to-end RAG: retrieve relevant chunks → generate
grounded answer using an LLM backend.

Usage:
    python scripts/06_run_rag.py --config configs/default.yaml --query "भोजपुरी के इतिहास का ह?"
    python scripts/06_run_rag.py --config configs/default.yaml --interactive
    python scripts/06_run_rag.py --config configs/default.yaml --batch data/synthetic/qa_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.io import load_jsonl, save_jsonl, ensure_dir
from src.data.chunker import Chunk
from src.retrieval.sparse_ngram_bm25 import CharNgramBM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.rag.generator import RAGGenerator
from src.rag.llm_backends import get_llm

logger = setup_logger(__name__)


def build_retriever(config, chunks, sparse_only=False):
    """Build the best available retriever (hybrid > dense > sparse)."""

    # Sparse component
    sparse = CharNgramBM25Retriever(
        ngram_range=config.sparse.ngram_range,
        k1=config.sparse.bm25_k1,
        b=config.sparse.bm25_b,
    )
    sparse.index(chunks)
    
    if sparse_only:
        return sparse

    # Dense component
    ft_path = Path(config.training.output_dir)
    model_name = (
        str(ft_path)
        if ft_path.exists() and (ft_path / "config.json").exists()
        else config.dense.model_name
    )

    dense = DenseRetriever(
        model_name=model_name,
        max_seq_length=config.dense.max_seq_length,
        batch_size=config.dense.batch_size,
        normalize_embeddings=config.dense.normalize_embeddings,
        name="dense",
    )
    dense.index(chunks)

    # Hybrid
    hybrid = HybridRetriever(
        retrievers=[sparse, dense],
        method=config.hybrid.method,
        rrf_k=config.hybrid.rrf_k,
        name="hybrid",
    )
    hybrid._indexed = True

    return hybrid


def main(
    config_path: str = "configs/default.yaml",
    query: str = None,
    interactive: bool = False,
    batch_path: str = None,
    sparse_only: bool = False,
) -> None:
    """Run RAG generation."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — RAG Generation")
    if sparse_only:
        logger.info("  (SPARSE-ONLY MODE — no dense model)")
    logger.info("=" * 60)

    # Load chunks
    chunks_path = Path(config.data.processed_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error(f"Chunks not found: {chunks_path}")
        logger.error("Run scripts/01_prepare_data.py first.")
        sys.exit(1)

    records = load_jsonl(str(chunks_path))
    chunks = [
        Chunk(
            chunk_id=r["chunk_id"], text=r["text"], doc_id=r["doc_id"],
            source=r["source"], chunk_index=r["chunk_index"],
            metadata=r.get("metadata", {}),
        )
        for r in records
    ]
    logger.info(f"Loaded {len(chunks)} chunks")

    # Build retriever
    logger.info("Building retriever...")
    retriever = build_retriever(config, chunks, sparse_only=sparse_only)

    # Initialize LLM
    logger.info(f"Initializing LLM backend: {config.generation.backend}")
    try:
        llm = get_llm(
            backend=config.generation.backend,
            model=config.generation.model,
            api_base_url=config.generation.api_base_url,
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
        )
    except (ValueError, ImportError) as e:
        logger.warning(f"LLM initialization failed: {e}")
        logger.info("Running in retrieval-only mode (no generation)")
        llm = None

    # Build RAG generator
    generator = RAGGenerator(
        retriever=retriever,
        llm=llm,
        prompt_template=config.generation.prompt_template,
        top_k_context=config.generation.top_k_context,
    )

    output_dir = ensure_dir("outputs/rag_results")

    # ---------------------------------------------------------------
    # Single query mode
    # ---------------------------------------------------------------
    if query:
        logger.info(f"\nQuery: {query}")
        if llm:
            response = generator.generate(query)
            logger.info(f"\nAnswer:\n{response.answer}")
            logger.info(f"\nSources ({len(response.sources)}):")
            for s in response.sources:
                logger.info(f"  [{s.rank}] {s.chunk_id} (score={s.score:.4f})")
                logger.info(f"      {s.text[:100]}...")

            # Save response
            save_jsonl([response.to_dict()], output_dir / "single_response.jsonl")
        else:
            results = retriever.retrieve(query, top_k=config.generation.top_k_context)
            logger.info(f"\nRetrieved {len(results)} chunks (no LLM):")
            for r in results:
                logger.info(f"  [{r.rank}] {r.chunk_id} (score={r.score:.4f})")
                logger.info(f"      {r.text[:100]}...")

    # ---------------------------------------------------------------
    # Interactive mode
    # ---------------------------------------------------------------
    elif interactive:
        logger.info("\nInteractive mode — type 'quit' to exit\n")
        while True:
            try:
                user_query = input("\n>>> Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_query.lower() in ("quit", "exit", "q"):
                break
            if not user_query:
                continue

            if llm:
                response = generator.generate(user_query)
                print(f"\n📝 Answer:\n{response.answer}")
                print(f"\n📚 Sources:")
                for s in response.sources[:3]:
                    print(f"  [{s.rank}] {s.text[:80]}...")
            else:
                results = retriever.retrieve(user_query, top_k=5)
                print(f"\n📚 Retrieved {len(results)} chunks:")
                for r in results:
                    print(f"  [{r.rank}] (score={r.score:.4f}) {r.text[:80]}...")

    # ---------------------------------------------------------------
    # Batch mode
    # ---------------------------------------------------------------
    elif batch_path:
        logger.info(f"Batch mode — processing queries from {batch_path}")
        qa_records = load_jsonl(batch_path)
        queries = [r["question"] for r in qa_records[:50]]  # limit to 50

        responses = []
        for i, q in enumerate(queries):
            if llm:
                resp = generator.generate(q)
                responses.append(resp.to_dict())
            else:
                results = retriever.retrieve(q, top_k=5)
                responses.append({
                    "query": q,
                    "retrieved": [
                        {"chunk_id": r.chunk_id, "score": r.score}
                        for r in results
                    ],
                })
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(queries)} queries")

        save_jsonl(responses, output_dir / "batch_responses.jsonl")
        logger.info(f"Saved {len(responses)} responses to {output_dir}")

    else:
        # Default: show a few sample queries
        sample_queries = [
            "भोजपुरी भाषा कहाँ बोलल जाला?",
            "छठ पूजा का ह?",
            "लिट्टी चोखा कइसे बनेला?",
            "भिखारी ठाकुर के बारे में बताईं।",
        ]
        logger.info("\nRunning sample queries:\n")
        for q in sample_queries:
            logger.info(f"Q: {q}")
            results = retriever.retrieve(q, top_k=3)
            for r in results:
                logger.info(f"  [{r.rank}] (score={r.score:.4f}) {r.text[:80]}...")
            logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG RAG Generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--query", type=str, default=None, help="Single query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", type=str, default=None, help="Batch JSONL path")
    parser.add_argument("--sparse-only", action="store_true", help="Use only sparse retrieval")
    args = parser.parse_args()
    main(args.config, query=args.query, interactive=args.interactive, batch_path=args.batch, sparse_only=args.sparse_only)
