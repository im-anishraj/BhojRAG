"""
Unit tests for hybrid retriever (RRF fusion).
"""

import pytest
from src.data.chunker import Chunk
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.hybrid import HybridRetriever


class MockRetriever(BaseRetriever):
    """Mock retriever for testing hybrid fusion logic."""

    def __init__(self, name: str, results: dict):
        super().__init__(name=name)
        self._mock_results = results
        self._indexed = True

    def index(self, chunks):
        self._corpus = chunks
        self._indexed = True

    def retrieve(self, query: str, top_k: int = 10):
        return self._mock_results.get(query, [])[:top_k]


@pytest.fixture
def mock_retrievers():
    """Create mock sparse and dense retrievers with controlled results."""
    sparse_results = {
        "test query": [
            RetrievalResult(chunk_id="c1", text="text1", score=5.0, rank=1),
            RetrievalResult(chunk_id="c2", text="text2", score=3.0, rank=2),
            RetrievalResult(chunk_id="c3", text="text3", score=1.0, rank=3),
        ]
    }
    dense_results = {
        "test query": [
            RetrievalResult(chunk_id="c2", text="text2", score=0.9, rank=1),
            RetrievalResult(chunk_id="c4", text="text4", score=0.7, rank=2),
            RetrievalResult(chunk_id="c1", text="text1", score=0.5, rank=3),
        ]
    }
    sparse = MockRetriever("sparse", sparse_results)
    dense = MockRetriever("dense", dense_results)
    return sparse, dense


class TestHybridRetriever:
    """Tests for RRF and weighted fusion."""

    def test_rrf_fusion(self, mock_retrievers):
        sparse, dense = mock_retrievers
        hybrid = HybridRetriever(
            retrievers=[sparse, dense], method="rrf", rrf_k=60,
        )
        hybrid._indexed = True

        results = hybrid.retrieve("test query", top_k=5)
        assert len(results) > 0

        # c2 appears in both systems (rank 2 sparse, rank 1 dense)
        # so it should have a high RRF score
        result_ids = [r.chunk_id for r in results]
        assert "c2" in result_ids

    def test_rrf_rank_ordering(self, mock_retrievers):
        sparse, dense = mock_retrievers
        hybrid = HybridRetriever(
            retrievers=[sparse, dense], method="rrf", rrf_k=60,
        )
        hybrid._indexed = True

        results = hybrid.retrieve("test query", top_k=5)
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_weighted_fusion(self, mock_retrievers):
        sparse, dense = mock_retrievers
        hybrid = HybridRetriever(
            retrievers=[sparse, dense],
            weights=[0.7, 0.3],
            method="weighted",
        )
        hybrid._indexed = True

        results = hybrid.retrieve("test query", top_k=5)
        assert len(results) > 0

    def test_empty_query(self, mock_retrievers):
        sparse, dense = mock_retrievers
        hybrid = HybridRetriever(
            retrievers=[sparse, dense], method="rrf",
        )
        hybrid._indexed = True

        results = hybrid.retrieve("nonexistent query", top_k=5)
        assert isinstance(results, list)

    def test_weight_mismatch_raises(self):
        with pytest.raises(ValueError, match="weights"):
            HybridRetriever(
                retrievers=[MockRetriever("a", {}), MockRetriever("b", {})],
                weights=[1.0],  # Only 1 weight for 2 retrievers
            )
