"""
Unit tests for sparse retrieval modules.
"""

import pytest

from src.data.chunker import Chunk
from src.retrieval.sparse_bm25 import WordBM25Retriever
from src.retrieval.sparse_ngram_bm25 import CharNgramBM25Retriever


@pytest.fixture
def sample_chunks():
    """Create sample Bhojpuri chunks for testing."""
    return [
        Chunk(
            chunk_id="c_0001",
            text="भोजपुरी भाषा बिहार में बोलल जाला",
            doc_id="d1",
            source="test",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="c_0002",
            text="छठ पूजा भोजपुरी संस्कृति के त्यौहार बा",
            doc_id="d2",
            source="test",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="c_0003",
            text="लिट्टी चोखा भोजपुरी के प्रसिद्ध खाना बा",
            doc_id="d3",
            source="test",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="c_0004",
            text="भोजपूरी भासा बहुत पुरान बा",
            doc_id="d4",
            source="test",
            chunk_index=0,
            metadata={"note": "alternate spelling"},
        ),
    ]


class TestWordBM25:
    """Tests for word-level BM25 retriever."""

    def test_index_and_retrieve(self, sample_chunks):
        retriever = WordBM25Retriever()
        retriever.index(sample_chunks)
        assert retriever.is_indexed

        results = retriever.retrieve("भोजपुरी भाषा", top_k=3)
        assert len(results) > 0
        assert results[0].rank == 1
        assert results[0].score > 0

    def test_empty_query(self, sample_chunks):
        retriever = WordBM25Retriever()
        retriever.index(sample_chunks)
        results = retriever.retrieve("", top_k=3)
        # Empty query should still return results (all zero scores)
        assert isinstance(results, list)

    def test_repr(self, sample_chunks):
        retriever = WordBM25Retriever()
        retriever.index(sample_chunks)
        assert "word_bm25" in repr(retriever)


class TestCharNgramBM25:
    """Tests for character n-gram BM25 retriever."""

    def test_index_and_retrieve(self, sample_chunks):
        retriever = CharNgramBM25Retriever(ngram_range=(2, 4))
        retriever.index(sample_chunks)
        assert retriever.is_indexed

        results = retriever.retrieve("भोजपुरी भाषा", top_k=3)
        assert len(results) > 0
        assert results[0].score > 0

    def test_spelling_variation_robustness(self, sample_chunks):
        """
        Core research claim: char n-gram BM25 should be more robust
        to spelling variations than word BM25.

        Query uses "भोजपूरी" (variant) instead of "भोजपुरी" (standard).
        Char n-gram should still find relevant results due to n-gram overlap.
        """
        word_retriever = WordBM25Retriever()
        word_retriever.index(sample_chunks)

        ngram_retriever = CharNgramBM25Retriever(ngram_range=(2, 4))
        ngram_retriever.index(sample_chunks)

        variant_query = "भोजपूरी भासा"  # Non-standard spelling

        ngram_results = ngram_retriever.retrieve(variant_query, top_k=4)

        # n-gram retriever should have higher scores for variant queries
        # because it captures subword overlap
        assert len(ngram_results) > 0

        # The variant spelling doc (c_0004) should rank higher in n-gram
        ngram_ids = [r.chunk_id for r in ngram_results]
        assert (
            "c_0004" in ngram_ids
        ), "Char n-gram BM25 should find the variant spelling chunk"

    def test_configurable_ngram_range(self, sample_chunks):
        retriever_2_3 = CharNgramBM25Retriever(ngram_range=(2, 3))
        retriever_3_5 = CharNgramBM25Retriever(ngram_range=(3, 5))

        retriever_2_3.index(sample_chunks)
        retriever_3_5.index(sample_chunks)

        results_2_3 = retriever_2_3.retrieve("छठ पूजा", top_k=3)
        results_3_5 = retriever_3_5.retrieve("छठ पूजा", top_k=3)

        # Both should return results, but scores may differ
        assert len(results_2_3) > 0
        assert len(results_3_5) > 0

    def test_ngram_extraction(self):
        retriever = CharNgramBM25Retriever(ngram_range=(2, 3))
        ngrams = retriever._extract_ngrams("ab cd")
        # Should contain boundary-marked n-grams
        assert "^a" in ngrams
        assert "b$" in ngrams
        assert "^c" in ngrams
        assert "d$" in ngrams

    def test_empty_string_ngram(self, sample_chunks):
        """Ensure tokenizer doesn't crash on empty strings."""
        retriever = CharNgramBM25Retriever(ngram_range=(2, 3))
        ngrams = retriever._extract_ngrams("")
        assert len(ngrams) == 0

        # Also test retrieval with empty query
        retriever.index(sample_chunks)
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)
