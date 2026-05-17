"""
Unit tests for dense retriever module.
Tests are designed to work without a GPU (using small models).
"""

import pytest

from src.data.chunker import Chunk
from src.retrieval.dense_retriever import DenseRetriever


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
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
    ]


class TestDenseRetriever:
    """Tests for dense retriever (mocked model for fast testing)."""

    def test_init_default(self):
        retriever = DenseRetriever()
        assert retriever.model_name == "google/muril-base-cased"
        assert not retriever.is_indexed

    def test_retrieve_without_index_raises(self):
        retriever = DenseRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("test query")

    def test_batch_retrieve_without_index_raises(self):
        retriever = DenseRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.batch_retrieve(["q1", "q2"])

    def test_custom_model_name(self):
        retriever = DenseRetriever(model_name="custom/model")
        assert retriever.model_name == "custom/model"

    def test_device_selection_cpu(self):
        retriever = DenseRetriever(device="cpu")
        assert retriever.device == "cpu"
