"""
Unit tests for IR evaluation metrics.
"""

import pytest
from src.eval.metrics import RetrievalMetrics


@pytest.fixture
def metrics():
    return RetrievalMetrics()


@pytest.fixture
def sample_data():
    """Standard IR evaluation test data."""
    results = {
        "q1": ["d1", "d2", "d3", "d4", "d5"],
        "q2": ["d3", "d1", "d5", "d2", "d4"],
        "q3": ["d5", "d4", "d3", "d2", "d1"],
    }
    gold = {
        "q1": {"d1", "d3"},
        "q2": {"d1"},
        "q3": {"d1", "d2"},
    }
    return results, gold


class TestMetrics:
    """Tests for retrieval metrics."""

    def test_mrr_perfect(self, metrics):
        results = {"q1": ["d1", "d2"]}
        gold = {"q1": {"d1"}}
        assert metrics.mrr_at_k(results, gold, k=5) == 1.0

    def test_mrr_rank2(self, metrics):
        results = {"q1": ["d2", "d1"]}
        gold = {"q1": {"d1"}}
        assert metrics.mrr_at_k(results, gold, k=5) == 0.5

    def test_mrr_not_found(self, metrics):
        results = {"q1": ["d2", "d3"]}
        gold = {"q1": {"d1"}}
        assert metrics.mrr_at_k(results, gold, k=2) == 0.0

    def test_mrr_zero_relevant_docs(self, metrics):
        """Edge case: What if the gold set has no relevant docs for a query?"""
        results = {"q1": ["d2", "d3"]}
        gold = {"q1": set()} # Empty set of relevant docs
        assert metrics.mrr_at_k(results, gold, k=5) == 0.0

    def test_recall_at_k(self, metrics, sample_data):
        results, gold = sample_data
        recall = metrics.recall_at_k(results, gold, k=5)
        assert 0.0 <= recall <= 1.0

    def test_recall_perfect(self, metrics):
        results = {"q1": ["d1", "d2"]}
        gold = {"q1": {"d1", "d2"}}
        assert metrics.recall_at_k(results, gold, k=5) == 1.0

    def test_precision_at_k(self, metrics, sample_data):
        results, gold = sample_data
        prec = metrics.precision_at_k(results, gold, k=5)
        assert 0.0 <= prec <= 1.0

    def test_ndcg_perfect_ranking(self, metrics):
        results = {"q1": ["d1", "d2", "d3"]}
        gold = {"q1": {"d1", "d2"}}
        ndcg = metrics.ndcg_at_k(results, gold, k=3)
        assert ndcg == 1.0  # Relevant docs at rank 1 and 2 = perfect

    def test_map(self, metrics, sample_data):
        results, gold = sample_data
        map_score = metrics.mean_average_precision(results, gold)
        assert 0.0 <= map_score <= 1.0

    def test_compute_all(self, metrics, sample_data):
        results, gold = sample_data
        scores = metrics.compute_all(results, gold, k=5)
        assert "mrr@5" in scores
        assert "recall@5" in scores
        assert "ndcg@5" in scores
        assert "precision@5" in scores
        assert "map" in scores

    def test_per_query_metrics(self, metrics, sample_data):
        results, gold = sample_data
        per_query = metrics.compute_per_query(results, gold, k=5)
        assert "q1" in per_query
        assert "mrr@5" in per_query["q1"]
