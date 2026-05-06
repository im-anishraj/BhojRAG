"""
Unit tests for configuration system.
"""

import pytest
import tempfile
from pathlib import Path
from src.utils.config import load_config, merge_configs, ExperimentConfig


class TestConfig:
    """Tests for YAML config loading and validation."""

    def test_load_default_config(self):
        config = load_config("configs/default.yaml")
        assert isinstance(config, ExperimentConfig)
        assert config.experiment.name == "bhojrag_baseline"
        assert config.experiment.seed == 42

    def test_sparse_config(self):
        config = load_config("configs/default.yaml")
        assert config.sparse.method == "char_ngram_bm25"
        assert config.sparse.ngram_range == (2, 4)

    def test_dense_config(self):
        config = load_config("configs/default.yaml")
        assert config.dense.model_name == "google/muril-base-cased"
        assert config.dense.embedding_dim == 768

    def test_training_config(self):
        config = load_config("configs/default.yaml")
        assert config.training.fp16 is True
        assert config.training.gradient_checkpointing is True

    def test_generation_config(self):
        config = load_config("configs/default.yaml")
        assert config.generation.backend == "api"

    def test_merge_configs(self):
        base = ExperimentConfig()
        overrides = {"experiment": {"name": "ablation_run_1"}}
        merged = merge_configs(base, overrides)
        assert merged.experiment.name == "ablation_run_1"
        assert merged.experiment.seed == 42  # unchanged

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_default_values(self):
        config = ExperimentConfig()
        assert config.hybrid.rrf_k == 60
        assert config.evaluation.error_analysis is True
