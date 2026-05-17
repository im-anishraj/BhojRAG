"""
BhojRAG Configuration Module
=============================
Type-safe YAML configuration parsing using Pydantic v2.
Supports nested configs, defaults, and validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class TransliterationConfig(BaseModel):
    """Transliteration settings for handling Latin/Hinglish Bhojpuri."""

    enabled: bool = True
    source_scripts: List[str] = ["latin", "devanagari"]
    target_script: str = "devanagari"


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    corpus_path: str = "data/raw/sample_corpus.txt"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    qa_pairs_path: str = "data/synthetic/qa_pairs.jsonl"

    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    lowercase: bool = False
    min_doc_length: int = 20

    chunk_size: int = 256
    chunk_overlap: int = 64
    chunking_method: Literal["fixed", "sentence"] = "fixed"

    transliteration: TransliterationConfig = TransliterationConfig()


class QAGenerationConfig(BaseModel):
    """Synthetic QA generation settings."""

    method: Literal["template", "llm"] = "template"
    num_questions_per_chunk: int = 2
    llm_backend: str = "api"
    llm_model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 256


class SparseConfig(BaseModel):
    """Sparse retriever configuration."""

    method: Literal["word_bm25", "char_ngram_bm25"] = "char_ngram_bm25"
    ngram_range: Tuple[int, int] = (2, 4)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k: int = 10

    @field_validator("ngram_range", mode="before")
    @classmethod
    def coerce_ngram_range(cls, v: Any) -> Tuple[int, int]:
        """Accept list from YAML and convert to tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v


class DenseConfig(BaseModel):
    """Dense retriever configuration."""

    model_name: str = "google/muril-base-cased"
    max_seq_length: int = 512
    embedding_dim: int = 768
    batch_size: int = 32
    normalize_embeddings: bool = True
    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    top_k: int = 10
    nlist: int = 100
    nprobe: int = 10


class HardNegativeConfig(BaseModel):
    """Hard negative mining settings for contrastive training."""

    enabled: bool = False
    num_negatives: int = 5
    strategy: str = "bm25"


class TrainingConfig(BaseModel):
    """Dense retriever fine-tuning configuration."""

    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    batch_size: int = 16
    eval_steps: int = 100
    save_steps: int = 500
    fp16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    loss: str = "MultipleNegativesRankingLoss"
    output_dir: str = "models/muril_finetuned"
    eval_fraction: float = 0.1
    hard_negatives: HardNegativeConfig = HardNegativeConfig()


class HybridConfig(BaseModel):
    """Hybrid retriever (RRF) configuration."""

    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    sparse_weight: float = 0.5
    dense_weight: float = 0.5
    top_k: int = 10


class LocalModelConfig(BaseModel):
    """Local LLM settings for RAG generation."""

    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantization: Optional[str] = None
    max_new_tokens: int = 512
    device: str = "auto"


class GenerationConfig(BaseModel):
    """RAG generation configuration."""

    backend: Literal["api", "local"] = "api"
    model: str = "gemini-1.5-flash"
    api_base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 512
    top_k_context: int = 5
    prompt_template: Literal["grounded_qa", "chain_of_thought", "hindi_bridge"] = (
        "grounded_qa"
    )
    local: LocalModelConfig = LocalModelConfig()


class AblationConfig(BaseModel):
    """Ablation study parameters."""

    ngram_sizes: List[List[int]] = [[2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]
    rrf_k_values: List[int] = [10, 30, 60, 100]
    dense_models: List[str] = ["google/muril-base-cased"]


class EvaluationConfig(BaseModel):
    """Evaluation and ablation configuration."""

    metrics: List[str] = ["mrr@10", "recall@5", "ndcg@10", "precision@5", "map"]
    eval_data_path: str = "data/synthetic/qa_pairs.jsonl"
    results_dir: str = "outputs/eval_results"
    error_analysis: bool = True
    save_per_query: bool = True
    ablation: AblationConfig = AblationConfig()


class PaperAssetsConfig(BaseModel):
    """Paper asset generation settings."""

    tables_dir: str = "paper_assets/tables"
    figures_dir: str = "paper_assets/figures"
    figure_format: str = "png"
    dpi: int = 300


class ExperimentMeta(BaseModel):
    """Top-level experiment metadata."""

    name: str = "bhojrag_baseline"
    seed: int = 42
    tracking: Literal["mlflow", "json"] = "mlflow"
    output_dir: str = "outputs"
    mlflow_uri: str = "mlruns"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """
    Root configuration aggregating all sub-configs.

    Usage:
        config = load_config("configs/default.yaml")
        print(config.dense.model_name)
    """

    experiment: ExperimentMeta = ExperimentMeta()
    data: DataConfig = DataConfig()
    qa_generation: QAGenerationConfig = QAGenerationConfig()
    sparse: SparseConfig = SparseConfig()
    dense: DenseConfig = DenseConfig()
    training: TrainingConfig = TrainingConfig()
    hybrid: HybridConfig = HybridConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paper_assets: PaperAssetsConfig = PaperAssetsConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        pydantic.ValidationError: If config values fail validation.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return ExperimentConfig(**raw)


def merge_configs(
    base: ExperimentConfig,
    overrides: Dict[str, Any],
) -> ExperimentConfig:
    """
    Merge override dict into a base config.
    Useful for CLI-driven experiment sweeps.

    Args:
        base: Base configuration.
        overrides: Flat or nested dict of overrides.

    Returns:
        New ExperimentConfig with overrides applied.
    """
    base_dict = base.model_dump()
    _deep_update(base_dict, overrides)
    return ExperimentConfig(**base_dict)


def _deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update dict d with dict u."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d
