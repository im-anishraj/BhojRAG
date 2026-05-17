"""
Script 03: Dense Retriever Fine-Tuning
=========================================
Fine-tunes MuRIL (or configured model) on synthetic Bhojpuri QA pairs
using MultipleNegativesRankingLoss (contrastive learning).

Usage:
    python scripts/03_train_dense.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.dense_trainer import DenseTrainer
from src.utils.config import load_config
from src.utils.logger import ExperimentTracker, setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def main(config_path: str = "configs/default.yaml") -> None:
    """Fine-tune the dense retriever model."""
    config = load_config(config_path)
    set_seed(config.experiment.seed)

    logger.info("=" * 60)
    logger.info("BhojRAG — Dense Retriever Fine-Tuning")
    logger.info("=" * 60)

    # Verify QA pairs exist
    qa_path = Path(config.data.qa_pairs_path)
    if not qa_path.exists():
        logger.error(
            f"QA pairs not found: {qa_path}\n" "Run scripts/02_generate_qa.py first."
        )
        sys.exit(1)

    # Setup experiment tracker
    tracker = ExperimentTracker(
        experiment_name=f"dense_training_{config.experiment.name}",
        tracking_mode=config.experiment.tracking,
        output_dir=config.experiment.output_dir,
        mlflow_uri=config.experiment.mlflow_uri,
    )
    tracker.start_run(run_name=f"train_{config.dense.model_name.split('/')[-1]}")

    # Initialize trainer
    trainer = DenseTrainer(
        model_name=config.dense.model_name,
        output_dir=config.training.output_dir,
        max_seq_length=config.dense.max_seq_length,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_fraction=config.training.eval_fraction,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        max_grad_norm=config.training.max_grad_norm,
        tracker=tracker,
    )

    logger.info(f"  Base model: {config.dense.model_name}")
    logger.info(f"  QA pairs: {qa_path}")
    logger.info(f"  Output: {config.training.output_dir}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  LR: {config.training.learning_rate}")
    logger.info(f"  FP16: {config.training.fp16}")

    # Train
    model_path = trainer.train(str(qa_path))

    tracker.end_run()

    logger.info("=" * 60)
    logger.info("Fine-tuning complete!")
    logger.info(f"  Model saved to: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG Dense Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args.config)
