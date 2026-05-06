"""
BhojRAG Dense Retriever Fine-Tuning
======================================
Contrastive learning pipeline for fine-tuning MuRIL (or any
sentence-transformer) on synthetic Bhojpuri QA pairs.

Uses MultipleNegativesRankingLoss for in-batch negative sampling.
Optimized for single-GPU with mixed precision and gradient checkpointing.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.utils.io import load_jsonl
from src.utils.logger import setup_logger, ExperimentTracker

logger = setup_logger(__name__)


class DenseTrainer:
    """
    Fine-tune a sentence-transformer model on QA pairs using
    contrastive learning (MultipleNegativesRankingLoss).
    
    Design choices:
        - In-batch negatives: every other QA pair in the batch
          serves as a negative, so batch_size matters.
        - Mixed precision (fp16) for GPU memory efficiency.
        - Gradient checkpointing to fit larger batches.
        - Warmup schedule for stable training.
    
    Usage:
        trainer = DenseTrainer(
            model_name="google/muril-base-cased",
            output_dir="models/muril_finetuned",
        )
        trainer.train(
            qa_pairs_path="data/synthetic/qa_pairs.jsonl",
            epochs=5,
        )
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        output_dir: str = "models/muril_finetuned",
        max_seq_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        fp16: bool = True,
        gradient_checkpointing: bool = True,
        eval_fraction: float = 0.1,
        eval_steps: int = 100,
        save_steps: int = 500,
        max_grad_norm: float = 1.0,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.eval_fraction = eval_fraction
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.tracker = tracker

    def train(self, qa_pairs_path: str) -> str:
        """
        Fine-tune the model on synthetic QA pairs.
        
        Args:
            qa_pairs_path: Path to JSONL file with QA pairs.
            
        Returns:
            Path to the saved fine-tuned model.
        """
        from sentence_transformers import (
            SentenceTransformer,
            InputExample,
            losses,
            evaluation,
        )

        # Load data
        logger.info(f"Loading QA pairs from {qa_pairs_path}")
        qa_pairs = load_jsonl(qa_pairs_path)

        # Split into train/eval
        split_idx = max(1, int(len(qa_pairs) * (1 - self.eval_fraction)))
        train_pairs = qa_pairs[:split_idx]
        eval_pairs = qa_pairs[split_idx:]
        logger.info(
            f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}"
        )

        # Prepare training examples
        train_examples = [
            InputExample(texts=[p["question"], p["answer"]])
            for p in train_pairs
        ]

        # Load model
        logger.info(f"Loading base model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        model.max_seq_length = self.max_seq_length

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            if hasattr(model._first_module(), "auto_model"):
                model._first_module().auto_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        # Training dataloader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size,
        )

        # Loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)
        logger.info("Using MultipleNegativesRankingLoss")

        # Evaluator (if we have eval data)
        evaluator = None
        if eval_pairs:
            eval_sentences1 = [p["question"] for p in eval_pairs]
            eval_sentences2 = [p["answer"] for p in eval_pairs]
            # Use InformationRetrievalEvaluator or simple EmbeddingSimilarity
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                eval_sentences1,
                eval_sentences2,
                [1.0] * len(eval_pairs),  # All pairs are positive
                name="bhojpuri_qa",
            )

        # Calculate warmup steps
        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log training config
        train_config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "train_size": len(train_pairs),
            "eval_size": len(eval_pairs),
        }
        if self.tracker:
            self.tracker.log_params(train_config)

        logger.info(f"Starting training: {train_config}")

        # Train
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            evaluator=evaluator,
            evaluation_steps=self.eval_steps,
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir),
            save_best_model=True,
            use_amp=self.fp16,
            optimizer_params={"lr": self.learning_rate},
            checkpoint_save_steps=self.save_steps,
            checkpoint_path=str(self.output_dir / "checkpoints"),
        )

        # Save training config
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(train_config, f, indent=2)

        logger.info(f"Model saved to {self.output_dir}")

        if self.tracker:
            self.tracker.log_artifact(str(self.output_dir))

        return str(self.output_dir)
