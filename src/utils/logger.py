"""
BhojRAG Logging & Experiment Tracking
======================================
Dual-mode tracking: MLflow for full experiment tracking,
JSON fallback for environments without MLflow.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a configured logger with console and optional file output.

    Args:
        name: Logger name (typically module name).
        log_file: Optional path to log file.
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """
    Unified experiment tracker that supports MLflow or JSON fallback.

    Usage:
        tracker = ExperimentTracker(
            experiment_name="dense_baseline",
            tracking_mode="mlflow",
            output_dir="outputs/dense_baseline",
        )
        tracker.start_run(run_name="muril_v1")
        tracker.log_params({"lr": 2e-5, "epochs": 5})
        tracker.log_metrics({"mrr@10": 0.78, "recall@5": 0.85})
        tracker.log_artifact("outputs/results.csv")
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_mode: str = "mlflow",
        output_dir: str = "outputs",
        mlflow_uri: str = "mlruns",
    ):
        self.experiment_name = experiment_name
        self.mode = tracking_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_data: Dict[str, Any] = {}
        self._run_active = False
        self._mlflow = None

        if self.mode == "mlflow":
            try:
                import mlflow

                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(experiment_name)
                self._mlflow = mlflow
            except ImportError:
                logging.warning("MLflow not installed. Falling back to JSON tracking.")
                self.mode = "json"

        self.logger = setup_logger(
            f"tracker.{experiment_name}",
            log_file=str(self.output_dir / "experiment.log"),
        )

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new experiment run."""
        self._run_data = {
            "run_name": run_name or f"run_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        self._run_active = True

        if self.mode == "mlflow" and self._mlflow:
            self._mlflow.start_run(run_name=self._run_data["run_name"])

        self.logger.info(f"Started run: {self._run_data['run_name']}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for the current run."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        self._run_data["params"].update(params)

        if self.mode == "mlflow" and self._mlflow:
            # MLflow requires string values and has a 500-char limit
            for k, v in params.items():
                self._mlflow.log_param(k, str(v)[:500])

        self.logger.info(f"Params: {params}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log evaluation metrics for the current run."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        self._run_data["metrics"].update(metrics)

        if self.mode == "mlflow" and self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

        self.logger.info(f"Metrics (step={step}): {metrics}")

    def log_artifact(self, file_path: str) -> None:
        """Log a file artifact (model checkpoint, result CSV, etc.)."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        self._run_data["artifacts"].append(str(file_path))

        if self.mode == "mlflow" and self._mlflow:
            self._mlflow.log_artifact(file_path)

        self.logger.info(f"Artifact logged: {file_path}")

    def end_run(self) -> None:
        """End the current run and persist results."""
        if not self._run_active:
            return

        self._run_data["end_time"] = datetime.now().isoformat()
        self._run_active = False

        if self.mode == "mlflow" and self._mlflow:
            self._mlflow.end_run()

        # Always save a JSON record as backup
        run_file = self.output_dir / f"{self._run_data['run_name']}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(self._run_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Run ended: {self._run_data['run_name']} → {run_file}")

    def get_run_data(self) -> Dict[str, Any]:
        """Return the current run's accumulated data."""
        return self._run_data.copy()
