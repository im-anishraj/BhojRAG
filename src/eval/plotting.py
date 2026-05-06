"""
BhojRAG Result Plotting
=========================
Generate publication-quality figures for paper assets.
Uses matplotlib + seaborn for consistent, journal-ready aesthetics.
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.utils.io import ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultPlotter:
    """
    Generate paper-ready plots from evaluation results.
    
    Plot types:
        - Retriever comparison bar chart
        - Ablation heatmap (n-gram size × metric)
        - RRF k sensitivity curve
        - Error analysis pie chart
    
    Usage:
        plotter = ResultPlotter(output_dir="paper_assets/figures", dpi=300)
        plotter.plot_retriever_comparison(results)
    """

    def __init__(
        self,
        output_dir: str = "paper_assets/figures",
        figure_format: str = "png",
        dpi: int = 300,
    ):
        self.output_dir = Path(output_dir)
        self.figure_format = figure_format
        self.dpi = dpi

    def plot_retriever_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Retrieval Performance Comparison",
        filename: str = "retrieval_comparison",
    ) -> str:
        """
        Bar chart comparing all retrievers across metrics.
        
        Args:
            results: Dict mapping retriever_name -> {metric: value}.
            title: Plot title.
            filename: Output filename (without extension).
            
        Returns:
            Path to saved figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        ensure_dir(self.output_dir)

        retrievers = list(results.keys())
        metrics = list(next(iter(results.values())).keys())
        n_retrievers = len(retrievers)
        n_metrics = len(metrics)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_metrics)
        width = 0.8 / n_retrievers

        colors = plt.cm.Set2(np.linspace(0, 1, n_retrievers))

        for i, retriever in enumerate(retrievers):
            values = [results[retriever][m] for m in metrics]
            offset = (i - n_retrievers / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values, width,
                label=retriever, color=colors[i], edgecolor="black",
                linewidth=0.5,
            )
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = self.output_dir / f"{filename}.{self.figure_format}"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved comparison plot to {out_path}")
        return str(out_path)

    def plot_ablation_heatmap(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        title: str = "N-gram Size Ablation",
        filename: str = "ablation_heatmap",
    ) -> str:
        """
        Heatmap showing metric values across ablation configurations.
        
        Args:
            ablation_results: Dict mapping config_label -> {metric: value}.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        ensure_dir(self.output_dir)

        configs = list(ablation_results.keys())
        metrics = list(next(iter(ablation_results.values())).keys())

        data = np.array([
            [ablation_results[c][m] for m in metrics]
            for c in configs
        ])

        fig, ax = plt.subplots(figsize=(10, max(4, len(configs) * 0.6)))
        sns.heatmap(
            data, annot=True, fmt=".3f",
            xticklabels=metrics, yticklabels=configs,
            cmap="YlOrRd", ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = self.output_dir / f"{filename}.{self.figure_format}"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved ablation heatmap to {out_path}")
        return str(out_path)

    def plot_error_analysis(
        self,
        summary: Dict[str, int],
        title: str = "Retrieval Error Analysis",
        filename: str = "error_analysis",
    ) -> str:
        """
        Pie chart showing breakdown of sparse/dense success patterns.
        
        Args:
            summary: Dict with keys: both_succeed, sparse_only_wins,
                     dense_only_wins, both_fail.
        """
        import matplotlib.pyplot as plt

        ensure_dir(self.output_dir)

        labels = ["Both Succeed", "Sparse Only", "Dense Only", "Both Fail"]
        sizes = [
            summary.get("both_succeed", 0),
            summary.get("sparse_only_wins", 0),
            summary.get("dense_only_wins", 0),
            summary.get("both_fail", 0),
        ]
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
        explode = (0.05, 0.05, 0.05, 0.05)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct="%1.1f%%", shadow=True, startangle=140,
            textprops={"fontsize": 11},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = self.output_dir / f"{filename}.{self.figure_format}"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved error analysis plot to {out_path}")
        return str(out_path)
