"""
Script 07: Generate Paper Assets
===================================
Generates publication-ready tables (LaTeX + CSV) and figures (PNG)
from evaluation results.

Usage:
    python scripts/07_generate_paper_assets.py --config configs/default.yaml
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.plotting import ResultPlotter
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_latex_table(
    results: dict,
    title: str,
    output_path: str,
    caption: str = "",
    label: str = "tab:results",
) -> None:
    """Generate a LaTeX table from evaluation results."""
    if not results:
        logger.warning("No results to generate table from.")
        return

    retrievers = list(results.keys())
    metrics = list(results[retrievers[0]].keys())

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption or title}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "\\textbf{System} & "
        + " & ".join(f"\\textbf{{{m}}}" for m in metrics)
        + " \\\\",
        "\\midrule",
    ]

    # Find best value per metric for bolding
    best_per_metric = {}
    for m in metrics:
        values = [results[r][m] for r in retrievers]
        best_per_metric[m] = max(values)

    for retriever in retrievers:
        cells = []
        for m in metrics:
            val = results[retriever][m]
            formatted = f"{val:.4f}"
            if val == best_per_metric[m]:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        display_name = retriever.replace("_", "\\_")
        lines.append(f"{display_name} & " + " & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"LaTeX table saved to {output_path}")


def main(config_path: str = "configs/default.yaml") -> None:
    """Generate all paper assets from evaluation results."""
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("BhojRAG — Paper Asset Generation")
    logger.info("=" * 60)

    results_dir = Path(config.evaluation.results_dir)
    tables_dir = ensure_dir(config.paper_assets.tables_dir)
    figures_dir = ensure_dir(config.paper_assets.figures_dir)

    plotter = ResultPlotter(
        output_dir=str(figures_dir),
        figure_format=config.paper_assets.figure_format,
        dpi=config.paper_assets.dpi,
    )

    # ---------------------------------------------------------------
    # 1. Main results table
    # ---------------------------------------------------------------
    aggregate_path = results_dir / "aggregate_results.csv"
    if aggregate_path.exists():
        logger.info("[1] Generating main results table...")

        # Parse CSV into dict
        results = {}
        with open(aggregate_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.pop("retriever")
                results[name] = {k: float(v) for k, v in row.items()}

        # LaTeX table
        generate_latex_table(
            results,
            title="Retrieval Performance Comparison",
            output_path=str(tables_dir / "main_results.tex"),
            caption=(
                "Retrieval performance of all systems on the Bhojpuri QA "
                "evaluation set. Best values per metric are bolded."
            ),
            label="tab:main_results",
        )

        # Also save as CSV
        with open(tables_dir / "main_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            metrics = list(next(iter(results.values())).keys())
            writer.writerow(["System"] + metrics)
            for name, vals in results.items():
                writer.writerow([name] + [f"{vals[m]:.4f}" for m in metrics])

        # Bar chart
        plotter.plot_retriever_comparison(
            results,
            title="Retrieval Performance Comparison — Bhojpuri QA",
            filename="retrieval_comparison",
        )
    else:
        logger.warning(f"No aggregate results at {aggregate_path}")
        logger.info("Run scripts/05_evaluate.py first.")

    # ---------------------------------------------------------------
    # 2. Ablation table + heatmap
    # ---------------------------------------------------------------
    ablation_path = results_dir / "ablation_ngram_results.json"
    if ablation_path.exists():
        logger.info("[2] Generating ablation assets...")
        with open(ablation_path, "r") as f:
            ablation_results = json.load(f)

        # LaTeX table
        generate_latex_table(
            ablation_results,
            title="Character N-gram Size Ablation",
            output_path=str(tables_dir / "ablation_ngram.tex"),
            caption=(
                "Effect of character n-gram range on sparse retrieval "
                "performance for Bhojpuri text."
            ),
            label="tab:ablation_ngram",
        )

        # Heatmap
        plotter.plot_ablation_heatmap(
            ablation_results,
            title="Character N-gram Size Ablation — Bhojpuri",
            filename="ablation_heatmap",
        )
    else:
        logger.info("No ablation results found (skipping).")

    # ---------------------------------------------------------------
    # 3. Error analysis plot
    # ---------------------------------------------------------------
    error_summary_path = results_dir / "error_analysis" / "summary.json"
    if error_summary_path.exists():
        logger.info("[3] Generating error analysis plot...")
        with open(error_summary_path, "r") as f:
            error_summary = json.load(f)

        plotter.plot_error_analysis(
            error_summary,
            title="Sparse vs. Dense Retrieval — Error Analysis",
            filename="error_analysis",
        )
    else:
        logger.info("No error analysis results found (skipping).")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Paper asset generation complete!")
    logger.info(f"  Tables: {tables_dir}")
    logger.info(f"  Figures: {figures_dir}")
    logger.info("=" * 60)

    # List generated files
    for d in [tables_dir, figures_dir]:
        files = list(Path(d).glob("*"))
        if files:
            logger.info(f"\n  {d}:")
            for f in files:
                logger.info(f"    - {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhojRAG Paper Assets")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
