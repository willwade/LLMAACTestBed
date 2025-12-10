#!/usr/bin/env python3
"""Aggregate and visualize sweep outputs created by sweep_wrapper.py."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def parse_run_params(filename: str) -> Dict[str, object]:
    """Extract sweep parameters from a results filename."""
    match = re.search(
        r"cf-(?P<context>[^_]+)_cw-(?P<cw>[^_]+)_tw-(?P<tw>[^_]+)_gr-(?P<gr>[^.]+)",
        filename,
    )
    if not match:
        return {
            "context_filter": "unknown",
            "conversation_window": None,
            "time_window_hours": None,
            "geo_radius_km": None,
        }

    params = match.groupdict()
    return {
        "context_filter": params["context"],
        "conversation_window": int(float(params["cw"])),
        "time_window_hours": float(params["tw"]),
        "geo_radius_km": float(params["gr"]),
    }


def load_results(results_dir: Path, pattern: str) -> pd.DataFrame:
    """Load all result CSVs matching the pattern and attach run metadata."""
    csv_paths: List[Path] = sorted(results_dir.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {results_dir} matching pattern '{pattern}'"
        )

    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        params = parse_run_params(csv_path.name)
        for key, value in params.items():
            df[key] = value
        df["run_id"] = csv_path.stem
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_runs(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute run-level averages for the key metrics."""
    metrics = [
        "embedding_similarity",
        "llm_judge_score",
        "character_accuracy",
        "word_accuracy",
    ]
    group_cols = [
        "run_id",
        "context_filter",
        "conversation_window",
        "time_window_hours",
        "geo_radius_km",
    ]
    return (
        results_df.groupby(group_cols)[metrics]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )


def plot_metric_by_conversation_window(
    summary_df: pd.DataFrame, output_dir: Path
) -> None:
    """Plot how metrics move with conversation_window for each context filter."""
    metrics = [
        "embedding_similarity",
        "llm_judge_score",
        "character_accuracy",
        "word_accuracy",
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=summary_df,
            x="conversation_window",
            y=metric,
            hue="context_filter",
            style="context_filter",
            marker="o",
        )
        plt.title(f"{metric.replace('_', ' ').title()} by Conversation Window")
        plt.xlabel("Conversation window (previous turns)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        outfile = output_dir / f"{metric}_by_conversation_window.png"
        plt.savefig(outfile, dpi=200)
        plt.close()


def plot_generation_boxplots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots per generation method to see spread across sweeps."""
    metrics = ["embedding_similarity", "llm_judge_score"]
    melted = results_df.melt(
        id_vars=["generation_method", "context_filter", "conversation_window"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=melted, x="generation_method", y="score", hue="metric", showfliers=False
    )
    plt.title("Metric Distribution by Generation Method (all sweeps)")
    plt.xlabel("Generation method")
    plt.ylabel("Score")
    plt.tight_layout()
    outfile = output_dir / "generation_method_boxplots.png"
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_heatmaps(results_df: pd.DataFrame, output_dir: Path, metric: str) -> None:
    """Heatmap of partial vs generation methods for each sweep setting."""
    heatmap_data = (
        results_df.groupby(
            [
                "context_filter",
                "conversation_window",
                "partial_method",
                "generation_method",
            ]
        )[metric]
        .mean()
        .reset_index()
    )

    for (context_filter, conv_window), group in heatmap_data.groupby(
        ["context_filter", "conversation_window"]
    ):
        pivot = group.pivot(
            index="partial_method",
            columns="generation_method",
            values=metric,
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title(
            f"{metric.replace('_', ' ').title()} — cf={context_filter}, cw={conv_window}"
        )
        plt.tight_layout()
        outfile = output_dir / f"heatmap_{metric}_cf-{context_filter}_cw-{conv_window}.png"
        plt.savefig(outfile, dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate sweep CSVs and produce reusable visualizations."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing sweep CSV outputs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results_cf-*.csv",
        help="Glob pattern for result files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="embedding_similarity",
        choices=[
            "embedding_similarity",
            "llm_judge_score",
            "character_accuracy",
            "word_accuracy",
        ],
        help="Metric to use for heatmaps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "figures",
        help="Where to write visualizations.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_df = load_results(args.results_dir, args.pattern)
    summary_df = summarize_runs(results_df)

    print("Loaded sweeps:")
    print(
        summary_df[
            [
                "run_id",
                "context_filter",
                "conversation_window",
                "time_window_hours",
                "geo_radius_km",
            ]
        ]
    )

    plot_metric_by_conversation_window(summary_df, args.output_dir)
    plot_generation_boxplots(results_df, args.output_dir)
    plot_heatmaps(results_df, args.output_dir, metric=args.metric)

    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
