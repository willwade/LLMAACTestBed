"""
Visualization Utilities

Common visualization patterns for evaluation results.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("default")
sns.set_palette("husl")

warnings.filterwarnings("ignore")


class ResultsVisualizer:
    """
    Visualizer for evaluation results.
    """

    def __init__(self, figsize: tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def create_performance_heatmap(
        self,
        results_df: pd.DataFrame,
        metric: str = "embedding_similarity",
        output_path: Path | None = None,
    ) -> plt.Figure | None:
        """
        Create heatmap of performance by method combination.

        Args:
            results_df: Results DataFrame
            metric: Metric to visualize
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if (
            "partial_method" not in results_df.columns
            or "generation_method" not in results_df.columns
        ):
            print("Cannot create heatmap: missing method columns")
            return None

        # Create pivot table
        pivot_df = results_df.pivot_table(
            values=metric, index="partial_method", columns="generation_method", aggfunc="mean"
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": metric.replace("_", " ").title()},
        )

        ax.set_title(f'Performance Heatmap - {metric.replace("_", " ").title()}')
        ax.set_xlabel("Generation Method")
        ax.set_ylabel("Partial Method")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()

        return fig

    def create_method_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "embedding_similarity",
        output_path: Path | None = None,
    ) -> plt.Figure | None:
        """
        Create bar chart comparing methods.

        Args:
            results_df: Results DataFrame
            metric: Metric to visualize
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Group by method if present
        if "generation_method" in results_df.columns:
            grouped = results_df.groupby("generation_method")[metric].agg(["mean", "std"])
            methods = grouped.index.tolist()
            means = grouped["mean"].values
            stds = grouped["std"].values
        else:
            # Single method
            means = [results_df[metric].mean()]
            stds = [results_df[metric].std()]
            methods = ["Method"]

        # Create bar chart
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7)

        # Add value labels on bars
        for bar, mean in zip(bars, means, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(stds) * 0.1,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
            )

        ax.set_title(f'Method Comparison - {metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Method")
        ax.set_ylim(0, max(means) + max(stds) * 0.5)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()

        return fig

    def create_score_distribution(
        self,
        results_df: pd.DataFrame,
        metric: str = "embedding_similarity",
        output_path: Path | None = None,
    ) -> plt.Figure | None:
        """
        Create distribution plot for scores.

        Args:
            results_df: Results DataFrame
            metric: Metric to visualize
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        ax1.hist(results_df[metric], bins=20, alpha=0.7, edgecolor="black")
        ax1.set_title(f'Score Distribution - {metric.replace("_", " ").title()}')
        ax1.set_xlabel(metric.replace("_", " ").title())
        ax1.set_ylabel("Frequency")
        ax1.axvline(results_df[metric].mean(), color="red", linestyle="--", label="Mean")
        ax1.axvline(results_df[metric].median(), color="green", linestyle="--", label="Median")
        ax1.legend()

        # Box plot
        if "generation_method" in results_df.columns:
            sns.boxplot(data=results_df, x="generation_method", y=metric, ax=ax2)
            ax2.set_title(f'Score Distribution by Method - {metric.replace("_", " ").title()}')
            ax2.tick_params(axis="x", rotation=45)
        else:
            ax2.boxplot(results_df[metric])
            ax2.set_title(f'Score Distribution - {metric.replace("_", " ").title()}')
            ax2.set_ylabel(metric.replace("_", " ").title())

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()

        return fig

    def create_temporal_analysis(
        self,
        results_df: pd.DataFrame,
        time_col: str = "timestamp",
        metric: str = "embedding_similarity",
        output_path: Path | None = None,
    ) -> plt.Figure | None:
        """
        Create temporal analysis plot.

        Args:
            results_df: Results DataFrame with timestamp column
            time_col: Name of timestamp column
            metric: Metric to visualize
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if time_col not in results_df.columns:
            print(f"Cannot create temporal plot: missing {time_col} column")
            return None

        # Convert timestamp if needed
        df = results_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Create time-based features
        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.day_name()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Performance by hour
        hourly_perf = df.groupby("hour")[metric].agg(["mean", "std"])
        ax1.errorbar(
            hourly_perf.index, hourly_perf["mean"], yerr=hourly_perf["std"], marker="o", capsize=5
        )
        ax1.set_title(f'Performance by Hour - {metric.replace("_", " ").title()}')
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel(metric.replace("_", " ").title())
        ax1.set_xticks(range(0, 24, 2))

        # Performance by day of week
        if "day_of_week" in df.columns:
            daily_perf = df.groupby("day_of_week")[metric].agg(["mean", "std"])
            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            daily_perf = daily_perf.reindex(days_order)
            ax2.bar(daily_perf.index, daily_perf["mean"], yerr=daily_perf["std"], capsize=5)
            ax2.set_title(f'Performance by Day - {metric.replace("_", " ").title()}')
            ax2.set_ylabel(metric.replace("_", " ").title())
            ax2.tick_params(axis="x", rotation=45)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()

        return fig

    def create_all_visualizations(
        self, results_df: pd.DataFrame, output_dir: Path, formats: list[str] | None = None
    ) -> list[str]:
        """
        Create all standard visualizations.

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
            formats: List of image formats

        Returns:
            List of generated figure paths
        """
        generated_files = []

        # Get available metrics
        metrics = [
            col
            for col in results_df.columns
            if col.endswith("_similarity") or col.endswith("_score") or col.endswith("_accuracy")
        ]

        for metric in metrics:
            # Performance heatmap
            if "partial_method" in results_df.columns and "generation_method" in results_df.columns:
                for fmt in formats:
                    path = output_dir / f"heatmap_{metric}.{fmt}"
                    self.create_performance_heatmap(results_df, metric, path)
                    generated_files.append(str(path))

            # Method comparison
            for fmt in formats:
                path = output_dir / f"comparison_{metric}.{fmt}"
                self.create_method_comparison(results_df, metric, path)
                generated_files.append(str(path))

            # Score distribution
            for fmt in formats:
                path = output_dir / f"distribution_{metric}.{fmt}"
                self.create_score_distribution(results_df, metric, path)
                generated_files.append(str(path))

            # Temporal analysis
            if "timestamp" in results_df.columns:
                for fmt in formats:
                    path = output_dir / f"temporal_{metric}.{fmt}"
                    self.create_temporal_analysis(results_df, metric, path)
                    generated_files.append(str(path))

        return generated_files


class ComparisonVisualizer:
    """
    Visualizer for comparing different experiments or conditions.
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        """
        Initialize comparison visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def compare_experiments(
        self,
        experiments_data: dict[str, pd.DataFrame],
        metric: str = "embedding_similarity",
        output_path: Path | None = None,
    ) -> plt.Figure | None:
        """
        Compare multiple experiments.

        Args:
            experiments_data: Dictionary mapping experiment names to DataFrames
            metric: Metric to compare
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        means = []
        stds = []
        names = []

        for name, df in experiments_data.items():
            if metric in df.columns:
                means.append(df[metric].mean())
                stds.append(df[metric].std())
                names.append(name)

        bars = ax.bar(names, means, yerr=stds, capsize=5, alpha=0.7)

        # Add value labels
        for bar, mean in zip(bars, means, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(stds) * 0.1,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
            )

        ax.set_title(f'Experiment Comparison - {metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()

        return fig
