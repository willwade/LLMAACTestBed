"""
Base Evaluator

Abstract base class for all evaluators in the framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.loaders import DataLoader
from ..llm_clients.base import BaseLLMClient
from .metrics import BaseMetric, MetricsCalculator
from .visualizers import ResultsVisualizer


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.

    Provides common functionality for running evaluations and managing results.
    """

    def __init__(self, llm_client: BaseLLMClient, config: dict[str, Any] | None = None):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for generation and scoring
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ResultsVisualizer()

        # Results storage
        self.results = []
        self.evaluation_start_time = None
        self.evaluation_end_time = None

    @abstractmethod
    def run_evaluation(self, *args, **kwargs) -> pd.DataFrame:
        """
        Run the evaluation with specific parameters.

        Returns:
            DataFrame with evaluation results
        """
        pass

    def setup_metrics(self, metrics: list[BaseMetric] | None = None):
        """
        Setup custom metrics for evaluation.

        Args:
            metrics: List of metric instances
        """
        if metrics:
            self.metrics_calculator.metrics = metrics

    def create_output_directory(self, base_dir: str = "results") -> Path:
        """
        Create output directory with timestamp.

        Args:
            base_dir: Base directory for results

        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "figures").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)

        return output_dir

    def save_results(
        self, results_df: pd.DataFrame, output_dir: Path, prefix: str = "results"
    ) -> list[str]:
        """
        Save results to files.

        Args:
            results_df: DataFrame with results
            output_dir: Output directory path
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        saved_files = []

        # Save detailed CSV
        csv_path = output_dir / "data" / f"{prefix}.csv"
        results_df.to_csv(csv_path, index=False)
        saved_files.append(str(csv_path))

        # Save summary statistics
        summary_df = self._create_summary(results_df)
        summary_path = output_dir / "data" / f"{prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        saved_files.append(str(summary_path))

        # Save configuration
        if self.config:
            import json

            config_path = output_dir / "data" / f"{prefix}_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            saved_files.append(str(config_path))

        return saved_files

    def _create_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics from results.

        Args:
            results_df: Detailed results DataFrame

        Returns:
            Summary statistics DataFrame
        """
        summary_data = []

        # Group by method combinations
        if "partial_method" in results_df.columns and "generation_method" in results_df.columns:
            grouped = results_df.groupby(["partial_method", "generation_method"])
        else:
            grouped = [(None, results_df)]

        for name, group in grouped:
            for metric_col in results_df.columns:
                if (
                    metric_col.endswith("_similarity")
                    or metric_col.endswith("_score")
                    or metric_col.endswith("_accuracy")
                ):
                    if metric_col in group.columns:
                        stats = {
                            "method": str(name) if name else "all",
                            "metric": metric_col,
                            "count": len(group),
                            "mean": group[metric_col].mean(),
                            "std": group[metric_col].std(),
                            "min": group[metric_col].min(),
                            "max": group[metric_col].max(),
                            "median": group[metric_col].median(),
                        }
                        summary_data.append(stats)

        return pd.DataFrame(summary_data)

    def log_evaluation_start(self):
        """Log the start of evaluation."""
        self.evaluation_start_time = datetime.now()
        print(f"Evaluation started at: {self.evaluation_start_time}")

    def log_evaluation_end(self):
        """Log the end of evaluation."""
        self.evaluation_end_time = datetime.now()
        if self.evaluation_start_time:
            duration = self.evaluation_end_time - self.evaluation_start_time
            print(f"Evaluation completed at: {self.evaluation_end_time}")
            print(f"Total duration: {duration}")

    def visualize_results(
        self, results_df: pd.DataFrame, output_dir: Path, formats: list[str] | None = None
    ) -> list[str]:
        """
        Generate visualizations for results.

        Args:
            results_df: Results DataFrame
            output_dir: Output directory
            formats: List of image formats

        Returns:
            List of generated figure paths
        """
        if not formats:
            formats = ["png"]

        return self.visualizer.create_all_visualizations(
            results_df, output_dir / "figures", formats=formats
        )
