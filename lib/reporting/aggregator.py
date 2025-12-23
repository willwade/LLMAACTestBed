"""
Results Aggregator

Aggregates results from multiple experiments for comprehensive analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ResultsAggregator:
    """
    Aggregates and combines results from different experimental phases.
    """

    def __init__(self):
        """Initialize the aggregator."""
        self.phase_results = {}
        self.combined_results = None

    def load_phase_results(self, phase_name: str, results_path: str | Path) -> pd.DataFrame:
        """
        Load results from a specific phase.

        Args:
            phase_name: Name of the phase (e.g., "phase1", "phase2")
            results_path: Path to results CSV file

        Returns:
            Loaded results DataFrame
        """
        results_df = pd.read_csv(results_path)

        # Add phase column if not present
        if "phase" not in results_df.columns:
            results_df["phase"] = phase_name

        # Add load timestamp
        results_df["load_timestamp"] = datetime.now()

        self.phase_results[phase_name] = results_df
        return results_df

    def combine_all_results(self) -> pd.DataFrame:
        """
        Combine results from all loaded phases.

        Returns:
            Combined DataFrame with all results
        """
        if not self.phase_results:
            raise ValueError("No phase results loaded. Use load_phase_results first.")

        # Combine all DataFrames
        combined_dfs = []
        for phase_name, results_df in self.phase_results.items():
            df = results_df.copy()
            if "phase" not in df.columns:
                df["phase"] = phase_name
            combined_dfs.append(df)

        self.combined_results = pd.concat(combined_dfs, ignore_index=True)
        return self.combined_results

    def get_summary_statistics(self, results_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Generate summary statistics for results.

        Args:
            results_df: DataFrame to summarize (uses combined if None)

        Returns:
            Dictionary with summary statistics
        """
        if results_df is None:
            if self.combined_results is None:
                self.combine_all_results()
            results_df = self.combined_results

        summary = {}

        # Overall statistics
        summary["total_evaluations"] = len(results_df)
        summary["phases"] = results_df["phase"].unique().tolist() if "phase" in results_df else []

        # Metric columns
        metric_cols = [
            col
            for col in results_df.columns
            if any(suffix in col.lower() for suffix in ["similarity", "score", "accuracy", "bleu"])
        ]

        # Statistics by metric
        for metric in metric_cols:
            if metric in results_df.columns:
                metric_data = results_df[metric].dropna()
                if len(metric_data) > 0:
                    summary[f"{metric}_stats"] = {
                        "mean": metric_data.mean(),
                        "std": metric_data.std(),
                        "min": metric_data.min(),
                        "max": metric_data.max(),
                        "median": metric_data.median(),
                        "count": len(metric_data),
                    }

        # Statistics by phase
        if "phase" in results_df.columns:
            summary["by_phase"] = {}
            for phase in results_df["phase"].unique():
                phase_df = results_df[results_df["phase"] == phase]
                phase_stats = {}

                for metric in metric_cols:
                    if metric in phase_df.columns:
                        metric_data = phase_df[metric].dropna()
                        if len(metric_data) > 0:
                            phase_stats[f"{metric}_mean"] = metric_data.mean()
                            phase_stats[f"{metric}_std"] = metric_data.std()

                summary["by_phase"][phase] = phase_stats

        return summary

    def compare_methods(
        self, results_df: pd.DataFrame | None = None, metric: str = "embedding_similarity"
    ) -> pd.DataFrame:
        """
        Compare performance across different methods.

        Args:
            results_df: Results DataFrame
            metric: Metric to compare

        Returns:
            Comparison DataFrame
        """
        if results_df is None:
            if self.combined_results is None:
                self.combine_all_results()
            results_df = self.combined_results

        # Group by method combinations
        grouping_cols = []
        for col in ["partial_method", "generation_method", "hypothesis", "context_type"]:
            if col in results_df.columns:
                grouping_cols.append(col)

        if not grouping_cols:
            # Simple comparison by phase
            comparison = results_df.groupby("phase")[metric].agg(["mean", "std", "count"])
        else:
            # Complex comparison by methods
            comparison = results_df.groupby(grouping_cols)[metric].agg(["mean", "std", "count"])

        return comparison.reset_index()

    def find_best_performers(
        self, results_df: pd.DataFrame | None = None, metric: str = "embedding_similarity"
    ) -> dict[str, Any]:
        """
        Find best performing configurations.

        Args:
            results_df: Results DataFrame
            metric: Metric to optimize

        Returns:
            Dictionary with best performers
        """
        if results_df is None:
            if self.combined_results is None:
                self.combine_all_results()
            results_df = self.combined_results

        if metric not in results_df.columns:
            return {}

        best_overall = results_df.loc[results_df[metric].idxmax()].to_dict()

        best_by_phase = {}
        if "phase" in results_df.columns:
            for phase in results_df["phase"].unique():
                phase_df = results_df[results_df["phase"] == phase]
                best_by_phase[phase] = phase_df.loc[phase_df[metric].idxmax()].to_dict()

        return {
            "best_overall": best_overall,
            "best_by_phase": best_by_phase,
            "best_score": best_overall[metric],
        }

    def generate_comparison_table(self, results_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Generate a comparison table for research paper.

        Args:
            results_df: Results DataFrame

        Returns:
            Comparison table DataFrame
        """
        if results_df is None:
            if self.combined_results is None:
                self.combine_all_results()
            results_df = self.combined_results

        # Prepare data for paper table
        table_data = []

        # Get unique method combinations
        if "phase" in results_df.columns:
            for phase in results_df["phase"].unique():
                phase_df = results_df[results_df["phase"] == phase]
                self._add_phase_to_table(phase_df, phase, table_data)
        else:
            self._add_phase_to_table(results_df, "combined", table_data)

        comparison_df = pd.DataFrame(table_data)

        # Sort by score
        score_cols = [
            col
            for col in comparison_df.columns
            if "similarity" in col or "score" in col or "accuracy" in col
        ]
        if score_cols:
            comparison_df = comparison_df.sort_values(score_cols[0], ascending=False)

        return comparison_df

    def _add_phase_to_table(
        self, phase_df: pd.DataFrame, phase_name: str, table_data: list[dict[str, Any]]
    ):
        """Add phase data to comparison table."""
        # Group by methods
        grouping_cols = []
        for col in ["partial_method", "generation_method", "hypothesis", "context_type"]:
            if col in phase_df.columns:
                grouping_cols.append(col)

        if grouping_cols:
            grouped = phase_df.groupby(grouping_cols)
        else:
            # Use all data as one group
            grouped = [(None, phase_df)]

        for group_key, group_df in grouped:
            # Calculate metrics
            row = {"phase": phase_name}

            if group_key:
                if isinstance(group_key, tuple):
                    for i, col in enumerate(grouping_cols):
                        row[col] = group_key[i]
                else:
                    row[grouping_cols[0]] = group_key

            # Add metric averages
            metric_cols = [
                col
                for col in group_df.columns
                if any(
                    suffix in col.lower() for suffix in ["similarity", "score", "accuracy", "bleu"]
                )
            ]

            for metric in metric_cols:
                if metric in group_df.columns:
                    row[f"{metric}_mean"] = group_df[metric].mean()
                    row[f"{metric}_std"] = group_df[metric].std()
                    row[f"{metric}_count"] = len(group_df)

            table_data.append(row)
