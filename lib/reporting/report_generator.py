"""
Report Generator

Generates comprehensive reports from aggregated results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..evaluation.visualizers import ComparisonVisualizer, ResultsVisualizer
from .aggregator import ResultsAggregator


class ReportGenerator:
    """
    Generates comprehensive reports from experiment results.
    """

    def __init__(self, output_dir: str | Path, title: str = "Context-Aware AAC Research Results"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
            title: Report title
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.aggregator = ResultsAggregator()
        self.visualizer = ResultsVisualizer()
        self.comparison_visualizer = ComparisonVisualizer()

    def generate_full_report(
        self, phase_results: dict[str, str | Path], include_figures: bool = True
    ) -> dict[str, Any]:
        """
        Generate a complete research report.

        Args:
            phase_results: Dictionary mapping phase names to result files
            include_figures: Whether to generate figures

        Returns:
            Dictionary with report information
        """
        print(f"Generating report: {self.title}")

        # Load all results
        for phase_name, results_path in phase_results.items():
            print(f"Loading {phase_name} results...")
            self.aggregator.load_phase_results(phase_name, results_path)

        # Combine results
        combined_results = self.aggregator.combine_all_results()
        print(f"Combined {len(combined_results)} evaluations")

        # Generate summary
        summary = self.aggregator.get_summary_statistics(combined_results)

        # Generate figures
        figures = {}
        if include_figures:
            print("Generating figures...")
            figures = self._generate_figures(combined_results, phase_results)

        # Generate tables
        tables = self._generate_tables(combined_results)

        # Save all components
        report_info = {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "figures": figures,
            "tables": tables,
            "total_evaluations": len(combined_results),
        }

        # Save report data
        self._save_report_data(report_info)

        # Generate HTML report
        self._generate_html_report(report_info)

        print(f"Report saved to: {self.output_dir}")
        return report_info

    def _generate_figures(
        self, combined_results: pd.DataFrame, phase_results: dict[str, str | Path]
    ) -> dict[str, str]:
        """Generate all figures for the report."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        figure_paths = {}

        # Overall performance comparison
        if "phase" in combined_results.columns:
            phase_comparison = {}
            for phase_name in combined_results["phase"].unique():
                phase_df = combined_results[combined_results["phase"] == phase_name]
                phase_comparison[phase_name] = phase_df

            fig_path = figures_dir / "phase_comparison.png"
            self.comparison_visualizer.compare_experiments(
                phase_comparison, metric="embedding_similarity", output_path=fig_path
            )
            figure_paths["phase_comparison"] = str(fig_path)

        # Generate figures for each metric
        metrics = [
            col
            for col in combined_results.columns
            if any(suffix in col.lower() for suffix in ["similarity", "score", "accuracy", "bleu"])
        ]

        for metric in metrics:
            if metric in combined_results.columns:
                # Distribution plot
                fig_path = figures_dir / f"{metric}_distribution.png"
                self.visualizer.create_score_distribution(
                    combined_results, metric=metric, output_path=fig_path
                )
                figure_paths[f"{metric}_distribution"] = str(fig_path)

                # Method comparison if applicable
                if "generation_method" in combined_results.columns:
                    fig_path = figures_dir / f"{metric}_method_comparison.png"
                    self.visualizer.create_method_comparison(
                        combined_results, metric=metric, output_path=fig_path
                    )
                    figure_paths[f"{metric}_method_comparison"] = str(fig_path)

        return figure_paths

    def _generate_tables(self, combined_results: pd.DataFrame) -> dict[str, str]:
        """Generate all tables for the report."""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        table_paths = {}

        # Main comparison table
        comparison_table = self.aggregator.generate_comparison_table(combined_results)
        comparison_path = tables_dir / "method_comparison.csv"
        comparison_table.to_csv(comparison_path, index=False)
        table_paths["method_comparison"] = str(comparison_path)

        # Summary statistics table
        summary = self.aggregator.get_summary_statistics(combined_results)

        # Create summary table
        summary_rows = []
        for key, value in summary.items():
            if key.endswith("_stats") and isinstance(value, dict):
                metric_name = key.replace("_stats", "")
                row = {
                    "metric": metric_name,
                    "mean": value.get("mean"),
                    "std": value.get("std"),
                    "min": value.get("min"),
                    "max": value.get("max"),
                    "median": value.get("median"),
                    "count": value.get("count"),
                }
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = tables_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_path, index=False)
            table_paths["summary_statistics"] = str(summary_path)

        # Best performers table
        best_performers = self.aggregator.find_best_performers(combined_results)
        best_df = pd.DataFrame([best_performers["best_overall"]])
        best_path = tables_dir / "best_performer.csv"
        best_df.to_csv(best_path, index=False)
        table_paths["best_performer"] = str(best_path)

        return table_paths

    def _save_report_data(self, report_info: dict[str, Any]):
        """Save report data as JSON."""
        report_path = self.output_dir / "report_data.json"

        # Convert non-serializable objects
        serializable_report = {}
        for key, value in report_info.items():
            if key == "summary":
                # Convert numpy types
                serializable_summary = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_summary[k] = {
                            k2: float(v2) if isinstance(v2, int | float) else v2
                            for k2, v2 in v.items()
                        }
                    else:
                        serializable_summary[k] = v
                serializable_report[key] = serializable_summary
            else:
                serializable_report[key] = value

        with open(report_path, "w") as f:
            json.dump(serializable_report, f, indent=2)

    def _generate_html_report(self, report_info: dict[str, Any]):
        """Generate an HTML report."""
        html_path = self.output_dir / "report.html"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        figure {{
            margin: 20px 0;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .metric {{
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p>Generated on: {report_info['generated_at']}</p>
    <p>Total evaluations: {report_info['total_evaluations']}</p>

    <h2>Summary Statistics</h2>
"""

        # Add summary statistics
        for key, value in report_info["summary"].items():
            if key.endswith("_stats") and isinstance(value, dict):
                metric_name = key.replace("_stats", "").replace("_", " ").title()
                html += f"""
    <div class="metric">
        <h3>{metric_name}</h3>
        <ul>
            <li>Mean: {value.get('mean', 'N/A'):.3f}</li>
            <li>Std: {value.get('std', 'N/A'):.3f}</li>
            <li>Min: {value.get('min', 'N/A'):.3f}</li>
            <li>Max: {value.get('max', 'N/A'):.3f}</li>
            <li>Median: {value.get('median', 'N/A'):.3f}</li>
        </ul>
    </div>
"""

        # Add figures
        if report_info["figures"]:
            html += "<h2>Figures</h2>"
            for fig_name, fig_path in report_info["figures"].items():
                fig_file = Path(fig_path).name
                html += f"""
    <figure>
        <img src="figures/{fig_file}" alt="{fig_name}">
        <figcaption>{fig_name.replace('_', ' ').title()}</figcaption>
    </figure>
"""

        html += """
</body>
</html>
"""

        with open(html_path, "w") as f:
            f.write(html)
