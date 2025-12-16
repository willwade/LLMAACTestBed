"""
Paper Formatter

Formats results for academic paper publication.
"""

from typing import Any

import pandas as pd

from .aggregator import ResultsAggregator


class PaperFormatter:
    """
    Formats experiment results for academic paper publication.
    """

    def __init__(self):
        """Initialize the paper formatter."""
        self.aggregator = ResultsAggregator()

    def format_latex_table(
        self,
        results_df: pd.DataFrame,
        caption: str = "Experimental Results",
        label: str = "tab:results"
    ) -> str:
        """
        Format results as a LaTeX table.

        Args:
            results_df: Results DataFrame
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        # Prepare data
        comparison_df = self.aggregator.generate_comparison_table(results_df)

        # Identify metric columns
        metric_cols = [col for col in comparison_df.columns
                      if any(suffix in col for suffix in ['_mean', '_std'])]

        # Create LaTeX table
        latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' + 'c' * len(metric_cols)}}}
\\toprule
"""

        # Header
        headers = ['Method'] + [self._format_header(col) for col in metric_cols]
        latex += " & ".join(headers) + " \\\\n\\midrule\n"

        # Rows
        for _, row in comparison_df.iterrows():
            row_data = []

            # Method name
            if 'phase' in row:
                method_parts = [str(row['phase'])]
            else:
                method_parts = []

            for col in ['partial_method', 'generation_method', 'hypothesis', 'context_type']:
                if col in row and pd.notna(row[col]):
                    method_parts.append(str(row[col]))

            method_name = " + ".join(method_parts) if method_parts else "Overall"
            row_data.append(method_name)

            # Metrics
            for col in metric_cols:
                mean_col = col.replace('_mean', '')
                std_col = col.replace('_mean', '_std')

                if mean_col in row and std_col in row:
                    mean_val = row[mean_col]
                    std_val = row[std_col]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        row_data.append(f"{mean_val:.3f} $\\pm$ {std_val:.3f}")
                    else:
                        row_data.append("N/A")
                else:
                    row_data.append("N/A")

            latex += " & ".join(row_data) + " \\\\\n"

        latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

        return latex

    def format_statistical_analysis(
        self,
        results_df: pd.DataFrame,
        metric: str = 'embedding_similarity'
    ) -> dict[str, Any]:
        """
        Format statistical analysis for paper.

        Args:
            results_df: Results DataFrame
            metric: Primary metric for analysis

        Returns:
            Dictionary with statistical analysis
        """
        if metric not in results_df.columns:
            return {}

        from scipy import stats

        analysis = {}

        # Overall statistics
        metric_data = results_df[metric].dropna()
        analysis['overall'] = {
            'n': len(metric_data),
            'mean': metric_data.mean(),
            'std': metric_data.std(),
            'ci': self._calculate_confidence_interval(metric_data),
            'normality_test': stats.shapiro(metric_data) if len(metric_data) >= 3 else None
        }

        # By phase analysis
        if 'phase' in results_df.columns:
            analysis['by_phase'] = {}
            phases = results_df['phase'].unique()

            for phase in phases:
                phase_data = results_df[results_df['phase'] == phase][metric].dropna()
                if len(phase_data) > 0:
                    analysis['by_phase'][phase] = {
                        'n': len(phase_data),
                        'mean': phase_data.mean(),
                        'std': phase_data.std(),
                        'ci': self._calculate_confidence_interval(phase_data)
                    }

            # ANOVA test if multiple phases
            if len(phases) > 2:
                phase_groups = [results_df[results_df['phase'] == phase][metric].dropna()
                              for phase in phases]
                try:
                    analysis['anova'] = stats.f_oneway(*phase_groups)
                except Exception:
                    analysis['anova'] = None

            # Pairwise t-tests
            analysis['pairwise_tests'] = {}
            for i, phase1 in enumerate(phases):
                for phase2 in phases[i+1:]:
                    data1 = results_df[results_df['phase'] == phase1][metric].dropna()
                    data2 = results_df[results_df['phase'] == phase2][metric].dropna()
                    if len(data1) > 0 and len(data2) > 0:
                        test_result = stats.ttest_ind(data1, data2)
                        analysis['pairwise_tests'][f"{phase1}_vs_{phase2}"] = {
                            't_statistic': test_result.statistic,
                            'p_value': test_result.pvalue
                        }

        return analysis

    def generate_results_summary(
        self,
        results_df: pd.DataFrame
    ) -> str:
        """
        Generate a text summary of results for paper.

        Args:
            results_df: Results DataFrame

        Returns:
            Results summary text
        """
        summary = []

        # Overall statistics
        total_evals = len(results_df)
        summary.append(f"We conducted {total_evals} evaluations across all experimental phases.")

        # Best performer
        best = self.aggregator.find_best_performers(results_df)
        if best and 'best_overall' in best:
            best_score = best['best_score']
            summary.append(f"The best performing configuration achieved a score of {best_score:.3f}.")

        # Phase comparisons
        if 'phase' in results_df.columns:
            phases = results_df['phase'].unique()
            summary.append("\nResults by phase:")

            for phase in sorted(phases):
                phase_df = results_df[results_df['phase'] == phase]
                phase_evals = len(phase_df)

                # Get best metric for phase
                metric_cols = [col for col in phase_df.columns
                              if any(suffix in col.lower() for suffix in
                                     ['similarity', 'score', 'accuracy'])]
                if metric_cols:
                    best_metric = metric_cols[0]
                    best_score = phase_df[best_metric].max()
                    mean_score = phase_df[best_metric].mean()

                    summary.append(f"  • {phase.title()}: {phase_evals} evaluations, "
                                 f"mean score = {mean_score:.3f}, best = {best_score:.3f}")

        # Method insights
        summary.append("\nKey findings:")

        # Context comparison
        if 'context_filter' in results_df.columns:
            context_means = results_df.groupby('context_filter')[metric_cols[0]].mean()
            best_context = context_means.idxmax()
            worst_context = context_means.idxmin()
            improvement = context_means[best_context] - context_means[worst_context]

            summary.append(f"  • Context filtering improved performance by {improvement:.3f} "
                         f"({worst_context}: {context_means[worst_context]:.3f} vs "
                         f"{best_context}: {context_means[best_context]:.3f})")

        return "\n".join(summary)

    def format_citation(self, experiment_name: str) -> str:
        """
        Format citation for the experiment.

        Args:
            experiment_name: Name of experiment

        Returns:
            Formatted citation
        """
        current_year = pd.Timestamp.now().year
        exp_key = experiment_name.lower().replace(' ', '_')

        # Build citation without f-string to avoid complexity
        citation_lines = [
            "@misc{" + exp_key + "_" + str(current_year) + "},",
            "  title={Context-Aware AAC Systems: " + experiment_name + "},",
            "  author={ContextAwareTestBed Research Team},",
            "  year={2025},",
            "  note={Unpublished manuscript}"
        ]
        return "\n".join(citation_lines)

    def _format_header(self, column_name: str) -> str:
        """Format column header for LaTeX."""
        if '_mean' in column_name:
            base_name = column_name.replace('_mean', '')
            return base_name.replace('_', ' ').title()
        elif '_std' in column_name:
            return "Std"
        else:
            return column_name.replace('_', ' ').title()

    def _calculate_confidence_interval(
        self,
        data: pd.Series,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for data."""
        from scipy import stats

        n = len(data)
        if n < 2:
            return (0, 0)

        mean = data.mean()
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        return (mean - h, mean + h)
