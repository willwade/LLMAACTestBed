"""
Results Analyzer for Phase 4 Experiments

Analyzes and visualizes results from keyword-to-utterance generation experiments.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyzes and visualizes experiment results."""

    def __init__(self, results_dir: Path):
        """
        Initialize results analyzer.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def load_all_results(self) -> dict[str, Any]:
        """Load all result files from the results directory."""
        results = {}

        # Load individual part results
        result_files = {
            "baseline": "part1_baseline_results.json",
            "contextual": "part2_contextual_results.json",
            "single_keyword": "part3_single_keyword_results.json",
            "summary": "experiment_summary.json"
        }

        for key, filename in result_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                with open(file_path) as f:
                    results[key] = json.load(f)

        return results

    def create_score_distribution_plot(self, scores: list[float], title: str, save_name: str):
        """Create distribution plot of scores."""
        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(scores, bins=range(1, 12), alpha=0.7, edgecolor='black')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Similarity Score (1-10)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(range(1, 11))
        plt.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_score = sum(scores) / len(scores)
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_context_comparison_plot(self, contextual_results: dict[str, Any]):
        """Create comparison plot for different context levels."""
        if "context_levels" not in contextual_results:
            return

        context_levels = contextual_results["context_levels"]
        scores_by_level = {}

        for level, results in context_levels.items():
            if "scores" in results:
                scores_by_level[level] = results["scores"]

        if not scores_by_level:
            return

        # Create box plot
        plt.figure(figsize=(12, 8))

        # Prepare data for box plot
        data_for_box = []
        labels = []

        for level, scores in scores_by_level.items():
            data_for_box.append(scores)
            labels.append(level.replace('_', ' ').title())

        plt.boxplot(data_for_box)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title('Score Distribution by Context Level', fontsize=14, fontweight='bold')
        plt.ylabel('Similarity Score (1-10)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "context_level_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create improvement plot
        if "improvements_by_level" in contextual_results:
            improvements = contextual_results["improvements_by_level"]
            if improvements:
                levels = list(improvements.keys())
                mean_improvements = [improvements[level]["mean_improvement"] for level in levels]

                plt.figure(figsize=(10, 6))
                bars = plt.bar(levels, mean_improvements)
                plt.title('Mean Score Improvement by Context Level', fontsize=14, fontweight='bold')
                plt.xlabel('Context Level', fontsize=12)
                plt.ylabel('Mean Improvement (points)', fontsize=12)
                plt.xticks(rotation=45)

                # Add value labels on bars
                for bar, improvement in zip(bars, mean_improvements, strict=False):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{improvement:.2f}', ha='center', va='bottom')

                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.figures_dir / "context_improvements.png", dpi=300, bbox_inches='tight')
                plt.close()

    def create_single_keyword_analysis(self, single_keyword_results: dict[str, Any]):
        """Create analysis plots for single keyword tests."""
        if "effectiveness_by_keyword" not in single_keyword_results:
            return

        keyword_stats = single_keyword_results["effectiveness_by_keyword"]

        # Create DataFrame for easier plotting
        df_data = []
        for keyword, stats in keyword_stats.items():
            df_data.append({
                'Keyword': keyword,
                'Mean Score': stats['mean_score'],
                'Success Rate': stats['success_rate'],
                'Total Tests': stats['total_tests']
            })

        df = pd.DataFrame(df_data)
        df = df.sort_values('Mean Score', ascending=True)

        # Create horizontal bar chart
        plt.figure(figsize=(12, 8))

        # Plot mean scores
        bars = plt.barh(df['Keyword'], df['Mean Score'])
        plt.title('Single Keyword Effectiveness (Mean Scores)', fontsize=14, fontweight='bold')
        plt.xlabel('Mean Similarity Score (1-10)', fontsize=12)
        plt.ylabel('Keyword', fontsize=12)

        # Add value labels
        for bar, score in zip(bars, df['Mean Score'], strict=False):
            plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', ha='left', va='center')

        plt.xlim(0, 11)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "single_keyword_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self, all_results: dict[str, Any]):
        """Create a summary dashboard with key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase 4 Keyword-to-Utterance Generation: Summary Dashboard',
                     fontsize=16, fontweight='bold')

        # 1. Overall performance comparison
        ax1 = axes[0, 0]
        experiment_scores = {}
        experiment_labels = []

        for exp_name, results in all_results.items():
            if exp_name == "summary" or "scores" not in str(results):
                continue
            if "summary_statistics" in results:
                stats = results["summary_statistics"]
                for part_name, part_stats in stats.items():
                    experiment_scores[part_name] = part_stats.get("mean_score", 0)
                    experiment_labels.append(part_name.replace("_", " ").title())

        if experiment_scores:
            colors = plt.cm.get_cmap('tab10')(range(len(experiment_scores)))
            bars = ax1.bar(range(len(experiment_scores)), list(experiment_scores.values()), color=colors)
            ax1.set_title('Mean Score by Experiment Part', fontweight='bold')
            ax1.set_ylabel('Mean Score (1-10)')
            ax1.set_xticks(range(len(experiment_scores)))
            ax1.set_xticklabels([k.replace("_", " ").title() for k in experiment_scores.keys()],
                              rotation=45, ha='right')
            ax1.set_ylim(0, 10)

            # Add value labels
            for bar, score in zip(bars, experiment_scores.values(), strict=False):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom')

        # 2. Success rates
        ax2 = axes[0, 1]
        success_rates = {}

        for exp_name, results in all_results.items():
            if exp_name == "summary" or "summary_statistics" not in results:
                continue
            stats = results["summary_statistics"]
            for part_name, part_stats in stats.items():
                success_rates[part_name] = part_stats.get("accuracy_over_70_percent", 0)

        if success_rates:
            colors = plt.cm.get_cmap('tab10')(range(len(success_rates)))
            bars = ax2.bar(range(len(success_rates)), list(success_rates.values()), color=colors)
            ax2.set_title('Success Rate (Score ≥ 7/10)', fontweight='bold')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_xticks(range(len(success_rates)))
            ax2.set_xticklabels([k.replace("_", " ").title() for k in success_rates.keys()],
                              rotation=45, ha='right')
            ax2.set_ylim(0, 100)

            # Add value labels
            for bar, rate in zip(bars, success_rates.values(), strict=False):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')

        # 3. Score distribution for baseline (if available)
        ax3 = axes[1, 0]
        if "baseline" in all_results and "scores" in all_results["baseline"]:
            scores = all_results["baseline"]["scores"]
            ax3.hist(scores, bins=range(1, 12), alpha=0.7, edgecolor='black', color='skyblue')
            ax3.set_title('Baseline Score Distribution', fontweight='bold')
            ax3.set_xlabel('Score (1-10)')
            ax3.set_ylabel('Frequency')
            ax3.set_xticks(range(1, 11))
            ax3.grid(axis='y', alpha=0.3)

        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')

        if "summary" in all_results and "summary_statistics" in all_results["summary"]:
            summary_stats = all_results["summary"]["summary_statistics"]

            # Create table data
            table_data = []
            headers = ['Experiment', 'Tests', 'Mean Score', 'Success Rate', 'Max Score']

            for exp_name, stats in summary_stats.items():
                table_data.append([
                    exp_name.replace("_", " ").title(),
                    stats.get("total_tests", 0),
                    f"{stats.get('mean_score', 0):.2f}",
                    f"{stats.get('accuracy_over_70_percent', 0):.1f}%",
                    stats.get("max_score", 0)
                ])

            if table_data:
                table = ax4.table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                ax4.set_title('Summary Statistics', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_analysis_report(self, all_results: dict[str, Any]) -> str:
        """Generate a text analysis report."""
        report = []
        report.append("# Phase 4 Keyword-to-Utterance Generation: Analysis Report")
        report.append("=" * 60)
        report.append("")

        # Summary statistics
        if "summary" in all_results and "summary_statistics" in all_results["summary"]:
            summary_stats = all_results["summary"]["summary_statistics"]

            report.append("## Summary Statistics")
            report.append("")

            for exp_name, stats in summary_stats.items():
                report.append(f"### {exp_name.replace('_', ' ').title()}")
                report.append(f"- Total Tests: {stats.get('total_tests', 0)}")
                report.append(f"- Mean Score: {stats.get('mean_score', 0):.2f}/10")
                report.append(f"- Success Rate (≥7/10): {stats.get('accuracy_over_70_percent', 0):.1f}%")
                report.append(f"- Max Score: {stats.get('max_score', 0)}")
                report.append("")

        # Key findings
        report.append("## Key Findings")
        report.append("")

        # Analyze contextual improvements
        if "contextual" in all_results and "improvements_by_level" in all_results["contextual"]:
            improvements = all_results["contextual"]["improvements_by_level"]
            if improvements:
                best_level = max(improvements.keys(), key=lambda k: improvements[k]["mean_improvement"])
                best_improvement = improvements[best_level]["mean_improvement"]

                report.append("### Context Impact")
                report.append(f"- Best performing context level: {best_level.replace('_', ' ').title()}")
                report.append(f"- Mean improvement: {best_improvement:.2f} points")
                report.append("")

        # Single keyword effectiveness
        if "single_keyword" in all_results and "effectiveness_by_keyword" in all_results["single_keyword"]:
            keyword_stats = all_results["single_keyword"]["effectiveness_by_keyword"]
            if keyword_stats:
                best_keyword = max(keyword_stats.keys(), key=lambda k: keyword_stats[k]["mean_score"])
                best_score = keyword_stats[best_keyword]["mean_score"]

                report.append("### Single Keyword Effectiveness")
                report.append(f"- Most effective keyword: '{best_keyword}'")
                report.append(f"- Mean score: {best_score:.2f}/10")
                report.append("")

        # Save report
        report_text = "\n".join(report)
        report_path = self.results_dir / "analysis_report.md"

        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text

    def run_full_analysis(self):
        """Run complete analysis and generate all visualizations."""
        print("📊 Running Phase 4 results analysis...")

        # Load all results
        all_results = self.load_all_results()

        if not all_results:
            print("❌ No results found to analyze")
            return

        print(f"✅ Loaded results for: {list(all_results.keys())}")

        # Generate individual part analyses
        if "baseline" in all_results:
            scores = all_results["baseline"].get("scores", [])
            if scores:
                self.create_score_distribution_plot(scores, "Baseline Score Distribution", "baseline_scores")
                print("✅ Created baseline score distribution")

        if "contextual" in all_results:
            self.create_context_comparison_plot(all_results["contextual"])
            print("✅ Created context level comparisons")

        if "single_keyword" in all_results:
            self.create_single_keyword_analysis(all_results["single_keyword"])
            print("✅ Created single keyword analysis")

        # Generate summary dashboard
        self.create_summary_dashboard(all_results)
        print("✅ Created summary dashboard")

        # Generate analysis report
        report = self.generate_analysis_report(all_results)
        print("✅ Generated analysis report")

        print(f"\n📁 All analysis files saved to: {self.results_dir}")
        print(f"📈 Figures saved to: {self.figures_dir}")

        return all_results, report
