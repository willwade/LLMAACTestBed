#!/usr/bin/env python3
"""
Phase 4: Keyword-to-Utterance Generation with Contextual Enhancement

This script runs experiments to test how effectively LLMs can generate
appropriate utterances from minimal keyword inputs with varying levels
of social and environmental context.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.llm_clients import create_llm_client
from lib.utils import load_env, setup_logging

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

# Import directly to avoid lib path issues
import pandas as pd
from evaluation.keyword_evaluator import KeywordEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 4 keyword-to-utterance generation experiments"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider to use",
    )

    parser.add_argument(
        "--model", type=str, help="Specific model to use (default depends on provider)"
    )

    parser.add_argument(
        "--part",
        type=int,
        choices=[1, 2, 3],
        help="Run specific part (1=baseline, 2=contextual, 3=single_keyword)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit number of keyword combinations to test (for debugging)",
    )

    parser.add_argument(
        "--output-dir", type=str, help="Output directory for results (default: results/timestamp)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def create_output_directory(base_dir: str | None = None) -> Path:
    """Create and return output directory with timestamp."""
    if base_dir:
        output_path = Path(base_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"experiment_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def load_experiment_data():
    """Load keyword data and social graph with fallback to synthetic samples."""
    root = Path(__file__).parent.parent.parent
    real_dir = root / "data" / "real" / "profiles" / "Dwayne"
    synth_dir = root / "data" / "synthetic" / "phase4"

    keyword_candidates = [
        real_dir / "DwayneKeyWords.tsv",
        synth_dir / "DwayneKeyWords.tsv",
    ]
    social_candidates = [
        real_dir / "social_graph.json",
        synth_dir / "social_graph.json",
    ]

    keywords_file = next((p for p in keyword_candidates if p.exists()), None)
    social_graph_file = next((p for p in social_candidates if p.exists()), None)

    if not keywords_file:
        raise FileNotFoundError(f"Keywords file not found in {keyword_candidates}")
    if not social_graph_file:
        raise FileNotFoundError(f"Social graph file not found in {social_candidates}")

    return keywords_file, social_graph_file


def run_part1_baseline(evaluator, keywords_df, output_dir, sample_size=None):
    """Run Part 1: Baseline keyword testing without context."""
    print("\n=== Part 1: Baseline Testing (Keywords Only) ===")

    results = evaluator.run_baseline_test(keywords_df=keywords_df, sample_size=sample_size)

    # Save results
    output_file = output_dir / "part1_baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline results saved to: {output_file}")
    return results


def run_part2_contextual(evaluator, keywords_df, social_graph, output_dir, sample_size=None):
    """Run Part 2: Contextual enhancement testing."""
    print("\n=== Part 2: Contextual Enhancement Testing ===")

    results = evaluator.run_contextual_test(
        keywords_df=keywords_df, social_graph=social_graph, sample_size=sample_size
    )

    # Save results
    output_file = output_dir / "part2_contextual_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Contextual results saved to: {output_file}")
    return results


def run_part3_single_keyword(evaluator, keywords_df, social_graph, output_dir, sample_size=None):
    """Run Part 3: Single keyword testing with optimal context."""
    print("\n=== Part 3: Single Keyword Testing ===")

    results = evaluator.run_single_keyword_test(
        keywords_df=keywords_df, social_graph=social_graph, sample_size=sample_size
    )

    # Save results
    output_file = output_dir / "part3_single_keyword_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Single keyword results saved to: {output_file}")
    return results


def generate_summary_report(
    all_results: dict[str, Any], output_dir: Path, llm_info: dict[str, str]
) -> None:
    """Generate summary report with all results."""
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "llm_provider": llm_info,
            "experiment_parts": list(all_results.keys()),
        },
        "summary_statistics": {},
        "detailed_results": all_results,
    }

    # Calculate summary statistics
    baseline_mean = None
    for part_name, results in all_results.items():
        if "scores" in results and results["scores"]:
            scores = results["scores"]
            summary["summary_statistics"][part_name] = {
                "total_tests": len(scores),
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "scores_over_7": sum(1 for s in scores if s >= 7),
                "accuracy_over_70_percent": sum(1 for s in scores if s >= 7) / len(scores) * 100,
            }
            if part_name == "part1_baseline":
                baseline_mean = summary["summary_statistics"][part_name]["mean_score"]
        # Handle contextual part with nested scores
        elif part_name == "part2_contextual" and "context_levels" in results:
            level_means = {}
            all_scores = []
            for level_name, level_data in results["context_levels"].items():
                scores = level_data.get("scores", [])
                if scores:
                    mean_score = sum(scores) / len(scores)
                    level_means[level_name] = mean_score
                    all_scores.extend(scores)
            if all_scores:
                summary["summary_statistics"][part_name] = {
                    "total_tests": len(all_scores),
                    "mean_score": sum(all_scores) / len(all_scores),
                    "max_score": max(all_scores),
                    "min_score": min(all_scores),
                    "scores_over_7": sum(1 for s in all_scores if s >= 7),
                    "accuracy_over_70_percent": sum(1 for s in all_scores if s >= 7)
                    / len(all_scores)
                    * 100,
                    "level_means": level_means,
                }
                if baseline_mean is not None:
                    summary["summary_statistics"][part_name]["deltas_vs_baseline"] = {
                        lvl: (mean - baseline_mean) for lvl, mean in level_means.items()
                    }
        # Handle single keyword test
        elif part_name == "part3_single_keyword" and "keyword_results" in results:
            scores = []
            for data in results["keyword_results"].values():
                scores.extend(data.get("scores", []))
            if scores:
                summary["summary_statistics"][part_name] = {
                    "total_tests": len(scores),
                    "mean_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "scores_over_7": sum(1 for s in scores if s >= 7),
                    "accuracy_over_70_percent": sum(1 for s in scores if s >= 7)
                    / len(scores)
                    * 100,
                }
                if baseline_mean is not None:
                    summary["summary_statistics"][part_name]["delta_vs_baseline"] = (
                        summary["summary_statistics"][part_name]["mean_score"] - baseline_mean
                    )

    # Save summary
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[summary] Report saved to: {summary_file}")

    # Print summary to console
    print("\n=== EXPERIMENT SUMMARY ===")
    for part_name, stats in summary["summary_statistics"].items():
        # Ensure stats is a dictionary with expected keys
        if isinstance(stats, dict) and "total_tests" in stats:
            print(f"\n{part_name.upper()}:")
            print(f"  Total tests: {stats['total_tests']}")
            print(f"  Mean score: {stats['mean_score']:.2f}/10")
            print(f"  Accuracy (>7/10): {stats['accuracy_over_70_percent']:.1f}%")
            if "level_means" in stats:
                print("  Context level means:")
                for lvl, mean in stats["level_means"].items():
                    delta = stats.get("deltas_vs_baseline", {}).get(lvl)
                    delta_str = f" (delta vs baseline {delta:+.2f})" if delta is not None else ""
                    print(f"    - {lvl}: {mean:.2f}{delta_str}")
            if "delta_vs_baseline" in stats:
                print(f"  Δ vs baseline: {stats['delta_vs_baseline']:+.2f}")


def main():
    """Main experiment runner."""
    # Load environment
    load_env()

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging("phase4_keyword_experiment", level=log_level)

    # Create LLM client
    logger.info(f"Creating LLM client: {args.provider}")
    client = create_llm_client(args.provider, args.model)
    llm_info = client.get_model_info()
    logger.info(f"Using LLM: {llm_info}")

    # Load experiment data
    keywords_file, social_graph_file = load_experiment_data()

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")

    # Initialize components
    evaluator = KeywordEvaluator(client, logger)

    # Load data
    keywords_df = pd.read_csv(keywords_file, sep="\t")
    # Normalize column names from real TSV (remove trailing spaces)
    keywords_df = keywords_df.rename(columns=lambda c: c.strip())
    # Standardize instruction column
    if "Instruction" not in keywords_df.columns and "Instruction " in keywords_df.columns:
        keywords_df = keywords_df.rename(columns={"Instruction ": "Instruction"})
    # Standardize keyword columns
    if "Key word " in keywords_df.columns and "Key word" not in keywords_df.columns:
        keywords_df = keywords_df.rename(columns={"Key word ": "Key word"})
    if "Key Word2 " in keywords_df.columns and "Key Word2" not in keywords_df.columns:
        keywords_df = keywords_df.rename(columns={"Key Word2 ": "Key Word2"})
    if "Key Word3 " in keywords_df.columns and "Key Word3" not in keywords_df.columns:
        keywords_df = keywords_df.rename(columns={"Key Word3 ": "Key Word3"})

    # Drop rows without instruction or without any keywords
    keywords_df = keywords_df.dropna(subset=["Instruction"])
    keywords_df["Instruction"] = keywords_df["Instruction"].astype(str).str.strip()
    keywords_df = keywords_df[keywords_df["Instruction"] != ""]
    keyword_cols = ["Key word", "Key Word2", "Key Word3"]
    keywords_df = keywords_df.dropna(
        how="all", subset=[c for c in keyword_cols if c in keywords_df.columns]
    )

    logger.info(f"Loaded {len(keywords_df)} keyword combinations")

    with open(social_graph_file) as f:
        social_graph = json.load(f)
    logger.info("Loaded Dwayne's social graph")

    # Apply sample size if specified
    if args.sample_size:
        keywords_df = keywords_df.head(args.sample_size)
        logger.info(f"Limited to {args.sample_size} keyword combinations")

    # Run experiments
    all_results = {}

    try:
        if args.part is None or args.part == 1:
            all_results["part1_baseline"] = run_part1_baseline(
                evaluator, keywords_df, output_dir, args.sample_size
            )

        if args.part is None or args.part == 2:
            all_results["part2_contextual"] = run_part2_contextual(
                evaluator, keywords_df, social_graph, output_dir, args.sample_size
            )

        if args.part is None or args.part == 3:
            all_results["part3_single_keyword"] = run_part3_single_keyword(
                evaluator, keywords_df, social_graph, output_dir, args.sample_size
            )

        # Generate summary report
        generate_summary_report(all_results, output_dir, llm_info)

        print("\n✅ Phase 4 experiment completed successfully!")
        print(f"📁 Results saved in: {output_dir}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
