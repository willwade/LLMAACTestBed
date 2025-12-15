#!/usr/bin/env python3
"""
Main script to run chat history-driven LLM evaluation.

This script provides a command-line interface for evaluating different text completion
methods for AAC users using real chat history data.

Usage:
    python run_evaluation.py --data path/to/chat_data.json [--sample-size 100] [--output results.csv]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from evaluation.chat_history_evaluator import ChatHistoryEvaluator


def create_default_methods():
    """Create default method dictionaries for partial generation, retrieval, and metrics."""
    # Partial utterance builders: how we truncate or sample the ground truth to simulate user input.
    partial_methods = {
        "prefix_3": lambda text: ChatHistoryEvaluator.create_prefix_partial(text, 3),
        "prefix_2": lambda text: ChatHistoryEvaluator.create_prefix_partial(text, 2),
        "keyword_2": lambda text: ChatHistoryEvaluator.create_keyword_partial(text, 2),
        "random": lambda text: ChatHistoryEvaluator.create_random_partial(text),
    }

    # Completion generators: which retrieval strategy (or none) to supply to the LLM.
    generation_methods = {
        "lexical": ChatHistoryEvaluator.generate_with_lexical_retrieval,
        "tfidf": ChatHistoryEvaluator.generate_with_tfidf_retrieval,
        "embedding": ChatHistoryEvaluator.generate_with_embedding_retrieval,
        "context_only": ChatHistoryEvaluator.generate_with_context_only,
    }

    # Metrics that require evaluator state (embedding model or LLM judge).
    def embedding_similarity_wrapper(evaluator, target, proposal):
        return evaluator.calculate_embedding_similarity(target, proposal)

    def judge_similarity_wrapper(evaluator, target, proposal):
        return evaluator.judge_similarity(target, proposal)

    evaluation_metrics = {
        "embedding_similarity": embedding_similarity_wrapper,
        "llm_judge_score": judge_similarity_wrapper,
        "character_accuracy": ChatHistoryEvaluator.calculate_character_accuracy,
        "word_accuracy": ChatHistoryEvaluator.calculate_word_accuracy,
    }

    return partial_methods, generation_methods, evaluation_metrics


def main():
    """Parse CLI args, configure evaluator, run evaluations, and report/save results."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-based text completion for AAC using chat history"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to chat history JSON file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM (if not set as environment variable)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of test examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--corpus-ratio",
        type=float,
        default=0.67,
        help="Fraction of data to use as corpus (default: 0.67)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Model name to use for generation and evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (CSV format)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of results",
    )
    parser.add_argument(
        "--context-filter",
        type=str,
        default="none",
        choices=["none", "time", "geo", "time_geo"],
        help="Filter corpus retrieval by temporal and/or geographic proximity",
    )
    parser.add_argument(
        "--time-window-hours",
        type=float,
        default=3.0,
        help="Hours window for temporal filtering when context-filter uses time",
    )
    parser.add_argument(
        "--geo-radius-km",
        type=float,
        default=10.0,
        help="Geographic radius (km) for filtering when context-filter uses geo",
    )
    parser.add_argument(
        "--conversation-window",
        type=int,
        default=0,
        help="Number of previous test utterances to include as local conversation context",
    )

    args = parser.parse_args()

    # Resolve data path (allow relative to CWD or script directory)
    data_path = Path(args.data)
    if not data_path.exists():
        alt_path = Path(__file__).resolve().parent / args.data
        if alt_path.exists():
            data_path = alt_path
        else:
            print(f"Error: Data file not found: {args.data}")
            sys.exit(1)

    # Create output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_results_{timestamp}.csv"

    print("Chat History-Driven LLM Evaluation")
    print("=" * 50)
    print(f"Data file: {data_path}")
    print(f"Sample size: {args.sample_size or 'all'}")
    print(f"Corpus ratio: {args.corpus_ratio}")
    print(f"Model: {args.model}")
    print(
        f"Context filter: {args.context_filter} (time_window_hours={args.time_window_hours}, geo_radius_km={args.geo_radius_km})"
    )
    print(f"Conversation window: {args.conversation_window}")
    print(f"Output: {args.output}")
    print("=" * 50)

    # Initialize evaluator
    try:
        evaluator = ChatHistoryEvaluator(
            chat_data_path=str(data_path),
            api_key=args.api_key,
            corpus_ratio=args.corpus_ratio,
            model_name=args.model,
        )
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        sys.exit(1)

    # Get default methods
    partial_methods, generation_methods, evaluation_metrics = create_default_methods()

    # Run evaluation
    try:
        print("\nRunning evaluation...")
        results_df = evaluator.run_evaluation(
            partial_methods=partial_methods,
            generation_methods=generation_methods,
            evaluation_metrics=evaluation_metrics,
            sample_size=args.sample_size,
            context_filter=args.context_filter,
            time_window_hours=args.time_window_hours,
            geo_radius_km=args.geo_radius_km,
            conversation_window=args.conversation_window,
        )

        if results_df.empty:
            print("No results generated. Please check your data and parameters.")
            sys.exit(1)

        print(f"\nEvaluation complete. Generated {len(results_df)} results.")

        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")

        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating visualizations...")
            evaluator.visualize_results(results_df)

            # Create additional analysis plots
            analyze_results(results_df, args.output.replace(".csv", "_analysis.png"))

        # Print summary statistics
        print_summary(results_df)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)


def analyze_results(results_df, output_path):
    """Create additional analysis plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Group by methods and calculate mean scores
    grouped_results = (
        results_df.groupby(["partial_method", "generation_method"])
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
                "character_accuracy": "mean",
                "word_accuracy": "mean",
                "target": "count",  # Count of samples
            }
        )
        .rename(columns={"target": "count"})
        .reset_index()
    )

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Performance Analysis", fontsize=16)

    # 1. Performance by partial method
    partial_perf = (
        results_df.groupby("partial_method")
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
            }
        )
        .reset_index()
        .melt(id_vars="partial_method", var_name="metric", value_name="score")
    )
    sns.barplot(data=partial_perf, x="partial_method", y="score", hue="metric", ax=axes[0, 0])
    axes[0, 0].set_title("Performance by Partial Method")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend(title="Metric")

    # 2. Performance by generation method
    gen_perf = (
        results_df.groupby("generation_method")
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
            }
        )
        .reset_index()
        .melt(id_vars="generation_method", var_name="metric", value_name="score")
    )
    sns.barplot(data=gen_perf, x="generation_method", y="score", hue="metric", ax=axes[0, 1])
    axes[0, 1].set_title("Performance by Generation Method")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend(title="Metric")

    # 3. Best combinations heatmap
    pivot = grouped_results.pivot(
        index="partial_method",
        columns="generation_method",
        values="embedding_similarity",
    )
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[1, 0])
    axes[1, 0].set_title("Embedding Similarity by Method Combination")

    # 4. Score distribution
    score_data = results_df.melt(
        id_vars=["partial_method", "generation_method"],
        value_vars=["embedding_similarity", "llm_judge_score"],
        var_name="metric",
        value_name="score",
    )
    sns.boxplot(data=score_data, x="metric", y="score", ax=axes[1, 1])
    axes[1, 1].set_title("Score Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Analysis plots saved to: {output_path}")


def print_summary(results_df):
    """Print summary statistics of evaluation results."""
    print("\nSummary Statistics:")
    print("-" * 50)

    # Overall statistics
    print(f"Total evaluations: {len(results_df)}")

    # Best and worst performers for each metric
    metrics = [
        "embedding_similarity",
        "llm_judge_score",
        "character_accuracy",
        "word_accuracy",
    ]
    for metric in metrics:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            worst_idx = results_df[metric].idxmin()

            best_row = results_df.loc[best_idx]
            worst_row = results_df.loc[worst_idx]

            print(f"\n{metric.replace('_', ' ').title()}:")
            print(
                f"  Best: {best_row['partial_method']} + {best_row['generation_method']} ({best_row[metric]:.3f})"
            )
            print(
                f"  Worst: {worst_row['partial_method']} + {worst_row['generation_method']} ({worst_row[metric]:.3f})"
            )

    # Method averages
    print("\nMethod Averages:")
    partial_avg = (
        results_df.groupby("partial_method")
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
                "character_accuracy": "mean",
                "word_accuracy": "mean",
            }
        )
        .round(3)
    )
    print(partial_avg)

    gen_avg = (
        results_df.groupby("generation_method")
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
                "character_accuracy": "mean",
                "word_accuracy": "mean",
            }
        )
        .round(3)
    )
    print(gen_avg)


if __name__ == "__main__":
    main()
