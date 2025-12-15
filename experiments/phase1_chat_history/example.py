#!/usr/bin/env python3
"""
Example script demonstrating how to use the Chat History Evaluation Framework.

This script shows how to:
1. Initialize the evaluator with chat data
2. Define custom methods for partial utterance generation
3. Run evaluation with different generation methods
4. Analyze and visualize results
"""

import os
import sys

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from lib.evaluation.chat_history_evaluator import ChatHistoryEvaluator


def main():
    """Main function demonstrating framework usage."""
    # Initialize evaluator with chat data
    chat_data_path = "data/test/baton-export-2025-11-24-nofullstop.json"

    print("Initializing Chat History Evaluator...")
    evaluator = ChatHistoryEvaluator(chat_data_path=chat_data_path, corpus_ratio=0.67)

    # Define partial utterance methods
    partial_methods = {
        "prefix_3": lambda text: evaluator.create_prefix_partial(text, 3),
        "prefix_2": lambda text: evaluator.create_prefix_partial(text, 2),
        "keyword_2": lambda text: evaluator.create_keyword_partial(text, 2),
    }

    # Define generation methods
    generation_methods = {
        "lexical": evaluator.generate_with_lexical_retrieval,
        "tfidf": evaluator.generate_with_tfidf_retrieval,
        "embedding": evaluator.generate_with_embedding_retrieval,
    }

    # Define evaluation metrics
    evaluation_metrics = {
        "embedding_similarity": evaluator.calculate_embedding_similarity,
        "llm_judge_score": evaluator.judge_similarity,
    }

    # Run evaluation on a small sample
    print("Running evaluation...")
    results_df = evaluator.run_evaluation(
        partial_methods=partial_methods,
        generation_methods=generation_methods,
        evaluation_metrics=evaluation_metrics,
        sample_size=5,  # Small sample for demonstration
    )

    if results_df.empty:
        print("No results generated. Please check your data and parameters.")
        return

    # Save results
    output_path = "example_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Display sample results
    print("\nSample Results:")
    display_cols = [
        "target",
        "partial",
        "proposal",
        "partial_method",
        "generation_method",
        "embedding_similarity",
        "llm_judge_score",
    ]
    print(results_df[display_cols])

    # Visualize results
    print("\nGenerating visualizations...")
    evaluator.visualize_results(results_df)

    # Additional analysis
    print("\nAnalyzing results...")

    # Group by methods and calculate mean scores
    grouped_results = (
        results_df.groupby(["partial_method", "generation_method"])
        .agg(
            {
                "embedding_similarity": "mean",
                "llm_judge_score": "mean",
                "target": "count",  # Count of samples
            }
        )
        .rename(columns={"target": "count"})
        .reset_index()
    )

    print("\nMethod Performance:")
    print(grouped_results)

    # Find best performing combination
    best_idx = results_df["embedding_similarity"].idxmax()
    best_result = results_df.loc[best_idx]

    print("\nBest Performing Combination:")
    print(f"Method: {best_result['partial_method']} + {best_result['generation_method']}")
    print(f"Target: '{best_result['target']}'")
    print(f"Partial: '{best_result['partial']}'")
    print(f"Proposal: '{best_result['proposal']}'")
    print(f"Embedding Similarity: {best_result['embedding_similarity']:.3f}")
    print(f"LLM Judge Score: {best_result['llm_judge_score']:.1f}")

    # Create custom visualization
    plt.figure(figsize=(10, 5))

    # Bar chart of method performance
    method_perf = grouped_results.groupby("partial_method")["embedding_similarity"].mean()
    method_perf.plot(kind="bar")
    plt.title("Performance by Partial Utterance Method")
    plt.ylabel("Average Embedding Similarity")
    plt.xlabel("Partial Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=150, bbox_inches="tight")

    print("\nVisualization saved to: method_comparison.png")


if __name__ == "__main__":
    main()
