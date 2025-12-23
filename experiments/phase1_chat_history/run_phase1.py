#!/usr/bin/env python3
"""
Phase 1: Chat History Evaluation

Runs chat history-based text completion evaluation using the reorganized framework.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized utilities
from lib.utils import load_env, setup_logging, create_output_directory  # noqa: E402, I001
from experiments.phase1_chat_history.evaluation.chat_evaluator import Phase1ChatEvaluator  # noqa: E402, I001


def _generate_plots(results_df, output_dir: Path) -> None:
    """Generate simple summary plots for Phase 1."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir.mkdir(parents=True, exist_ok=True)

        # Histogram of judge scores
        plt.figure(figsize=(6, 4))
        sns.histplot(results_df["llm_judge_score"], bins=10, kde=False)
        plt.title("LLM Judge Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "judge_score_distribution.png", dpi=200)
        plt.close()

        # Boxplot of embedding similarity by generation method
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=results_df, x="generation_method", y="embedding_similarity")
        plt.title("Embedding Similarity by Generation Method")
        plt.xlabel("Generation Method")
        plt.ylabel("Embedding Similarity")
        plt.tight_layout()
        plt.savefig(output_dir / "embedding_similarity_by_method.png", dpi=200)
        plt.close()
    except Exception:
        # Plots are optional; ignore failures
        return


def main():
    """Main entry point for Phase 1 evaluation."""
    # Load environment variables
    load_env()

    # Setup logging
    logger = setup_logging("phase1_main")

    parser = argparse.ArgumentParser(description="Run Phase 1 Chat History Evaluation")

    # Data arguments
    parser.add_argument("--data", type=str, default=None, help="Path to chat history JSON file")
    parser.add_argument("--sample-size", type=int, help="Number of test examples to evaluate")

    # Config arguments
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "config.yaml"),
        help="Path to configuration file",
    )

    # Output arguments
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Create evaluator
    print("[start] Phase 1: Chat History Evaluation")
    print(f"Config: {args.config}")
    print(f"Data: {args.data or 'from config'}")
    print(f"Sample size: {args.sample_size or 'from config'}")
    print()

    try:
        # Initialize Phase 1 evaluator
        evaluator = Phase1ChatEvaluator(config_path=args.config)

        # Run evaluation
        results = evaluator.run_evaluation(chat_data_path=args.data, sample_size=args.sample_size)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = create_output_directory("results")
            output_path = output_dir / "results.json"

        evaluator.save_results(results, str(output_path))
        print(f"[ok] Results saved to: {output_path}")
        # Plots
        if isinstance(results, object) and hasattr(results, "to_csv"):
            plot_dir = output_path.parent if output_path.suffix else output_path
            _generate_plots(results, Path(plot_dir) / "figures")

        # Print summary
        print("\nSummary:")
        if hasattr(results, "describe"):
            print(f"  Total evaluations: {len(results)}")
            if "partial_method" in results.columns and "generation_method" in results.columns:
                print(
                    f"  Methods tested: {len(results[['partial_method', 'generation_method']].drop_duplicates())}"
                )
        elif hasattr(results, "__len__"):
            print(f"  Total evaluations: {len(results)}")

        print("\n[done] Phase 1 evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Phase 1 evaluation failed: {e}")
        print(f"[error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
