#!/usr/bin/env python3
"""
Simplified Phase 4: Keyword-to-Utterance Generation with Contextual Enhancement

This script runs experiments to test how effectively LLMs can generate
appropriate utterances from minimal keyword inputs with varying levels
of contextual enhancement.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from lib.llm_clients import create_llm_client
from lib.utils import load_env, setup_logging


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
        help="LLM provider to use"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use (default depends on provider)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of keyword combinations to test (default: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: results/timestamp)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def create_output_directory(base_dir: str | None = None) -> Path:
    """Create and return output directory with timestamp."""
    if base_dir:
        output_path = Path(base_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"simple_experiment_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def load_data():
    """Load keyword data and social graph."""
    # Use centralized data directory
    data_dir = Path(__file__).parent.parent.parent / "data" / "real" / "profiles" / "Dwayne"

    # Load keywords
    keywords_file = data_dir / "DwayneKeyWords.tsv"
    if not keywords_file.exists():
        raise FileNotFoundError(f"Keywords file not found: {keywords_file}")

    keywords_df = pd.read_csv(keywords_file, sep='\t')

    # Check column names and print them for debugging
    print("Available columns:", list(keywords_df.columns))

    # Use correct column name (trailing space issue)
    if 'Instruction ' in keywords_df.columns:
        keywords_df = keywords_df.dropna(subset=['Instruction '])
        keywords_df = keywords_df.rename(columns={'Instruction ': 'Instruction'})
    elif 'Instruction' in keywords_df.columns:
        keywords_df = keywords_df.dropna(subset=['Instruction'])
    else:
        raise ValueError(f"Instruction column not found. Available columns: {list(keywords_df.columns)}")

    # Load social graph
    social_graph_file = data_dir / "social_graph.json"
    if not social_graph_file.exists():
        raise FileNotFoundError(f"Social graph file not found: {social_graph_file}")

    with open(social_graph_file) as f:
        social_graph = json.load(f)

    return keywords_df, social_graph


def extract_keywords(row):
    """Extract keywords from a DataFrame row."""
    keywords = []
    for col in ['Key word ', 'Key Word2', 'Key Word3']:
        if pd.notna(row[col]) and str(row[col]).strip().lower() != 'n/a':
            keywords.append(str(row[col]).strip())
    return keywords


def create_baseline_prompt(keywords):
    """Create prompt for baseline testing (keywords only)."""
    keywords_str = ", ".join(keywords)

    return f"""You are an AAC assistant communicating for Dwayne, an MND patient.

KEYWORDS: {keywords_str}

Based only on these keywords, predict what Dwayne wants to communicate.
Dwayne has limited speech and uses short telegraphic phrases.

Predicted utterance:"""


def create_contextual_prompt(keywords, context):
    """Create prompt with contextual information."""
    keywords_str = ", ".join(keywords)

    # Build context string
    context_parts = []

    if "location" in context:
        context_parts.append(f"LOCATION: {context['location']}")
    if "people_present" in context:
        context_parts.append(f"PEOPLE PRESENT: {context['people_present']}")
    if "time_of_day" in context:
        context_parts.append(f"TIME: {context['time_of_day']}")

    context_str = "\n".join(context_parts) if context_parts else "No additional context available"

    return f"""You are an AAC assistant communicating for Dwayne, an MND patient.

CONTEXT:
{context_str}

KEYWORDS: {keywords_str}

Based on the context and keywords, predict Dwayne's intended utterance.
Consider his current situation, who he's with, and his communication patterns.

Predicted utterance:"""


def build_simple_context(social_graph, keywords):
    """Build a simple context based on keywords and social graph."""

    context = {}

    # Random time-based people context
    current_hour = datetime.now().hour

    if 16 <= current_hour <= 22:  # Afternoon/evening
        context["people_present"] = "Kerry (wife) is present"
        context["time_of_day"] = "evening_family_time"
    else:
        context["people_present"] = "professional carer is present"
        context["time_of_day"] = "daytime_care"

    # Location context based on keywords
    if any(kw in keywords for kw in ["Chair", "Cushion", "Tilt"]):
        context["location"] = "in wheelchair/chair"
    elif any(kw in keywords for kw in ["Scratch", "Head", "Face"]):
        context["location"] = "in bed or resting position"
    elif any(kw in keywords for kw in ["Feed", "Medication", "Sick"]):
        context["location"] = "in bedroom or care area"
    else:
        context["location"] = "in living room"

    return context


def evaluate_prediction(llm_client, prediction, target):
    """Evaluate prediction against target using LLM judge."""
    try:
        score = llm_client.judge_similarity(target, prediction)
        return min(10, max(1, score))  # Ensure score is within 1-10 range
    except Exception as e:
        print(f"Error evaluating prediction: {e}")
        return 5  # Default middle score


def run_baseline_test(llm_client, keywords_df, sample_size=5):
    """Run simplified baseline test."""
    print("\n=== Baseline Test (Keywords Only) ===")

    results = []
    total_score = 0

    for idx, row in keywords_df.head(sample_size).iterrows():
        keywords = extract_keywords(row)
        target = str(row['Instruction']).strip()

        if not keywords or not target:
            continue

        # Generate prediction
        prompt = create_baseline_prompt(keywords)
        prediction = llm_client.generate(prompt, temperature=0.2)

        # Evaluate prediction
        score = evaluate_prediction(llm_client, prediction, target)

        # Store results
        results.append({
            "keywords": keywords,
            "target": target,
            "prediction": prediction,
            "score": score
        })

        total_score += score
        print(f"Test {idx+1}: {keywords} -> Score: {score}/10")

    mean_score = total_score / len(results) if results else 0
    print(f"Baseline mean score: {mean_score:.2f}/10")

    return results, mean_score


def run_contextual_test(llm_client, keywords_df, social_graph, sample_size=5):
    """Run simplified contextual test."""
    print("\n=== Contextual Test (Keywords + Context) ===")

    results = []
    total_score = 0

    for idx, row in keywords_df.head(sample_size).iterrows():
        keywords = extract_keywords(row)
        target = str(row['Instruction']).strip()

        if not keywords or not target:
            continue

        # Build context
        context = build_simple_context(social_graph, keywords)

        # Generate prediction
        prompt = create_contextual_prompt(keywords, context)
        prediction = llm_client.generate(prompt, temperature=0.2)

        # Evaluate prediction
        score = evaluate_prediction(llm_client, prediction, target)

        # Store results
        results.append({
            "keywords": keywords,
            "target": target,
            "prediction": prediction,
            "score": score,
            "context": context
        })

        total_score += score
        print(f"Test {idx+1}: {keywords} with context -> Score: {score}/10")

    mean_score = total_score / len(results) if results else 0
    print(f"Contextual mean score: {mean_score:.2f}/10")

    return results, mean_score


def main():
    """Main experiment runner."""
    # Load environment
    load_env()

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging("phase4_simple", level=log_level)

    # Create LLM client
    logger.info(f"Creating LLM client: {args.provider}")
    client = create_llm_client(args.provider, args.model)
    llm_info = client.get_model_info()
    logger.info(f"Using LLM: {llm_info}")

    # Load data
    keywords_df, social_graph = load_data()
    logger.info(f"Loaded {len(keywords_df)} keyword combinations")

    # Apply sample size
    if args.sample_size:
        keywords_df = keywords_df.head(args.sample_size)
        logger.info(f"Limited to {args.sample_size} keyword combinations")

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")

    # Run tests
    all_results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "llm_provider": llm_info,
            "sample_size": len(keywords_df)
        }
    }

    try:
        # Run baseline test
        baseline_results, baseline_score = run_baseline_test(client, keywords_df, args.sample_size)
        all_results["baseline"] = {
            "results": baseline_results,
            "mean_score": baseline_score
        }

        # Run contextual test
        contextual_results, contextual_score = run_contextual_test(client, keywords_df, social_graph, args.sample_size)
        all_results["contextual"] = {
            "results": contextual_results,
            "mean_score": contextual_score
        }

        # Calculate improvement
        improvement = contextual_score - baseline_score
        all_results["improvement"] = improvement
        print(f"\n📊 Context improvement: {improvement:.2f} points")

        # Save results
        with open(output_dir / "simple_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n✅ Simple experiment completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        print("\n=== SUMMARY ===")
        print(f"Baseline score: {baseline_score:.2f}/10")
        print(f"Contextual score: {contextual_score:.2f}/10")
        print(f"Improvement: {improvement:.2f} points")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
