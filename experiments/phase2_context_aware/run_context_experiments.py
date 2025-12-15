#!/usr/bin/env python3
"""
Phase 2: Context-Aware Experiments

Runs context injection experiments using the shared framework.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.context import ContextBuilder, ProfileManager, PromptBuilder
from lib.evaluation import BaseEvaluator
from lib.llm_clients import create_llm_client
from lib.utils import ExperimentConfig, setup_logging, load_env
from lib.data import DataLoader


class ContextAwareEvaluator(BaseEvaluator):
    """
    Evaluator for context-aware experiments (Phase 2).
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize evaluator.

        Args:
            config: Experiment configuration
        """
        config_dict = config.to_dict()
        load_env()

        # Initialize LLM client
        self.llm_client = create_llm_client(provider=config.provider, model=config.model_name)

        # Initialize base evaluator
        super().__init__(self.llm_client, config_dict)

        # Initialize components
        self.config = config_dict
        self.experiment_config = config
        self.data_loader = DataLoader()
        self.profile_manager = ProfileManager()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder(template_style="aac")

        # Setup logging
        self.logger = setup_logging(
            name="phase2_evaluation", level="INFO" if config.verbose else "WARNING"
        )

    def run_evaluation(self, *args, **kwargs) -> pd.DataFrame:
        """Placeholder to satisfy BaseEvaluator interface (not used in Phase 2)."""
        return pd.DataFrame()

    def run_hypothesis_test(self, profile_path: str, transcript_path: str) -> pd.DataFrame:
        """
        Run the H1-H5 hypothesis test from original experiment.

        Args:
            profile_path: Path to user profile
            transcript_path: Path to transcript scenarios

        Returns:
            Results DataFrame
        """
        self.log_evaluation_start()

        # Load profile and transcripts
        profile = self.profile_manager.load_profile(profile_path)
        self.context_builder.profile = profile
        transcripts = self.data_loader.load_transcript(transcript_path)

        self.logger.info(f"Loaded profile for {profile.name}")
        self.logger.info(f"Loaded {len(transcripts)} test scenarios")

        results = []

        for idx, scenario in enumerate(transcripts):
            self.logger.info(f"Processing scenario {idx + 1}/{len(transcripts)}")

            # Extract scenario data
            target = scenario.get("target", scenario.get("expected", ""))
            interlocutor = scenario.get("interlocutor", "Unknown")
            location = scenario.get("location", {})
            current_time = self._parse_time(scenario.get("time", ""))
            speech_input = scenario.get("speech", scenario.get("input", ""))

            # Test each hypothesis
            hypotheses_results = self._test_hypotheses(
                speech_input, target, interlocutor, location, current_time
            )

            # Add scenario metadata
            for result in hypotheses_results:
                result.update(
                    {
                        "scenario_id": idx + 1,
                        "interlocutor": interlocutor,
                        "location": str(location),
                        "time": str(current_time),
                        "speech_input": speech_input,
                        "target": target,
                    }
                )

            results.extend(hypotheses_results)

        results_df = pd.DataFrame(results)
        self.log_evaluation_end()

        return results_df

    def _test_hypotheses(
        self,
        speech_input: str,
        target: str,
        interlocutor: str,
        location: dict[str, Any],
        current_time: datetime,
    ) -> list[dict[str, Any]]:
        """Test all 5 hypotheses (H1-H5)."""
        results = []

        # H1: Time only
        h1_context = self.context_builder.build_context(
            current_time=current_time, context_levels=["time"]
        )
        h1_prediction = self._generate_prediction(speech_input, h1_context)
        h1_score = self.llm_client.judge_similarity(target, h1_prediction)

        results.append(
            {
                "hypothesis": "H1",
                "context_type": "time_only",
                "prediction": h1_prediction,
                "score": h1_score,
            }
        )

        # H2: Time + Social
        h2_context = self.context_builder.build_context(
            current_time=current_time, interlocutor=interlocutor, context_levels=["time", "social"]
        )
        h2_prediction = self._generate_prediction(speech_input, h2_context)
        h2_score = self.llm_client.judge_similarity(target, h2_prediction)

        results.append(
            {
                "hypothesis": "H2",
                "context_type": "time_social",
                "prediction": h2_prediction,
                "score": h2_score,
            }
        )

        # H3: Time + Social + Location
        h3_context = self.context_builder.build_context(
            current_time=current_time,
            interlocutor=interlocutor,
            location=location,
            context_levels=["time", "social", "location"],
        )
        h3_prediction = self._generate_prediction(speech_input, h3_context)
        h3_score = self.llm_client.judge_similarity(target, h3_prediction)

        results.append(
            {
                "hypothesis": "H3",
                "context_type": "time_social_location",
                "prediction": h3_prediction,
                "score": h3_score,
            }
        )

        # H4: Speech + Profile (Original H5)
        h4_context = self.context_builder.build_context(
            conversation_history=[speech_input], context_levels=["profile"]
        )
        h4_prediction = self._generate_prediction(speech_input, h4_context)
        h4_score = self.llm_client.judge_similarity(target, h4_prediction)

        results.append(
            {
                "hypothesis": "H4",
                "context_type": "speech_profile",
                "prediction": h4_prediction,
                "score": h4_score,
            }
        )

        # H5: Full Context (Original H4)
        h5_context = self.context_builder.build_context(
            current_time=current_time,
            interlocutor=interlocutor,
            location=location,
            conversation_history=[speech_input],
            context_levels=["profile", "time", "social", "location"],
        )
        h5_prediction = self._generate_prediction(speech_input, h5_context)
        h5_score = self.llm_client.judge_similarity(target, h5_prediction)

        results.append(
            {
                "hypothesis": "H5",
                "context_type": "full_context",
                "prediction": h5_prediction,
                "score": h5_score,
            }
        )

        return results

    def run_ablation_test(self, profile_path: str, vague_transcript_path: str) -> pd.DataFrame:
        """
        Run profile ablation test (Smart vs Raw).

        Args:
            profile_path: Path to user profile
            vague_transcript_path: Path to vague input scenarios

        Returns:
            Results DataFrame
        """
        self.logger.info("Running profile ablation test")

        # Load data
        profile = self.profile_manager.load_profile(profile_path)
        vague_scenarios = self.data_loader.load_transcript(vague_transcript_path)

        results = []

        for idx, scenario in enumerate(vague_scenarios):
            input_speech = scenario.get("input", "")
            target = scenario.get("target", "")
            interlocutor = scenario.get("interlocutor", "Unknown")

            # Smart model (with profile)
            self.context_builder.profile = profile
            smart_context = self.context_builder.build_context(
                interlocutor=interlocutor, context_levels=["profile", "social"]
            )
            smart_prediction = self._generate_prediction(input_speech, smart_context)

            # Raw model (no profile)
            self.context_builder.profile = None
            raw_context = self.context_builder.build_context(
                interlocutor=interlocutor, context_levels=[]
            )
            raw_prediction = self._generate_prediction(input_speech, raw_context)

            # Score both
            smart_score = self.llm_client.judge_similarity(target, smart_prediction)
            raw_score = self.llm_client.judge_similarity(target, raw_prediction)

            results.append(
                {
                    "scenario_id": idx + 1,
                    "input": input_speech,
                    "target": target,
                    "interlocutor": interlocutor,
                    "smart_prediction": smart_prediction,
                    "smart_score": smart_score,
                    "raw_prediction": raw_prediction,
                    "raw_score": raw_score,
                    "improvement": smart_score - raw_score,
                }
            )

        return pd.DataFrame(results)

    def run_temporal_disambiguation_test(self, profile_path: str) -> pd.DataFrame:
        """
        Run temporal disambiguation test.

        Args:
            profile_path: Path to user profile

        Returns:
            Results DataFrame
        """
        self.logger.info("Running temporal disambiguation test")

        profile = self.profile_manager.load_profile(profile_path)
        self.context_builder.profile = profile

        # Test case: "Do you want the usual?" at different times
        test_cases = [
            {"time": "08:00", "expected": "Meds please"},
            {"time": "20:00", "expected": "Mask on"},
            {"time": "12:00", "expected": "Food please"},
            {"time": "22:00", "expected": "Help me turn"},
        ]

        results = []
        input_phrase = "Do you want the usual?"

        for case in test_cases:
            time_str = case["time"]
            expected = case["expected"]

            # Parse time
            current_time = datetime.strptime(f"2024-01-01 {time_str}", "%Y-%m-%d %H:%M")

            # Build context with time
            context = self.context_builder.build_context(
                current_time=current_time, context_levels=["profile", "time"]
            )

            # Generate prediction
            prediction = self._generate_prediction(input_phrase, context)
            score = self.llm_client.judge_similarity(expected, prediction)

            results.append(
                {
                    "time": time_str,
                    "input": input_phrase,
                    "expected": expected,
                    "prediction": prediction,
                    "score": score,
                }
            )

        return pd.DataFrame(results)

    def _generate_prediction(self, input_text: str, context: dict[str, Any]) -> str:
        """Generate prediction using LLM with context."""
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            partial_input=input_text, context=context, task="completion"
        )

        return self.llm_client.generate(
            user_prompt, system_prompt, temperature=self.experiment_config.temperature
        )

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        if not time_str:
            return datetime.now()

        try:
            return datetime.strptime(f"2024-01-01 {time_str}", "%Y-%m-%d %H:%M")
        except:
            return datetime.now()


def main():
    """Main entry point for Phase 2 experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 Context-Aware Experiments")

    parser.add_argument(
        "--profile",
        type=str,
        default="../../data/synthetic/profiles/dave_context.json",
        help="Path to user profile",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        default="../../data/synthetic/transcripts/transcript_data_2.json",
        help="Path to transcript scenarios",
    )
    parser.add_argument(
        "--vague",
        type=str,
        default="../../data/synthetic/transcripts/transcript_vague.json",
        help="Path to vague input scenarios",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["hypothesis", "ablation", "temporal", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output", type=str, default="../../data/synthetic/outputs", help="Output directory"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="LLM model to use"
    )
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "gemini"], help="LLM provider"
    )

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        model_name=args.model, provider=args.provider, output_dir=args.output, verbose=True
    )

    # Create evaluator
    evaluator = ContextAwareEvaluator(config)

    # Create output directory
    output_dir = evaluator.create_output_directory(args.output)

    # Run experiments
    all_results = {}

    if args.experiment in ["hypothesis", "all"]:
        print("\n=== Running Hypothesis Test (H1-H5) ===")
        hypothesis_results = evaluator.run_hypothesis_test(args.profile, args.transcripts)
        all_results["hypothesis"] = hypothesis_results
        evaluator.save_results(hypothesis_results, output_dir, prefix="hypothesis_test")
        print(f"Completed: {len(hypothesis_results)} evaluations")

    if args.experiment in ["ablation", "all"]:
        print("\n=== Running Profile Ablation Test ===")
        ablation_results = evaluator.run_ablation_test(args.profile, args.vague)
        all_results["ablation"] = ablation_results
        evaluator.save_results(ablation_results, output_dir, prefix="ablation_test")
        print(f"Completed: {len(ablation_results)} comparisons")

    if args.experiment in ["temporal", "all"]:
        print("\n=== Running Temporal Disambiguation Test ===")
        temporal_results = evaluator.run_temporal_disambiguation_test(args.profile)
        all_results["temporal"] = temporal_results
        evaluator.save_results(temporal_results, output_dir, prefix="temporal_test")
        print(f"Completed: {len(temporal_results)} time tests")

    # Generate visualizations
    for name, results_df in all_results.items():
        evaluator.visualize_results(results_df, output_dir)

    print(f"\nAll experiments completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
