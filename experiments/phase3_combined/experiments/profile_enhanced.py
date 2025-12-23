"""
Profile-enhanced chat history experiment for Phase 3.

Enhances real chat data with synthetic user profiles to test
if context improves prediction accuracy.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.context import ContextBuilder, UserProfile
from lib.evaluation import MetricsCalculator
from lib.llm_clients import create_llm_client
from lib.utils import load_config, load_env, setup_logging

from lib.data import ChatHistoryLoader, ProfileLoader


class ProfileEnhancedExperiment:
    """Experiment combining real chat history with user profiles."""

    def __init__(
        self,
        config_path: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        max_chats: int | None = 20,
    ):
        """Initialize the profile-enhanced experiment."""
        # Load environment variables
        load_env()

        self.config = load_config(config_path or "configs/profile_enhanced.yaml")
        self.logger = setup_logging(
            name="phase3_profile_enhanced",
            level=self.config.get("logging", {}).get("level", "INFO"),
        )

        # Initialize LLM client
        llm_config = self.config.get("llm", {})
        self.llm_client = create_llm_client(
            provider=llm_provider or llm_config.get("provider"),
            model=llm_model or llm_config.get("model"),
        )
        self.logger.info(f"Using LLM: {self.llm_client.provider_name}:{self.llm_client.model_name}")
        self.chat_loader = ChatHistoryLoader()
        self.profile_loader = ProfileLoader()
        self.metrics_calc = MetricsCalculator()
        self.context_builder = ContextBuilder()

        # Load data
        self.chat_data = self._load_chat_data()
        self.profiles = self._load_profiles()
        self.mappings = self._load_mappings()
        self.max_chats = max_chats

    def _load_chat_data(self) -> list[dict]:
        """Load real chat history data."""
        chat_path = Path("data/real/chat_history/processed/dataset.json")
        if not chat_path.exists():
            self.logger.warning(f"Chat data not found at {chat_path}")
            return []

        try:
            with open(chat_path) as f:
                data = json.load(f)
            chats = data.get("chats") or data.get("conversations") or []
            if isinstance(chats, list):
                return chats
            self.logger.warning(
                "Chat data not in expected list format; skipping profile-enhanced run"
            )
            return []
        except Exception as exc:
            self.logger.warning(f"Failed to load chat data: {exc}")
            return []

    def _load_profiles(self) -> dict[str, dict]:
        """Load user profiles."""
        profiles_path = Path("data/synthetic/profiles/")
        if not profiles_path.exists():
            self.logger.warning(f"Profiles directory not found at {profiles_path}")
            return {}

        profiles = {}
        for profile_file in profiles_path.glob("*.json"):
            profile_dict = self.profile_loader.load(profile_file)
            if not profile_dict:
                continue
            user_id = profile_dict.get("user_id") or profile_file.stem
            profile_dict["user_id"] = user_id
            try:
                profile_obj = UserProfile(profile_dict)
                profile_obj.user_id = user_id  # type: ignore[attr-defined]
            except Exception as exc:
                self.logger.warning(f"Skipping profile {profile_file}: {exc}")
                continue
            profiles[user_id] = profile_obj

        self.logger.info(f"Loaded {len(profiles)} user profiles")
        return profiles

    def _load_mappings(self) -> dict[str, str]:
        """Load chat-to-profile mappings."""
        mappings_path = Path("data/synthetic/chat_profile_mappings/mappings.json")
        if not mappings_path.exists():
            self.logger.warning(f"Chat-profile mappings not found at {mappings_path}")
            return {}

        import json

        with open(mappings_path) as f:
            mappings = json.load(f)

        self.logger.info(f"Loaded {len(mappings)} chat-profile mappings")
        return mappings

    def run_experiment(self) -> dict[str, Any]:
        """Run the profile-enhanced experiment."""
        self.logger.info("Starting Phase 3: Profile-Enhanced Experiment")

        results: dict[str, Any] = {
            "experiment_type": "profile_enhanced",
            "total_chats": len(self.chat_data),
            "enhanced_chats": 0,
            "baseline_scores": [],
            "enhanced_scores": [],
            "improvements": [],
        }

        # Process each chat
        chats_iter = self.chat_data[: self.max_chats] if self.max_chats else self.chat_data

        for chat in chats_iter:
            chat_id = chat.get("chat_id")
            if chat_id is None:
                continue
            user_id = self._get_user_id(chat_id)

            if not user_id or user_id not in self.profiles:
                # Skip chats without profiles
                continue

            profile = self.profiles[user_id]

            # Baseline prediction (chat history only)
            baseline_score = self._evaluate_baseline(chat)
            results["baseline_scores"].append(baseline_score)

            # Enhanced prediction (with profile context)
            enhanced_score = self._evaluate_with_profile(chat, profile)
            results["enhanced_scores"].append(enhanced_score)

            # Calculate improvement
            improvement = enhanced_score - baseline_score
            results["improvements"].append(improvement)
            results["enhanced_chats"] = int(results["enhanced_chats"]) + 1

        # Calculate summary statistics
        if results["improvements"]:
            results["mean_improvement"] = sum(results["improvements"]) / len(
                results["improvements"]
            )
            results["positive_improvements"] = sum(1 for imp in results["improvements"] if imp > 0)
            results["improvement_rate"] = results["positive_improvements"] / len(
                results["improvements"]
            )

        self.logger.info(f"Experiment completed. Enhanced {results['enhanced_chats']} chats.")
        if results["improvements"]:
            self.logger.info(f"Mean improvement: {results['mean_improvement']:.3f}")
            self.logger.info(f"Improvement rate: {results['improvement_rate']:.2%}")

        return results

    def _get_user_id(self, chat_id: str) -> str | None:
        """Get user ID from chat ID using mappings."""
        return self.mappings.get(chat_id)

    def _evaluate_baseline(self, chat: dict) -> float:
        """Evaluate baseline prediction without profile context."""
        # Get last few messages for context
        messages = chat.get("messages", [])
        if len(messages) < 2:
            return 0.0

        # Use last message as target
        target = messages[-1]["content"]
        context_messages = messages[:-2]  # Exclude last two for context

        # Build prompt from chat history only
        context = "\n".join(
            [
                f"{msg.get('role', 'unknown')}: {msg['content']}"
                for msg in context_messages[-5:]  # Last 5 messages
            ]
        )

        prompt = f"""Given this chat history:
{context}

Predict the next message:"""

        try:
            prediction = self.llm_client.generate(prompt)
            score = self.llm_client.judge_similarity(target, prediction)
            return score / 100.0  # Normalize to 0-1
        except Exception as e:
            self.logger.error(f"Error in baseline evaluation: {e}")
            return 0.0

    def _evaluate_with_profile(self, chat: dict, profile: dict) -> float:
        """Evaluate prediction with profile context."""
        messages = chat.get("messages", [])
        if len(messages) < 2:
            return 0.0

        target = messages[-1]["content"]
        context_messages = messages[:-2]

        # Build context with profile
        self.context_builder.profile = profile
        profile_context = (
            self.context_builder._build_profile_context()
            if hasattr(self.context_builder, "_build_profile_context")
            else {}
        )
        chat_context = "\n".join(
            [f"{msg.get('role', 'unknown')}: {msg['content']}" for msg in context_messages[-5:]]
        )

        prompt = f"""User Profile:
{profile_context}

Recent Chat History:
{chat_context}

Predict the next message considering the user's profile:"""

        try:
            prediction = self.llm_client.generate(prompt)
            score = self.llm_client.judge_similarity(target, prediction)
            return score / 100.0
        except Exception as e:
            self.logger.error(f"Error in enhanced evaluation: {e}")
            return 0.0

    def save_results(self, results: dict, output_path: str | None = None):
        """Save experiment results."""
        if output_path is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/phase3_profile_enhanced_{timestamp}.json"

        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")


def main():
    """Run the profile-enhanced experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3 Profile-Enhanced Experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, help="Output path for results")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai"],
        help="LLM provider to use (overrides config)",
    )
    args = parser.parse_args()

    experiment = ProfileEnhancedExperiment(args.config, args.provider)
    results = experiment.run_experiment()
    experiment.save_results(results, args.output)


if __name__ == "__main__":
    main()
