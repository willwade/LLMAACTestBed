"""
Social context experiment for Phase 3.

Tests if social graph information improves prediction accuracy
by adding relationship context to chat history.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.evaluation import MetricsCalculator
from lib.llm_clients import create_llm_client
from lib.utils import Config, load_config, load_env, setup_logging

from lib.data import ChatHistoryLoader, ProfileLoader


class SocialContextExperiment:
    """Experiment testing social graph context impact."""

    def __init__(
        self,
        config_path: str | None = None,
        provider: str | None = None,
        max_chats: int | None = 20,
    ):
        """Initialize the social context experiment."""
        load_env()
        try:
            self.config = load_config(config_path or "configs/social_context.yaml")
        except FileNotFoundError:
            self.config = Config()
        self.provider = provider
        self.logger = setup_logging(
            name="phase3_social_context",
            level=(self.config.get("logging.level") if isinstance(self.config, Config) else None)
            or "INFO",
        )

        # Initialize components
        llm_provider = provider or (
            self.config.get("llm.provider") if isinstance(self.config, Config) else None
        )
        llm_model = self.config.get("llm.model") if isinstance(self.config, Config) else None
        self.llm_client = create_llm_client(provider=llm_provider, model=llm_model)
        self.chat_loader = ChatHistoryLoader()
        self.profile_loader = ProfileLoader()
        self.metrics_calc = MetricsCalculator()
        self.chat_profile_map = self._load_chat_profile_map()
        self.max_chats = max_chats

        # Load data
        self.chat_data = self._load_chat_data()
        self.social_graphs = self._load_social_graphs()

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
                "Chat data not in expected conversation format; skipping social context run"
            )
            return []
        except Exception as exc:
            self.logger.warning(f"Failed to load chat data: {exc}")
            return []

    def _load_chat_profile_map(self) -> dict[str, str]:
        """Load chat->profile mapping."""
        map_path = Path("data/synthetic/chat_profile_mappings/mappings.json")
        if not map_path.exists():
            return {}
        try:
            with open(map_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_social_graphs(self) -> dict[str, dict]:
        """Load social graphs."""
        # Use centralized synthetic data directory
        repo_root = Path(__file__).resolve().parents[3]
        graphs_path = repo_root / "data" / "synthetic" / "social_graphs"
        if not graphs_path.exists():
            self.logger.warning(f"Social graphs directory not found at {graphs_path}")
            return {}

        graphs = {}
        import json

        for graph_file in graphs_path.glob("*.json"):
            with open(graph_file) as f:
                graph = json.load(f)
                if "user_id" in graph:
                    graphs[graph["user_id"]] = graph

        self.logger.info(f"Loaded {len(graphs)} social graphs")
        return graphs

    def run_experiment(self) -> dict[str, Any]:
        """Run the social context experiment."""
        self.logger.info("Starting Phase 3: Social Context Experiment")

        results: dict[str, Any] = {
            "experiment_type": "social_context",
            "total_chats": len(self.chat_data),
            "social_enhanced": 0,
            "baseline_scores": [],
            "social_scores": [],
            "energy_costs": [],
            "improvements": [],
        }

        # Process each chat
        chats_iter = self.chat_data[: self.max_chats] if self.max_chats else self.chat_data
        for chat in chats_iter:
            user_id = self._extract_user_id(chat)

            if not user_id or user_id not in self.social_graphs:
                continue

            social_graph = self.social_graphs[user_id]

            # Get interlocutor from last message
            interlocutor = self._extract_interlocutor(chat)

            # Baseline evaluation
            baseline_score = self._evaluate_baseline(chat)
            results["baseline_scores"].append(baseline_score)

            # Social context evaluation
            social_result = self._evaluate_with_social(chat, social_graph, interlocutor)
            results["social_scores"].append(social_result["score"])
            results["energy_costs"].append(social_result["energy_cost"])

            improvement = social_result["score"] - baseline_score
            results["improvements"].append(improvement)
            results["social_enhanced"] += 1

        # Calculate statistics
        if results["improvements"]:
            results["mean_improvement"] = sum(results["improvements"]) / len(
                results["improvements"]
            )
            results["mean_energy_cost"] = sum(results["energy_costs"]) / len(
                results["energy_costs"]
            )
            results["positive_improvements"] = sum(1 for imp in results["improvements"] if imp > 0)
            results["improvement_rate"] = results["positive_improvements"] / len(
                results["improvements"]
            )

        self.logger.info(f"Social context enhanced {results['social_enhanced']} chats")
        if results["improvements"]:
            self.logger.info(f"Mean improvement: {results['mean_improvement']:.3f}")
            self.logger.info(f"Mean energy cost: {results['mean_energy_cost']:.2f}")

        return results

    def _extract_user_id(self, chat: dict) -> str | None:
        """Extract user ID from chat data."""
        chat_id = chat.get("chat_id")
        return chat.get("user_id") or self.chat_profile_map.get(chat_id)

    def _extract_interlocutor(self, chat: dict) -> str | None:
        """Extract the last interlocutor from chat."""
        messages = chat.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "other":
                return msg.get("speaker")
        return None

    def _evaluate_baseline(self, chat: dict) -> float:
        """Baseline evaluation without social context."""
        messages = chat.get("messages", [])
        if len(messages) < 2:
            return 0.0

        target = messages[-1]["content"]
        context = "\n".join(
            [f"{msg.get('role', 'unknown')}: {msg['content']}" for msg in messages[-5:-1]]
        )

        prompt = f"Chat history:\n{context}\n\nPredict next message:"

        try:
            prediction = self.llm_client.generate(prompt)
            score = self.llm_client.judge_similarity(target, prediction)
            return score / 100.0
        except Exception:
            return 0.0

    def _evaluate_with_social(
        self, chat: dict, social_graph: dict, interlocutor: str | None
    ) -> dict:
        """Evaluate with social context and calculate energy cost."""
        messages = chat.get("messages", [])
        if len(messages) < 2:
            return {"score": 0.0, "energy_cost": 0.0}

        target = messages[-1]["content"]

        # Build social context
        social_context = self._build_social_context(social_graph, interlocutor)

        # Calculate energy cost
        energy_cost = self._calculate_energy_cost(social_graph, interlocutor)

        chat_history = "\n".join(
            [f"{msg.get('role', 'unknown')}: {msg['content']}" for msg in messages[-5:-1]]
        )

        prompt = f"""Social Context:
{social_context}

Recent Chat:
{chat_history}

Predict next message considering the relationship:"""

        try:
            prediction = self.llm_client.generate(prompt)
            score = self.llm_client.judge_similarity(target, prediction)
            return {"score": score / 100.0, "energy_cost": energy_cost}
        except Exception:
            return {"score": 0.0, "energy_cost": energy_cost}

    def _build_social_context(self, social_graph: dict, interlocutor: str | None) -> str:
        """Build social context description."""
        context_parts = []

        # Add relationship if known
        if interlocutor and "relationships" in social_graph:
            relationships = social_graph["relationships"]
            if interlocutor in relationships:
                rel = relationships[interlocutor]
                context_parts.append(
                    f"Relationship with {interlocutor}: {rel.get('type', 'unknown')} "
                    f"(closeness: {rel.get('closeness', 0)}/5, "
                    f"frequency: {rel.get('frequency', 'unknown')})"
                )

        # Add communication preferences
        if "communication_preferences" in social_graph:
            prefs = social_graph["communication_preferences"]
            context_parts.append(
                f"Communication style: {prefs.get('style', 'neutral')}, "
                f"formality: {prefs.get('formality', 'neutral')}"
            )

        return (
            "\n".join(context_parts) if context_parts else "No specific social context available."
        )

    def _calculate_energy_cost(self, social_graph: dict, interlocutor: str | None) -> float:
        """Calculate energy cost for social interaction."""
        base_cost = 1.0

        if interlocutor and "relationships" in social_graph:
            rel = social_graph["relationships"].get(interlocutor, {})
            closeness = rel.get("closeness", 3)

            # Higher closeness = lower energy cost
            cost_modifier = 1.0 - (closeness - 3) * 0.2
            base_cost *= max(0.4, cost_modifier)

        return base_cost

    def save_results(self, results: dict, output_path: str | None = None):
        """Save experiment results."""
        if output_path is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/phase3_social_context_{timestamp}.json"

        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")


def main():
    """Run the social context experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3 Social Context Experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, help="Output path for results")
    args = parser.parse_args()

    experiment = SocialContextExperiment(args.config)
    results = experiment.run_experiment()
    experiment.save_results(results, args.output)


if __name__ == "__main__":
    main()
