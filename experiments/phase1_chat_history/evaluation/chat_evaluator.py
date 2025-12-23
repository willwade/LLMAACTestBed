"""
Phase 1 Chat History Evaluator

Wraps the shared ChatHistoryEvaluator with Phase 1 specific configuration
and data loading.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add lib to path
lib_path = str(Path(__file__).parent.parent.parent / "lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

try:
    from evaluation.chat_history_evaluator import ChatHistoryEvaluator
except ImportError:
    # Fallback for import issues
    from lib.evaluation.chat_history_evaluator import ChatHistoryEvaluator


class Phase1ChatEvaluator:
    """Phase 1 specific chat history evaluator."""

    def __init__(self, config_path: str | None = None):
        """Initialize Phase 1 evaluator.

        Args:
            config_path: Path to config file
        """
        self.config_path = config_path or str(
            Path(__file__).parent.parent / "configs" / "config.yaml"
        )
        self.config = self._load_config()
        self.evaluator = None

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml

            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                "data": {
                    "chat_data_path": "../../data/real/chat_history/raw/baton-export-2025-11-24-nofullstop.json",
                    "corpus_ratio": 0.67,
                },
                "model": {
                    "provider": "gemini",
                    "name": "gemini-2.0-flash-exp",
                    "temperature": 0.2,
                    "embedding_model": "all-MiniLM-L6-v2",
                },
                "evaluation": {
                    "sample_size": 10,
                    "partial_utterance_methods": {
                        "prefix_3": {"type": "prefix", "n_words": 3},
                        "prefix_2": {"type": "prefix", "n_words": 2},
                        "keyword_2": {"type": "keyword", "n_keywords": 2},
                        "random": {"type": "random", "min_words": 1, "max_words": 3},
                    },
                    "generation_methods": {
                        "lexical": {"type": "lexical", "top_k": 3},
                        "tfidf": {"type": "tfidf", "top_k": 3},
                        "embedding": {"type": "embedding", "top_k": 3},
                        "context_only": {"type": "context_only"},
                    },
                    "evaluation_metrics": {
                        "embedding_similarity": {"enabled": True},
                        "llm_judge_score": {"enabled": True},
                        "character_accuracy": {"enabled": True},
                        "word_accuracy": {"enabled": True},
                    },
                },
                "output": {
                    "results_dir": "results",
                    "save_plots": True,
                    "plot_format": "png",
                    "plot_dpi": 300,
                },
            }

    def initialize_evaluator(self, chat_data_path: str | None = None) -> ChatHistoryEvaluator:
        """Initialize the chat history evaluator.

        Args:
            chat_data_path: Override path to chat data file
            deduplicate_corpus: Drop duplicate utterances before splitting

        Returns:
            Initialized ChatHistoryEvaluator
        """
        data_path = chat_data_path or self.config["data"]["chat_data_path"]
        corpus_ratio = self.config["data"]["corpus_ratio"]
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "gemini-2.0-flash-exp")
        llm_provider = model_config.get("provider")

        self.evaluator = ChatHistoryEvaluator(
            chat_data_path=data_path,
            corpus_ratio=corpus_ratio,
            model_name=model_name,
            llm_provider=llm_provider,
            deduplicate_corpus=self.config.get("data", {}).get("deduplicate_corpus", False),
        )

        return self.evaluator

    def create_default_methods(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Create default method dictionaries.

        Returns:
            Tuple of (partial_methods, generation_methods, evaluation_metrics)
        """
        evaluator = self.evaluator or self.initialize_evaluator()

        # Partial utterance methods
        partial_methods = {
            "prefix_3": lambda text: evaluator.create_prefix_partial(text, 3),
            "prefix_2": lambda text: evaluator.create_prefix_partial(text, 2),
            "keyword_2": lambda text: evaluator.create_keyword_partial(text, 2),
            "random": lambda text: evaluator.create_random_partial(text, 1, 3),
        }

        # Generation methods
        generation_methods = {
            "lexical": lambda partial, context, n_candidates=1: evaluator.lexical_generate(
                partial, context, top_k=3, n_candidates=n_candidates
            ),
            "tfidf": lambda partial, context, n_candidates=1: evaluator.tfidf_generate(
                partial, context, top_k=3, n_candidates=n_candidates
            ),
            "embedding": lambda partial, context, n_candidates=1: evaluator.embedding_generate(
                partial, context, top_k=3, n_candidates=n_candidates
            ),
            "context_only": lambda partial,
            context,
            n_candidates=1: evaluator.context_only_generate(
                partial, context, n_candidates=n_candidates
            ),
        }

        # Evaluation metrics
        evaluation_metrics = {
            "embedding_similarity": lambda pred, target: evaluator.embedding_similarity(
                pred, target
            ),
            "llm_judge_score": lambda pred, target: evaluator.llm_judge_score(pred, target),
            "character_accuracy": lambda pred, target: evaluator.character_accuracy(pred, target),
            "character_accuracy_ci": lambda pred, target: evaluator.character_accuracy_ci(
                pred, target
            ),
            "word_accuracy": lambda pred, target: evaluator.word_accuracy(pred, target),
            "word_precision": lambda pred, target: evaluator.word_precision(pred, target),
            "word_recall": lambda pred, target: evaluator.word_recall(pred, target),
            "word_f1": lambda pred, target: evaluator.word_f1(pred, target),
            "weighted_word_f1": lambda pred,
            target,
            partial=None: evaluator.calculate_weighted_word_f1(target, pred, partial or ""),
            "completion_gain": lambda pred,
            target,
            partial=None: evaluator.calculate_completion_gain(target, pred, partial or ""),
        }

        return partial_methods, generation_methods, evaluation_metrics

    def run_evaluation(
        self,
        chat_data_path: str | None = None,
        sample_size: int | None = None,
        skip_short_prefixes: bool | None = None,
        n_candidates: int | None = None,
    ) -> Any:
        """Run the Phase 1 evaluation.

        Args:
            chat_data_path: Override path to chat data file
            sample_size: Number of examples to evaluate
            skip_short_prefixes: If True, skip prefix_* methods when they cannot truncate
            n_candidates: Number of completions to generate per method

        Returns:
            Evaluation results DataFrame
        """
        evaluator = self.initialize_evaluator(chat_data_path)
        partial_methods, generation_methods, evaluation_metrics = self.create_default_methods()

        eval_config = self.config.get("evaluation", {})
        sample_size = sample_size if sample_size is not None else eval_config.get("sample_size")
        # Defaults from config (with sensible fallbacks)
        defaults = eval_config.get("generation_methods", {}).get("defaults", {})
        if n_candidates is None:
            n_candidates = defaults.get("n_candidates", 1)
        if skip_short_prefixes is None:
            skip_short_prefixes = defaults.get("skip_short_prefixes", False)

        # Run evaluation with default methods
        results = evaluator.run_evaluation(
            partial_methods=partial_methods,
            generation_methods=generation_methods,
            evaluation_metrics=evaluation_metrics,
            sample_size=sample_size,
            skip_short_prefixes=skip_short_prefixes,
            n_candidates=n_candidates,
        )

        return results

    def save_results(self, results: Any, output_path: str) -> None:
        """Save evaluation results to file.

        Args:
            results: Evaluation results (DataFrame or dict)
            output_path: Path to save results
        """
        if output_path.endswith(".json"):
            # Convert DataFrame to dict for JSON
            if hasattr(results, "to_dict"):
                data = results.to_dict("records")
            else:
                data = results
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif output_path.endswith(".csv"):
            # Handle DataFrame
            if hasattr(results, "to_csv"):
                results.to_csv(output_path, index=False)
            else:
                # Convert to DataFrame
                import pandas as pd

                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
