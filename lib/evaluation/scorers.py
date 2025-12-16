"""
Scoring Utilities

Different scoring approaches for evaluating predictions.
"""

from abc import ABC, abstractmethod

from ..llm_clients.base import BaseLLMClient


class BaseScorer(ABC):
    """
    Abstract base class for scorers.
    """

    @abstractmethod
    def score(self, prediction: str, target: str, context: str = "") -> float:
        """
        Score a single prediction against target.

        Args:
            prediction: Model prediction
            target: Ground truth
            context: Optional context

        Returns:
            Score value
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the scorer."""
        pass


class LLMScorer(BaseScorer):
    """
    LLM-based scorer using semantic similarity.
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize with LLM client.

        Args:
            llm_client: LLM client for scoring
        """
        self.llm_client = llm_client

    def score(self, prediction: str, target: str, context: str = "") -> float:
        """
        Score using LLM judgment.

        Args:
            prediction: Model prediction
            target: Ground truth
            context: Optional context

        Returns:
            Score from 1-10
        """
        return self.llm_client.judge_similarity(target, prediction)

    def name(self) -> str:
        return "llm_judge_score"


class RuleBasedScorer(BaseScorer):
    """
    Rule-based scorer for AAC-specific criteria.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Initialize with scoring rules.

        Args:
            weights: Weights for different criteria
        """
        self.weights = weights or {
            "word_overlap": 0.3,
            "length_similarity": 0.2,
            "keyword_match": 0.3,
            "urgency_match": 0.2,
        }

    def score(self, prediction: str, target: str, context: str = "") -> float:
        """
        Score using rule-based criteria.

        Args:
            prediction: Model prediction
            target: Ground truth
            context: Optional context

        Returns:
            Score from 0-1
        """
        score = 0.0

        # Word overlap
        pred_words = set(prediction.lower().split())
        target_words = set(target.lower().split())
        if target_words:
            overlap = len(pred_words & target_words) / len(target_words)
            score += self.weights["word_overlap"] * overlap

        # Length similarity
        pred_len = len(prediction.split())
        target_len = len(target.split())
        if target_len > 0:
            length_sim = 1 - abs(pred_len - target_len) / target_len
            score += self.weights["length_similarity"] * max(0, length_sim)

        # Keyword match (for AAC-specific terms)
        urgent_keywords = {"help", "urgent", "now", "please", "need", "want"}
        pred_urgent = any(kw in prediction.lower() for kw in urgent_keywords)
        target_urgent = any(kw in target.lower() for kw in urgent_keywords)
        if pred_urgent == target_urgent:
            score += self.weights["urgency_match"]

        # Normalize to 0-1
        return min(1.0, score)

    def name(self) -> str:
        return "rule_based_score"


class CompositeScorer(BaseScorer):
    """
    Composite scorer combining multiple scoring methods.
    """

    def __init__(self, scorers: list[BaseScorer], weights: list[float] | None = None):
        """
        Initialize with multiple scorers.

        Args:
            scorers: List of scorers to combine
            weights: Weights for each scorer (default: equal weights)
        """
        self.scorers = scorers
        self.weights = weights or [1.0 / len(scorers)] * len(scorers)

        if len(self.weights) != len(self.scorers):
            raise ValueError("Number of weights must match number of scorers")

    def score(self, prediction: str, target: str, context: str = "") -> float:
        """
        Score using weighted combination of all scorers.

        Args:
            prediction: Model prediction
            target: Ground truth
            context: Optional context

        Returns:
            Combined score
        """
        total_score = 0.0
        total_weight = 0.0

        for scorer, weight in zip(self.scorers, self.weights, strict=False):
            try:
                score = scorer.score(prediction, target, context)
                # Normalize score to 0-1 if needed
                if scorer.name() == "llm_judge_score":
                    score = score / 10.0  # Convert 1-10 to 0-1
                total_score += weight * score
                total_weight += weight
            except Exception as e:
                print(f"Error in scorer {scorer.name()}: {e}")
                continue

        return total_score / total_weight if total_weight > 0 else 0.0

    def name(self) -> str:
        scorer_names = [s.name() for s in self.scorers]
        return f"composite_({'_'.join(scorer_names)})"
