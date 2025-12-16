"""
LLM Judge Module

Provides judging capabilities for comparing model outputs against targets.
"""

from typing import Any

from .base import BaseLLMClient


class LLMPairwiseJudge:
    """
    Judge for evaluating model outputs using an LLM.
    """

    def __init__(self, client: BaseLLMClient):
        """
        Initialize the judge with an LLM client.

        Args:
            client: LLM client for judging
        """
        self.client = client

    def evaluate_batch(
        self, targets: list[str], predictions: list[str], context: str = ""
    ) -> list[int]:
        """
        Evaluate multiple predictions against targets.

        Args:
            targets: List of ground truth responses
            predictions: List of model predictions
            context: Optional context for evaluation

        Returns:
            List of integer scores
        """
        scores = []
        for target, prediction in zip(targets, predictions, strict=False):
            score = self.client.judge_similarity(target, prediction)
            scores.append(score)
        return scores

    def compare_predictions(
        self, target: str, prediction_a: str, prediction_b: str, context: str = ""
    ) -> tuple[int, int, dict[str, Any]]:
        """
        Compare two predictions for the same target.

        Args:
            target: Ground truth response
            prediction_a: First prediction
            prediction_b: Second prediction
            context: Optional context

        Returns:
            Tuple of (score_a, score_b, comparison_details)
        """
        score_a = self.client.judge_similarity(target, prediction_a)
        score_b = self.client.judge_similarity(target, prediction_b)

        comparison_prompt = f"""
TARGET: "{target}"
PREDICTION A: "{prediction_a}" (Score: {score_a}/10)
PREDICTION B: "{prediction_b}" (Score: {score_b}/10)

Context: {context}

Which prediction better serves an AAC user's needs? Explain briefly.
"""

        try:
            explanation = self.client.generate(comparison_prompt)
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

        details = {
            "winner": "A" if score_a > score_b else "B" if score_b > score_a else "Tie",
            "explanation": explanation,
            "context": context,
        }

        return score_a, score_b, details
