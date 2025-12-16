"""
Evaluation Framework

Shared evaluation metrics, scorers, and visualizations.
"""

from .base_evaluator import BaseEvaluator
from .chat_history_evaluator import ChatHistoryEvaluator
from .metrics import (
    BLEUScoreMetric,
    CharacterAccuracyMetric,
    EmbeddingSimilarityMetric,
    MetricsCalculator,
    WordAccuracyMetric,
)
from .scorers import BaseScorer, CompositeScorer, LLMScorer
from .visualizers import ComparisonVisualizer, ResultsVisualizer

__all__ = [
    "EmbeddingSimilarityMetric",
    "CharacterAccuracyMetric",
    "WordAccuracyMetric",
    "BLEUScoreMetric",
    "MetricsCalculator",
    "BaseScorer",
    "LLMScorer",
    "CompositeScorer",
    "ResultsVisualizer",
    "ComparisonVisualizer",
    "BaseEvaluator",
    "ChatHistoryEvaluator",
]
