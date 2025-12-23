"""
Evaluation Metrics

Implementation of various metrics for evaluating text completion systems.
"""

from abc import ABC, abstractmethod

import nltk
import numpy as np
import pandas as pd
from Levenshtein import ratio as levenshtein_ratio
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """
        Calculate metric for each prediction-target pair.

        Args:
            predictions: List of predicted texts
            targets: List of target texts

        Returns:
            List of metric scores
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

    def aggregate(self, scores: list[float]) -> dict[str, float]:
        """
        Aggregate scores into summary statistics.

        Args:
            scores: List of metric scores

        Returns:
            Dictionary with mean, std, median, etc.
        """
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "q25": np.percentile(scores, 25),
            "q75": np.percentile(scores, 75),
        }


class EmbeddingSimilarityMetric(BaseMetric):
    """
    Cosine similarity using sentence embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding metric.

        Args:
            model_name: Name of sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self._embeddings_cache = {}

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """
        Calculate cosine similarity between embeddings.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            List of cosine similarities (0-1)
        """
        similarities = []

        for pred, target in zip(predictions, targets, strict=False):
            # Get embeddings (with caching)
            if pred not in self._embeddings_cache:
                self._embeddings_cache[pred] = self.model.encode(pred)
            if target not in self._embeddings_cache:
                self._embeddings_cache[target] = self.model.encode(target)

            pred_emb = self._embeddings_cache[pred]
            target_emb = self._embeddings_cache[target]

            # Calculate cosine similarity
            similarity = np.dot(pred_emb, target_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(target_emb)
            )
            similarities.append(float(similarity))

        return similarities

    def name(self) -> str:
        return "embedding_similarity"


class CharacterAccuracyMetric(BaseMetric):
    """
    Character-level accuracy using Levenshtein distance.
    """

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """
        Calculate character accuracy.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            List of character accuracies (0-1)
        """
        accuracies = []
        for pred, target in zip(predictions, targets, strict=False):
            accuracy = levenshtein_ratio(pred.lower(), target.lower())
            accuracies.append(accuracy)
        return accuracies

    def name(self) -> str:
        return "character_accuracy"


class WordAccuracyMetric(BaseMetric):
    """
    Word-level accuracy based on word overlap.
    """

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """
        Calculate word accuracy.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            List of word accuracies (0-1)
        """
        accuracies = []
        for pred, target in zip(predictions, targets, strict=False):
            pred_words = set(pred.lower().split())
            target_words = set(target.lower().split())

            if len(target_words) == 0:
                accuracies.append(1.0 if len(pred_words) == 0 else 0.0)
                continue

            intersection = pred_words.intersection(target_words)
            accuracy = len(intersection) / len(target_words)
            accuracies.append(accuracy)

        return accuracies

    def name(self) -> str:
        return "word_accuracy"


class WordPrecisionMetric(BaseMetric):
    """
    Word-level precision based on overlap.
    """

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        scores = []
        for pred, target in zip(predictions, targets, strict=False):
            pred_words = set(pred.lower().split())
            target_words = set(target.lower().split())

            if len(pred_words) == 0:
                scores.append(0.0)
                continue

            intersection = pred_words.intersection(target_words)
            scores.append(len(intersection) / len(pred_words))
        return scores

    def name(self) -> str:
        return "word_precision"


class WordRecallMetric(BaseMetric):
    """
    Word-level recall based on overlap.
    """

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        scores = []
        for pred, target in zip(predictions, targets, strict=False):
            pred_words = set(pred.lower().split())
            target_words = set(target.lower().split())

            if len(target_words) == 0:
                scores.append(1.0 if len(pred_words) == 0 else 0.0)
                continue

            intersection = pred_words.intersection(target_words)
            scores.append(len(intersection) / len(target_words))
        return scores

    def name(self) -> str:
        return "word_recall"


class WordF1Metric(BaseMetric):
    """
    Word-level F1 combining precision and recall.
    """

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        scores = []
        for pred, target in zip(predictions, targets, strict=False):
            precision_metric = WordPrecisionMetric()
            recall_metric = WordRecallMetric()
            precision = precision_metric.calculate([pred], [target])[0]
            recall = recall_metric.calculate([pred], [target])[0]

            if precision + recall == 0:
                scores.append(0.0)
                continue

            scores.append(2 * (precision * recall) / (precision + recall))
        return scores

    def name(self) -> str:
        return "word_f1"


class BLEUScoreMetric(BaseMetric):
    """
    BLEU score for evaluating text generation.
    """

    def __init__(self, weights: tuple | None = None):
        """
        Initialize BLEU metric.

        Args:
            weights: Weights for n-gram precision (default: 4-gram)
        """
        self.weights = weights or (0.25, 0.25, 0.25, 0.25)

    def calculate(self, predictions: list[str], targets: list[str]) -> list[float]:
        """
        Calculate BLEU scores.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            List of BLEU scores (0-1)
        """
        scores = []
        for pred, target in zip(predictions, targets, strict=False):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            target_tokens = [nltk.word_tokenize(target.lower())]

            # Calculate BLEU
            try:
                score = sentence_bleu(target_tokens, pred_tokens, weights=self.weights)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return scores

    def name(self) -> str:
        return "bleu_score"


class MetricsCalculator:
    """
    Calculator for managing multiple metrics.
    """

    def __init__(self, metrics: list[BaseMetric] | None = None):
        """
        Initialize with list of metrics.

        Args:
            metrics: List of metric instances
        """
        self.metrics = metrics or [
            EmbeddingSimilarityMetric(),
            CharacterAccuracyMetric(),
            WordAccuracyMetric(),
            BLEUScoreMetric(),
        ]

    def calculate_all(self, predictions: list[str], targets: list[str]) -> dict[str, list[float]]:
        """
        Calculate all configured metrics.

        Args:
            predictions: List of predictions
            targets: List of targets

        Returns:
            Dictionary mapping metric names to scores
        """
        results = {}
        for metric in self.metrics:
            scores = metric.calculate(predictions, targets)
            results[metric.name()] = scores

        return results

    def create_results_dataframe(
        self,
        predictions: list[str],
        targets: list[str],
        additional_data: dict[str, list] | None = None,
    ) -> pd.DataFrame:
        """
        Create a results DataFrame with all metrics.

        Args:
            predictions: List of predictions
            targets: List of targets
            additional_data: Additional columns to include

        Returns:
            DataFrame with predictions, targets, and all metric scores
        """
        data = {"target": targets, "prediction": predictions}

        # Add metrics
        metric_results = self.calculate_all(predictions, targets)
        data.update(metric_results)

        # Add additional data
        if additional_data:
            data.update(additional_data)

        return pd.DataFrame(data)
