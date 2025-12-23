"""
Data Processing Utilities

Common data processing functions for experiments.
"""

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def calculate_statistics(scores: Sequence[float]) -> dict[str, float]:
    """
    Calculate basic statistics for a list of scores.

    Args:
        scores: List of scores

    Returns:
        Dictionary with statistics
    """
    if not scores:
        return {}

    return {
        "mean": sum(scores) / len(scores),
        "max": max(scores),
        "min": min(scores),
        "count": len(scores),
        "success_rate": sum(1 for s in scores if s >= 7) / len(scores) * 100,
    }


def validate_llm_response(response: str) -> bool:
    """
    Validate that LLM response is reasonable.

    Args:
        response: LLM response string

    Returns:
        True if response appears valid
    """
    if not response:
        return False

    response = response.strip()

    # Check for common error indicators
    error_indicators = [
        "error:",
        "not available",
        "i cannot",
        "i'm sorry",
        "i don't understand",
        "i cannot answer",
    ]

    response_lower = response.lower()
    for indicator in error_indicators:
        if indicator in response_lower:
            return False

    # Check minimum length
    if len(response) < 3:
        return False

    return True


def format_keywords_for_prompt(keywords: list[str]) -> str:
    """
    Format keywords for inclusion in prompts.

    Args:
        keywords: List of keywords

    Returns:
        Formatted string
    """
    if not keywords:
        return ""

    return ", ".join(keywords)


def create_experiment_metadata(
    provider: str, model: str, timestamp: str, phase: int, experiment_type: str
) -> dict[str, Any]:
    """
    Create metadata for experiment tracking.

    Args:
        provider: LLM provider name
        model: Model name
        timestamp: Experiment timestamp
        phase: Experiment phase number
        experiment_type: Type of experiment

    Returns:
        Metadata dictionary
    """
    return {
        "provider": provider,
        "model": model,
        "timestamp": timestamp,
        "framework_version": "1.0",
        "experiment_phase": phase,
        "experiment_type": experiment_type,
    }


def clean_dataframe(
    df: pd.DataFrame, required_columns: list[str], subset_columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Clean and validate a DataFrame.

    Args:
        df: DataFrame to clean
        required_columns: Columns that must be present
        subset_columns: Columns to check for NaN values

    Returns:
        Cleaned DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with NaN in specified columns
    if subset_columns:
        df = df.dropna(subset=subset_columns)

    return df
