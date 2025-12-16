"""
Data Loading Utilities

Unified data loading for different data types and formats.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load data from the given path."""
        pass


class ProfileLoader(BaseDataLoader):
    """
    Loader for user profile JSON files.
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """
        Load user profile from JSON file.

        Args:
            path: Path to profile JSON file

        Returns:
            Dictionary containing user profile data
        """
        with open(path) as f:
            return json.load(f)

    def validate_profile(self, profile: dict[str, Any]) -> bool:
        """
        Validate that profile has required structure.

        Args:
            profile: Profile dictionary

        Returns:
            True if valid, False otherwise
        """
        required_sections = ["identity", "medical_context", "social_graph"]
        return all(section in profile for section in required_sections)


class ChatHistoryLoader(BaseDataLoader):
    """
    Loader for chat history JSON files.
    """

    def load(self, path: str | Path) -> pd.DataFrame:
        """
        Load chat history and convert to DataFrame.

        Args:
            path: Path to chat history JSON file

        Returns:
            DataFrame with chat data
        """
        with open(path) as f:
            data = json.load(f)

        sentences = data.get("sentences", [])
        rows = []

        for sentence in sentences:
            base_row = {
                "uuid": sentence.get("uuid"),
                "content": sentence.get("content"),
            }

            # Handle metadata
            metadata = sentence.get("metadata", [])
            if metadata and isinstance(metadata, list) and len(metadata) > 0:
                meta = metadata[0]  # Take first metadata entry
                base_row.update(
                    {
                        "timestamp": meta.get("timestamp"),
                        "latitude": meta.get("latitude"),
                        "longitude": meta.get("longitude"),
                    }
                )

            rows.append(base_row)

        df = pd.DataFrame(rows)

        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df


class TranscriptLoader(BaseDataLoader):
    """
    Loader for transcript/scenario JSON files.
    """

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load transcript scenarios.

        Args:
            path: Path to transcript JSON file

        Returns:
            List of scenario dictionaries
        """
        with open(path) as f:
            data = json.load(f)

        # Handle different transcript formats
        if "scenarios" in data:
            return data["scenarios"]
        elif "transcripts" in data:
            return data["transcripts"]
        elif isinstance(data, list):
            return data
        else:
            return [data]  # Single scenario


class DataLoader:
    """
    Factory class for loading different types of data.
    """

    @staticmethod
    def load_profile(path: str | Path) -> dict[str, Any]:
        """Load user profile."""
        loader = ProfileLoader()
        return loader.load(path)

    @staticmethod
    def load_chat_history(path: str | Path) -> pd.DataFrame:
        """Load chat history."""
        loader = ChatHistoryLoader()
        return loader.load(path)

    @staticmethod
    def load_transcript(path: str | Path) -> list[dict[str, Any]]:
        """Load transcript scenarios."""
        loader = TranscriptLoader()
        return loader.load(path)
