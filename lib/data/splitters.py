"""
Data Splitting Utilities

Different strategies for splitting data into train/test sets.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseSplitter(ABC):
    """
    Abstract base class for data splitters.
    """

    @abstractmethod
    def split(self, data: Any, test_ratio: float = 0.33) -> tuple[Any, Any]:
        """
        Split data into train and test sets.

        Args:
            data: Data to split
            test_ratio: Fraction of data for test set

        Returns:
            Tuple of (train_data, test_data)
        """
        pass


class ChronologicalSplitter(BaseSplitter):
    """
    Split data chronologically based on timestamps.
    """

    def __init__(self, timestamp_col: str = "timestamp"):
        """
        Initialize chronological splitter.

        Args:
            timestamp_col: Name of timestamp column
        """
        self.timestamp_col = timestamp_col

    def split(
        self, data: pd.DataFrame, test_ratio: float = 0.33
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame chronologically.

        Args:
            data: DataFrame with timestamp column
            test_ratio: Fraction for test set (most recent)

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.timestamp_col not in data.columns:
            # Fallback to random split if no timestamp
            splitter = RandomSplitter()
            return splitter.split(data, test_ratio)

        # Sort by timestamp
        data_sorted = data.sort_values(self.timestamp_col).reset_index(drop=True)

        # Calculate split point
        split_idx = int(len(data_sorted) * (1 - test_ratio))

        train_df = data_sorted.iloc[:split_idx]
        test_df = data_sorted.iloc[split_idx:]

        return train_df, test_df


class RandomSplitter(BaseSplitter):
    """
    Random split of data.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random splitter.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def split(
        self, data: pd.DataFrame, test_ratio: float = 0.33
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame randomly.

        Args:
            data: DataFrame to split
            test_ratio: Fraction for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        data_shuffled = data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        split_idx = int(len(data_shuffled) * (1 - test_ratio))

        train_df = data_shuffled.iloc[:split_idx]
        test_df = data_shuffled.iloc[split_idx:]

        return train_df, test_df


class ConversationSplitter(BaseSplitter):
    """
    Split data by conversation boundaries.
    """

    def __init__(self, timestamp_col: str = "timestamp", max_gap_minutes: int = 60):
        """
        Initialize conversation splitter.

        Args:
            timestamp_col: Name of timestamp column
            max_gap_minutes: Maximum gap between messages in same conversation
        """
        self.timestamp_col = timestamp_col
        self.max_gap_minutes = max_gap_minutes

    def split(
        self, data: pd.DataFrame, test_ratio: float = 0.33
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by conversations.

        Args:
            data: DataFrame with timestamp column
            test_ratio: Fraction for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.timestamp_col not in data.columns:
            # Fallback to chronological split
            splitter = ChronologicalSplitter(self.timestamp_col)
            return splitter.split(data, test_ratio)

        # Identify conversation boundaries
        data_sorted = data.sort_values(self.timestamp_col).reset_index(drop=True)
        data_sorted["time_diff"] = data_sorted[self.timestamp_col].diff()
        data_sorted["new_conversation"] = (
            data_sorted["time_diff"] > pd.Timedelta(minutes=self.max_gap_minutes)
        ).fillna(False)

        # Assign conversation IDs
        data_sorted["conversation_id"] = data_sorted["new_conversation"].cumsum()

        # Get unique conversations
        conversations = data_sorted["conversation_id"].unique()
        n_test = int(len(conversations) * test_ratio)

        # Split conversations
        test_conversations = conversations[-n_test:]  # Most recent conversations
        train_conversations = conversations[:-n_test]

        # Create train/test splits
        train_df = data_sorted[data_sorted["conversation_id"].isin(train_conversations)]
        test_df = data_sorted[data_sorted["conversation_id"].isin(test_conversations)]

        # Drop helper columns
        for df in [train_df, test_df]:
            df.drop(columns=["time_diff", "new_conversation", "conversation_id"], inplace=True)

        return train_df, test_df
