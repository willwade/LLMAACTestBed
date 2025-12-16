"""
Data Processing Utilities

Common data processing operations for chat history and transcripts.
"""

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Base data processor with common utilities.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove common artifacts
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        return text

    @staticmethod
    def extract_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Extract time-based features from timestamp column.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            return df

        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract features
        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["month"] = df[timestamp_col].dt.month
        df["is_weekend"] = df[timestamp_col].dt.dayofweek >= 5

        # Time periods
        df["time_period"] = df["hour"].apply(
            lambda h: (
                "morning"
                if 6 <= h < 12
                else "afternoon"
                if 12 <= h < 18
                else "evening"
                if 18 <= h < 22
                else "night"
            )
        )

        return df


class ChatHistoryProcessor(DataProcessor):
    """
    Processor specific to chat history data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with chat history DataFrame.

        Args:
            df: Chat history DataFrame
        """
        self.df = df.copy()

    def preprocess(self) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.

        Returns:
            Preprocessed DataFrame
        """
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=["content"])

        # Clean text content
        self.df["content"] = self.df["content"].apply(self.clean_text)

        # Remove empty content
        self.df = self.df[self.df["content"] != ""]

        # Extract time features
        self.df = self.extract_time_features(self.df)

        # Sort by timestamp
        if "timestamp" in self.df.columns:
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        return self.df

    def add_conversation_windows(self, window_size: int = 3) -> pd.DataFrame:
        """
        Add conversation window context.

        Args:
            window_size: Number of previous messages to include

        Returns:
            DataFrame with conversation windows
        """
        if "timestamp" not in self.df.columns:
            return self.df

        self.df["conversation_window"] = (
            self.df["content"]
            .rolling(window=window_size + 1, min_periods=1)
            .apply(lambda x: " | ".join(x[:-1]), raw=False)
        )

        return self.df

    def filter_by_time_window(
        self, target_time: datetime, window_hours: float = 2.0
    ) -> pd.DataFrame:
        """
        Filter data by time window around target time.

        Args:
            target_time: Center time for window
            window_hours: Hours before and after target time

        Returns:
            Filtered DataFrame
        """
        if "timestamp" not in self.df.columns:
            return self.df

        time_delta = timedelta(hours=window_hours)
        mask = (self.df["timestamp"] >= target_time - time_delta) & (
            self.df["timestamp"] <= target_time + time_delta
        )

        return self.df[mask]

    def filter_by_location(
        self, target_lat: float, target_lon: float, radius_km: float = 0.5
    ) -> pd.DataFrame:
        """
        Filter data by geographic radius.

        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            radius_km: Radius in kilometers

        Returns:
            Filtered DataFrame
        """
        if "latitude" not in self.df.columns or "longitude" not in self.df.columns:
            return self.df

        # Calculate distance using haversine formula
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km

            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))

            return R * c

        distances = self.df.apply(
            lambda row: haversine_distance(
                target_lat, target_lon, row["latitude"], row["longitude"]
            ),
            axis=1,
        )

        return self.df[distances <= radius_km]
