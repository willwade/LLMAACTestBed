"""
Context Filters

Classes for filtering and selecting relevant context data.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


class ContextFilter(ABC):
    """
    Abstract base class for context filters.
    """

    @abstractmethod
    def filter(self, data: Any, **kwargs) -> Any:
        """Apply filter to data."""
        pass


class TimeFilter(ContextFilter):
    """
    Filter data based on temporal proximity.
    """

    def __init__(self, window_hours: float = 2.0):
        """
        Initialize time filter.

        Args:
            window_hours: Time window in hours
        """
        self.window_hours = window_hours

    def filter(
        self, data: pd.DataFrame, reference_time: datetime, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Filter DataFrame to items within time window.

        Args:
            data: DataFrame with timestamp column
            reference_time: Reference time
            timestamp_col: Name of timestamp column

        Returns:
            Filtered DataFrame
        """
        if timestamp_col not in data.columns:
            return data

        time_delta = timedelta(hours=self.window_hours)
        mask = (data[timestamp_col] >= reference_time - time_delta) & (
            data[timestamp_col] <= reference_time + time_delta
        )

        return data[mask]

    def get_relevant_time_periods(
        self, reference_time: datetime, num_periods: int = 3
    ) -> list[tuple[datetime, datetime]]:
        """
        Get relevant time periods around reference time.

        Args:
            reference_time: Reference time
            num_periods: Number of periods to return

        Returns:
            List of (start_time, end_time) tuples
        """
        periods = []
        for i in range(num_periods):
            days_offset = i - num_periods // 2
            start = reference_time + timedelta(days=days_offset, hours=-self.window_hours)
            end = reference_time + timedelta(days=days_offset, hours=self.window_hours)
            periods.append((start, end))

        return periods


class LocationFilter(ContextFilter):
    """
    Filter data based on geographic proximity.
    """

    def __init__(self, radius_km: float = 0.5):
        """
        Initialize location filter.

        Args:
            radius_km: Radius in kilometers
        """
        self.radius_km = radius_km

    def filter(
        self,
        data: pd.DataFrame,
        reference_location: dict[str, float],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to items within geographic radius.

        Args:
            data: DataFrame with location columns
            reference_location: Dictionary with 'latitude' and 'longitude'
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            Filtered DataFrame
        """
        if lat_col not in data.columns or lon_col not in data.columns:
            return data

        ref_lat = reference_location.get("latitude")
        ref_lon = reference_location.get("longitude")

        if ref_lat is None or ref_lon is None:
            return data

        # Calculate distances
        distances = data.apply(
            lambda row: self._haversine_distance(ref_lat, ref_lon, row[lat_col], row[lon_col]),
            axis=1,
        )

        return data[distances <= self.radius_km]

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c


class SocialFilter(ContextFilter):
    """
    Filter data based on social context.
    """

    def __init__(self, social_graph: dict[str, Any] | None = None):
        """
        Initialize social filter.

        Args:
            social_graph: Social graph information
        """
        self.social_graph = social_graph or {}

    def filter(
        self, data: pd.DataFrame, interlocutor: str, context_col: str = "content"
    ) -> pd.DataFrame:
        """
        Filter DataFrame based on relevance to interlocutor.

        Args:
            data: DataFrame with content
            interlocutor: Name of current interlocutor
            context_col: Column containing text content

        Returns:
            Filtered DataFrame with relevance scores
        """
        if context_col not in data.columns:
            return data

        # Get person info
        person_info = self.social_graph.get(interlocutor, {})
        keywords = self._extract_keywords(person_info)

        if not keywords:
            return data

        # Calculate relevance scores
        def relevance_score(text):
            if not isinstance(text, str):
                return 0
            score = 0
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            return score

        data = data.copy()
        data["relevance_score"] = data[context_col].apply(relevance_score)

        # Filter to relevant items (score > 0)
        relevant_data = data[data["relevance_score"] > 0]

        return relevant_data.sort_values("relevance_score", ascending=False)

    def _extract_keywords(self, person_info: dict[str, Any]) -> list[str]:
        """Extract relevant keywords from person info."""
        keywords = []

        # Add relation-specific keywords
        relation = person_info.get("relation", "").lower()
        if "nurse" in relation or "medical" in relation:
            keywords.extend(["medication", "pain", "symptoms", "medical", "nurse", "doctor"])
        elif "family" in relation or "mother" in relation or "wife" in relation:
            keywords.extend(["family", "home", "personal", "feelings", "emotional"])
        elif "therapist" in relation:
            keywords.extend(["therapy", "exercise", "progress", "treatment"])

        # Add hobby/interest keywords
        if "hobbies" in person_info:
            keywords.extend(person_info["hobbies"])

        # Add role-specific keywords
        role = person_info.get("role", "").lower()
        if role:
            keywords.append(role)

        return list(set(keywords))  # Remove duplicates


class ConversationWindowFilter(ContextFilter):
    """
    Filter for maintaining conversation window context.
    """

    def __init__(self, window_size: int = 3):
        """
        Initialize conversation window filter.

        Args:
            window_size: Number of previous messages to include
        """
        self.window_size = window_size

    def filter(
        self, conversation_history: list[dict[str, Any]], current_index: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Filter conversation history to window around current message.

        Args:
            conversation_history: List of conversation messages
            current_index: Index of current message (defaults to last)

        Returns:
            Filtered conversation window
        """
        if current_index is None:
            current_index = len(conversation_history) - 1

        start_idx = max(0, current_index - self.window_size)
        end_idx = min(len(conversation_history), current_index + 1)

        return conversation_history[start_idx:end_idx]

    def build_context_string(
        self, conversation_window: list[dict[str, Any]], content_field: str = "content"
    ) -> str:
        """
        Build context string from conversation window.

        Args:
            conversation_window: List of messages in window
            content_field: Field name for message content

        Returns:
            Context string
        """
        if not conversation_window:
            return ""

        context_parts = []
        for msg in conversation_window[:-1]:  # Exclude current message
            content = msg.get(content_field, "")
            if content:
                context_parts.append(content)

        return " | ".join(context_parts)
