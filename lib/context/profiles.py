"""
User Profile Management

Classes for managing user profiles and context data.
"""

import json
from pathlib import Path
from typing import Any


class UserProfile:
    """
    Represents a user's profile with medical, social, and vocabulary context.
    """

    def __init__(self, profile_data: dict[str, Any]):
        """
        Initialize user profile.

        Args:
            profile_data: Dictionary containing profile information
        """
        self.data = profile_data
        self._validate_profile()

    def _validate_profile(self):
        """Validate profile structure."""
        required_sections = ["identity", "medical_context", "social_graph"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required profile section: {section}")

    @property
    def name(self) -> str:
        """Get user's name."""
        return self.data.get("identity", {}).get("name", "Unknown")

    @property
    def condition(self) -> str:
        """Get user's medical condition."""
        return self.data.get("identity", {}).get("condition", "Unknown")

    @property
    def age(self) -> int | None:
        """Get user's age."""
        return self.data.get("identity", {}).get("age")

    @property
    def communication_style(self) -> str:
        """Get communication style description."""
        return self.data.get("identity", {}).get("current_status", "")

    def get_medical_equipment(self) -> list[str]:
        """Get list of medical equipment."""
        return self.data.get("medical_context", {}).get("equipment", [])

    def get_symptoms(self) -> list[str]:
        """Get list of symptoms."""
        return self.data.get("medical_context", {}).get("symptoms", [])

    def get_medications(self) -> list[str]:
        """Get list of medications."""
        return self.data.get("medical_context", {}).get("medications", [])

    def get_social_graph(self) -> dict[str, Any]:
        """Get social graph information."""
        return self.data.get("social_graph", {})

    def get_person_info(self, name: str) -> dict[str, Any] | None:
        """
        Get information about a specific person in social graph.

        Args:
            name: Name of the person

        Returns:
            Dictionary with person information or None
        """
        social_graph = self.get_social_graph()
        return social_graph.get(name)

    def get_vocabulary_preferences(self) -> dict[str, Any]:
        """Get vocabulary and style preferences."""
        return self.data.get("vocabulary_preferences", {})

    def get_common_requests(self) -> list[str]:
        """Get list of common requests/phrases."""
        vocab = self.get_vocabulary_preferences()
        return vocab.get("common_requests", [])

    def get_recent_events(self) -> dict[str, Any]:
        """Get recent events context."""
        return self.data.get("recent_events", {})

    def get_communication_context(self) -> dict[str, Any]:
        """
        Get comprehensive communication context.

        Returns:
            Dictionary with all relevant context for communication
        """
        return {
            "identity": {
                "name": self.name,
                "condition": self.condition,
                "communication_style": self.communication_style,
            },
            "medical_needs": {
                "equipment": self.get_medical_equipment(),
                "symptoms": self.get_symptoms(),
                "medications": self.get_medications(),
            },
            "social_context": self.get_social_graph(),
            "vocabulary": self.get_vocabulary_preferences(),
            "recent_context": self.get_recent_events(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return profile as dictionary."""
        return self.data.copy()


class ProfileManager:
    """
    Manager for loading and managing multiple user profiles.
    """

    def __init__(self, profile_dir: str | Path | None = None):
        """
        Initialize profile manager.

        Args:
            profile_dir: Directory containing profile files
        """
        self.profile_dir = Path(profile_dir) if profile_dir else None
        self.profiles = {}

    def load_profile(self, profile_path: str | Path) -> UserProfile:
        """
        Load a single profile from file.

        Args:
            profile_path: Path to profile JSON file

        Returns:
            UserProfile instance
        """
        with open(profile_path) as f:
            profile_data = json.load(f)

        profile = UserProfile(profile_data)
        self.profiles[profile.name] = profile
        return profile

    def load_all_profiles(self) -> dict[str, UserProfile]:
        """
        Load all profiles from the configured directory.

        Returns:
            Dictionary mapping profile names to UserProfile instances
        """
        if not self.profile_dir or not self.profile_dir.exists():
            return self.profiles

        for profile_file in self.profile_dir.glob("*.json"):
            try:
                self.load_profile(profile_file)
            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")

        return self.profiles

    def get_profile(self, name: str) -> UserProfile | None:
        """
        Get profile by name.

        Args:
            name: Profile name

        Returns:
            UserProfile instance or None
        """
        return self.profiles.get(name)

    def create_template_profile(self, name: str, condition: str = "MND") -> UserProfile:
        """
        Create a template profile with basic structure.

        Args:
            name: User's name
            condition: Medical condition

        Returns:
            UserProfile instance with template data
        """
        template_data = {
            "identity": {
                "name": name,
                "age": 45,
                "condition": condition,
                "former_occupation": "Unknown",
                "current_status": "AAC user, telegraphic speech patterns",
            },
            "medical_context": {"equipment": [], "symptoms": [], "medications": []},
            "social_graph": {},
            "vocabulary_preferences": {
                "style": "Telegraphic, keyword-focused",
                "common_requests": [],
            },
            "recent_events": {},
        }

        profile = UserProfile(template_data)
        self.profiles[name] = profile
        return profile

    def save_profile(self, profile: UserProfile, save_path: str | Path | None = None):
        """
        Save profile to file.

        Args:
            profile: UserProfile to save
            save_path: Optional custom save path
        """
        if save_path is None and self.profile_dir:
            save_path = self.profile_dir / f"{profile.name.lower()}_profile.json"
        elif save_path is None:
            raise ValueError("No save path specified and no profile directory configured")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
