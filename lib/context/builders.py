"""
Context Builders

Classes for building context prompts and managing context injection.
"""

from datetime import datetime
from typing import Any

from .filters import LocationFilter, SocialFilter, TimeFilter
from .profiles import UserProfile


class ContextBuilder:
    """
    Builder for constructing context from various sources.
    """

    def __init__(self, profile: UserProfile | None = None):
        """
        Initialize context builder.

        Args:
            profile: Optional user profile
        """
        self.profile = profile
        self.filters = {
            "time": TimeFilter(),
            "location": LocationFilter(),
            "social": SocialFilter(),
        }

    def build_context(
        self,
        current_time: datetime | None = None,
        location: dict[str, float] | None = None,
        interlocutor: str | None = None,
        conversation_history: list[str] | None = None,
        context_levels: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build comprehensive context.

        Args:
            current_time: Current timestamp
            location: Dictionary with 'latitude' and 'longitude'
            interlocutor: Name of person speaking to user
            conversation_history: Recent conversation history
            context_levels: List of context types to include

        Returns:
            Dictionary with built context
        """
        context = {}

        if context_levels is None:
            context_levels = ["profile", "social", "time", "location"]

        # Add user profile context
        if "profile" in context_levels and self.profile:
            context["user_profile"] = self._build_profile_context()

        # Add social context
        if "social" in context_levels and interlocutor and self.profile:
            context["social_context"] = self._build_social_context(interlocutor)

        # Add temporal context
        if "time" in context_levels and current_time:
            context["temporal_context"] = self._build_temporal_context(current_time)

        # Add location context
        if "location" in context_levels and location:
            context["location_context"] = self._build_location_context(location)

        # Add conversation context
        if conversation_history:
            context["conversation_context"] = {
                "recent_messages": conversation_history[-5:],  # Last 5 messages
                "message_count": len(conversation_history),
            }

        return context

    def _build_profile_context(self) -> dict[str, Any]:
        """Build user profile context."""
        if not self.profile:
            return {}

        return {
            "identity": self.profile.data.get("identity", {}),
            "medical_needs": {
                "equipment": self.profile.get_medical_equipment(),
                "symptoms": self.profile.get_symptoms(),
                "urgency_indicators": self._get_urgency_indicators(),
            },
            "communication_style": self.profile.get_vocabulary_preferences(),
        }

    def _build_social_context(self, interlocutor: str) -> dict[str, Any]:
        """Build social context for interlocutor."""
        if not self.profile:
            return {}

        person_info = self.profile.get_person_info(interlocutor)
        if not person_info:
            return {"interlocutor": interlocutor, "relationship": "unknown"}

        return {
            "interlocutor": interlocutor,
            "relationship": person_info.get("relation", "unknown"),
            "communication_style": person_info.get("personality", ""),
            "context_notes": person_info.get("dynamic", ""),
            "energy_cost": self._estimate_interaction_energy(person_info),
        }

    def _build_temporal_context(self, current_time: datetime) -> dict[str, Any]:
        """Build temporal context."""
        hour = current_time.hour
        time_period = self._get_time_period(hour)
        day_of_week = current_time.strftime("%A")

        # Map time periods to likely needs
        likely_needs = self._infer_temporal_needs(hour, day_of_week)

        return {
            "current_time": current_time.strftime("%H:%M"),
            "time_period": time_period,
            "day_of_week": day_of_week,
            "likely_needs": likely_needs,
        }

    def _build_location_context(self, location: dict[str, float]) -> dict[str, Any]:
        """Build location context."""
        return {
            "location": location,
            "location_type": self._infer_location_type(location),
            "environmental_factors": self._infer_environmental_factors(location),
        }

    def _get_urgency_indicators(self) -> list[str]:
        """Get urgency indicators from medical context."""
        indicators = []
        if self.profile:
            symptoms = self.profile.get_symptoms()
            if "temperature dysregulation" in [s.lower() for s in symptoms]:
                indicators.append("temperature_changes")
            if "shortness of breath" in [s.lower() for s in symptoms]:
                indicators.append("breathing_difficulty")
        return indicators

    def _estimate_interaction_energy(self, person_info: dict[str, Any]) -> str:
        """Estimate energy cost for interaction."""
        relation = person_info.get("relation", "").lower()
        personality = person_info.get("personality", "").lower()

        if "medical" in relation or "nurse" in relation:
            return "low"
        elif "mother" in relation or "emotional" in personality:
            return "high"
        else:
            return "medium"

    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _infer_temporal_needs(self, hour: int, day_of_week: str) -> list[str]:
        """Infer likely needs based on time."""
        needs = []

        if 6 <= hour < 9:
            needs.extend(["medications", "morning_care"])
        elif 12 <= hour < 14:
            needs.extend(["lunch", "positioning"])
        elif 18 <= hour < 21:
            needs.extend(["dinner", "evening_care"])
        elif 21 <= hour or hour < 6:
            needs.extend(["comfort", "night_care"])

        if day_of_week == "Monday":
            needs.append("weekly_routine")

        return needs

    def _infer_location_type(self, location: dict[str, float]) -> str:
        """Infer location type from coordinates (placeholder)."""
        # In a real implementation, this would use reverse geocoding
        return "unknown"

    def _infer_environmental_factors(self, location: dict[str, float]) -> list[str]:
        """Infer environmental factors (placeholder)."""
        return []


class PromptBuilder:
    """
    Builder for creating prompts with context injection.
    """

    def __init__(self, template_style: str = "aac"):
        """
        Initialize prompt builder.

        Args:
            template_style: Style of prompt templates ('aac', 'general', etc.)
        """
        self.template_style = template_style
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, str]:
        """Load prompt templates."""
        if self.template_style == "aac":
            return {
                "system": "You are an AI assistant for an AAC user with {condition}. Focus on understanding telegraphic speech and providing practical, actionable responses.",
                "context": "User Context:\n{context}\n\nCurrent Situation:\n{situation}",
                "prediction": "Given the partial input '{partial_utterance}', predict the user's full intent or need.",
                "completion": "Complete the user's message based on their context and needs.",
            }
        else:
            return {
                "system": "You are a helpful AI assistant.",
                "context": "Context: {context}",
                "prediction": "Complete the partial message: '{partial_utterance}'",
                "completion": "Continue the message in a helpful way.",
            }

    def build_prompt(
        self, partial_input: str, context: dict[str, Any], task: str = "prediction"
    ) -> tuple[str, str]:
        """
        Build a complete prompt with context.

        Args:
            partial_input: Partial user input
            context: Context dictionary
            task: Type of task ('prediction' or 'completion')

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Build system prompt
        system_prompt = self.templates["system"].format(
            condition=context.get("user_profile", {})
            .get("identity", {})
            .get("condition", "communication difficulties")
        )

        # Build context string
        context_str = self._format_context(context)

        # Build situation string
        situation_str = self._format_situation(context)

        # Build user prompt
        if task == "prediction":
            user_prompt = f"""
{self.templates["context"].format(context=context_str, situation=situation_str)}

{self.templates["prediction"].format(partial_utterance=partial_input)}

Respond with the most likely full message or need.
"""
        else:
            user_prompt = f"""
{self.templates["context"].format(context=context_str, situation=situation_str)}

{self.templates["completion"]}

Partial input: "{partial_input}"
"""

        return system_prompt, user_prompt

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context dictionary into readable string."""
        parts = []

        if "user_profile" in context:
            profile = context["user_profile"]
            parts.append(f"User: {profile.get('identity', {}).get('name', 'Unknown')}")
            parts.append(f"Condition: {profile.get('identity', {}).get('condition', 'Unknown')}")
            parts.append(
                f"Communication: {profile.get('communication_style', {}).get('style', 'Unknown')}"
            )

            medical = profile.get("medical_needs", {})
            if medical.get("equipment"):
                parts.append(f"Equipment: {', '.join(medical['equipment'])}")
            if medical.get("symptoms"):
                parts.append(f"Symptoms: {', '.join(medical['symptoms'])}")

        if "social_context" in context:
            social = context["social_context"]
            parts.append(
                f"Speaking with: {social.get('interlocutor', 'Unknown')} ({social.get('relationship', 'Unknown')})"
            )

        return "\n".join(parts)

    def _format_situation(self, context: dict[str, Any]) -> str:
        """Format current situation into readable string."""
        parts = []

        if "temporal_context" in context:
            temp = context["temporal_context"]
            parts.append(f"Time: {temp.get('current_time')} ({temp.get('time_period')})")

        if "location_context" in context:
            loc = context["location_context"]
            parts.append(f"Location: {loc.get('location_type', 'Unknown')}")

        if "conversation_context" in context:
            conv = context["conversation_context"]
            if conv.get("recent_messages"):
                parts.append(f"Recent messages: {' | '.join(conv['recent_messages'][-3:])}")

        return "\n".join(parts) if parts else "No additional context"
