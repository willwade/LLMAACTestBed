"""
Context Builder for Phase 4 Experiments

Builds contextual information for keyword-to-utterance generation
based on Dwayne's social graph and situational factors.
"""

import random
from datetime import datetime
from typing import Any


class ContextBuilder:
    """Builds contextual information for keyword predictions."""

    def __init__(self, logger=None):
        """Initialize context builder."""
        self.logger = logger

    def build_context(self, level: str, social_graph: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
        """
        Build context based on specified level.

        Args:
            level: Context level to build
            social_graph: Dwayne's social graph data
            keywords: Current keywords being processed

        Returns:
            Dictionary containing contextual information
        """
        context = {}

        if level == "location_only":
            context.update(self._get_location_context())
        elif level == "location_people":
            context.update(self._get_location_context())
            context.update(self._get_people_context(social_graph))
        elif level == "location_equipment":
            context.update(self._get_location_context())
            context.update(self._get_equipment_context(social_graph, keywords))
        elif level == "social_relationships":
            context.update(self._get_social_context(social_graph))
        elif level == "full_context":
            context.update(self._get_full_context(social_graph, keywords))

        return context

    def _get_location_context(self) -> dict[str, Any]:
        """Generate location-based context."""
        locations = ["bed", "chair", "table", "living_room", "bathroom", "kitchen"]
        positions = ["sitting", "lying_down", "tilted_forward", "tilted_back"]

        # Random but realistic location selection
        location = random.choice(locations)

        context = {
            "location": location,
            "time_of_day": self._get_time_context(),
        }

        # Add position if relevant to location
        if location in ["bed", "chair"]:
            context["position"] = random.choice(positions)

        return context

    def _get_people_context(self, social_graph: dict[str, Any]) -> dict[str, Any]:
        """Generate people present context."""
        people_present = []
        relationship_context = ""

        # Determine who might be present based on time
        current_hour = datetime.now().hour

        if 9 <= current_hour <= 17:  # Daytime
            # 70% chance of professional carer during day
            if random.random() < 0.7:
                people_present.append("professional_carer")
                relationship_context += "Professional carer with formal training is present. "

            # 30% chance of family member visiting
            if random.random() < 0.3:
                family_member = random.choice(["Kerry", "Evalyn", "Beryl"])
                people_present.append(family_member)
                relationship_context += f"{family_member} is visiting. "

        elif 16 <= current_hour <= 22:  # Afternoon/evening
            # Kerry likely present after school
            if random.random() < 0.8:
                people_present.append("Kerry")
                relationship_context += "Kerry (wife) is present. "

            # Evalyn home from school
            if current_hour >= 16 and random.random() < 0.9:
                people_present.append("Evalyn")
                relationship_context += "Evalyn (daughter) is home from school. "

        else:  # Night/early morning
            # Professional carer or Kerry
            if random.random() < 0.6:
                people_present.append("professional_carer")
                relationship_context += "Professional carer for night support. "
            else:
                people_present.append("Kerry")
                relationship_context += "Kerry providing night care. "

        if not people_present:
            people_present.append("professional_carer")
            relationship_context += "Professional carer present. "

        return {
            "people_present": ", ".join(people_present),
            "relationship_context": relationship_context.strip(),
            "caregiver_type": "family" if "Kerry" in people_present else "professional"
        }

    def _get_equipment_context(self, social_graph: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
        """Generate equipment context based on keywords and location."""
        equipment_context = ""
        nearby_equipment = []

        # Equipment commonly needed based on keywords
        keyword_equipment = {
            "Gel": "ibuprofen_gel",
            "Feed": ["feeding_pump", "rig_tube", "fortisip_feed"],
            "Cough": ["cough_assist", "suction_machine"],
            "Phone": ["laptop", "communication_ipad", "grid_software"],
            "Charge": ["phone_charger", "laptop_charger"],
            "Wipe": ["tissues", "wet_wipes"],
            "Move": ["ceiling_hoist", "transfer_equipment"],
            "Transfer": ["commode", "urinal_bottle"],
            "Sick": ["metal_bowl", "vomit_management"],
            "Table": ["adjustable_table", "overbed_table"],
            "Chair": ["electric_chair", "positioning_cushion"]
        }

        # Find equipment relevant to keywords
        for keyword in keywords:
            if keyword in keyword_equipment:
                equipment = keyword_equipment[keyword]
                if isinstance(equipment, list):
                    nearby_equipment.extend(equipment)
                else:
                    nearby_equipment.append(equipment)

        # Add standard equipment
        standard_equipment = ["suction_machine", "video_monitor", "call_bell"]
        nearby_equipment.extend(standard_equipment)

        # Remove duplicates and format
        unique_equipment = list(set(nearby_equipment))
        if unique_equipment:
            equipment_context = f"Nearby equipment: {', '.join(unique_equipment)}"

        return {
            "equipment_context": equipment_context,
            "nearby_equipment": unique_equipment
        }

    def _get_social_context(self, social_graph: dict[str, Any]) -> dict[str, Any]:
        """Generate social relationship context."""
        context_parts = []

        # Family relationships
        family = social_graph.get("family", {})
        if "immediate_family" in family:
            context_parts.append("Family: Kerry (wife), Evalyn (daughter), extended family network")

        # Care team
        care_team = social_graph.get("care_team", {})
        if "professional_carers" in care_team:
            context_parts.append("Care team: Professional carers with formal training, strong bonds formed")

        # Communication preferences
        interaction_prefs = social_graph.get("interaction_preferences", {})
        if "care_approach" in interaction_prefs:
            approach = interaction_prefs["care_approach"]
            if "communication" in approach:
                context_parts.append(f"Communication: {approach['communication']}")

        # Personality
        personality = social_graph.get("personality", {})
        if "character" in personality:
            context_parts.append(f"Personality: {personality['character']}")

        return {
            "social_context": " | ".join(context_parts),
            "relationship_dynamics": "Strong family bonds, professional carer relationships built over time"
        }

    def _get_time_context(self) -> str:
        """Get time of day context."""
        current_hour = datetime.now().hour

        if 6 <= current_hour < 9:
            return "morning_routine"
        elif 9 <= current_hour < 12:
            return "late_morning"
        elif 12 <= current_hour < 13:
            return "lunchtime"
        elif 13 <= current_hour < 16:
            return "afternoon"
        elif 16 <= current_hour < 18:
            return "after_school_family_time"
        elif 18 <= current_hour < 22:
            return "evening_family_time"
        else:
            return "night_time"

    def _get_full_context(self, social_graph: dict[str, Any], keywords: list[str]) -> dict[str, Any]:
        """Generate comprehensive context combining all elements."""
        context = {}

        # Location and time
        context.update(self._get_location_context())
        context.update(self._get_people_context(social_graph))
        context.update(self._get_equipment_context(social_graph, keywords))

        # Medical context
        medical_context = social_graph.get("medical_context", {})
        symptoms = medical_context.get("symptoms", {})

        health_status = "MND patient with "
        if isinstance(symptoms, dict):
            if "mobility" in symptoms:
                health_status += "mobility challenges, "
            if "speech" in symptoms:
                health_status += "slurred speech, "
            if "fatigue" in symptoms:
                health_status += "fatigue, "

        context["medical_context"] = health_status.rstrip(", ")

        # Recent activities based on time
        time_context = context.get("time_of_day", "afternoon")
        recent_activities = self._get_recent_activities(time_context)
        context["recent_activities"] = recent_activities

        # Communication needs
        communication = social_graph.get("communication", {})
        comm_needs = "Uses telegraphic speech, requires active listening and patience"
        if "verbal" in communication:
            ability = communication["verbal"].get("ability", "")
            if ability:
                comm_needs += f", {ability}"

        context["communication_needs"] = comm_needs

        # Personality and preferences
        personality = social_graph.get("personality", {})
        if "humour" in personality:
            context["personality_notes"] = personality["humour"]

        return context

    def _get_recent_activities(self, time_context: str) -> str:
        """Get recent activities based on time of day."""
        activities = {
            "morning_routine": "Recently completed morning hygiene and first feed",
            "late_morning": "Set up for daytime activities, family may visit",
            "lunchtime": "Just finished or about to have second feed",
            "afternoon": "Resting or engaging in daytime activities",
            "after_school_family_time": "Evalyn just returned from school, family interaction time",
            "evening_family_time": "Family dinner and relaxation time",
            "night_time": "Evening medications and bedtime preparation"
        }

        return activities.get(time_context, "Daily care routine in progress")
