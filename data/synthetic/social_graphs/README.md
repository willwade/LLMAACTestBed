# Social Graphs

This directory contains synthetic social relationship graphs for AAC users.

## Purpose

Social graphs define the relationships and communication patterns for AAC users, enabling:
- Social context-aware predictions
- Energy cost calculations for interactions
- Relationship-based response generation
- Social appropriateness evaluation

## File Format

Each JSON file represents one user's social graph:

```json
{
  "user_id": "user_123",
  "relationships": {
    "caregiver_1": {
      "type": "caregiver",
      "closeness": 5,
      "frequency": "daily",
      "communication_style": "supportive",
      "topics": ["health", "daily_care", "emotions"]
    },
    "family_member": {
      "type": "family",
      "closeness": 4,
      "frequency": "weekly",
      "communication_style": "casual",
      "topics": ["family_news", "personal_stories"]
    }
  },
  "communication_preferences": {
    "style": "mixed",
    "formality": "context_dependent",
    "topics": ["technology", "hobbies", "current_events"]
  },
  "energy_costs": {
    "base": 1.0,
    "close_friends": 0.6,
    "family": 0.7,
    "caregivers": 0.8,
    "strangers": 1.5
  }
}
```

## Relationship Types

- **caregiver**: Healthcare professionals, support workers
- **family**: Immediate and extended family members
- **friends**: Close and casual friends
- **professional**: Therapists, doctors, colleagues
- **service**: Customer service interactions

## Closeness Scale

1: Distant/formal relationship
2: Acquaintance
3: Casual contact
4: Regular contact
5: Very close/intimate

## Usage in Experiments

- Used by `experiments/phase3_combined/experiments/social_context.py`
- Determines context to inject into prompts
- Calculates interaction energy costs
- Guides response appropriateness