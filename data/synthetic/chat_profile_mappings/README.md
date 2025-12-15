# Chat Profile Mappings

This directory contains mappings between real chat data and synthetic user profiles.

## Purpose

These JSON files map chat IDs from the real chat history to user profile IDs from the synthetic profiles. This allows us to enhance real chat data with rich profile context for Phase 3 experiments.

## File Format

```json
{
  "chat_001": "user_123",
  "chat_002": "user_456",
  ...
}
```

## Usage

- Used by `experiments/phase3_combined/experiments/profile_enhanced.py`
- Enables linking real conversations with synthetic profiles
- Maintains privacy by not storing actual user identities

## Creation Process

Mappings should be created based on:
1. Chat patterns and linguistic features
2. Conversation topics and complexity
3. Communication frequency and timing
4. Other observable characteristics that can be matched to synthetic profiles

## Privacy

- No real user identifiers are stored
- Mappings are anonymized
- Original chat data remains private and gitignored