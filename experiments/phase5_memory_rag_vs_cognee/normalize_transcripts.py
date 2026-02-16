#!/usr/bin/env python3
"""Normalize transcript_data_2.json into a consistent schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "synthetic" / "transcripts" / "transcript_data_2.json"
OUTPUT_PATH = REPO_ROOT / "data" / "synthetic" / "transcripts" / "transcript_data_2_improved.json"


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata", {}) or {}
    dialogue = raw.get("dialogue_history", {}) or {}

    target = (
        raw.get("target")
        or raw.get("target_ground_truth")
        or raw.get("expected")
        or raw.get("target_text")
        or ""
    )
    last_utterance = (
        dialogue.get("last_utterance")
        or raw.get("last_utterance")
        or raw.get("speech")
        or raw.get("input")
        or ""
    )
    interlocutor = dialogue.get("previous_speaker") or raw.get("interlocutor")

    participants = (
        metadata.get("active_participants")
        or raw.get("active_participants")
        or ([] if not interlocutor else [interlocutor])
    )

    return {
        "id": raw.get("id"),
        "target": target,
        "last_utterance": last_utterance,
        "interlocutor": interlocutor,
        "time": metadata.get("time") or raw.get("time"),
        "location": metadata.get("location") or raw.get("location"),
        "active_participants": participants,
        "raw": raw,
    }


def main() -> None:
    data = json.loads(INPUT_PATH.read_text())
    improved = [normalize_item(item) for item in data]
    OUTPUT_PATH.write_text(json.dumps(improved, indent=2))
    print(f"Wrote {len(improved)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
