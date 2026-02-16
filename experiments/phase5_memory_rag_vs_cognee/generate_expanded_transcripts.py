#!/usr/bin/env python3
"""Generate an expanded synthetic transcript set with memory dependencies."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "data" / "synthetic" / "profiles" / "dave_context.json"
BASE_PATH = REPO_ROOT / "data" / "synthetic" / "transcripts" / "transcript_data_2_improved.json"
OUTPUT_PATH = REPO_ROOT / "data" / "synthetic" / "transcripts" / "transcript_data_2_improved_expanded.json"

SEED = 1337
TOTAL_TURNS = 60


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    random.seed(SEED)
    profile = load_json(PROFILE_PATH)
    base = load_json(BASE_PATH)

    common_requests = profile.get("vocabulary_preferences", {}).get("common_requests", [])
    equipment = profile.get("medical_context", {}).get("equipment", [])
    symptoms = profile.get("medical_context", {}).get("symptoms", [])

    interlocutors = ["Kelsey", "Sarah", "Dawn"]
    locations = ["Living Room", "Bedroom", "Clinic"]

    templates = [
        "Do you want the {thing}?",
        "Same as before?",
        "Need anything right now?",
        "Should I do the usual?",
        "Are you still feeling {symptom}?",
        "Want the {equipment} adjusted?",
        "Do you want me to turn that on?",
        "Is this what you mean?",
    ]

    now = datetime(2024, 1, 1, 8, 0)
    sessions = 4
    turns_per_session = TOTAL_TURNS // sessions

    expanded = []
    turn_id = 1000

    for s in range(sessions):
        location = random.choice(locations)
        interlocutor = random.choice(interlocutors)
        last_request = None
        session_start = now + timedelta(hours=2 * s)

        for t in range(turns_per_session):
            turn_id += 1
            time_str = (session_start + timedelta(minutes=7 * t)).strftime("%H:%M")

            template = random.choice(templates)
            thing = random.choice(common_requests) if common_requests else "help"
            symptom = random.choice(symptoms) if symptoms else "tired"
            equip = random.choice(equipment) if equipment else "mask"

            if "Same as before" in template or "usual" in template:
                if last_request:
                    target = last_request
                else:
                    target = thing
            elif "symptom" in template:
                target = "Pain" if "pain" in symptom.lower() else "Help"
            elif "equipment" in template:
                target = equip
            elif "turn that on" in template:
                target = "Fan on" if "fan" in thing.lower() else thing
            else:
                target = thing

            last_request = target

            last_utterance = template.format(thing=thing, symptom=symptom, equipment=equip)

            expanded.append(
                {
                    "id": turn_id,
                    "target": target,
                    "last_utterance": last_utterance,
                    "interlocutor": interlocutor,
                    "time": time_str,
                    "location": location,
                    "active_participants": [interlocutor],
                }
            )

    # Prepend base examples for variety
    expanded = base + expanded

    OUTPUT_PATH.write_text(json.dumps(expanded, indent=2))
    print(f"Wrote {len(expanded)} turns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
