#!/usr/bin/env python3
"""
Convert baton chat history export (flat sentences) into conversation-style data for Phase 3.

Takes the raw baton-export-2025-11-24-nofullstop.json and:
1) Sorts sentences by timestamp.
2) Splits into sessions per anonymous user, with a gap threshold to start a new chat.
3) Emits conversation records: {chat_id, user_id, messages: [...]}
4) Writes processed output to data/real/chat_history/processed/dataset.json
5) Extends chat_profile_mappings to point new chat_ids to dave_context by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

DEFAULT_RAW = Path("data/real/chat_history/raw/baton-export-2025-11-24-nofullstop.json")
DEFAULT_PROCESSED = Path("data/real/chat_history/processed/dataset.json")
DEFAULT_MAPPINGS = Path("data/synthetic/chat_profile_mappings/mappings.json")


def load_raw(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data.get("sentences", [])


def parse_ts(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        try:
            return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            return None


def build_conversations(
    sentences: list[dict[str, Any]], gap_minutes: int, default_user_id: str = "dave_context"
) -> list[dict[str, Any]]:
    # Group by anonymousUUID, sort by timestamp
    by_user: dict[str, list[dict[str, Any]]] = {}
    for s in sentences:
        user = s.get("anonymousUUID") or "unknown_user"
        by_user.setdefault(user, []).append(s)

    convs: list[dict[str, Any]] = []
    chat_index = 1

    for user, items in by_user.items():
        items.sort(
            key=lambda x: parse_ts(x.get("metadata", [{}])[0].get("timestamp")) or dt.datetime.min
        )
        current: list[dict[str, Any]] = []
        last_ts: dt.datetime | None = None

        for s in items:
            meta = s.get("metadata", [{}])[0] if s.get("metadata") else {}
            ts = parse_ts(meta.get("timestamp"))
            if last_ts and ts and (ts - last_ts).total_seconds() > gap_minutes * 60:
                if current:
                    convs.append(
                        {
                            "chat_id": f"chat_{chat_index:03d}",
                            "user_id": default_user_id,
                            "messages": current,
                        }
                    )
                    chat_index += 1
                    current = []
            last_ts = ts
            current.append(
                {
                    "role": "user",
                    "content": s.get("content", ""),
                    "timestamp": meta.get("timestamp"),
                    "latitude": meta.get("latitude"),
                    "longitude": meta.get("longitude"),
                    "uuid": s.get("uuid"),
                }
            )

        if current:
            convs.append(
                {
                    "chat_id": f"chat_{chat_index:03d}",
                    "user_id": default_user_id,
                    "messages": current,
                }
            )
            chat_index += 1

    return convs


def update_mappings(
    mappings_path: Path, chat_ids: list[str], profile: str = "dave_context"
) -> None:
    existing: dict[str, str] = {}
    if mappings_path.exists():
        existing = json.loads(mappings_path.read_text())

    for cid in chat_ids:
        existing.setdefault(cid, profile)

    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    mappings_path.write_text(json.dumps(existing, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Convert baton export to conversation dataset")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Path to raw baton export")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_PROCESSED, help="Path for processed dataset"
    )
    parser.add_argument(
        "--mappings", type=Path, default=DEFAULT_MAPPINGS, help="Path to chat-profile mappings"
    )
    parser.add_argument(
        "--gap-minutes", type=int, default=45, help="Gap (minutes) to start a new chat session"
    )
    args = parser.parse_args()

    sentences = load_raw(args.raw)
    convs = build_conversations(sentences, args.gap_minutes)

    payload = {
        "version": "1.0",
        "source": args.raw.name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "session_gap_minutes": args.gap_minutes,
        "chat_count": len(convs),
        "chats": convs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    update_mappings(args.mappings, [c["chat_id"] for c in convs])

    print(
        f"[ok] Converted {len(sentences)} sentences into {len(convs)} chats. "
        f"Saved dataset to {args.output}"
    )
    print(f"[ok] Updated mappings at {args.mappings} (default profile: dave_context)")


if __name__ == "__main__":
    main()
