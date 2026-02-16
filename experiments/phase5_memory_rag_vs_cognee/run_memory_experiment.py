#!/usr/bin/env python3
"""
Phase 5: Memory comparison (baseline vs RAG vs Cognee)

This experiment simulates conversation sessions where memory is cleared at the
start of each session. Each turn only has access to memory from previous turns
in the same session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from lib.context import ContextBuilder, ProfileManager, PromptBuilder
from lib.data import DataLoader
from lib.llm_clients import create_llm_client
from lib.utils import load_env, setup_logging

try:
    import cognee  # type: ignore

    HAS_COGNEE = True
except Exception:
    HAS_COGNEE = False


@dataclass
class Turn:
    turn_id: int | str
    last_utterance: str
    target: str
    interlocutor: str | None
    time: datetime | None
    location: str | dict[str, Any] | None
    participants: list[str]
    raw: dict[str, Any]


def _split_stage_directions(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []
    notes = [match.strip() for match in re.findall(r"\(([^()]*)\)", text) if match.strip()]
    spoken = re.sub(r"\s*\([^()]*\)", "", text).strip()
    spoken = re.sub(r"\s{2,}", " ", spoken)
    return spoken, notes


def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z ]", " ", name)
    return " ".join(name.split())


def _parse_time(time_str: str | None) -> datetime | None:
    if not time_str:
        return None
    for fmt in ["%H:%M", "%H:%M:%S"]:
        try:
            return datetime.strptime(f"2024-01-01 {time_str}", f"%Y-%m-%d {fmt}")
        except ValueError:
            continue
    return None


def _resolve_interlocutor(name: str | None, profile) -> str | None:
    if not name:
        return None
    normalized = _normalize_name(name)
    known = {_normalize_name(n): n for n in profile.get_social_graph().keys()}
    aliases = {
        "nurse sarah": "Sarah",
        "sarah": "Sarah",
        "dawn ipad": "Dawn",
    }
    if normalized in aliases:
        return aliases[normalized]
    return known.get(normalized, name)


def normalize_turn(raw: dict[str, Any], profile) -> Turn:
    metadata = raw.get("metadata", {})
    dialogue = raw.get("dialogue_history", {})

    target = (
        raw.get("target_ground_truth")
        or raw.get("target")
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
    cleaned_last_utterance, _notes = _split_stage_directions(str(last_utterance))
    interlocutor = dialogue.get("previous_speaker") or raw.get("interlocutor")
    participants = metadata.get("active_participants") or raw.get("active_participants") or []
    if not interlocutor and participants:
        interlocutor = participants[0]

    time_str = metadata.get("time") or raw.get("time")
    time = _parse_time(time_str)

    location = metadata.get("location") or raw.get("location")

    return Turn(
        turn_id=raw.get("id", "unknown"),
        last_utterance=cleaned_last_utterance,
        target=str(target),
        interlocutor=_resolve_interlocutor(interlocutor, profile),
        time=time,
        location=location,
        participants=participants,
        raw=raw,
    )


def group_sessions(turns: list[Turn], max_gap_minutes: int = 30) -> list[list[Turn]]:
    sessions: list[list[Turn]] = []
    current: list[Turn] = []
    prev_time: datetime | None = None
    prev_key: tuple[str, tuple[str, ...]] | None = None

    for turn in turns:
        participants_key = tuple(sorted(_normalize_name(p) for p in turn.participants))
        loc = turn.location
        loc_key = _normalize_name(str(loc)) if loc is not None else ""
        key = (loc_key, participants_key)

        gap_minutes = None
        if prev_time and turn.time:
            gap_minutes = (turn.time - prev_time).total_seconds() / 60

        is_new_session = (
            not current
            or (gap_minutes is not None and gap_minutes > max_gap_minutes)
            or (prev_key is not None and key != prev_key)
        )

        if is_new_session:
            if current:
                sessions.append(current)
            current = []

        current.append(turn)
        prev_time = turn.time
        prev_key = key

    if current:
        sessions.append(current)

    return sessions


def render_social_graph(profile) -> list[str]:
    texts = []
    social_graph = profile.get_social_graph()
    for name, info in social_graph.items():
        relation = info.get("relation") or info.get("type") or "unknown"
        personality = info.get("personality") or ""
        topics = info.get("topics") or info.get("hobbies") or []
        dynamic = info.get("dynamic") or info.get("notes") or ""
        pieces = [f"{name}: relation={relation}"]
        if personality:
            pieces.append(f"personality={personality}")
        if topics:
            pieces.append(f"topics={', '.join(topics)}")
        if dynamic:
            pieces.append(f"notes={dynamic}")
        texts.append("; ".join(pieces))

    prefs = profile.get_vocabulary_preferences()
    if prefs:
        texts.append(f"Communication prefs: {json.dumps(prefs)}")

    events = profile.get_recent_events()
    if events:
        texts.append(f"Recent events: {json.dumps(events)}")

    return texts


def format_turn_memory(turn: Turn) -> str:
    speaker = turn.interlocutor or "partner"
    return f"Partner ({speaker}): {turn.last_utterance}\nUser reply: {turn.target}"


class RagMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.texts: list[str] = []
        self.embeddings: np.ndarray | None = None

    def clear(self) -> None:
        self.texts = []
        self.embeddings = None

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        embedding = self.model.encode([text], normalize_embeddings=True)
        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.texts.append(text)

    def search(self, query: str) -> list[str]:
        if not self.texts or self.embeddings is None:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_vec.T).squeeze()
        idxs = np.argsort(scores)[::-1][: self.top_k]
        return [self.texts[i] for i in idxs]


def build_prompts(prompt_builder: PromptBuilder, partial_input: str, context: dict[str, Any], memory: list[str]) -> tuple[str, str]:
    system_prompt, user_prompt = prompt_builder.build_prompt(
        partial_input=partial_input, context=context, task="completion"
    )

    if memory:
        memory_block = "\n".join(f"- {m}" for m in memory)
        user_prompt = (
            f"{user_prompt}\n\nRetrieved memory (use only if relevant):\n{memory_block}\n"
        )

    return system_prompt, user_prompt


def collect_conversation_history(history: list[str], turn: Turn) -> list[str]:
    history = list(history)
    if turn.last_utterance:
        history.append(f"Partner: {turn.last_utterance}")
    return history


def timed_generate(llm_client, system_prompt: str, user_prompt: str, temperature: float) -> tuple[str, float]:
    start = time.perf_counter()
    prediction = llm_client.generate(user_prompt, system_prompt, temperature=temperature)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return prediction, elapsed_ms


def timed_judge(llm_client, target: str, prediction: str) -> tuple[int, float]:
    start = time.perf_counter()
    score = llm_client.judge_similarity(target, prediction)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return score, elapsed_ms


def approx_tokens(prompt: str, response: str) -> int:
    return int((len(prompt) + len(response)) / 4)


async def cognee_clear_memory() -> None:
    if not HAS_COGNEE:
        return
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def extract_cognee_snippets(results: list[Any], max_snippets: int = 3) -> list[str]:
    snippets = []
    for result in results[:max_snippets]:
        if isinstance(result, dict):
            text = (
                result.get("text")
                or result.get("content")
                or result.get("chunk")
                or result.get("node", {}).get("text")
                or json.dumps(result)
            )
        else:
            text = str(result)
        text = text.strip()
        if text:
            snippets.append(text[:500])
    return snippets


def extract_keywords(text: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "or",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "my",
        "me",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "is",
        "are",
        "was",
        "were",
        "be",
        "please",
        "now",
        "here",
        "this",
        "that",
    }
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
    return list(dict.fromkeys(keywords))


def keyword_recall(target: str, prediction: str) -> float | None:
    target_keys = extract_keywords(target)
    if not target_keys:
        return None
    pred_lower = prediction.lower()
    hits = sum(1 for key in target_keys if key in pred_lower)
    return hits / len(target_keys)


def bootstrap_mean_ci(values: list[float], seed: int = 1337, samples: int = 2000) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    data = np.array(values, dtype=float)
    means = []
    for _ in range(samples):
        resample = rng.choice(data, size=len(data), replace=True)
        means.append(float(np.mean(resample)))
    means.sort()
    return {
        "mean": float(np.mean(data)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def compute_paired_stats(
    results: list[dict[str, Any]],
    baseline_mode: str,
    compare_mode: str,
    metric: str,
) -> dict[str, Any]:
    baseline = {}
    compare = {}
    for row in results:
        key = (row["session_id"], row["turn_id"])
        if row["memory_mode"] == baseline_mode and row.get(metric) is not None:
            baseline[key] = float(row[metric])
        if row["memory_mode"] == compare_mode and row.get(metric) is not None:
            compare[key] = float(row[metric])

    paired = []
    for key, base_val in baseline.items():
        if key in compare:
            paired.append(compare[key] - base_val)

    if not paired:
        return {"count": 0}

    t_stat, p_val = stats.ttest_rel(
        [baseline[k] for k in baseline if k in compare],
        [compare[k] for k in compare if k in baseline],
    )
    ci = bootstrap_mean_ci(paired)
    return {
        "count": len(paired),
        "mean_diff": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    load_env(args.env_file)

    logger = setup_logging(name="phase5_memory", level="INFO")
    llm_client = create_llm_client(provider=args.provider, model=args.model)
    prompt_builder = PromptBuilder(template_style="aac")

    profile_manager = ProfileManager()
    profile = profile_manager.load_profile(args.profile)
    context_builder = ContextBuilder(profile=profile)

    transcripts = DataLoader.load_transcript(args.transcripts)
    turns = [normalize_turn(t, profile) for t in transcripts]
    turns = [t for t in turns if t.last_utterance and t.target]

    sessions = group_sessions(turns, max_gap_minutes=args.session_gap_minutes)
    logger.info(f"Loaded {len(turns)} turns across {len(sessions)} sessions")

    rag_memory = RagMemory(model_name=args.embedding_model, top_k=args.top_k)
    static_memory = render_social_graph(profile)
    global_memory_pool = [format_turn_memory(t) for t in turns]
    rng = random.Random(args.random_seed)

    results: list[dict[str, Any]] = []

    for session_idx, session in enumerate(sessions, start=1):
        logger.info(f"Session {session_idx}: {len(session)} turns")

        rag_memory.clear()
        for item in static_memory:
            rag_memory.add(item)

        if args.use_cognee and HAS_COGNEE:
            await cognee_clear_memory()
            await cognee.add(static_memory)
            await cognee.cognify()
            if args.cognee_memify:
                await cognee.memify()

        conversation_history: list[str] = []
        session_memory_texts: list[str] = []

        for turn_idx, turn in enumerate(session, start=1):
            if args.max_turns and turn_idx > args.max_turns:
                break

            conversation_context = collect_conversation_history(conversation_history, turn)
            context = context_builder.build_context(
                current_time=turn.time,
                location=turn.location if isinstance(turn.location, dict) else None,
                interlocutor=turn.interlocutor,
                conversation_history=conversation_context,
                context_levels=["profile", "time", "social"],
            )

            # Baseline (no memory)
            system_prompt, user_prompt = build_prompts(
                prompt_builder, turn.last_utterance, context, memory=[]
            )
            baseline_pred, baseline_gen_ms = timed_generate(
                llm_client, system_prompt, user_prompt, args.temperature
            )
            baseline_score, baseline_judge_ms = timed_judge(
                llm_client, turn.target, baseline_pred
            )
            baseline_recall = keyword_recall(turn.target, baseline_pred)
            baseline_prompt_chars = len(system_prompt) + len(user_prompt)
            baseline_pred_chars = len(baseline_pred)
            results.append(
                {
                    "session_id": session_idx,
                    "turn_id": turn.turn_id,
                    "memory_mode": "baseline",
                    "prediction": baseline_pred,
                    "score": baseline_score,
                    "keyword_recall": baseline_recall,
                    "gen_latency_ms": baseline_gen_ms,
                    "judge_latency_ms": baseline_judge_ms,
                    "prompt_chars": baseline_prompt_chars,
                    "response_chars": baseline_pred_chars,
                    "approx_tokens": approx_tokens(user_prompt, baseline_pred),
                    "target": turn.target,
                    "last_utterance": turn.last_utterance,
                    "interlocutor": turn.interlocutor,
                }
            )

            # RAG memory
            rag_query = f"{turn.last_utterance}\nInterlocutor: {turn.interlocutor or 'unknown'}"
            rag_search_start = time.perf_counter()
            rag_snippets = rag_memory.search(rag_query)
            rag_search_ms = (time.perf_counter() - rag_search_start) * 1000
            system_prompt, user_prompt = build_prompts(
                prompt_builder, turn.last_utterance, context, memory=rag_snippets
            )
            rag_pred, rag_gen_ms = timed_generate(
                llm_client, system_prompt, user_prompt, args.temperature
            )
            rag_score, rag_judge_ms = timed_judge(llm_client, turn.target, rag_pred)
            rag_recall = keyword_recall(turn.target, rag_pred)
            rag_prompt_chars = len(system_prompt) + len(user_prompt)
            rag_pred_chars = len(rag_pred)
            results.append(
                {
                    "session_id": session_idx,
                    "turn_id": turn.turn_id,
                    "memory_mode": "rag",
                    "prediction": rag_pred,
                    "score": rag_score,
                    "keyword_recall": rag_recall,
                    "search_latency_ms": rag_search_ms,
                    "gen_latency_ms": rag_gen_ms,
                    "judge_latency_ms": rag_judge_ms,
                    "prompt_chars": rag_prompt_chars,
                    "response_chars": rag_pred_chars,
                    "approx_tokens": approx_tokens(user_prompt, rag_pred),
                    "retrieved_memory_count": len(rag_snippets),
                    "target": turn.target,
                    "last_utterance": turn.last_utterance,
                    "interlocutor": turn.interlocutor,
                }
            )

            # Random memory (same count)
            random_candidates = list(global_memory_pool)
            rng.shuffle(random_candidates)
            random_snippets = random_candidates[: len(rag_snippets)]
            system_prompt, user_prompt = build_prompts(
                prompt_builder, turn.last_utterance, context, memory=random_snippets
            )
            random_pred, random_gen_ms = timed_generate(
                llm_client, system_prompt, user_prompt, args.temperature
            )
            random_score, random_judge_ms = timed_judge(
                llm_client, turn.target, random_pred
            )
            random_recall = keyword_recall(turn.target, random_pred)
            random_prompt_chars = len(system_prompt) + len(user_prompt)
            random_pred_chars = len(random_pred)
            results.append(
                {
                    "session_id": session_idx,
                    "turn_id": turn.turn_id,
                    "memory_mode": "random_memory",
                    "prediction": random_pred,
                    "score": random_score,
                    "keyword_recall": random_recall,
                    "gen_latency_ms": random_gen_ms,
                    "judge_latency_ms": random_judge_ms,
                    "prompt_chars": random_prompt_chars,
                    "response_chars": random_pred_chars,
                    "approx_tokens": approx_tokens(user_prompt, random_pred),
                    "retrieved_memory_count": len(random_snippets),
                    "target": turn.target,
                    "last_utterance": turn.last_utterance,
                    "interlocutor": turn.interlocutor,
                }
            )

            # Shuffled memory (exclude current session memory if possible)
            shuffled_candidates = [m for m in global_memory_pool if m not in session_memory_texts]
            if len(shuffled_candidates) < len(rag_snippets):
                shuffled_candidates = list(global_memory_pool)
            rng.shuffle(shuffled_candidates)
            shuffled_snippets = shuffled_candidates[: len(rag_snippets)]
            system_prompt, user_prompt = build_prompts(
                prompt_builder, turn.last_utterance, context, memory=shuffled_snippets
            )
            shuffled_pred, shuffled_gen_ms = timed_generate(
                llm_client, system_prompt, user_prompt, args.temperature
            )
            shuffled_score, shuffled_judge_ms = timed_judge(
                llm_client, turn.target, shuffled_pred
            )
            shuffled_recall = keyword_recall(turn.target, shuffled_pred)
            shuffled_prompt_chars = len(system_prompt) + len(user_prompt)
            shuffled_pred_chars = len(shuffled_pred)
            results.append(
                {
                    "session_id": session_idx,
                    "turn_id": turn.turn_id,
                    "memory_mode": "shuffled_memory",
                    "prediction": shuffled_pred,
                    "score": shuffled_score,
                    "keyword_recall": shuffled_recall,
                    "gen_latency_ms": shuffled_gen_ms,
                    "judge_latency_ms": shuffled_judge_ms,
                    "prompt_chars": shuffled_prompt_chars,
                    "response_chars": shuffled_pred_chars,
                    "approx_tokens": approx_tokens(user_prompt, shuffled_pred),
                    "retrieved_memory_count": len(shuffled_snippets),
                    "target": turn.target,
                    "last_utterance": turn.last_utterance,
                    "interlocutor": turn.interlocutor,
                }
            )

            # Cognee memory
            if args.use_cognee and HAS_COGNEE:
                cognee_search_start = time.perf_counter()
                cognee_results = await cognee.search(
                    query_text=rag_query, top_k=args.top_k
                )
                cognee_search_ms = (time.perf_counter() - cognee_search_start) * 1000
                cognee_snippets = extract_cognee_snippets(cognee_results, args.top_k)
                system_prompt, user_prompt = build_prompts(
                    prompt_builder, turn.last_utterance, context, memory=cognee_snippets
                )
                cognee_pred, cognee_gen_ms = timed_generate(
                    llm_client, system_prompt, user_prompt, args.temperature
                )
                cognee_score, cognee_judge_ms = timed_judge(
                    llm_client, turn.target, cognee_pred
                )
                cognee_recall = keyword_recall(turn.target, cognee_pred)
                cognee_prompt_chars = len(system_prompt) + len(user_prompt)
                cognee_pred_chars = len(cognee_pred)
                results.append(
                    {
                        "session_id": session_idx,
                        "turn_id": turn.turn_id,
                        "memory_mode": "cognee",
                        "prediction": cognee_pred,
                        "score": cognee_score,
                        "keyword_recall": cognee_recall,
                        "search_latency_ms": cognee_search_ms,
                        "gen_latency_ms": cognee_gen_ms,
                        "judge_latency_ms": cognee_judge_ms,
                        "prompt_chars": cognee_prompt_chars,
                        "response_chars": cognee_pred_chars,
                        "approx_tokens": approx_tokens(user_prompt, cognee_pred),
                        "retrieved_memory_count": len(cognee_snippets),
                        "target": turn.target,
                        "last_utterance": turn.last_utterance,
                        "interlocutor": turn.interlocutor,
                    }
                )

            # Add this turn to memory for future turns
            memory_text = format_turn_memory(turn)
            rag_memory.add(memory_text)
            session_memory_texts.append(memory_text)
            if args.use_cognee and HAS_COGNEE:
                await cognee.add([memory_text])
                if args.cognee_update_each_turn:
                    await cognee.cognify()
                    if args.cognee_memify:
                        await cognee.memify()

            conversation_history.append(f"User: {turn.target}")

    summary = summarize_results(results)
    return {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "use_cognee": args.use_cognee and HAS_COGNEE,
            "session_gap_minutes": args.session_gap_minutes,
        },
        "summary": summary,
        "results": results,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[float]] = {}
    by_mode_recall: dict[str, list[float]] = {}
    by_mode_latency: dict[str, list[float]] = {}
    by_mode_tokens: dict[str, list[float]] = {}
    for row in results:
        by_mode.setdefault(row["memory_mode"], []).append(float(row["score"]))
        if row.get("keyword_recall") is not None:
            by_mode_recall.setdefault(row["memory_mode"], []).append(
                float(row["keyword_recall"])
            )
        if row.get("gen_latency_ms") is not None:
            by_mode_latency.setdefault(row["memory_mode"], []).append(
                float(row["gen_latency_ms"])
            )
        if row.get("approx_tokens") is not None:
            by_mode_tokens.setdefault(row["memory_mode"], []).append(
                float(row["approx_tokens"])
            )

    summary = {}
    for mode, scores in by_mode.items():
        if scores:
            summary[mode] = {
                "count": len(scores),
                "avg_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
            }
            if mode in by_mode_recall and by_mode_recall[mode]:
                summary[mode]["avg_keyword_recall"] = float(
                    np.mean(by_mode_recall[mode])
                )
            if mode in by_mode_latency and by_mode_latency[mode]:
                summary[mode]["avg_gen_latency_ms"] = float(
                    np.mean(by_mode_latency[mode])
                )
            if mode in by_mode_tokens and by_mode_tokens[mode]:
                summary[mode]["avg_approx_tokens"] = float(
                    np.mean(by_mode_tokens[mode])
                )

    if "baseline" in summary and "rag" in summary:
        summary["rag_vs_baseline"] = summary["rag"]["avg_score"] - summary["baseline"]["avg_score"]
    if "baseline" in summary and "cognee" in summary:
        summary["cognee_vs_baseline"] = summary["cognee"]["avg_score"] - summary["baseline"]["avg_score"]
    if "baseline" in summary and "random_memory" in summary:
        summary["random_vs_baseline"] = summary["random_memory"]["avg_score"] - summary["baseline"]["avg_score"]
    if "baseline" in summary and "shuffled_memory" in summary:
        summary["shuffled_vs_baseline"] = summary["shuffled_memory"]["avg_score"] - summary["baseline"]["avg_score"]

    summary["paired_stats"] = {
        "rag_score": compute_paired_stats(results, "baseline", "rag", "score"),
        "random_score": compute_paired_stats(results, "baseline", "random_memory", "score"),
        "shuffled_score": compute_paired_stats(results, "baseline", "shuffled_memory", "score"),
        "cognee_score": compute_paired_stats(results, "baseline", "cognee", "score"),
        "rag_keyword_recall": compute_paired_stats(
            results, "baseline", "rag", "keyword_recall"
        ),
        "random_keyword_recall": compute_paired_stats(
            results, "baseline", "random_memory", "keyword_recall"
        ),
        "shuffled_keyword_recall": compute_paired_stats(
            results, "baseline", "shuffled_memory", "keyword_recall"
        ),
        "cognee_keyword_recall": compute_paired_stats(
            results, "baseline", "cognee", "keyword_recall"
        ),
    }

    return summary


def write_reports(output_path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    df = pd.DataFrame(results)
    csv_path = output_path.with_suffix(".csv")
    html_path = output_path.with_suffix(".html")

    df.to_csv(csv_path, index=False)

    summary_table = pd.DataFrame(
        [
            {"memory_mode": k, **v}
            for k, v in summary.items()
            if isinstance(v, dict) and k not in {"paired_stats"}
        ]
    )

    paired_stats = summary.get("paired_stats", {})
    paired_table = pd.DataFrame(
        [
            {"comparison": k, **v}
            for k, v in paired_stats.items()
            if isinstance(v, dict)
        ]
    )

    with html_path.open("w", encoding="utf-8") as f:
        f.write("<html><head><title>Phase5 Memory Report</title></head><body>")
        f.write("<h1>Phase 5 Memory Report</h1>")
        f.write("<h2>Summary</h2>")
        f.write(summary_table.to_html(index=False))
        f.write("<h2>Paired Stats</h2>")
        f.write(paired_table.to_html(index=False))
        f.write("<h2>Sample Results</h2>")
        sample = df.sample(min(20, len(df)), random_state=1337)
        f.write(sample.to_html(index=False))
        f.write("</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: memory experiment")
    parser.add_argument(
        "--profile",
        type=str,
        default="data/synthetic/profiles/dave_context.json",
        help="Path to user profile",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        default="data/synthetic/transcripts/transcript_data_2_improved_expanded.json",
        help="Path to transcript scenarios",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--use-cognee", action="store_true")
    parser.add_argument("--cognee-memify", action="store_true")
    parser.add_argument("--cognee-update-each-turn", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (defaults to .env.local if present)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase5_memory_comparison.json",
        help="Output JSON path",
    )

    args = parser.parse_args()
    args.max_turns = args.max_turns or None

    if args.env_file is None:
        default_env = Path(".env.local")
        args.env_file = str(default_env) if default_env.exists() else None

    result = asyncio.run(run_experiment(args))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    write_reports(output_path, result["results"], result["summary"])

    print(f"Saved results to {output_path}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
