#!/usr/bin/env python3
"""Generate a presentation-ready HTML report from results JSON files."""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_HTML = RESULTS_DIR / "results_report.html"

EXPECTED_EXPERIMENTS = {
    "phase1_chat_history": "Phase 1: chat history retrieval",
    "phase2_context": "Phase 2: context hypotheses",
    "phase3_profile": "Phase 3: profile-enhanced",
    "phase3_social": "Phase 3: social context",
    "phase4_keyword": "Phase 4: keyword → utterance",
    "phase5_memory": "Phase 5: memory (baseline/RAG/Cognee)",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def summarize_phase1(files: list[Path]) -> tuple[str, list[str]]:
    if not files:
        return "<p>No Phase 1 results found.</p>", []

    frames = []
    for path in files:
        data = load_json(path)
        if isinstance(data, list):
            frames.append(pd.DataFrame(data))

    if not frames:
        return "<p>No Phase 1 results found.</p>", []

    df = pd.concat(frames, ignore_index=True)
    charts = []

    # Chart: mean judge score by context_filter
    if "context_filter" in df.columns and "llm_judge_score" in df.columns:
        pivot = df.groupby("context_filter")[["llm_judge_score"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(pivot["context_filter"], pivot["llm_judge_score"], color="#4C78A8")
        ax.set_title("Phase 1: Mean LLM Judge Score by Context Filter")
        ax.set_ylabel("Mean score")
        ax.set_xlabel("Context filter")
        ax.set_ylim(0, 10)
        charts.append(fig_to_base64(fig))

    # Chart: embedding similarity by generation method
    if "generation_method" in df.columns and "embedding_similarity" in df.columns:
        pivot = df.groupby("generation_method")[["embedding_similarity"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(pivot["generation_method"], pivot["embedding_similarity"], color="#F58518")
        ax.set_title("Phase 1: Mean Embedding Similarity by Generation Method")
        ax.set_ylabel("Mean similarity")
        ax.set_xlabel("Generation method")
        charts.append(fig_to_base64(fig))

    summary = f"""
    <p><strong>Records:</strong> {len(df):,}</p>
    <p><strong>Context filters:</strong> {', '.join(sorted(df['context_filter'].dropna().unique().tolist()))}</p>
    <p><strong>Generation methods:</strong> {', '.join(sorted(df['generation_method'].dropna().unique().tolist()))}</p>
    """
    return summary, charts


def summarize_phase3(files: list[Path]) -> tuple[str, list[str]]:
    if not files:
        return "<p>No Phase 3 results found.</p>", []

    charts = []
    rows = []
    for path in files:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        exp_type = data.get("experiment_type", "unknown")
        if exp_type in ("social_context", "profile_enhanced"):
            base_scores = data.get("baseline_scores", [])
            enh_scores = data.get("social_scores", data.get("enhanced_scores", []))
            rows.append(
                {
                    "experiment_type": exp_type,
                    "total_chats": data.get("total_chats"),
                    "enhanced": data.get("social_enhanced", data.get("enhanced_chats")),
                    "mean_improvement": data.get("mean_improvement"),
                    "improvement_rate": data.get("improvement_rate"),
                }
            )

            if base_scores and enh_scores:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(base_scores, bins=10, alpha=0.6, label="Baseline", color="#54A24B")
                ax.hist(enh_scores, bins=10, alpha=0.6, label="Enhanced", color="#E45756")
                ax.set_title(f"Phase 3 ({exp_type}): Score Distribution")
                ax.set_xlabel("Score")
                ax.set_ylabel("Count")
                ax.legend()
                charts.append(fig_to_base64(fig))

    if not rows:
        return "<p>No Phase 3 results found.</p>", []

    df = pd.DataFrame(rows)
    summary = df.to_html(index=False)
    return summary, charts


def summarize_phase2(files: list[Path]) -> tuple[str, list[str]]:
    if not files:
        return "<p>No Phase 2 results found.</p>", []

    charts = []
    frames = []
    for path in files:
        try:
            if path.suffix.lower() == ".json":
                data = load_json(path)
                if isinstance(data, list):
                    frames.append(pd.DataFrame(data))
            elif path.suffix.lower() == ".csv":
                frames.append(pd.read_csv(path))
        except Exception:
            continue

    if not frames:
        return "<p>No parseable Phase 2 results found.</p>", []

    df = pd.concat(frames, ignore_index=True)
    summary = [f"<p><strong>Rows:</strong> {len(df):,}</p>"]

    score_col = "score" if "score" in df.columns else "smart_score" if "smart_score" in df.columns else None
    if "hypothesis" in df.columns and score_col:
        pivot = df.groupby("hypothesis")[[score_col]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(pivot["hypothesis"], pivot[score_col], color="#4C78A8")
        ax.set_title("Phase 2: Mean Score by Hypothesis")
        ax.set_ylabel("Mean score")
        ax.set_xlabel("Hypothesis")
        charts.append(fig_to_base64(fig))
        summary.append(f"<p><strong>Hypotheses detected:</strong> {', '.join(pivot['hypothesis'].astype(str).tolist())}</p>")

    if "improvement" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["improvement"].dropna(), bins=12, color="#F58518")
        ax.set_title("Phase 2: Improvement Distribution")
        ax.set_xlabel("Improvement")
        ax.set_ylabel("Count")
        charts.append(fig_to_base64(fig))
        summary.append(f"<p><strong>Mean improvement:</strong> {df['improvement'].mean():.4f}</p>")

    return "".join(summary), charts


def summarize_phase4(files: list[Path]) -> tuple[str, list[str]]:
    if not files:
        return "<p>No Phase 4 results found.</p>", []

    charts = []
    summaries: list[str] = []
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            data = load_json(path)
        except Exception:
            continue

        if isinstance(data, dict) and "summary_statistics" in data:
            stats = data["summary_statistics"]
            stats_rows = []
            for part, values in stats.items():
                if isinstance(values, dict):
                    stats_rows.append({"part": part, **values})
            if stats_rows:
                df = pd.DataFrame(stats_rows)
                summaries.append(df.to_html(index=False))
                if "mean_score" in df.columns:
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.bar(df["part"], df["mean_score"], color="#72B7B2")
                    ax.set_title("Phase 4: Mean Score by Experiment Part")
                    ax.set_ylabel("Mean score")
                    ax.set_xlabel("Part")
                    charts.append(fig_to_base64(fig))
        elif isinstance(data, dict):
            summaries.append(f"<pre>{json.dumps(data, indent=2)[:4000]}</pre>")

    if not summaries:
        return "<p>No parseable Phase 4 results found.</p>", []

    return "".join(summaries), charts


def summarize_phase5(files: list[Path]) -> tuple[str, list[str]]:
    if not files:
        return "<p>No Phase 5 results found.</p>", []

    charts = []
    summaries = []
    for path in files:
        data = load_json(path)
        if not isinstance(data, dict) or "summary" not in data:
            continue
        summary = data["summary"]
        summaries.append(summary)

        # If per-mode averages exist
        mode_rows = []
        for mode, values in summary.items():
            if not isinstance(values, dict):
                continue
            if "avg_score" in values:
                mode_rows.append({"memory_mode": mode, **values})

        if mode_rows:
            df = pd.DataFrame(mode_rows)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(df["memory_mode"], df["avg_score"], color="#72B7B2")
            ax.set_title("Phase 5: Avg Score by Memory Mode")
            ax.set_ylabel("Avg score")
            ax.set_xlabel("Memory mode")
            ax.set_ylim(0, 10)
            charts.append(fig_to_base64(fig))

            if "avg_keyword_recall" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(df["memory_mode"], df["avg_keyword_recall"], color="#B279A2")
                ax.set_title("Phase 5: Avg Keyword Recall by Mode")
                ax.set_ylabel("Avg keyword recall")
                ax.set_xlabel("Memory mode")
                ax.set_ylim(0, 1)
                charts.append(fig_to_base64(fig))

    if not summaries:
        return "<p>No Phase 5 results found.</p>", []

    summary_html = "".join(
        f"<pre>{json.dumps(s, indent=2)}</pre>" for s in summaries
    )
    return summary_html, charts


def find_results() -> dict[str, list[Path]]:
    phase1 = []
    phase2 = []
    phase3 = []
    phase4 = []
    phase5 = []

    for path in RESULTS_DIR.rglob("*.json"):
        if path.name.startswith("phase3_"):
            phase3.append(path)
        elif path.name.endswith("results.json"):
            phase1.append(path)
        elif path.name.startswith("phase5_") or "memory" in path.name:
            phase5.append(path)
        elif "phase2" in path.name or "hypothesis" in path.name or "ablation" in path.name:
            phase2.append(path)
        elif "phase4" in path.name or "keyword" in path.name:
            phase4.append(path)

    # Look for phase2/4 outputs in synthetic outputs if present
    outputs_dir = REPO_ROOT / "data" / "synthetic" / "outputs"
    if outputs_dir.exists():
        for path in outputs_dir.rglob("*.*"):
            name = path.name.lower()
            if name.endswith(".csv") or name.endswith(".json"):
                if "hypothesis" in name or "ablation" in name or "temporal" in name:
                    phase2.append(path)
                if "keyword" in name or "phase4" in name:
                    phase4.append(path)

    return {"phase1": phase1, "phase2": phase2, "phase3": phase3, "phase4": phase4, "phase5": phase5}


def render_report() -> None:
    results = find_results()
    missing = []

    if not results["phase1"]:
        missing.append("Phase 1: chat history retrieval")
    if not results["phase2"]:
        missing.append("Phase 2: context hypotheses")
    if not results["phase3"]:
        missing.append("Phase 3: profile/social context")
    if not results["phase4"]:
        missing.append("Phase 4: keyword → utterance")
    if not results["phase5"]:
        missing.append("Phase 5: memory")

    phase1_summary, phase1_charts = summarize_phase1(results["phase1"])
    phase2_summary, phase2_charts = summarize_phase2(results["phase2"])
    phase3_summary, phase3_charts = summarize_phase3(results["phase3"])
    phase4_summary, phase4_charts = summarize_phase4(results["phase4"])
    phase5_summary, phase5_charts = summarize_phase5(results["phase5"])

    html_parts = []
    html_parts.append("<html><head><title>LLMAAC Experiment Report</title>")
    html_parts.append(
        "<style>"
        "body{font-family:Arial, sans-serif; margin:40px;} "
        "h1,h2{color:#111;} "
        "img{max-width:800px; display:block; margin:16px 0;} "
        ".section{margin-bottom:32px;}"
        "table{border-collapse:collapse;} "
        "table,th,td{border:1px solid #ccc;padding:6px;}"
        "pre{background:#f7f7f7;padding:12px;border:1px solid #e0e0e0;}"
        "</style></head><body>"
    )
    html_parts.append(f"<h1>LLMAAC Experiment Report</h1><p>Generated {datetime.now().isoformat()}</p>")

    html_parts.append("<div class='section'><h2>Executive Summary</h2>")
    html_parts.append(
        "<p>This report summarizes completed experiments across phases and highlights where data is missing. "
        "It is intended for a presentation audience with both lay and technical sections.</p>"
    )
    if missing:
        html_parts.append("<p><strong>Missing data:</strong> " + ", ".join(missing) + "</p>")
    else:
        html_parts.append("<p><strong>All expected phases have results.</strong></p>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Layman Descriptions</h2>")
    html_parts.append(
        "<p><strong>Phase 1:</strong> Tests how well the system can finish a sentence when given partial input and basic context.</p>"
        "<p><strong>Phase 2:</strong> Tests different context combinations (time, who is present, location, profile) to see what helps disambiguate intent.</p>"
        "<p><strong>Phase 3:</strong> Tests whether adding information about relationships (social context) or personal profiles improves response accuracy.</p>"
        "<p><strong>Phase 4:</strong> Tests converting keywords into useful utterances, with and without richer contextual information.</p>"
        "<p><strong>Phase 5:</strong> Tests memory strategies (no memory vs RAG vs Cognee) to see if recent conversation improves accuracy.</p>"
    )
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Phase 1 Results</h2>")
    html_parts.append(phase1_summary)
    for chart in phase1_charts:
        html_parts.append(f"<img src='data:image/png;base64,{chart}' />")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Phase 2 Results</h2>")
    html_parts.append(phase2_summary)
    for chart in phase2_charts:
        html_parts.append(f"<img src='data:image/png;base64,{chart}' />")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Phase 3 Results</h2>")
    html_parts.append(phase3_summary)
    for chart in phase3_charts:
        html_parts.append(f"<img src='data:image/png;base64,{chart}' />")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Phase 4 Results</h2>")
    html_parts.append(phase4_summary)
    for chart in phase4_charts:
        html_parts.append(f"<img src='data:image/png;base64,{chart}' />")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Phase 5 Results</h2>")
    html_parts.append(phase5_summary)
    for chart in phase5_charts:
        html_parts.append(f"<img src='data:image/png;base64,{chart}' />")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Technical Review</h2>")
    html_parts.append(
        "<p><strong>Phase 1:</strong> Aggregates embedding similarity and LLM-judge scores across context filters and generation methods. "
        "Focus is on whether context filters improve semantic alignment.</p>"
        "<p><strong>Phase 2:</strong> H1-H5 context ablations are useful, but transcript schema consistency is critical; missing or mismatched fields can silently degrade validity.</p>"
        "<p><strong>Phase 3:</strong> Compares baseline vs enhanced scores. The key signal is mean improvement and improvement rate. "
        "Energy cost is tracked when available.</p>"
        "<p><strong>Phase 4:</strong> Evaluates keyword-to-utterance quality with context levels, but result quality depends on representative keyword distributions and reliable ground truth utterances.</p>"
        "<p><strong>Phase 5:</strong> Compares baseline, RAG, and Cognee (if run). Includes shuffled/random-memory baselines to control for prompt length. "
        "Paired t-tests and bootstrap CIs are included in the raw JSON output.</p>"
        "<p><strong>Potential issues and missing pieces:</strong></p>"
        "<ul>"
        "<li><strong>Data quality and scale:</strong> Current synthetic data is small and partially templated, limiting external validity.</li>"
        "<li><strong>Schema drift risk:</strong> Several scripts consume different field names (`target`, `target_ground_truth`, `speech`, `last_utterance`), which can create hidden evaluation errors.</li>"
        "<li><strong>LLM judge bias/variance:</strong> Single-model judging can be unstable and style-sensitive; scores can drift by prompt wording rather than intent fidelity.</li>"
        "<li><strong>Generator-judge coupling:</strong> Using similar model families for generation and judging can inflate agreement and hide errors.</li>"
        "<li><strong>Phase 3 score scaling bug risk:</strong> `judge_similarity` returns 1-10 in clients, but Phase 3 divides by 100, compressing signal to near-zero.</li>"
        "<li><strong>No human adjudication loop:</strong> Missing manual review sample makes it hard to validate automatic metrics for clinical communication quality.</li>"
        "<li><strong>Limited partner-action instrumentation:</strong> Stage directions are often unavailable in deployment; these should remain optional context, not required input.</li>"
        "</ul>"
    )
    html_parts.append("</div>")

    html_parts.append("</body></html>")

    OUTPUT_HTML.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Saved report to {OUTPUT_HTML}")


if __name__ == "__main__":
    render_report()
