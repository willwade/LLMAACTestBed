# Phase 2: Context-Aware Experiments

Synthetic scenarios that stress-test how time, social graph, location, and profile context improve AAC intent prediction. Phase 2 validates the H1–H5 hypotheses by varying which context is injected into the prompt and measuring accuracy with an LLM judge.

## What This Phase Tests
- **H1 (Time only)**: Does routine-aware time context help disambiguate telegraphic speech?
- **H2 (Time + Social)**: Do interlocutor relationships sharpen predictions?
- **H3 (Time + Social + Location)**: Does adding environment further improve accuracy?
- **H4 (Speech + Profile)**: How much lift comes from profile knowledge alone?
- **H5 (Full Context)**: What is the ceiling when all context types are combined?

## Files
- `run_context_experiments.py`: Primary runner built on the shared `lib/` stack (ContextBuilder, PromptBuilder, BaseEvaluator). Supports hypothesis testing (H1–H5), profile ablation (smart vs raw), and temporal disambiguation. Saves CSVs and visualizations to the chosen output directory.
- `run_strict_aac.py`: Standalone “full spectrum” script (H1–H5) that logs scores to `aac_full_spectrum_results.csv` using the `llm` package directly.
- `run_speech_ablation.py`: Minimal ablation comparing profile-aware vs generic predictions on vague speech inputs.
- `run_synthesis_test.py`: Small temporal routine probe (“Do you want the usual?”) at morning vs evening.
- `plot_results.py`: Visualizes `aac_full_spectrum_results.csv`, highlighting context gain between speech-only and full-context runs.

## Data
- Profiles: `data/synthetic/profiles/dave_context.json`
- Transcripts: `data/synthetic/transcripts/transcript_data_2.json` (structured scenarios) and `data/synthetic/transcripts/transcript_vague.json` (vague inputs)

## Running the main runner (recommended)
From the repo root:

```bash
uv run python experiments/phase2_context_aware/run_context_experiments.py \
  --provider openai \
  --model gpt-4o-mini \
  --profile data/synthetic/profiles/dave_context.json \
  --transcripts data/synthetic/transcripts/transcript_data_2.json \
  --vague data/synthetic/transcripts/transcript_vague.json \
  --experiment all \
  --output results/phase2
```

`--experiment` options: `hypothesis` (H1–H5), `ablation` (profile vs no profile), `temporal` (time disambiguation), or `all`. Results CSVs and plots are written under the chosen output directory.

## Running the legacy standalone scripts
These use hard-coded filenames at the top of each script. Edit `PROFILE_FILE`, `TRANSCRIPT_FILE`, and `MODEL_NAME` as needed (e.g., point them to `data/synthetic/profiles/dave_context.json` and `data/synthetic/transcripts/...`), then run:

```bash
# Full spectrum H1–H5 (writes aac_full_spectrum_results.csv)
uv run python experiments/phase2_context_aware/run_strict_aac.py

# Speech-only ablation
uv run python experiments/phase2_context_aware/run_speech_ablation.py

# Temporal routine probe
uv run python experiments/phase2_context_aware/run_synthesis_test.py
```

After `run_strict_aac.py`, render the comparison chart:

```bash
uv run python experiments/phase2_context_aware/plot_results.py
```

## Outputs
- CSVs of hypothesis runs, ablation comparisons, and temporal probes
- Plots illustrating context gain (speech-only vs full-context)
- Console summaries showing per-scenario predictions and LLM-judge scores
