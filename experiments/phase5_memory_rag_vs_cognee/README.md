# Phase 5: Memory Experiment (RAG vs Cognee)

This experiment simulates conversation sessions where memory is cleared at the start of each session and only previous turns in the same session are available to the model.
It also includes shuffled/random-memory baselines and paired statistical tests.

## Run

Baseline + RAG only:

```bash
uv run python experiments/phase5_memory_rag_vs_cognee/run_memory_experiment.py \
  --provider openai \
  --model gpt-4o-mini
```

Include Cognee (after install):

```bash
uv pip install cognee
uv run python experiments/phase5_memory_rag_vs_cognee/run_memory_experiment.py \
  --provider openai \
  --model gpt-4o-mini \
  --use-cognee \
  --cognee-update-each-turn

Use the improved transcript schema (default is expanded 65 turns):

```bash
uv run python experiments/phase5_memory_rag_vs_cognee/run_memory_experiment.py \
  --transcripts data/synthetic/transcripts/transcript_data_2_improved_expanded.json
```
```

## Notes
- Cognee is optional; the script will skip it if not installed.
- Results are written to `results/phase5_memory_comparison.json` by default.
- Summary includes paired t-tests + bootstrap CI for score and keyword recall.
- CSV/HTML reports are generated alongside the JSON.
