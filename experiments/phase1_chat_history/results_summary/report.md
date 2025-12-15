# Chat History LLM Evaluation - Sweep Report (200-sample run)

## Why we did this
- Build an evaluation harness that shows how well LLMs can propose completions from partial user input while staying true to the user’s style, with and without extra context (retrieval, time, recent turns).
- Dataset: one real user chat-history export; chronologically split into corpus (for retrieval) vs held-out test turns.

## What we ran
- Scripts: `sweep_wrapper.py` -> `run_evaluation.py` -> `visualize_sweep_results.py`.
- Samples: 200 requested; produced ~110 unique held-out targets, 8,648 rows across sweeps.
- Swept parameters (5 runs; files `results_cf200-*.csv`):
  - `context_filter`: `none`, `time` (±2h; geo radius 0.5km had no effect here as no geo data in the dataset).
  - `conversation_window`: 0, 1, 3 (for `time`, only 0 and 1 were run).
  - Partial methods: `prefix_3`, `prefix_2`, `keyword_2`, `random`.
  - Generation methods: `tfidf`, `embedding`, `lexical`, `context_only` (no retrieval).
- Outputs:
  - CSVs: `results/results_cf200-*.csv`.
  - Figures in `results/figures/`:
    - `*_by_conversation_window.png` for each metric,
    - `generation_method_boxplots.png`,
    - `heatmap_embedding_similarity_cf-<context>_cw-<N>.png` per sweep.

## How the pipeline works
1) Split the chat history chronologically: first ~67% is the corpus for retrieval; last ~33% is the test set.  
2) Build the partial input (what the user has typed so far):
   - `prefix_3` / `prefix_2`: first 3 or 2 words.
   - `keyword_2`: two most frequent “heavy” words.
   - `random`: 1–3 random words, in original order.
3) Optional context filtering: for `time`, keep only corpus utterances within ±2 hours of the test timestamp (geo was effectively unused here).
4) Retrieve similar past utterances:
   - `lexical`: direct word overlap.
   - `tfidf`: term-frequency / inverse-document-frequency similarity (rare-but-important words weigh more).
   - `embedding`: semantic similarity via `all-MiniLM-L6-v2` sentence embeddings.
5) Prompt the model with the partial, optional time text, recent turns (if `conversation_window` > 0), and retrieved examples, then ask it to complete. Core prompt (temperature 0.2):
   ```text
   System: You are an intelligent AAC text completion system. Complete the user's partial text based on provided context and examples.
   User: Complete the following partial text: '<partial>'
   Context: <time/geo text if any>
   Recent conversation context:
   <previous turns if conversation_window > 0>
   Here are some examples of similar utterances:
   1. <retrieved example 1>
   ...
   Provide a completion that matches the user's likely intent. Only return the completed text, no explanation.
   ```
6) Score each proposal vs the true target:
   - `embedding_similarity` (0–1 cosine of sentence embeddings),
   - `llm_judge_score` (LLM rater 1–10; mostly clustered near 1–2),
   - `character_accuracy`, `word_accuracy` (overlap-based exactness).
7) Aggregate and plot: `visualize_sweep_results.py` averages by sweep and writes figures.

## Key results (means across runs)

| context_filter | conv_window | embed_sim | LLM_judge | char_acc | word_acc |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 0.309 | 1.93 | 0.410 | 0.343 |
| none | 1 | **0.320** | **2.00** | **0.424** | **0.363** |
| none | 3 | 0.311 | 1.97 | 0.410 | 0.366 |
| time | 0 | 0.319 | 1.97 | 0.424 | 0.354 |
| time | 1 | 0.317 | 1.97 | 0.424 | 0.358 |

Method-level averages:
- Partial methods: `prefix_3` best on all metrics (embed_sim 0.342, judge 2.20, word_acc 0.451). `prefix_2` is weakest semantically; `random` beats it on similarity but not on exactness.
- Generation methods: `tfidf` and `embedding` tie for top (embed_sim ~0.348, judge ~2.0). `lexical` slightly lower; `context_only` lags (embed_sim 0.241).
- Best combinations: `prefix_3` + (`tfidf` or `embedding`), especially with `conversation_window=1`, reaching embed_sim ~0.39 and judge ~2.3 in the heatmaps.

Metric distribution reminder:
- LLM judge is mostly at the floor (median 1, 75th percentile 2). There is large headroom before we match user style/intent well.

## What this means for the aim
- Retrieval helps: tfidf/embedding beats context-only, supporting history-aware prompting.
- Partial length matters: 3-word prefixes are strongest; shorter prefixes or sparse keywords reduce quality.
- Local conversation context: 1 prior turn helps a bit; 3 does not. Time filtering (±2h) neither helped nor hurt here.
- Quality remains modest: even the best setup scores low on the judge scale; better prompting/modeling is needed to reach user-style fidelity.

## Figures to consult
- `results/figures/embedding_similarity_by_conversation_window.png`
- `results/figures/llm_judge_score_by_conversation_window.png`
- `results/figures/generation_method_boxplots.png`
- Heatmaps e.g. `results/figures/heatmap_embedding_similarity_cf-none_cw-0.png` (partial × generation combinations; darker = better).

## Plain-language glossary
- `prefix_3` / `prefix_2`: first 3 or 2 words of the user’s sentence.
- `keyword_2`: two most salient words (by simple frequency).
- `random`: 1–3 random words from the sentence.
- `tfidf`: retrieval that ranks past utterances by TF-IDF similarity—rare-but-shared words matter most.
- `embedding`: retrieval by semantic similarity using vector embeddings (`all-MiniLM-L6-v2`).
- `lexical`: retrieval by direct word overlap counts.
- `context_only`: no retrieval; just the time/geo/conversation context.
- `conversation_window`: how many previous test turns are fed into the prompt as local chat history.
- `context_filter=time`: only retrieve utterances within a ±2-hour band of the current message’s timestamp (geo filter inactive here).

## Concrete examples from the CSVs
- Sampled rows:
  - `prefix_3` + `tfidf` (file: `results_cf200-none_cw-0_tw-2.0_gr-0.5.csv`):  
    - partial: `So I am`  
    - proposal: `So I am feeling a bit overwhelmed right now.`  
    - target: `So I am bursting for a widdle`
  - `keyword_2` + `embedding`:  
    - partial: `She could`  
    - proposal: `She could feel the excitement building inside her.`  
    - target: `She could not have swept today`
  - `random` + `lexical`:  
    - partial: `could not`  
    - proposal: `could not believe what I was seeing, it felt surreal and overwhelming.`  
    - target: `She could not have swept today`

- Example of the actual prompt (for the first row above: `prefix_3` + `tfidf`, `context_filter=none`, `conversation_window=0`). The retrieval examples shown are the top-3 TF-IDF matches from the corpus for the partial “So I am”:
  ```text
  System: You are an intelligent AAC text completion system. Complete the user's partial text based on provided context and examples.
  User: Complete the following partial text: 'So I am'

  Here are some examples of similar utterances:
  1. So they want to be between 3 point 9 and 10, so those are the parameters.
  2. Well I am fucked
  3. I am going for a wee again

  Provide a completion that matches the user's likely intent. Only return the completed text, no explanation.
  ```
  Model output from the run: `So I am feeling a bit overwhelmed right now.`  
  Ground truth: `So I am bursting for a widdle`

