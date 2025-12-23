# Chat History-Driven LLM Evaluation Framework

This framework evaluates Large Language Model (LLM)-based text completion systems for Augmentative and Alternative Communication (AAC) users using real chat history data. It provides a systematic approach to compare different methods for generating message proposals based on partial utterances.

## Overview

The evaluation framework:

1. **Splits data chronologically** into corpus (for conditioning) and test sets
2. **Generates proposals** for partial test utterances using different methods
3. **Scores proposals** against the true full utterances using multiple metrics

## Key Components

### Partial Utterance Generation
- **Prefix Truncation**: Uses the first N words of the utterance
- **Keyword Extraction**: Uses the semantically heaviest words
- **Random Selection**: Selects random words from the utterance

### Corpus-Based Retrieval
- **Lexical Search**: Retrieves examples containing exact word matches
- **TF-IDF Search**: Retrieves examples using TF-IDF similarity
- **Embedding Search**: Retrieves examples using semantic similarity

### Evaluation Metrics
- **Embedding Similarity**: Sentence-level similarity using embeddings
- **LLM-Judge Scoring**: Semantic evaluation using an LLM
- **Character Accuracy**: Character-level accuracy (case-sensitive and case-insensitive variants)
- **Word Precision/Recall/F1**: Word overlap precision, recall, and F1
- **Weighted Word F1**: Word F1 scaled by how much of the target remained after the partial
- **Completion Gain**: Fraction of remaining target words correctly added beyond the partial

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ContextAwareTestBed.git
cd ContextAwareTestBed

# Install dependencies with uv
uv sync
```

## Usage

### Command Line Interface

```bash
# Basic usage (process all test rows)
uv run experiments/phase1_chat_history/run_evaluation.py --data path/to/chat_data.json

# Recommended defaults: 3 candidates, skip too-short prefixes
uv run experiments/phase1_chat_history/run_evaluation.py \
  --data path/to/chat_data.json \
  --n-candidates 3 \
  --skip-short-prefixes \
  --output results.csv \
  --visualize

# With custom model
uv run experiments/phase1_chat_history/run_evaluation.py \
  --data path/to/chat_data.json \
  --model gemini-1.5-pro \
  --corpus-ratio 0.8

# With contextual filters and conversation window
uv run experiments/phase1_chat_history/run_evaluation.py \
  --data path/to/chat_data.json \
  --n-candidates 3 \
  --skip-short-prefixes \
  --context-filter time_geo \
  --time-window-hours 2 \
  --geo-radius-km 5 \
  --conversation-window 3
```

`--context-filter` controls how retrieval is narrowed using metadata (`none`, `time`, `geo`, `time_geo`). `--conversation-window` includes the previous N test utterances in the prompt to test local dialogue conditioning.

### Python API

```python
from utils.evaluation_utils import ChatHistoryEvaluator

# Initialize evaluator
evaluator = ChatHistoryEvaluator(
    chat_data_path="path/to/chat_data.json",
    corpus_ratio=0.67
)

# Define methods
partial_methods = {
    'prefix_3': lambda text: evaluator.create_prefix_partial(text, 3),
    'keyword_2': lambda text: evaluator.create_keyword_partial(text, 2)
}

generation_methods = {
    'lexical': lambda partial, context, n: evaluator.lexical_generate(partial, context, top_k=3, n_candidates=n),
    'embedding': lambda partial, context, n: evaluator.embedding_generate(partial, context, top_k=3, n_candidates=n)
}

evaluation_metrics = {
    'embedding_similarity': evaluator.calculate_embedding_similarity,
    'llm_judge_score': evaluator.judge_similarity,
    'word_f1': evaluator.calculate_word_f1,
    'weighted_word_f1': lambda target, proposal, partial: evaluator.calculate_weighted_word_f1(target, proposal, partial),
    'completion_gain': lambda target, proposal, partial: evaluator.calculate_completion_gain(target, proposal, partial)
}

# Run evaluation
results = evaluator.run_evaluation(
    partial_methods=partial_methods,
    generation_methods=generation_methods,
    evaluation_metrics=evaluation_metrics,
    sample_size=None,  # use all rows
    n_candidates=3,
    skip_short_prefixes=True,
    context_filter="time_geo",
    time_window_hours=2,
    geo_radius_km=5,
    conversation_window=3,
)

# Visualize results
evaluator.visualize_results(results)
```

### Notebook Use

Open `ChatHistoryEvaluation.ipynb` or `Demo.ipynb` and reuse the snippet above inside a cell. Set your `LLM_GEMINI_KEY` in the environment first, then call `run_evaluation` with `context_filter`/`conversation_window` to explore how temporal/location conditioning and local dialogue context affect scores. Plotting works the same via `visualize_results`.

## Configuration

You can customize the evaluation parameters using a configuration file:

```yaml
# Example config.yaml
data:
  chat_data_path: "path/to/chat_data.json"
  corpus_ratio: 0.67

model:
  name: "gemini-2.0-flash-exp"
  temperature: 0.2

evaluation:
  sample_size: 50
  partial_utterance_methods:
    prefix_3:
      type: "prefix"
      n_words: 3
  generation_methods:
    lexical:
      type: "lexical"
      top_k: 3
  evaluation_metrics:
    embedding_similarity:
      enabled: true
```

## Data Format

The chat history data should be in the following JSON format:

```json
{
  "version": "1.0",
  "exportDate": "2023-11-24",
  "sentenceCount": 1000,
  "sentences": [
    {
      "uuid": "unique-id",
      "anonymousUUID": "anonymous-id",
      "content": "I need to adjust my neck brace",
      "metadata": [
        {
          "timestamp": "2023-11-24T14:30:00.000Z",
          "latitude": 40.7128,
          "longitude": -74.0060
        }
      ]
    }
  ]
}
```

## Results

The framework generates results in CSV format with the following columns:

- `target`: The original full utterance
- `partial`: The partial utterance input
- `proposal`: The generated completion
- `partial_method`: Method used to create the partial utterance
- `generation_method`: Method used to generate the proposal
- `embedding_similarity`: Embedding similarity score (0-1)
- `llm_judge_score`: LLM judge score (1-10)
- `character_accuracy`: Character accuracy score (0-1)
- `word_accuracy`: Word accuracy score (0-1)

## Visualizations

The framework generates various visualizations:

1. **Performance Heatmaps**: Show performance by method combination
2. **Method Comparison**: Compare performance across different methods
3. **Score Distribution**: Show distribution of scores
4. **Error Analysis**: Analyze common error patterns
5. **Temporal Analysis**: Analyze performance by time of day

## Security and Privacy

**IMPORTANT**: This framework is designed to work with real user chat history data that may contain sensitive personal information. Please ensure:

1. **Data Anonymization**: All personally identifiable information should be removed or anonymized
2. **Access Control**: Limit access to the data and results
3. **No Version Control**: Do not commit the chat history data or evaluation results to version control systems

## Extensions

The framework is designed to be extensible. You can easily add:

1. **New Partial Utterance Methods**: Implement your own approach to creating partial utterances
2. **New Retrieval Methods**: Implement different approaches to finding relevant examples
3. **New Evaluation Metrics**: Add additional metrics for specific use cases
4. **Contextual Filtering**: Filter retrieval candidates by metadata like time or location
