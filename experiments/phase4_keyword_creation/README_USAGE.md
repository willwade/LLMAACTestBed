# Phase 4: Keyword-to-Utterance Generation

## Quick Start

### 1. Test Setup
```bash
# Test that everything is working
uv run experiments/phase4_keyword_creation/test_setup.py
```

### 2. Run Experiments

#### Quick Test (5 samples)
```bash
# Baseline test only
uv run experiments/phase4_keyword_creation/run_experiment.py --part 1 --sample-size 5 --provider gemini

# All parts with small sample
uv run experiments/phase4_keyword_creation/run_experiment.py --sample-size 5 --provider gemini
```

#### Full Experiment
```bash
# Run all parts with all data (74 keyword combinations)
uv run experiments/phase4_keyword_creation/run_experiment.py --provider gemini

# Run specific parts
uv run experiments/phase4_keyword_creation/run_experiment.py --part 1 --provider gemini  # Baseline
uv run experiments/phase4_keyword_creation/run_experiment.py --part 2 --provider gemini  # Contextual
uv run experiments/phase4_keyword_creation/run_experiment.py --part 3 --provider gemini  # Single keyword
```

#### Different LLM Providers
```bash
# With OpenAI (requires OPENAI_API_KEY)
uv run experiments/phase4_keyword_creation/run_experiment.py --provider openai

# Custom model
uv run experiments/phase4_keyword_creation/run_experiment.py --provider openai --model gpt-4-turbo
```

### 3. Analyze Results

Results are saved in `experiments/phase4_keyword_creation/results/experiment_[timestamp]/`

#### Automatic Analysis
```bash
# Run analysis (if experiment already completed)
python -c "
from experiments.phase4_keyword_creation.evaluation.results_analyzer import ResultsAnalyzer
from pathlib import Path
import glob

# Find latest results directory
results_dirs = glob.glob('experiments/phase4_keyword_creation/results/experiment_*')
latest = max(results_dirs)
analyzer = ResultsAnalyzer(Path(latest))
analyzer.run_full_analysis()
"
```

#### Manual Analysis
```bash
# View summary
cat results/latest/experiment_summary.json

# View analysis report
cat results/latest/analysis_report.md

# Check figures
ls results/latest/figures/
```

## Experiment Structure

### Part 1: Baseline Testing
- **Input**: 2-3 keywords only
- **No context**: Pure keyword-to-utterance prediction
- **Purpose**: Establish baseline performance

### Part 2: Contextual Enhancement
- **Context Levels**:
  - Location only
  - Location + People present
  - Location + Equipment
  - Social relationships
  - Full context (all above)
- **Purpose**: Measure impact of context

### Part 3: Single Keyword Testing
- **Input**: 1 high-value keyword + full context
- **Focus**: Pain, Scratch, Move, Sick, Help, Chair, Transfer, Feed
- **Purpose**: Test minimal input with optimal context

## Data Files

- `data/DwayneKeyWords.tsv`: 74 keyword combinations with ground truth
- `data/social_graph.json`: Dwayne's comprehensive social context

## Output Files

### Results
- `part1_baseline_results.json`: Baseline testing results
- `part2_contextual_results.json`: Contextual enhancement results
- `part3_single_keyword_results.json`: Single keyword results
- `experiment_summary.json`: Overall summary with statistics

### Analysis
- `analysis_report.md`: Text summary of findings
- `figures/`: Visualizations and plots
  - `baseline_scores.png`: Score distribution
  - `context_level_comparison.png`: Context comparison
  - `single_keyword_effectiveness.png`: Keyword analysis
  - `summary_dashboard.png`: Overview dashboard

## Key Metrics

- **Similarity Score**: 1-10 scale (LLM judge evaluation)
- **Success Rate**: Percentage of scores ≥ 7/10
- **Context Improvement**: Score improvement with context
- **Keyword Effectiveness**: Performance of individual keywords

## Customization

### Adding New Context Levels
Edit `evaluation/context_builder.py` to add new context types in the `build_context()` method.

### Modifying Keywords
Update `data/DwayneKeyWords.tsv` or use the `--sample-size` parameter for testing.

### Changing Evaluation Criteria
Modify the `_evaluate_prediction()` method in `evaluation/keyword_evaluator.py`.

## Troubleshooting

### Common Issues

1. **"Keywords file not found"**: Ensure data files are copied correctly
2. **LLM connection errors**: Check API keys and network connection
3. **Import errors**: Verify all dependencies are installed

### Debug Mode
```bash
# Run with verbose logging
uv run experiments/phase4_keyword_creation/run_experiment.py --verbose --sample-size 1
```