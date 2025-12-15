# ContextAwareTestBed: Unified Research Framework for AAC Systems

A comprehensive framework for evaluating context-aware Large Language Model systems for Augmentative and Alternative Communication (AAC) users with Motor Neurone Disease (MND).

## Quick Start

### Prerequisites
- **UV** (modern Python package manager)
- Python 3.11+
- API keys for at least one LLM provider

### Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/your-repo/ContextAwareTestBed.git
cd ContextAwareTestBed

# Install dependencies and create virtual environment
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running Experiments

```bash
# Test LLM client functionality (quick test)
uv run python -c "from lib.llm_clients import create_llm_client; print('✅ LLM setup works')"

# Run individual experiments (direct execution)
uv run experiments/phase1_chat_history/run_evaluation.py --provider gemini --sample-size 10

# For specific experiment scripts, see experiments/ directories
# Note: Some experiments may need additional setup or data files
```

## 🔬 Research Framework Overview

This repository implements a three-phase research approach for evaluating context-aware AAC systems:

### Phase 1: Chat History Evaluation
- **Location**: `experiments/phase1_chat_history/`
- **Data**: Real chat history from AAC users (331 utterances)
- **Focus**: Baseline performance with retrieval-based text completion
- **Key Metric**: Semantic similarity and embedding-based retrieval accuracy

### Phase 2: Context-Aware Experiments
- **Location**: `experiments/phase2_context/`
- **Data**: Synthetic scenarios with rich user profiles
- **Focus**: Context injection hypotheses (H1-H5)
- **Key Features**: Profile, temporal, location, and social context

### Phase 3: Combined Integration
- **Location**: `experiments/phase3_combined/`
- **Data**: Real chat + synthetic social graphs and profiles
- **Focus**: Real-world validation and energy cost analysis
- **Innovation**: Social context-aware conversation prediction

## 🏗️ Repository Structure

```
ContextAwareTestBed/
├── lib/                          # Shared library code
│   ├── llm_clients/             # Multi-provider LLM abstraction
│   │   ├── base.py              # Abstract BaseLLMClient interface
│   │   ├── gemini_client.py     # Google Gemini implementation
│   │   ├── openai_client.py     # OpenAI implementation
│   │   └── factory.py           # LLMClientFactory for provider selection
│   ├── context/                 # Context filtering and building
│   │   ├── filters.py           # Time, location, and social filters
│   │   └── profiles.py          # User profile management
│   ├── data/                    # Data loading utilities
│   │   └── loaders.py           # Unified data loading interface
│   ├── evaluation/              # Metrics and scoring
│   │   ├── metrics.py           # Similarity and embedding metrics
│   │   ├── scorers.py           # LLM-based similarity scoring
│   │   └── visualizers.py       # Results visualization
│   ├── reporting/               # Results aggregation
│   │   └── aggregator.py        # Cross-phase result aggregation
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── logger_setup.py      # Logging setup
├── experiments/                  # All experiment implementations
│   ├── phase1_chat_history/     # Chat history-driven evaluation
│   │   ├── run.py               # Phase 1 main script
│   │   └── chat_history_evaluator.py
│   ├── phase2_context/          # Context-aware experiments
│   └── phase3_combined/         # Combined integration
├── data/                        # All data files (centralized)
│   ├── synthetic/               # Generated test scenarios
│   │   ├── profiles/            # User profiles
│   │   ├── transcripts/         # Test scenarios
│   │   └── social_graphs/       # Social relationships
│   └── real/                    # Real-world data
│       └── chat_history/        # Chat data (331 utterances)
├── results/                     # Experiment outputs
├── docs/                        # Documentation
├── pyproject.toml               # UV configuration and dependencies
├── mypy.ini                     # Type checking configuration
└── .env.example                 # Environment variable template
```

## ⚡ Key Features

### Multi-Provider LLM Support
- **Provider Agnostic**: Single interface for OpenAI, Gemini, Anthropic
- **Easy Switching**: `--provider` flag for experiments
- **Consistent API**: All clients implement `BaseLLMClient` interface
- **Retry Logic**: Built-in exponential backoff for reliability

### Comprehensive Evaluation
- **Dual-LLM Architecture**: Generator + Judge for scoring
- **Multiple Metrics**: Embedding similarity, LLM-judge scores, retrieval accuracy
- **Energy Cost Analysis**: Social context interaction costs
- **Cross-Phase Comparison**: Unified result aggregation

### Real-World Data Integration
- **331 Real Utterances**: From actual AAC chat history
- **Privacy Protected**: Real data never committed to repository
- **Synthetic Profiles**: Rich user profiles for controlled testing
- **Social Graphs**: Relationship-based context modeling

## 🔧 Development Workflow

### Code Quality Standards
```bash
# Run linting (mandatory before committing)
uv run ruff check
uv run ruff format

# Run type checking (recommended)
uv run mypy lib/

# Run both together
uv run ruff check && uv run mypy lib/
```

### Environment Configuration
```bash
# Set API keys in .env file
OPENAI_API_KEY=your-openai-key
LLM_GEMINI_KEY=your-gemini-key
ANTHROPIC_API_KEY=your-anthropic-key

# Set default LLM provider
DEFAULT_LLM_PROVIDER=gemini
```

### Adding New Experiments
1. Create new directory in `experiments/`
2. Implement main script with `--provider` argument
3. Use `lib.llm_clients.create_llm_client()` for LLM access
4. Store results in `results/` with timestamps

### Adding New LLM Providers
1. Create client class in `lib/llm_clients/`
2. Implement `BaseLLMClient` interface
3. Register in `LLMClientFactory._clients`
4. Add dependencies to `pyproject.toml`

## 📊 Results and Analysis

### Expected Outcomes
- **Profile Context**: 80% improvement in prediction accuracy
- **Social Context**: Improved conversation flow prediction
- **Temporal Context**: Critical for routine-based requests
- **Energy Cost**: Quantified social interaction overhead

### Result Files
- `results/`: Timestamped experiment outputs
- `results_summary/`: Aggregated cross-phase analysis
- CSV, JSON, and HTML report formats
- LaTeX table generation for papers

### Key Research Findings
1. **H5 (Speech + Profile)** performs exceptionally well
2. **H4 (Full Context)** is safest for critical medical scenarios
3. Social context reduces prediction ambiguity
4. Telegraphic speech requires keyword interpretation, not full NL understanding

## 🧪 Testing and Validation

### Quick Validation Test
```bash
# Test LLM client functionality
uv run python -c "
from lib.llm_clients import create_llm_client
client = create_llm_client('gemini')
response = client.generate('Count to 3')
print(f'LLM test: {response}')
"
```

### Experiment Testing
```bash
# Test LLM client with different providers
uv run python -c "
from lib.llm_clients import create_llm_client
for provider in ['gemini', 'openai']:
    try:
        client = create_llm_client(provider)
        response = client.generate('Count to 3')
        print(f'✅ {provider}: {response}')
    except Exception as e:
        print(f'❌ {provider}: {e}')
"

# Test chat history evaluation (requires data file)
uv run python -c "
from lib.evaluation.chat_history_evaluator import ChatHistoryEvaluator
import os
data_path = 'data/real/chat_history/processed/dataset.json'
if os.path.exists(data_path):
    evaluator = ChatHistoryEvaluator(chat_data_path=data_path, llm_provider='gemini')
    print('✅ Chat history evaluator ready with real data')
else:
    print('ℹ️ Chat history data not found - place data in data/real/chat_history/processed/')
"
```

## 📚 Research Papers and Publications

### Supported Publication Types
- **CHI, ASSETS, UbiComp**: HCI and accessibility innovations
- **ACM TACCESS**: Accessibility computing journal
- **IEEE THMS**: Human-machine systems
- **Journal of AAC**: Augmentative communication research

### Reproducibility
- Complete experiment code and configuration
- Synthetic data generation scripts
- Results aggregation and visualization
- Multi-provider LLM support for validation

## 🔒 Privacy and Security

- **Real Data Protection**: Chat history files in `.gitignore`
- **API Key Security**: Environment variables only
- **Anonymization**: All synthetic data is fake
- **No PII**: No personally identifiable information in repository

## 🤝 Contributing

We welcome contributions! Please:

1. Follow code quality standards (`ruff`, `mypy`)
2. Add type hints to new code
3. Update documentation
4. Use UV for dependency management
5. Test with multiple LLM providers

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **AAC Community** for use cases and feedback
- **Research Participants** for data contributions
- **OpenAI and Google** for LLM APIs
- **UV Team** for modern Python package management

---

## 📖 Documentation

- **`AGENTS.md`**: Detailed guide for Claude Code instances
- **`docs/`**: Additional documentation and methodology
- **`UV_MIGRATION.md`**: Migration notes from pip to UV
- **Inline Documentation**: Comprehensive docstrings throughout codebase