# LLM Provider Configuration

This document explains how to configure and use different LLM providers with the ContextAwareTestBed framework.

## Supported Providers

- **Gemini** (Google): Default provider, excellent for AAC context
- **OpenAI**: GPT-4 and other OpenAI models
- **Anthropic**: Claude models (experimental)

## Setup

### 1. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required for the provider you want to use
GEMINI_API_KEY=your-gemini-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Set defaults
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_GEMINI_MODEL=gemini-2.0-flash-exp
DEFAULT_OPENAI_MODEL=gpt-4
DEFAULT_ANTHROPIC_MODEL=claude-3-sonnet-20241022
```

### 2. UV Setup

Install with uv (recommended):

```bash
# Install the project with uv
uv sync

# Or install specific provider dependencies
uv sync --extra dev  # For development
```

## Usage

### Command Line

Run experiments with specific providers:

```bash
# Use default provider from .env
run-phase3

# Specify provider explicitly
run-phase3 --provider openai
run-phase3 --provider gemini

# Or run the Python script directly
python experiments/phase3_combined/experiments/profile_enhanced.py --provider openai
```

### Python API

```python
from llm import create_llm_client
from utils import load_env

# Load environment variables
load_env()

# Create client with default provider
client = create_llm_client()

# Create client with specific provider
client = create_llm_client(provider="openai", model="gpt-4-turbo")

# Generate text
response = client.generate("Hello, how are you?")
print(response)

# Judge similarity
score = client.judge_similarity("I need water", "Can I have water please?")
print(f"Similarity: {score}/100")
```

## Provider Comparison

### Gemini (Google)
- **Strengths**: Fast, context-aware, good with AAC use cases
- **Models**: gemini-2.0-flash-exp, gemini-1.5-pro
- **API Cost**: Generally lower than competitors
- **Recommended for**: Most AAC experiments

### OpenAI
- **Strengths**: Well-tested, robust, good at structured tasks
- **Models**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **API Cost**: Higher, especially for GPT-4
- **Recommended for**: Validation experiments, comparison studies

### Anthropic (Claude)
- **Strengths**: Large context window, good with nuanced text
- **Models**: claude-3-sonnet, claude-3-opus
- **API Cost**: Similar to OpenAI
- **Status**: Experimental in this framework

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Fast iteration | gemini-2.0-flash-exp | Speed and cost |
| High accuracy | gpt-4-turbo | Best performance |
| Large context | claude-3-sonnet | 200K context |
| Cost-sensitive | gemini-1.5-flash | Lowest cost |

## Results Tracking

The framework automatically tracks which model and provider were used:

```json
{
  "experiment_results": {
    "metadata": {
      "llm_provider": "gemini",
      "llm_model": "gemini-2.0-flash-exp",
      "timestamp": "2025-01-12T10:30:00Z"
    },
    "scores": {...}
  }
}
```

This enables easy comparison across different models and providers in your analysis.