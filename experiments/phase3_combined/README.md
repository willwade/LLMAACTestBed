# Phase 3: Combined Experiments

This phase integrates real chat history data with synthetic social graphs and user profiles to create comprehensive evaluation scenarios.

## Purpose

Phase 3 bridges the gap between:
- **Phase 1**: Real chat data but no social context
- **Phase 2**: Rich social context but synthetic scenarios

## Research Questions

1. **Context Integration**: How does real chat data behave when combined with detailed user profiles?
2. **Social Graph Impact**: Does knowing social relationships improve prediction accuracy on real conversations?
3. **Temporal Patterns**: Can temporal disambiguation from Phase 2 be applied to real chat timestamps?
4. **Hybrid Approaches**: What combinations of retrieval + context work best for real-world AAC use?

## Experiment Design

### 1. **Profile-Enhanced Chat History**
- Take real chat history from `data/real/chat_history/`
- Enhance with profiles from `data/synthetic/profiles/`
- Compare vs. baseline chat-only predictions

### 2. **Social Context Injection**
- Add social graphs from `data/synthetic/social_graphs/`
- Test if interlocutor identification improves predictions
- Measure energy cost vs. accuracy gain

### 3. **Temporal Filtering**
- Apply time-based filtering to real chat data
- Test if temporal patterns hold in real usage
- Measure performance across different times of day

### 4. **Hybrid Retrieval**
- Combine corpus-based retrieval with profile context
- Test if retrieval benefits from user knowledge
- Optimize retrieval strategies

## Implementation Structure

```
phase3_combined/
├── README.md                    # This file
├── requirements.txt            # Phase 3 specific deps
├── configs/
│   ├── profile_enhanced.yaml  # Config for profile experiments
│   ├── social_context.yaml    # Config for social experiments
│   ├── temporal_filter.yaml   # Config for time experiments
│   └── hybrid_retrieval.yaml  # Config for hybrid experiments
├── experiments/
│   ├── profile_enhanced.py     # Core experiment
│   ├── social_context.py       # Social graph experiments
│   ├── temporal_filtering.py   # Time-based experiments
│   └── hybrid_retrieval.py      # Retrieval + context
├── utils/
│   ├── profile_enhancer.py      # Tools to enhance profiles
│   ├── social_graph_builder.py  # Build social graphs from chat
│   └── chat_mapper.py           # Map chat data to profiles
├── run_phase3.py               # Master script for Phase 3
└── analysis/
    ├── cross_phase_comparison.py # Compare with Phases 1 & 2
    └── real_world_validation.py   # Validate findings
```

## Data Organization

All data is stored in the root `/data` directory:

```
data/
├── synthetic/
│   ├── profiles/               # User profiles
│   ├── social_graphs/          # Social relationships
│   ├── transcripts/           # Test scenarios
│   └── chat_profile_mappings/   # Maps chat users to profiles
└── real/
    ├── chat_history/
    │   ├── raw/                 # Original chat data
    │   ├── processed/           # Enhanced with context
    │   └── annotated/           # Social annotations
    └── enhanced_profiles/       # Real profiles from data
```

## Data Requirements

### Real Chat Data
- From Phase 1: `data/real/chat_history/raw/`
- Additional datasets if available
- Must have timestamps for temporal analysis

### User Profiles
- Base profiles from Phase 2: `data/synthetic/profiles/`
- Enhanced profiles in `data/real/enhanced_profiles/`
- Chat-to-profile mappings in `data/synthetic/chat_profile_mappings/`

### Social Graphs
- Generated profiles in `data/synthetic/social_graphs/`
- Extracted from chat patterns
- Annotated relationships
- Energy cost ratings

## Key Metrics

### New Metrics for Phase 3
- **Context Accuracy**: How well does real context match predictions?
- **Social Appropriateness**: Are responses socially suitable?
- **Energy Savings**: How much energy does context save?
- **Latency**: Time to generate predictions

### Existing Metrics
- Embedding similarity
- LLM judge scores
- Character/word accuracy
- BLEU scores

## Research Outcomes

### Expected Findings
1. **Context Value**: Quantify improvement from adding profiles
2. **Social Impact**: Measure benefit of social context
3. **Temporal Patterns**: Validate time-based insights from real data
4. **Optimal Strategies**: Identify best combinations for real use

### Paper Contributions
1. **Real-world Validation**: Show that lab findings hold in practice
2. **Integration Framework**: Methodology for combining data types
3. **Performance Analysis**: Comprehensive comparison across all phases
4. **Design Guidelines**: Recommendations for AAC system designers

## Running Experiments

```bash
# Run all Phase 3 experiments
python run_phase3.py

# Run specific experiment types
python experiments/profile_enhanced.py --config configs/profile_enhanced.yaml
python experiments/social_context.py --config configs/social_context.yaml
python experiments/temporal_filtering.py --config configs/temporal_filter.yaml
python experiments/hybrid_retrieval.py --config configs/hybrid_retrieval.yaml
```

## Integration with Other Phases

### Data Flow
1. **From Phase 1**: Real chat history
2. **From Phase 2**: User profiles and social graphs
3. **In Phase 3**: Enhanced experiments with both

### Comparison Framework
- Use `analysis/cross_phase_comparison.py`
- Compare metrics across all three phases
- Generate unified visualizations

## Success Criteria

1. **Technical**: All experiments run without errors
2. **Data**: Successfully integrate real and synthetic data
3. **Analysis**: Clear insights into context value
4. **Papers**: Ready for publication-quality results

## Timeline

- **Week 1**: Data preparation and mapping
- **Week 2**: Core experiments implementation
- **Week 3**: Analysis and comparison
- **Week 4**: Paper writing and visualization