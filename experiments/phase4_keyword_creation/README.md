# Phase 4: Keyword-to-Utterance Generation with Contextual Enhancement

## Overview

Phase 4 investigates how effectively LLMs can generate appropriate utterances from minimal keyword inputs, simulating the real-world AAC user experience. This experiment builds on the comprehensive social graph and communication patterns of Dwayne, a real MND patient, to test the boundaries of contextual understanding in language models.

## Research Questions

1. **Primary Question**: How accurately can LLMs predict intended utterances from 1-3 keywords given varying levels of social and environmental context?

2. **Secondary Questions**:
   - What is the impact of social graph context on prediction accuracy?
   - How does environmental context (location, people nearby) affect utterance generation?
   - What is the minimal context required for accurate predictions?
   - How do different LLM providers compare in this task?

## Experiment Structure

### Part 1: Baseline Keyword Testing
**Objective**: Evaluate LLM performance with minimal keyword input (no additional context)

**Method**:
- Input: 3 keywords from Dwayne's communication dataset
- Ground Truth: Expected utterance from DwayneKeyWords.tsv
- Evaluation: Semantic similarity scoring (1-10 scale)
- LLM Providers: Gemini, OpenAI (comparative analysis)

**Example**:
```
Input Keywords: ["Back", "Gel", "Pain"]
Ground Truth: "Put Ibruprofen gel on back (currently needed lower back around the spine towards the left)"
LLM Prediction: "Apply pain relief gel to my back"
Similarity Score: 8/10
```

### Part 2: Contextual Enhancement Testing
**Objective**: Measure improvement in accuracy with incremental context addition

**Context Levels**:
1. **Level 1**: Keywords only (baseline)
2. **Level 2**: Keywords + Location context
3. **Level 3**: Keywords + Location + People nearby
4. **Level 4**: Keywords + Full environmental context
5. **Level 5**: Keywords + Social graph context (relationships, roles)
6. **Level 6**: Keywords + Full context (environmental + social)

**Context Categories**:

#### Environmental Context
- **Location**: Bed, Chair, Table, Kitchen, Living Room, Bathroom
- **Position**: Sitting, Lying, Tilted forward/back
- **Nearby Objects**: Phone, Laptop, Suction Machine, Food/Water
- **Time of Day**: Morning (care routine), Daytime (activities), Evening (family time)

#### Social Context
- **People Present**: Kerry (partner), Carers, Family Members, Friends
- **Relationship Roles**: Primary caregiver, Professional carer, Family, Visitor
- **Interaction History**: Previous communication patterns, established routines
- **Social Dynamics**: Trust levels, communication efficiency, emotional context

### Part 3: Single Keyword Testing
**Objective**: Test the limits of prediction with minimal input (1 keyword + context)

**Focus Areas**:
- High-frequency commands ("Pain", "Scratch", "Move")
- Context-dependent commands ("Chair", "Table", "Transfer")
- Emergency commands ("Sick", "Help", "Emergency")

## Data Sources

### Primary Data
1. **DwayneKeyWords.tsv**: 74 keyword combinations with ground truth instructions
2. **Social Graph**: Comprehensive relational data (family, care team, pets, routines)
3. **Environmental Context**: Home layout, equipment positioning, daily patterns

### Social Graph Components
- **Family Relationships**: Kerry (wife), Evalyn (daughter, 12), Extended family (parents, siblings, nieces/nephews)
- **Care Team**: Professional carers with formal training, Medical team (MND nurses, GP, district nurses)
- **Achievement**: British Empire Medal recipient for bus industry service
- **Pet Relationships**: Kona ("doggy daughter" - integral family member)
- **Daily Routines**: Detailed care schedule from 09:00-00:00 with feeding, medication, and family time
- **Medical Context**: Complex MND symptoms, RIG feeding tube, extensive medication regimen
- **Communication**: Slurred telegraphic speech, Grid software, binary response system

### Keyword Categories
1. **Physical Care**: Scratch, Wipe, Position, Temperature
2. **Medical Needs**: Pain, Medication, Cough Assist, Suction
3. **Environmental Control**: Chair, Table, Doors, Equipment
4. **Communication**: Phone setup, Entertainment
5. **Pet Care**: Kona feeding, toilet, training commands
6. **Emergency**: Sick, Transfer assistance, Respiratory support

## Methodology

### Experimental Design

**Sample Size**: All 74 keyword combinations from Dwayne's dataset
**Evaluation Metric**: LLM-based semantic similarity scoring (1-10 scale)
**Statistical Analysis**: Paired t-tests comparing context levels

### Testing Protocol

1. **Baseline Establishment**: Run Part 1 with all keywords across all LLM providers
2. **Context Incremental Testing**: Progressively add context levels (Part 2)
3. **Minimal Input Testing**: Single keyword with optimal context (Part 3)
4. **Cross-Provider Comparison**: Test with OpenAI GPT-4 and Google Gemini

### Prompt Engineering

**Standard Template**:
```
You are an AAC assistant communicating for Dwayne, an MND patient.

CONTEXT:
{context_variables}

KEYWORDS: {keywords}

Based on the context and keywords, predict Dwayne's intended utterance.
Consider his:
- Current physical position and location
- People present and their relationship to him
- Time of day and routine patterns
- Communication history and preferences
- Medical equipment and care needs

Predicted utterance:
```

### Context Variables

**Location Context**:
```
Location: {bed/chair/table/bathroom}
Position: {sitting/lying/tilted}
Nearby equipment: {list of relevant equipment}
Time of day: {morning/daytime/evening}
```

**Social Context**:
```
People present: {Kerry/Evalyn/family/professional_carers}
Primary caregiver: {yes - Kerry/family OR professional_carer}
Care team training: {formal_training vs family_experience}
Family dynamics: {wife_daugher_extended_family_visitors}
Recent interactions: {care_routines_family_time_medical_needs}
Relationship trust level: {5 for Kerry/family, 2 for professionals, builds over time}
```

**Environmental Context**:
```
Room setup: {furniture arrangement}
Equipment status: {powered/positioned}
Accessibility needs: {current requirements}
Weather/temperature: {environmental factors}
```

## Expected Outcomes

### Hypotheses

1. **H1**: Social context will significantly improve prediction accuracy over environmental context alone
2. **H2**: Relationship-based context (caregiver identity) will provide the highest accuracy boost
3. **H3**: Single keywords with rich context will outperform three keywords without context
4. **H4**: Emergency-related keywords will have higher baseline accuracy than routine care commands
5. **H5**: LLM providers will show different strengths in contextual understanding

### Success Metrics

- **Accuracy Improvement**: >40% improvement from baseline to full context
- **Context Value**: Quantify contribution of each context type
- **Minimal Viable Context**: Identify optimal context-to-accuracy ratio
- **Cross-Provider Variance**: Measure differences between LLM providers

## Data Analysis

### Evaluation Framework

**Primary Metrics**:
- Semantic similarity scores (1-10 scale)
- Accuracy improvement percentages
- Context contribution analysis

**Secondary Metrics**:
- Response appropriateness (safety-critical scenarios)
- Communication efficiency (word count, clarity)
- Error type analysis (misinterpretation patterns)

### Statistical Analysis

**Comparisons**:
- Paired t-tests between context levels
- ANOVA for multi-level context comparison
- Provider performance comparison
- Keyword category analysis

**Visualization**:
- Accuracy improvement curves by context level
- Heat maps of context effectiveness by keyword category
- Provider comparison radar charts
- Error analysis Sankey diagrams

## Implementation Details

### File Structure
```
experiments/phase4_keyword_creation/
├── README.md                          # This document
├── run_experiment.py                 # Main experiment runner
├── evaluation/
│   ├── keyword_evaluator.py          # Core evaluation logic
│   ├── context_builder.py            # Context assembly
│   └── similarity_scorer.py          # LLM-based scoring
├── data/
│   ├── dwayne_keywords.tsv           # Ground truth data
│   ├── social_graph.json             # Comprehensive social context
│   └── context_templates/            # Context pattern templates
└── results/
    ├── baseline_results/             # Part 1 outcomes
    ├── contextual_results/           # Part 2 outcomes
    ├── single_keyword_results/       # Part 3 outcomes
    └── analysis/                     # Statistical analysis outputs
```

### Dependencies
- `lib.llm_clients`: Multi-provider LLM interface
- `lib.evaluation`: Similarity scoring and metrics
- `lib.context`: Social graph and context management
- `pandas`: Data analysis and result aggregation
- `matplotlib/seaborn`: Result visualization

## Ethical Considerations

### Data Privacy
- All personal data anonymized where possible
- Sensitive medical information handled appropriately
- Consent obtained for all personal data usage

### Accessibility Focus
- Results directly benefit AAC user community
- Involves real patients and caregivers in validation
- Maintains dignity and respect throughout process

### Clinical Relevance
- Findings inform actual AAC system design
- Context models applicable to other MND patients
- Potential to improve real-world communication outcomes

## Future Directions

### Expansion Opportunities
1. **Multi-Patient Testing**: Apply framework to other AAC users
2. **Real-Time Testing**: Implement in live AAC systems
3. **Dynamic Context**: Test with changing environmental conditions
4. **Voice Integration**: Add speech-to-text input scenarios
5. **Caregiver Training**: Develop caregiver assistance tools

### Research Applications
1. **Personalization**: Individual user adaptation algorithms
2. **Learning Systems**: Context-aware prediction improvement
3. **Emergency Response**: Critical scenario optimization
4. **Quality of Life**: Long-term impact studies on communication effectiveness

## Related Publications

This experiment builds on and extends several research areas:
- AAC communication enhancement strategies
- Context-aware language model applications
- MND patient care and communication needs
- Social graph modeling in healthcare
- Human-AI interaction in accessibility contexts

---

**Note**: This experiment uses real patient data with appropriate consent and ethical oversight. All sensitive information has been anonymized where necessary, and the research directly contributes to improving communication outcomes for people with MND and other conditions requiring AAC support.