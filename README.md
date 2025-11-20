# **Context-Aware AAC Testbed: LLM Experimentation**

**Objective:** To determine if injecting specific contextual data (User Profile, Time, Location) into a Large Language Model (LLM) improves the accuracy and utility of phrase predictions for a user with Motor Neurone Disease (MND).

## **The Hypothesis**

Standard predictive text and generic LLMs fail AAC users because they prioritize "polite conversation" over "functional tools." We hypothesize that by layering **Static Context** (User Profile) and **Dynamic Context** (Time/Location) over speech input, we can move from generic chat to precise intent prediction.

**Subject Persona:** "Dave" – Late-stage MND, telegraphic speech, developer background. **Critical Constraints:** High fatigue (limited breath for speech), temperature dysregulation, dependence on specific equipment (NIV Mask, Fan).

## **Repository Structure**

### **Data & Context**

* **`Dave_context.json`**: The "Brain." Contains the static knowledge graph: medical needs, equipment list (Fan, Mask), social graph (Kelsey, Dawn), and routine markers.  
* **`transcript_data_2.json`**: Ground truth scenarios for Experiment 1 (The strict context test).  
* **`transcript_vague.json`**: Ambiguous input scenarios for Experiment 2 (e.g., "Do you want this?").

### **Experiments (Scripts)**

* **`run_strict_aac.py`**: Runs Exp 1\. Compares 5 levels of context (H1-H5) and uses a secondary LLM to score the semantic accuracy (1-10).  
* **`run_speech_ablation.py`**: Runs Exp 2\. Compares a "Smart" Profile-Aware model against a "Raw" Generic model.  
* **`run_synthesis_test.py`**: Runs Exp 3\. Tests temporal disambiguation by injecting specific time variables.  
* **`plot_results.py`**: Generates the visualization chart from the CSV output.

### **Results**

* **`aac_full_spectrum_results.csv`**: Raw scoring data from Experiment 1\.  
* **`context_advantage_chart.png`**: Visual comparison of Speech-Only vs. Full-Context accuracy.

## **The Knowledge Graph (User Profile)**

The core of our "Smart" system is the **Static Profile** (`Dave_context.json`). This JSON structure acts as the "Long Term Memory" for the LLM, providing it with the medical and social nuances required to interpret telegraphic speech.

{  
  "identity": {  
    "name": "Dave",  
    "age": 45,  
    "condition": "MND (Motor Neurone Disease)",  
    "former\_occupation": "Software Developer",  
    "current\_status": "Wheelchair user, fatigued, reduced speech volume, telegraphic speech patterns."  
  },  
  "medical\_context": {  
    "equipment": \[  
      "Riser-recliner chair",  
      "Neck brace (foam, often uncomfortable/digging in)",  
      "NIV Mask (Non-Invasive Ventilation)",  
      "Syringe driver (arm)",  
      "Air mattress",  
      "Desk fan"  
    \],  
    "symptoms": \[  
      "Temperature dysregulation (rapidly hot/cold)",  
      "Weak neck muscles (head drop)",  
      "Low voice volume",  
      "Shortness of breath"  
    \],  
    "medications": \[  
      "Baclofen (muscle relaxant)",  
      "Liquid meds via tube"  
    \]  
  },  
  "social\_graph": {  
    "Kelsey": {  
      "relation": "Wife / Primary Carer",  
      "occupation": "QA Tester",  
      "personality": "Efficient, protective, stressed, anticipates needs, dislikes admin.",  
      "dynamic": "Dave feels guilty about her workload; she manages his 'inputs' (meds/calls) and 'outputs' (interpreting speech)."  
    },  
    "Dawn": {  
      "relation": "Mother",  
      "hobbies": \["Knitting"\],  
      "personality": "Emotional, anxious, loves Dave but requires emotional management.",  
      "communication\_barrier": "Dave finds Facetime exhausting; she struggles to hear/lipread him."  
    },  
    "Mia": {  
      "relation": "Step-daughter",  
      "age": "School age",  
      "notes": "Arrival home usually increases noise/chaos."  
    },  
    "Sarah": {  
      "relation": "Specialist Nurse",  
      "role": "Wound care, medication review."  
    }  
  },  
  "vocabulary\_preferences": {  
    "style": "Telegraphic, keyword-focused, developer metaphors (bugs, drivers, shutting down).",  
    "common\_requests": \[  
      "Fan on",  
      "Window open",  
      "Neck brace adjust",  
      "TV volume down",  
      "Water/Straw",  
      "Mask on"  
    \]  
  },  
  "recent\_events": {  
    "news": "Flooding in Derbyshire/local area.",  
    "admin\_issues": "Pharmacy lost the ticket for liquid Baclofen.",  
    "medical": "Pressure sore check on heel (healing) and arm (reaction to adhesive)."  
  }  
}

## **Experiment 1: The Baseline (Context vs. Noise)**

**Script:** `uv run run_strict_aac.py`

**Logic:** This script iterates through specific interaction scenarios. It generates predictions based on 5 distinct Hypotheses (H) ranging from "Blind Guessing" to "Full Context". A "Judge" LLM then scores the prediction against the Ground Truth intent.

| Hypothesis | Inputs Provided to LLM |
| ----- | ----- |
| **H1 (Time)** | Time Only (Blind) |
| **H2 (Social)** | Time \+ People |
| **H3 (Loc)** | Time \+ People \+ Location |
| **H5 (Speech)** | Previous Speech \+ **Static Profile** |
| **H4 (Full)** | Previous Speech \+ **Static Profile** \+ Time \+ Loc \+ People |

**Key Findings:**

* **H5 (Speech \+ Profile) is strong:** Scoring 9s/10s on clear requests. Knowing the user's profile is 80% of the battle.  
* **H4 (Full Context) is safer:** In Scenario 103 ("Fan Incident"), Kelsey says "You'll freeze." H5 (Speech Only) agreed and predicted "Window shut." H4 used the medical profile \+ environmental context to correctly predict **"Fan on"**(due to temperature dysregulation).

*Figure 1: While Speech+Profile (Grey) performs well generally, Full Context (Green) closes the gap in safety-critical or socially complex scenarios (ID 102, 105).*

## **Experiment 2: The Ablation Test (The Value of Profile)**

**Script:** `uv run run_speech_ablation.py`

**Logic:** We stripped away the `Dave_context.json` to see how a "Raw" LLM (like ChatGPT/Siri) handles vague cues compared to our "Smart" system.

**Results:**

| Input Speech | SMART (With Profile) | RAW (No Profile) | Analysis |
| ----- | ----- | ----- | ----- |
| *"You look a bit flushed."* | **Fan on.** | *"I do feel a bit warm."* | **The "Tool" vs. "Chatbot" Gap.** The Raw model offers polite conversation (useless). The Smart model uses the medical graph to trigger a tool. |
| *"You sound rattly."* | **Mask on.** | *"Yes, please."* | **Specific vs. Generic.** The Smart model identifies the specific equipment needed. |
| *"Do you want the usual?"* | **Yeah please.** | *"Yes, please."* | **The Temporal Failure.** Both models failed to identify *what* "the usual" is because the Timestamp was missing. |

## **Experiment 3: The Synthesis (The Value of Time)**

**Script:** `uv run run_synthesis_test.py`

**Logic:** We fed the exact same speech input (`"Do you want the usual?"`) into the model but injected different **Time** variables into the system prompt.

**Results:**

* **Input:** "Do you want the usual?" \+ **Time: 08:00** $\\rightarrow$ **Prediction:** "Meds please."  
* **Input:** "Do you want the usual?" \+ **Time: 20:00** $\\rightarrow$ **Prediction:** "Mask on."

**Conclusion:** Static profiles provide the *vocabulary*, but Dynamic Context (Time) provides the *selection logic*.

## **Architecture Priorities for Context-Aware AAC**

Based on these findings, an effective AI-AAC system must prioritize data collection in this order:

### **Priority 1: The "Static Profile" (The Brain)**

**Verdict: CRITICAL / MUST HAVE**

* **Why:** Without this, the system is a chatbot, not a tool. It handles the "Tool vs. Chatbot" gap.  
* **Data:** Medical Needs (symptoms), Equipment List (Fan, Mask), Vocabulary Rules (Telegraphic speech).

### **Priority 2: The "Social Graph" (The Filter)**

**Verdict: HIGH VALUE**

* **Why:** Prevents social burnout. Allows the system to distinguish between "Nurse" (Medical intent) and "Mom" (Social intent).  
* **Data:** Entity Names, Roles, and **Energy Cost** (High load vs. Low load people).

### **Priority 3: Temporal Context (The Switch)**

**Verdict: HIGH VALUE / LOW EFFORT**

* **Why:** Solves ambiguity ("The usual") cheaply using the system clock.  
* **Data:** Routine Markers mapped to probabilities (08:00 \= Meds).

### **Priority 4: Interlocutor Voice Data (The Input)**

**Verdict: MEDIUM PRIORITY**

* **Why:** We do not need to *clone* the partner's voice, only Identify *who* is speaking (Diarization) to trigger the Social Graph rules.  
* **Data:** Accurate Speech-to-Text and simple Speaker Identification tags.

