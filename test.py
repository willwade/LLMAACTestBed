import json
import llm
import sys

# 1. Load Data
try:
    with open('dwayne_context.json', 'r') as f:
        social_graph = json.load(f)
    with open('transcript_data.json', 'r') as f:
        transcript = json.load(f)
except FileNotFoundError:
    print("Error: JSON files not found.")
    sys.exit(1)

# 2. Define the Prediction Function
def predict_dwayne(model, hypothesis_name, system_prompt, user_prompt):
    response = model.prompt(user_prompt, system=system_prompt)
    return response.text()

# 3. Setup the LLM (Using GPT-4 or a local model like Llama2/Mistral via 'llm')
# Ideally, use a model capable of good reasoning. 
# If using CLI default:
model = llm.get_model() 

print(f"Running experiments using model: {model.model_id}\n")

# 4. Iterate through Transcript looking for DWAYNE's turns
for turn_index, turn in enumerate(transcript):
    
    if turn['speaker'] != "Dwayne":
        continue

    # Get previous turn for context (if it exists)
    prev_turn = transcript[turn_index - 1] if turn_index > 0 else None
    
    print(f"--- Experimenting on Transcript ID {turn['id']} (Target: '{turn['target_utterance']}') ---")

    # Base System Prompt (Always includes the Social Graph)
    base_system = f"""
    You are an AAC (Augmentative and Alternative Communication) AI assistant for Dwayne.
    
    DWAYNE'S PROFILE:
    {json.dumps(social_graph, indent=2)}
    
    YOUR GOAL:
    Predict what Dwayne wants to say or expand his utterance into a full sentence.
    Output ONLY the predicted sentence.
    """

    # --- HYPOTHESIS 1: JUST TIME OF DAY ---
    h1_prompt = f"The time is {turn['timestamp']}. What does Dwayne say?"
    h1_pred = predict_dwayne(model, "H1", base_system, h1_prompt)
    
    # --- HYPOTHESIS 2: TIME & WHO ---
    participants = ", ".join([p for p in turn['participants'] if p != "Dwayne"])
    h2_prompt = f"The time is {turn['timestamp']}. Dwayne is with {participants}. What does he say?"
    h2_pred = predict_dwayne(model, "H2", base_system, h2_prompt)

    # --- HYPOTHESIS 3: TIME, WHO & LOCATION ---
    h3_prompt = f"Time: {turn['timestamp']}. Location: {turn['location']}. Participants: {participants}. What does he say?"
    h3_pred = predict_dwayne(model, "H3", base_system, h3_prompt)

    # --- HYPOTHESIS 4: H3 + KELLY/PREVIOUS SPEECH ---
    prev_speech = f" The previous speaker said: '{prev_turn['text']}'" if prev_turn else ""
    h4_prompt = f"Time: {turn['timestamp']}. Location: {turn['location']}. Participants: {participants}.{prev_speech} What does Dwayne say?"
    h4_pred = predict_dwayne(model, "H4", base_system, h4_prompt)

    # --- HYPOTHESIS 5: JUST KELLY/PREVIOUS SPEECH (No Time/Loc) ---
    h5_prompt = f"The previous speaker said: '{prev_turn['text']}' if prev_turn else 'Silence'. What does Dwayne say?"
    h5_pred = predict_dwayne(model, "H5", base_system, h5_prompt)

    # 5. Output Results for this Turn
    print(f"Actual Short Utterance: '{turn['actual_utterance']}'")
    print(f"Target Intent:          '{turn['target_utterance']}'")
    print("-" * 20)
    print(f"H1 (Time Only):         {h1_pred.strip()}")
    print(f"H2 (Time + Who):        {h2_pred.strip()}")
    print(f"H3 (Time + Who + Loc):  {h3_pred.strip()}")
    print(f"H4 (Full Context):      {h4_pred.strip()}")
    print(f"H5 (Speech Only):       {h5_pred.strip()}")
    print("\n" + "="*60 + "\n")