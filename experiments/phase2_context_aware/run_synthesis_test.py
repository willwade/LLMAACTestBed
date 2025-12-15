import json

import llm

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-pro"
PROFILE_FILE = "Dave_context.json"


def run_experiment():
    try:
        with open(PROFILE_FILE) as f:
            profile = json.load(f)
    except FileNotFoundError:
        print(f"Error: {PROFILE_FILE} not found.")
        return

    model = llm.get_model(MODEL_NAME)

    scenarios = [
        {
            "speech": "Do you want the usual?",
            "time": "08:00",
            "context_note": "Morning",
        },
        {
            "speech": "Do you want the usual?",
            "time": "20:00",
            "context_note": "Evening",
        },
    ]

    print(f"Running Temporal Context Test on {MODEL_NAME}")
    print("=" * 60)

    for sc in scenarios:
        # FIX: Define the prompt INSIDE the loop using an f-string.
        # This avoids the conflict between JSON braces and .format() braces.
        current_system = f"""
        You are an AAC assistant for Dave.
        
        USER PROFILE:
        {json.dumps(profile, indent=2)}
        
        CURRENT CONTEXT:
        Time: {sc['time']}
        
        INSTRUCTIONS:
        Predict Dave's response based on the input speech. 
        If the speech is vague (e.g. "the usual"), use the TIME and the PROFILE to guess the routine.
        Output only the short phrase response.
        """

        input_text = f"Kelsey said: '{sc['speech']}'"

        # Run prediction
        prediction = model.prompt(input_text, system=current_system, temperature=0.1).text().strip()

        print(f"Time: {sc['time']} ({sc['context_note']})")
        print(f"Input: '{sc['speech']}'")
        print(f"Prediction: {prediction}")
        print("-" * 40)


if __name__ == "__main__":
    run_experiment()
