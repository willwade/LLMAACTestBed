import json
import llm
import sys

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-pro"
PROFILE_FILE = "Dave_context.json"
TRANSCRIPT_FILE = "transcript_vague.json"


def run_experiment():
    # Load Data
    try:
        with open(PROFILE_FILE, "r") as f:
            profile = json.load(f)
        with open(TRANSCRIPT_FILE, "r") as f:
            transcript = json.load(f)
    except FileNotFoundError:
        print("Missing JSON files.")
        return

    model = llm.get_model(MODEL_NAME)

    # SYSTEM PROMPT 1: Dave-AWARE (The "H5" from before)
    smart_system = f"""
    You are an AAC assistant for Dave.
    PROFILE: {json.dumps(profile)}
    Predict his response based on the input speech. Short phrases only.
    """

    # SYSTEM PROMPT 2: GENERIC (The "Raw Speech" Test)
    dumb_system = """
    You are a helpful predictive text assistant. 
    Predict the next logical response based on the input speech. 
    Keep it short.
    """

    print(f"Running 'Speech Only' Ablation Test on {MODEL_NAME}")
    print("=" * 60)
    print(f"{'INPUT SPEECH':<40} | {'SMART (Profile)':<20} | {'RAW (No Profile)'}")
    print("-" * 80)

    for turn in transcript:
        user_input = f"Previous speaker said: '{turn['last_utterance']}'"

        # 1. Run Smart
        smart_pred = (
            model.prompt(user_input, system=smart_system, temperature=0.1)
            .text()
            .strip()
        )

        # 2. Run Raw
        raw_pred = (
            model.prompt(user_input, system=dumb_system, temperature=0.1).text().strip()
        )

        # Output
        print(f"'{turn['last_utterance'][:35]:<38}' | {smart_pred:<20} | {raw_pred}")


if __name__ == "__main__":
    run_experiment()
