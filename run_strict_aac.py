import json
import llm
import sys
import csv

# --- CONFIGURATION ---
# Use the Thinking model for best reasoning, or "gemini-2.5-pro" for speed
GEN_MODEL_NAME = "gemini-2.5-pro"
JUDGE_MODEL_NAME = "gemini-2.5-pro"

PROFILE_FILE = "Dave_context.json"
TRANSCRIPT_FILE = "transcript_data_2.json"
OUTPUT_CSV = "aac_full_spectrum_results.csv"


def load_data():
    try:
        with open(PROFILE_FILE, "r") as f:
            profile = json.load(f)
        with open(TRANSCRIPT_FILE, "r") as f:
            transcript = json.load(f)
        return profile, transcript
    except FileNotFoundError:
        print("Error: Data files not found.")
        sys.exit(1)


def generate_prediction(model, system_prompt, user_prompt):
    try:
        # Low temp for consistency
        response = model.prompt(user_prompt, system=system_prompt, temperature=0.2)
        text = response.text().strip()
        # Clean thinking tags if present
        if "</thinking>" in text:
            text = text.split("</thinking>")[-1].strip()
        return text.replace("\n", " ")
    except Exception as e:
        return f"Error: {e}"


def evaluate_intent_match(model, target, prediction):
    """
    Judges the semantic closeness of the prediction to the target (1-10).
    """
    judge_system = "You are a semantic evaluator for an AAC system."
    judge_prompt = f"""
    Compare these two phrases.
    1. TARGET INTENT: "{target}"
    2. AI PREDICTION: "{prediction}"
    
    Rate the similarity of the INTENT (Actionability/Meaning) on a scale of 1 to 10.
    1 = Completely wrong/harmful.
    5 = Vague or related topic but wrong action.
    10 = Perfect match.
    
    Return ONLY the integer.
    """
    try:
        response = model.prompt(judge_prompt, system=judge_system, temperature=0.0)
        score = "".join(filter(str.isdigit, response.text()))
        return int(score) if score else 0
    except:
        return 0


def run_experiment():
    profile, transcript = load_data()
    gen_model = llm.get_model(GEN_MODEL_NAME)
    judge_model = llm.get_model(JUDGE_MODEL_NAME)

    base_system_prompt = f"""
    You are a Predictive AAC System for a user named Dave.
    USER PROFILE:
    {json.dumps(profile, indent=2)}
    INSTRUCTIONS:
    1. Predict the most likely short phrase Dave wants to say.
    2. Base prediction ONLY on the INPUT DATA provided.
    3. Output ONLY the predicted phrase.
    """

    print(f"Running Full Spectrum Experiment...")
    print("=" * 70)

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        # Headers for all 5 Hypotheses
        fieldnames = [
            "ID",
            "Target",
            "H1_Time",
            "H1_Score",
            "H2_Who",
            "H2_Score",
            "H3_Loc",
            "H3_Score",
            "H5_Speech",
            "H5_Score",
            "H4_Full",
            "H4_Score",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for turn in transcript:
            print(f"\nProcessing ID {turn['id']} ({turn['target_ground_truth']})...")

            # Data Points
            time = turn["metadata"]["time"]
            participants = ", ".join(turn["metadata"]["active_participants"])
            location = turn["metadata"]["location"]
            prev_utterance = (
                f"Previous Speaker said: '{turn['dialogue_history']['last_utterance']}'"
            )

            # --- H1: TIME ONLY ---
            h1_pred = generate_prediction(
                gen_model, base_system_prompt, f"INPUT: Time: {time}"
            )
            h1_score = evaluate_intent_match(
                judge_model, turn["target_ground_truth"], h1_pred
            )

            # --- H2: TIME + WHO ---
            h2_pred = generate_prediction(
                gen_model,
                base_system_prompt,
                f"INPUT: Time: {time}. People: {participants}",
            )
            h2_score = evaluate_intent_match(
                judge_model, turn["target_ground_truth"], h2_pred
            )

            # --- H3: TIME + WHO + LOC ---
            h3_pred = generate_prediction(
                gen_model,
                base_system_prompt,
                f"INPUT: Time: {time}. People: {participants}. Location: {location}",
            )
            h3_score = evaluate_intent_match(
                judge_model, turn["target_ground_truth"], h3_pred
            )

            # --- H5: SPEECH ONLY (No environmental context) ---
            h5_pred = generate_prediction(
                gen_model, base_system_prompt, f"INPUT: {prev_utterance}"
            )
            h5_score = evaluate_intent_match(
                judge_model, turn["target_ground_truth"], h5_pred
            )

            # --- H4: FULL CONTEXT (All inputs) ---
            h4_pred = generate_prediction(
                gen_model,
                base_system_prompt,
                f"INPUT: Time: {time}. People: {participants}. Location: {location}. {prev_utterance}",
            )
            h4_score = evaluate_intent_match(
                judge_model, turn["target_ground_truth"], h4_pred
            )

            # Write to CSV
            writer.writerow(
                {
                    "ID": turn["id"],
                    "Target": turn["target_ground_truth"],
                    "H1_Time": h1_pred,
                    "H1_Score": h1_score,
                    "H2_Who": h2_pred,
                    "H2_Score": h2_score,
                    "H3_Loc": h3_pred,
                    "H3_Score": h3_score,
                    "H5_Speech": h5_pred,
                    "H5_Score": h5_score,
                    "H4_Full": h4_pred,
                    "H4_Score": h4_score,
                }
            )

            # Console Summary
            print(f"  H1 (Time):   {h1_pred:<20} Score: {h1_score}")
            print(f"  H2 (+Who):   {h2_pred:<20} Score: {h2_score}")
            print(f"  H3 (+Loc):   {h3_pred:<20} Score: {h3_score}")
            print(f"  H5 (Speech): {h5_pred:<20} Score: {h5_score}")
            print(f"  H4 (Full):   {h4_pred:<20} Score: {h4_score}")

    print("\n" + "=" * 70)
    print(f"Experiment Complete. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiment()
