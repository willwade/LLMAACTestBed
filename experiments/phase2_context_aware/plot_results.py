import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the results
csv_file = "aac_full_spectrum_results.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print("CSV not found. Please run 'run_full_spectrum_aac.py' first.")
    exit()

# We want to compare H5 (Speech Only) vs H4 (Full Context)
# Let's create a "Delta" column if it doesn't exist
df["Context_Gain"] = df["H4_Score"] - df["H5_Score"]

# Setup the plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create a Bar Chart comparing the scores per ID
# We need to melt the dataframe to make it friendly for Seaborn
df_melted = df.melt(
    id_vars=["ID", "Target"],
    value_vars=["H5_Score", "H4_Score"],
    var_name="Hypothesis",
    value_name="Accuracy_Score",
)

# Custom colors: Grey for Speech Only, Green for Context
palette = {"H5_Score": "#95a5a6", "H4_Score": "#2ecc71"}

ax = sns.barplot(x="ID", y="Accuracy_Score", hue="Hypothesis", data=df_melted, palette=palette)

# Add labels
plt.title("Impact of Context Awareness on AAC Prediction Accuracy", fontsize=16)
plt.xlabel("Scenario ID", fontsize=12)
plt.ylabel("Semantic Accuracy Score (1-10)", fontsize=12)
plt.ylim(0, 11)
plt.legend(title="Model Configuration", labels=["H5: Speech Only", "H4: Full Context"])

# Annotate the bars with the target intent for clarity
for i, row in df.iterrows():
    # We place the text above the pair of bars
    target_text = row["Target"]
    # Simple logic to place text roughly in the middle of the pair
    plt.text(i, 10.2, target_text, ha="center", fontsize=9, rotation=0, color="#34495e")

plt.tight_layout()
plt.savefig("context_advantage_chart.png")
print("Chart saved as 'context_advantage_chart.png'")
plt.show()
