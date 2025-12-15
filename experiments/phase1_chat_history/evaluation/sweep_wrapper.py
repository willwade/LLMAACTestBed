import os
import subprocess

# Parameters to sweep
context_filters = ["none", "time"]
conversation_windows = [0, 1, 3]
time_windows = [2.0]  # hours
geo_radii = [0.5]  # km

# Data and model settings
data_file = "baton-export-2025-11-24-nofullstop.json"
model_name = "gpt-4o-mini"
sample_size = "20"  # Set your desired sample size here

# Base output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Construct base command
base_command = [
    "python",
    "run_evaluation.py",
    "--data",
    data_file,
    "--model",
    model_name,
    "--sample-size",
    sample_size,  # Add the sample size argument
]

# Iterate through all combinations of parameters
for context_filter in context_filters:
    for conv_window in conversation_windows:
        for time_window in time_windows:
            for geo_radius in geo_radii:
                # Construct a descriptive filename
                output_file = os.path.join(
                    output_dir,
                    f"results_cf-{context_filter}_cw-{conv_window}_tw-{time_window}_gr-{geo_radius}.csv",
                )

                # Construct the command for this combination
                command = base_command + [
                    "--context-filter",
                    context_filter,
                    "--conversation-window",
                    str(conv_window),
                    "--time-window-hours",
                    str(time_window),
                    "--geo-radius-km",
                    str(geo_radius),
                    "--output",
                    output_file,  # Specify the output file
                ]

                # Print the command to be executed
                print(f"Running: {' '.join(command)}")

                # Execute the command
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {e}")
                    # Optionally, continue to the next iteration or exit

print("Sweep complete!")
