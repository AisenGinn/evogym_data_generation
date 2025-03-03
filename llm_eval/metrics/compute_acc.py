import os
import json
import pandas as pd
import argparse
from collections import defaultdict

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def compute_and_save_acc(root_folder_path, recur):   
    """
    Computes accuracy metrics for different environments by loading multiple JSON files.
    
    Args:
        root_folder_path (str): The path to the root directory containing environment subfolders.
        recur (str): Whether to include repeated structures in questions.
    """

    success_rates = defaultdict(lambda: {"easy_correct": 0, "easy_total": 0, "hard_correct": 0, "hard_total": 0})

    # Iterate through each environment folder
    for env_name in os.listdir(root_folder_path):
        env_path = os.path.join(root_folder_path, env_name)
        
        if not os.path.isdir(env_path):  # Ensure it's a valid directory
            continue

        for difficulty in ["easy", "hard"]:  # Loop over difficulty before filtering files
            # Process each JSON file in the environment folder that matches the recurrence pattern
            json_files = [
                os.path.join(env_path, f) for f in os.listdir(env_path)
                if f.endswith(".json") and difficulty in f and any(f"_{recur}_{i}" in f for i in range(1, 4))
            ]
            
            if len(json_files) != 3:  # Ensure all three recurrence files are available
                print(f"Skipping {env_name} {difficulty} - Incomplete repeat files found.")
                continue

            # Sort files to ensure order (repeat_1, repeat_2, repeat_3)
            json_files.sort()

            # Load answer data
            data1, data2, data3 = load_json(json_files[0]), load_json(json_files[1]), load_json(json_files[2])

            # Extract entries from each file matching the difficulty
            for data in [data1, data2, data3]:
                for entry in data:
                    if entry["final_answer"] == entry["correct_answer"]:
                        success_rates[env_name][f"{difficulty}_correct"] += 1
                    success_rates[env_name][f"{difficulty}_total"] += 1

    # Prepare data for CSV
    csv_data = []
    for env, counts in success_rates.items():
        easy_success_rate = (counts["easy_correct"] / counts["easy_total"]) * 100 if counts["easy_total"] > 0 else 0
        hard_success_rate = (counts["hard_correct"] / counts["hard_total"]) * 100 if counts["hard_total"] > 0 else 0
        csv_data.append([env, easy_success_rate, hard_success_rate])

    # Convert to DataFrame
    df = pd.DataFrame(csv_data, columns=["Environment", "Easy Success Rate (%)", "Hard Success Rate (%)"])

    # Save results to CSV
    output_path = os.path.join(root_folder_path, f"{recur}_success_rates.csv")
    df.to_csv(output_path, index=False)

    print(f"Success rates saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy for different environments.")
    parser.add_argument("--recur", type=str, choices=["repeat", "norepeat"], default="norepeat", help="Whether to include repeated structure in questions")   
    parser.add_argument("--answer_dir", type=str, default="/media/hdd2/users/changhe/saved_answers/grok2", help="Path containing answer files.") 
    
    args = parser.parse_args()
    compute_and_save_acc(args.answer_dir, args.recur)

if __name__ == "__main__":
    main()
