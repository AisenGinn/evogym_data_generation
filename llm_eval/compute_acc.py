import os
import json
import pandas as pd
import argparse
from collections import defaultdict

def compute_and_save_acc(root_folder_path):   # Define the root folder containing the environment subfolders
    root_folder = root_folder_path  # Change this to your actual folder path

    # Dictionary to store results for each environment
    success_rates = defaultdict(lambda: {"easy_correct": 0, "easy_total": 0, "hard_correct": 0, "hard_total": 0})

    # Iterate through each environment folder
    for env_name in os.listdir(root_folder):
        env_path = os.path.join(root_folder, env_name)
        
        if os.path.isdir(env_path):  # Ensure it's a directory
            # Process each JSON file in the environment folder
            for file_name in os.listdir(env_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(env_path, file_name)
                    
                    # Determine difficulty (easy or hard)
                    if "easy" in file_name:
                        difficulty = "easy"
                    elif "hard" in file_name:
                        difficulty = "hard"
                    else:
                        continue  # Skip files that do not match the pattern
                    
                    # Read JSON data
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Ensure data is a list of dictionaries
                    if isinstance(data, list):
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
    output_path = os.path.join(root_folder, "success_rates.csv")
    df.to_csv(output_path, index=False)

    print(f"Success rates saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy for different environments.")   
    parser.add_argument("--answer_dir", type=str, default="/media/hdd2/users/changhe/saved_answers/gpto3mini", help="Output path for the generated answer JSON.") 
    
    args = parser.parse_args()
    compute_and_save_acc(args.answer_dir)

if __name__ == "__main__":
    main()
