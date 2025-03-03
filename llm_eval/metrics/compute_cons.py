import os
import json
import pandas as pd
import argparse
from collections import defaultdict

ANSWER_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    "A": 0, "B": 1, "C": 2, "D": 3
}

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def map_question_to_answer_batches(questions, answers, batch_size=20):
    """
    Maps batch indices from questions to answers, handling skipped batches.

    Args:
        questions (str or list): File path to questions JSON or already loaded data.
        answers (str or list): File path to answers JSON or already loaded data.
        batch_size (int): The number of questions per batch.

    Returns:
        dict: A mapping of question batch_start -> answer batch_start.
        list: List of confirmed batch starts.
        list: List of skipped batch starts in questions.
    """

    # Load JSON data if file paths are given
    if isinstance(questions, str):
        questions = load_json(questions)
    if isinstance(answers, str):
        answers = load_json(answers)

    confirm_batch_start = []  # Stores batch starts that align in both files
    skipped_batches = []       # Tracks batch indices skipped in answers
    question_to_answer_mapping = {}  # Maps question batch_start -> answer batch_start

    question_idx = 0  # Tracks batch_start in questions
    answer_idx = 0    # Tracks batch_start in answers

    while question_idx < len(questions):
        confirm = True
        batch_questions = questions[question_idx: question_idx + batch_size]

        # Ensure we don't go out of bounds in answers
        if answer_idx >= len(answers):  
            #print(f"⚠️ Skipping batch_start {question_idx} - No corresponding batch in answers.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        batch_answer = answers[answer_idx: answer_idx + batch_size]

        if len(batch_questions) != len(batch_answer):  
            print(f"⚠️ Skipping batch_start {question_idx} due to batch size mismatch.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        # Validate answers match in both batches
        for i in range(len(batch_questions)):  
            question_answer = ANSWER_DICT[batch_questions[i]['correct_answer']]
            gt = batch_answer[i]["correct_answer"]

            if question_answer != gt:
                confirm = False
                break  # Skip this batch if any mismatch is found

        if confirm:
            confirm_batch_start.append(question_idx)
            question_to_answer_mapping[question_idx] = answer_idx  # Map question batch_start -> answer batch_start
            answer_idx += batch_size  # Increment answer index **only for valid batches**

        # Always increment question index to check the next batch in questions
        question_idx += batch_size

    return question_to_answer_mapping

def compute_and_save_consistency(question_dir, answer_dir, recur):  
    """
    Compute consistency for different environments and save the results as a CSV file.
    
    Args:
        answer_dir_path (str): Path to the root folder containing environment subfolders.
    """

    # Dictionary to store consistency results
    batch_size= 20
    consistency_scores = defaultdict(lambda: {"easy_consistency": 0, "easy_total": 0, 
                                              "hard_consistency": 0, "hard_total": 0})

    # Iterate through each environment folder
    for env_name in os.listdir(answer_dir):
        env_path = os.path.join(answer_dir, env_name)
        q_env_path = os.path.join(question_dir, env_name)
        
        # Skip if either the answer or question path is not a directory.
        if not os.path.isdir(env_path) or not os.path.isdir(q_env_path):
            continue
        
        # Construct file paths for the easy and hard question JSON files.
        q_easy_file = os.path.join(q_env_path, f"{env_name}_QA_better_easy_2_{recur}.json")
        q_hard_file = os.path.join(q_env_path, f"{env_name}_QA_better_hard_2_{recur}.json")
        
        # Check if these files exist
        if not os.path.exists(q_easy_file) or not os.path.exists(q_hard_file):
            print(f"Skipping {env_name} - Missing question file(s).")
            continue

        # Load question data
        q_easy_data = load_json(q_easy_file)
        q_hard_data = load_json(q_hard_file)
        
        if os.path.isdir(env_path):  # Ensure it's a valid directory
            for difficulty in ["easy", "hard"]:
                # Collect file paths for the three repeat files
                repeat_files = [
                    os.path.join(env_path, f) for f in os.listdir(env_path)
                    if f.endswith(".json") and difficulty in f and any(f"_{recur}_{i}" in f for i in range(1, 4))
                ]

                if len(repeat_files) != 3:
                    print(f"Skipping {env_name} {difficulty} - Incomplete repeat files found.")
                    continue  # Skip if we don't have exactly 3 repeat files
                
                # Sort to ensure repeat_1, repeat_2, repeat_3 order
                repeat_files.sort()
                
                # define question data
                q_data = q_easy_data if difficulty=="easy" else q_hard_data
                
                # load answer data
                data1, data2, data3 = load_json(repeat_files[0]), load_json(repeat_files[1]), load_json(repeat_files[2])
                
                
                mapping1 = map_question_to_answer_batches(q_data, data1, batch_size)
                mapping2 = map_question_to_answer_batches(q_data, data2, batch_size)
                mapping3 = map_question_to_answer_batches(q_data, data3, batch_size)
                
                common_batches = set(mapping1.keys()) & set(mapping2.keys()) & set(mapping3.keys())
                
                # define total number of questions use to compute consistency
                N = len(common_batches) * batch_size
                consistency_sum = 0
                
                for batch_start in common_batches:
                    a1_batch = data1[mapping1[batch_start] : mapping1[batch_start]+batch_size]
                    a2_batch = data2[mapping2[batch_start] : mapping2[batch_start]+batch_size]
                    a3_batch = data3[mapping3[batch_start] : mapping3[batch_start]+batch_size]
                    
                    assert len(a1_batch) == len(a2_batch) == len(a3_batch)
                    
                    # compute batch consistency
                    batch_consistency = sum(
                        (int(a1_batch[i]["final_answer"] == a2_batch[i]["final_answer"]) + int(a1_batch[i]["final_answer"] == a3_batch[i]["final_answer"])) / 2
                    for i in range(len(a1_batch))
                    )
                    
                    consistency_sum += batch_consistency

                # Store results
                consistency_scores[env_name][f"{difficulty}_consistency"] += consistency_sum
                consistency_scores[env_name][f"{difficulty}_total"] += N

    # Prepare data for CSV
    csv_data = []
    for env, counts in consistency_scores.items():
        easy_consistency = (counts["easy_consistency"] / counts["easy_total"]) * 100 if counts["easy_total"] > 0 else 0
        hard_consistency = (counts["hard_consistency"] / counts["hard_total"]) * 100 if counts["hard_total"] > 0 else 0
        csv_data.append([env, easy_consistency, hard_consistency])

    # Convert to DataFrame
    df = pd.DataFrame(csv_data, columns=["Environment", "Easy Consistency (%)", "Hard Consistency (%)"])

    # Save results to CSV
    output_path = os.path.join(answer_dir, f"{recur}_consistency_results.csv")
    df.to_csv(output_path, index=False)

    print(f"Consistency results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute consistency for different environments.")
    parser.add_argument("--question_dir", type=str, default="/media/hdd2/users/changhe/saved_questions", help="Path to saved JSON answer files.")
    parser.add_argument("--recur", type=str, choices=["repeat", "norepeat"], default="norepeat", help="Whether include repeated structure in questions")   
    parser.add_argument("--answer_dir", type=str, default="/media/hdd2/users/changhe/saved_answers/grok2", help="Path to saved JSON answer files.") 
    
    args = parser.parse_args()
    compute_and_save_consistency(args.question_dir, args.answer_dir, args.recur)

if __name__ == "__main__":
    main()
