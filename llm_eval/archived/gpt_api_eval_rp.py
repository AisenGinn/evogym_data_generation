import json
import re
import time
from tqdm import tqdm
from openai import OpenAI
import argparse
import os
import ast
import random
import math
import copy

# This mapping helps convert between numeric indices and letters.
ANSWER_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    "A": 0, "B": 1, "C": 2, "D": 3
}

# Initialize GPT O3 Mini High API client.
def init_gpt_client(api_key):
    client = OpenAI(api_key=api_key)
    return client

# Load and prepare questions from a JSON file.
def load_and_prepare_questions(file_path):
    with open(file_path, 'r') as file:
        questions = json.load(file)

    # Combine additional descriptions with the main question.
    for q in questions:
        combined_description = (
            f"{q['task_description']} "
            f"{q['structure_description']} "
            f"{q['actuation_description']}"
        )
        # Append the combined description to the question text.
        q["question"] = q['question'] + " " + combined_description

    return questions

def extract_answers_and_calculate_accuracy(raw_output, batch_questions, total_correct_count, total_count):
    """
    Extracts answers from the GPT response (expected to be a Python list of letter choices),
    compares them with the correct answers, and calculates accuracy.

    Args:
        raw_output (str): The GPT response containing a Python list of answers.
        batch_questions (list): The list of questions corresponding to the batch.

    Returns:
        list: A list of dictionaries containing final answer, correct answer, and accuracy.
        int: Updated total_correct_count.
        int: Updated total_count.
    """
    try:
        # Extract the list of answers from the response.
        answer_list = ast.literal_eval(raw_output.strip())  # Convert string to list
        if not isinstance(answer_list, list) or len(answer_list) != len(batch_questions):
            raise ValueError("Invalid format: Response is not a list or length mismatch.")

        results = []
        for i, (answer, question) in enumerate(zip(answer_list, batch_questions)):
            total_count += 1
            final_answer = answer.upper()  # Ensure uppercase format
            correct_answer = ANSWER_DICT[question["correct_answer"]]  # Get correct letter

            is_correct = (final_answer == correct_answer)
            total_correct_count += int(is_correct)
            accuracy = total_correct_count / total_count

            result_dict = {
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "accuracy": f"{accuracy * 100:.2f}%"
            }
            results.append(result_dict)

        return results, total_correct_count, total_count

    except (SyntaxError, ValueError, IndexError) as e:
        print(f"Error parsing GPT response: {e}")
        return [], total_correct_count, total_count

def evaluate_multiple_choice_batch(client, questions, result_path, times):
    """
    Evaluates multiple questions in batches.
    Each unique question is repeated 'times' times within a batch.
    The total number of questions per batch is the smallest multiple of 'times' that is at least 20.
    Only up to 500 unique questions are evaluated.
    """

    # Compute batch parameters.
    # total number of questions in a batch must be a multiple of times and at least 20.
    batch_total = math.ceil(20 / times) * times  
    unique_per_batch = batch_total // times

    with open(result_path, 'w') as output_file:
        output_file.write("[\n")
        total_correct_count = 0
        total_count = 0        
        iter_count = 0
        max_iter = math.ceil(500 / batch_total)

        # Iterate over unique questions in batches.
        for batch_start in tqdm(range(0, len(questions), unique_per_batch), total=max_iter):
            if iter_count == max_iter:
                break
            batch_unique_questions = questions[batch_start: batch_start + unique_per_batch]
            # Create a batch where each unique question is repeated 'times' times with random shuffling.
            batch_questions = []
            for q in batch_unique_questions:
                for rep in range(times):
                    new_q = copy.deepcopy(q)
                    # Get the original correct index.
                    original_index = new_q["correct_answer"] # correct index
                    original_choice = new_q["choices"][original_index] # correct choice
                    # Randomly shuffle the choices.
                    random.shuffle(new_q["choices"])
                    # Update correct answer based on new position.
                    new_index = new_q["choices"].index(original_choice) # new index of correct choice
                    new_q["correct_answer"] =new_index # correct index
                    batch_questions.append(new_q)

            # Build the prompt for the current batch.
            prompt = ""
            for index, q in enumerate(batch_questions):
                choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
                prompt += f"{index+1}: {q['question']}\nChoices:\n{choices_text}\n\n"
            prompt += (f"Here are {len(batch_questions)} questions, answer all of them "
                       "and return the letter choice answers in a Python list format without any other text.")

            # Call the GPT API until a valid response is obtained.
            while True:
                try:
                    responses = client.chat.completions.create(
                        model="o3-mini-2025-01-31",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    raw_output = responses.choices[0].message.content  # Extract response
                    results, total_correct_count, total_count = extract_answers_and_calculate_accuracy(
                        raw_output, batch_questions, total_correct_count, total_count
                    )

                    # Save results.
                    for result in results:
                        output_file.write(json.dumps(result) + ",\n")
                    output_file.flush()
                    break

                except Exception as e:
                    if "Expecting value: line 1 column 1 (char 0)" in str(e):
                        print("API error, retrying in 3 seconds...")
                        time.sleep(3)
                        continue
                    else:
                        print(f"API Error: {e}")
                        break
            
            iter_count += 1
                    
    # Fix JSON formatting: Remove last comma and add closing bracket.
    with open(result_path, 'rb+') as output_file:
        output_file.seek(0, 2)  # Move to end of file.
        output_file.seek(output_file.tell() - 2, 0)  # Move 2 bytes back to remove last comma.
        output_file.truncate()  # Remove last comma.
        output_file.write(b"\n]")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions using GPT API.")    
    parser.add_argument("--env_id", type=str, required=True,
                        help="Specify an environment ID to run a single environment, or 'all' for all environments.")
    parser.add_argument("--num_choices", type=int, choices=[2, 4], default=2,
                        help="Number of choices per question (2 or 4).")
    parser.add_argument("--mode", type=str, choices=["easy", "hard"], default="easy",
                        help="Difficulty level of the questions. Easy choices will have larger differences in reward values.")
    parser.add_argument("--description", type=str, choices=["better", "worse"], default="better",
                        help="Ask LLMs to pick better or worse performance choices.")
    parser.add_argument("--times", type=int, choices=[1, 2, 6, 8, 10], default=1,
                        help="The number of times each question is repeated in a batch.")
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_questions",
                        help="Path to the question folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_answers_shuffle/gpto3mini",
                        help="Output path for the generated answer JSON.")
    
    args = parser.parse_args()
    env_list = ["Walker-v0", "BidirectionalWalker-v0", "Jumper-v0", "Balancer-v0", "UpStepper-v0", "GapJumper-v0",
                "Carrier-v0", "Carrier-v1", "Pusher-v0", "Pusher-v1", "Climber-v0", "Climber-v1"]

    env_names = env_list if args.env_id == "all" else [args.env_id]
    
    # Initialize GPT API client.
    api_key = "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"  
    client = init_gpt_client(api_key)
            
    for env_name in env_names:
        source_QA_path = os.path.join(args.data_dir,
            f"{env_name}/{env_name}_QA_{args.description}_{args.mode}_{args.num_choices}_repeat.json")
        output_env_dir = os.path.join(args.output_dir, env_name)
        os.makedirs(output_env_dir, exist_ok=True)
        # Evaluate questions.
        questions = load_and_prepare_questions(source_QA_path)        
        output_ANS_path = os.path.join(output_env_dir,
            f"{env_name}_ANS_{args.description}_{args.mode}_{args.num_choices}_repeat{args.times}_.json")
        evaluate_multiple_choice_batch(client, questions, output_ANS_path, times=args.times)

if __name__ == "__main__":
    main()
