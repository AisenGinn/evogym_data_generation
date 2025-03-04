from google import genai
import json
import math
import numpy as np
import time
from tqdm import tqdm
import copy
import argparse
import os
import ast 
import re
import random

# This mapping helps convert between numeric indices and letters.
ANSWER_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    "A": 0, "B": 1, "C": 2, "D": 3
}

# Updated task descriptions
task_description_dict = {
    "Walker-v0": "The robot walks as far as possible on flat terrain.",
    "BridgeWalker-v0": "The robot walks as far as possible on a soft rope-bridge.",
    "BidirectionalWalker-v0": "The robot walks bidirectionally towards changing goals.",
    
    "Carrier-v0": "The robot catches a box initialized above it and carries it as far as possible.",
    "Carrier-v1": "The robot carries a box to a table and places it on the table.",
    "Pusher-v0": "The robot pushes a box initialized in front of it.",
    "Pusher-v1": "The robot pushes or drags a box initialized behind it in the forward direction.",
    
    "Climber-v0": "The robot climbs as high as possible on a vertical wall.",
    # "Climber-v1": "The robot climbs through a vertical channel made of mixed rigid and soft materials.",
    
    "UpStepper-v0": "The robot climbs up stairs of varying lengths.",
    "GapJumper-v0": "The robot traverses a series of floating platforms, each spaced 5 units apart, all at the same height.",

    "Jumper-v0": "The robot jumps as high as possible in place on flat terrain.",
    "Balancer-v0": "The robot is initialized on top of a thin pole and balances on it.",

}

difficulty_dict = {
    # Locomotion Tasks
    "Walker-v0": "easy",
    "BridgeWalker-v0": "medium",
    "BidirectionalWalker-v0": "hard",
    "UpStepper-v0": "medium",
    "GapJumper-v0": "hard",

    # Object Manipulation Tasks
    "Carrier-v0": "medium",
    "Carrier-v1": "hard",
    "Pusher-v0": "easy",
    "Pusher-v1": "medium",
    # "Thrower-v0": "hard",
    
    # Climbing Tasks
    "Climber-v0": "medium",
    # "Climber-v1": "hard",
    
    # Balance and Jumping Tasks
    "Jumper-v0": "medium",
    "Balancer-v0": "hard",
}

def load_data(source_path, recur="norepeat", num_choices=2):
    """
    Load and preprocess the data.
    
    Args:
        source_path (str): Path to the JSON file.
        num_choices (int): Number of choices per question (default: 2).
    
    Returns:
        list: Preprocessed data sorted by reward with duplicate 5x5 structures removed.
    """
    with open(source_path, 'r') as file:
        sourcedata = json.load(file)

    # Sort in descending order by reward
    sourcedata.sort(key=lambda x: x["reward"], reverse=True)

    # Ensure the data length is a multiple of num_choices
    sourcedata = sourcedata[:len(sourcedata) - (len(sourcedata) % num_choices)]
    
    if recur == "norepeat":
        # Remove entries with duplicate 5x5 structures
        seen_structures = set()
        unique_data = []
        for entry in sourcedata:
            # Convert the 5x5 structure (list of lists) to a tuple of tuples (hashable)
            structure_tuple = tuple(tuple(row) for row in entry["structure"])
            if structure_tuple not in seen_structures:
                seen_structures.add(structure_tuple)
                unique_data.append(entry)
        return unique_data

    return sourcedata

def create_questions_context(sourcedata, num_choices, shuffle=False):
    """
    Create questions with the specified number of choices (2 or 4).
    If mode is "high", the choices come from largely separated reward buckets.
    If mode is "low", each bucket is further subdivided so that the choices are closer in reward.
    
    Args:
        sourcedata (list): Preprocessed data sorted by reward.
        mode (str): "high" or "low" difference in reward values.
        num_choices (int): Number of choices per question.
    
    Returns:
        list: A list of dictionaries containing questions.
    """
    
    splits = num_choices
    questions = []

    # Original behavior: split data into num_choices groups and zip them.
    sub_lists = np.array_split(sourcedata, splits)
    sub_lists = [list(sublist) for sublist in sub_lists]
    # questiondata = [list(items) for items in zip(*sub_lists)]
    low_reward_list = sub_lists[1][:600]
    #high_reward_list = sub_lists[2][:250]
    reference_list = sub_lists[0][:60]

    total_list = low_reward_list #+ high_reward_list

    if shuffle: random.shuffle(total_list)

    # For each reference element, create 20 pairs.
    for i, ref in enumerate(reference_list):
        # Slice out 20 elements from total_list for this reference.
        group = total_list[i * 20 : (i + 1) * 20]

        # Create 20 pairs where each pair is [total_list_item, reference]
        paired_options = [[option, ref] for option in group]
    
        for eachquestion in paired_options:
            # Randomize the order.
            random.shuffle(eachquestion)
            
            task = eachquestion[0]['env_name']
            structures = [option['structure'] for option in eachquestion]
            rewards = [option['reward'] for option in eachquestion]
            
            question = {
                "question": f"Which of the following robot structures will perform better in the evogym {task} task?",
                "choices": structures,
                "correct_answer": rewards.index(max(rewards)),
                "reward": rewards,
                "env_name": task,
                "difficulty": difficulty_dict.get(task, "unknown"),
                "task_description": task_description_dict.get(task, "Task description not available."),
                "structure_description": (
                    "The robot structure uses a unified multi-material voxel-based representation. "
                    "The robot is represented as a 5x5 material matrix. The entries of the material matrix "
                    "are integers corresponding to a voxel type from the set {Empty, Rigid, Soft, Horizontal Actuator, Vertical Actuator}. "
                    "0 stands for Empty voxel, 1 stands for Rigid voxel, 2 stands for Soft voxel, "
                    "3 stands for Horizontal Actuator voxel, 4 stands for Vertical Actuator voxel. "
                    "All pairs of adjacent voxels are connected to each other."
                ),
                "actuation_description": (
                    "The robot's controller is optimized to maximize task performance by actuating its structure through precise deformation."
                )
            }
            questions.append(question)

    return questions

# Initialize GPT O3 Mini High API client.
def init_gemini_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

# Load and prepare questions from a JSON file.
def prepare_questions(questions):
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
        list: A list of dictionaries containing question ID, final answer, correct answer, and accuracy.
        float: The overall accuracy of the batch.
    """
    try:
        # Extract the list of answers from the response
        raw_output = re.sub(r"```(?:python)?", "", raw_output).strip()
        answer_list = ast.literal_eval(raw_output.strip())  # Convert string to list
        if not isinstance(answer_list, list) or len(answer_list) != len(batch_questions):
            raise ValueError("Invalid format: Response is not a list or length mismatch.")

        results = []
        #correct_count = 0  # Track correct answers

        for i, (answer, question) in enumerate(zip(answer_list, batch_questions)):
            total_count += 1
            final_answer = answer.upper()  # Ensure uppercase format
            correct_answer = ANSWER_DICT[question["correct_answer"]]  # Get correct letter

            is_correct = (final_answer == correct_answer)
            total_correct_count += int(is_correct)
            accuracy = total_correct_count / total_count

            result_dict = {
                #"question_id": question["question_id"],
                "final_answer": final_answer,
                "correct_answer": correct_answer,
                "accuracy": f"{accuracy * 100:.2f}%"
            }

            results.append(result_dict)

        #overall_accuracy = correct_count / len(batch_questions)  # Calculate batch accuracy
        return results, total_correct_count, total_count

    except (SyntaxError, ValueError, IndexError) as e:
        print(f"Error parsing GPT response: {e}")
        return [], total_correct_count, total_count  # Return empty results and 0 accuracy on failure

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

            #print(prompt)
            # Call the GPT API until a valid response is obtained.
            while True:
                try:
                    responses = client.models.generate_content(
                        model="gemini-2.0-flash",  # Ensure the correct model name
                        contents = prompt
                    )

                    # Extract the model's reply.
                    raw_output = responses.text  # Extract response
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
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_data",
                        help="Path to the data folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_answers_reference/gemini2.0",
                        help="Output path for the generated answer JSON.")
    parser.add_argument("--times", type=int, choices=[1, 2, 3], default=1, help="The number of times of answers.")
    parser.add_argument("--shuffle", action="store_true", help="whether shuffle the high reward difference and low reward difference.")
    
    args = parser.parse_args()
    env_list = ["Walker-v0", "BidirectionalWalker-v0", "Jumper-v0", "Balancer-v0", "UpStepper-v0", "GapJumper-v0",
                "Carrier-v0", "Carrier-v1", "Pusher-v0", "Pusher-v1", "Climber-v0", "Climber-v1"]

    env_names = env_list if args.env_id == "all" else [args.env_id]
    
    # Initialize GPT API client.
    api_key = "YOUR_API_KEY"
    client = init_gemini_client(api_key)
            
    for env_name in env_names:
        source_path = os.path.join(args.data_dir, f"test_ga_{env_name}/{env_name}_results.json")
        source_data = load_data(source_path)
        questions = create_questions_context(source_data, args.num_choices, args.shuffle)
        questions = prepare_questions(questions)
        output_env_dir = os.path.join(args.output_dir, env_name)
        os.makedirs(output_env_dir, exist_ok=True)
        # Evaluate questions.)
        ex = "shuffle" if args.shuffle else "noshuffle"        
        output_ANS_path = os.path.join(output_env_dir,
            f"{env_name}_ANS_{ex}.json")
        evaluate_multiple_choice_batch(client, questions, output_ANS_path, times=args.times)

if __name__ == "__main__":
    main()
