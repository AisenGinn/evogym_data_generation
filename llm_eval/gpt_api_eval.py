import json
import re
import time
from tqdm import tqdm
from openai import OpenAI
import argparse
import os

# This mapping helps convert between numeric indices and letters.
ANSWER_DICT = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
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

# def extract_thinking_and_answer(text):
#     """
#     Extract the reasoning (thinking part) and final answer from the model's response.
#     - The thinking part is everything before "The answer is X."
#     - The answer part is just "A" or "B".
#     """
#     match = re.search(r"(.*)\s*The answer is\s+([AB])\.", text, re.DOTALL)
#     if match:
#         thinking_part = match.group(1).strip()
#         answer_letter = match.group(2)
#         return thinking_part, ANSWER_DICT[answer_letter] if answer_letter in ANSWER_DICT else None
#     else:
#         return text.strip(), None  # If extraction fails, return full text as thinking part.
    
def extract_thinking_and_answer(raw_output):
    """
    Extracts the reasoning part and the final answer from the raw output.
    Ensures the answer is case-insensitive.
    """
    raw_output = raw_output.strip()
    lower_output = raw_output.lower()  # Convert to lowercase

    # Look for the pattern "the answer is X"
    if "the answer is " in lower_output:
        parts = lower_output.split("the answer is ")
        thinking_part = parts[0].strip()  # Extract reasoning before answer
        final_answer = parts[1].strip()[0].upper()  # Extract first letter and standardize to uppercase
    else:
        thinking_part = raw_output
        final_answer = None  # If no valid answer is found

    return thinking_part, ANSWER_DICT[final_answer]

def evaluate_multiple_choice(client, questions, result_path):    
    with open(result_path, 'w') as output_file:
        output_file.write("[\n")  # Start JSON list    
        correct_count = 0  # Track correct answers
        for index, q in tqdm(enumerate(questions), total=len(questions)):
            # Prepare the choices string (only two choices are assumed).
            choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])

            # Updated prompt to ensure reasoning + final answer format.
            # **Updated Safe Prompt**
            prompt = (
                f"{q['question']}\nChoices:\n{choices_text}\n"
                f"Without any other text, give your answer with \"The answer is X\" where X is the correct letter choice."
            )

            while True:
                try:
                    response = client.chat.completions.create(
                        model="o3-mini-2025-01-31",  # Ensure the correct model name
                        messages=[{"role": "user", "content": prompt}]
                    )

                    # Extract the model's reply.
                    raw_output = response.choices[0].message.content

                    # Separate reasoning (thinking part) and final answer.
                    thinking_part, final_answer = extract_thinking_and_answer(raw_output)
                    correct_answer = q["correct_answer"]

                    is_correct = (final_answer == correct_answer)
                    correct_count += int(is_correct)
                    accuracy = correct_count / (index + 1)

                    # Create a structured result dictionary.
                    result_dict = {
                        "question_id": index + 1,
                        "thinking_part": thinking_part,  # Store detailed reasoning
                        "final_answer": ANSWER_DICT[final_answer] if final_answer is not None else "Invalid",
                        "correct_answer": ANSWER_DICT[correct_answer],
                        "accuracy": f"{accuracy * 100:.2f}%"
                    }

                    output_file.write(json.dumps(result_dict) + ",\n")
                    output_file.flush()
                    
                    break  # Exit the retry loop on success

                except Exception as e:
                    if "Expecting value: line 1 column 1 (char 0)" in str(e):
                        print("Encountered API error, waiting 3 seconds before retrying...")
                        time.sleep(3)
                        continue  # Retry the request
                    else:
                        print(f"API Error: {e}")
                        break  # Exit for other errors
                    
    # **Fix JSON formatting: Remove last comma and add closing bracket**
    with open(result_path, 'rb+') as output_file:
        output_file.seek(0, 2)  # Move to end of file
        output_file.seek(output_file.tell() - 2, 0)  # Move 2 bytes back to remove last comma
        output_file.truncate()  # Remove last comma
        output_file.write(b"\n]")  # Close the JSON list

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions using gpt api.")    
    parser.add_argument("--env_id", type=str, required=True, help="Specify an environment ID to run a single environment, or 'all' for all environments.")
    parser.add_argument("--num_choices", type=int, choices=[2, 4], default=2, help="Number of choices per question (2 or 4).")
    parser.add_argument("--mode", type=str, choices=["easy", "hard"], default="easy", help="difficulty level of the questions. Easy choices will have larger differences in reward values.")
    parser.add_argument("--description", type=str, choices=["better", "worse"], default="better", help="ask LLMs to pick better or worse performance choices.")
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_questions", help="Path to the question folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_answers", help="Output path for the generated answer JSON.")
    
    args = parser.parse_args()
    env_list = [
        "Walker-v0",
        "BridgeWalker-v0",
        "Jumper-v0",
        "Balancer-v0",
        "UpStepper-v0",
        "GapJumper-v0",
        "Carrier-v0",
        "Carrier-v1",
        "Pusher-v0",
        "Pusher-v1",
        "Climber-v0",
        "Climber-v1",
    ]

    env_names = env_list if args.env_id == "all" else [args.env_id]
    
    # initialize gpt api client
    api_key = "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"  
    client = init_gpt_client(api_key)
            
    for env_name in env_names:
        source_QA_path = os.path.join(args.data_dir, f"{env_name}/{env_name}_QA_{args.description}_{args.mode}_{args.num_choices}.json")
        output_env_dir = f"{args.output_dir}/{env_name}"
        os.makedirs(output_env_dir, exist_ok=True)
        output_ANS_path = os.path.join(output_env_dir, f"{env_name}_ANS_{args.description}_{args.mode}_{args.num_choices}.json")
        # Evaluate questions
        questions = load_and_prepare_questions(source_QA_path)
        evaluate_multiple_choice(client, questions, output_ANS_path)

if __name__ == "__main__":
    main()
