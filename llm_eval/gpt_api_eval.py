import json
import re
import time
from tqdm import tqdm
from openai import OpenAI

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
            f"{q['task description']} "
            f"{q['structure description']} "
            f"{q['actuation description']}"
        )
        # Append the combined description to the question text.
        q["question"] = q['question'] + " " + combined_description

    return questions

def extract_thinking_and_answer(text):
    """
    Extract the reasoning (thinking part) and final answer from the model's response.
    - The thinking part is everything before "The answer is X."
    - The answer part is just "A" or "B".
    """
    match = re.search(r"(.*)\s*The answer is\s+([AB])\.", text, re.DOTALL)
    if match:
        thinking_part = match.group(1).strip()
        answer_letter = match.group(2)
        return thinking_part, ANSWER_DICT[answer_letter] if answer_letter in ANSWER_DICT else None
    else:
        return text.strip(), None  # If extraction fails, return full text as thinking part.

def evaluate_multiple_choice(client, questions, result_path):
    correct_count = 0

    with open(result_path, 'w') as output_file:
        for index, q in tqdm(enumerate(questions), total=len(questions)):
            if index == 501:
                break
            # Prepare the choices string (only two choices are assumed).
            choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])

            # Updated prompt to ensure reasoning + final answer format.
            # **Updated Safe Prompt**
            prompt = (
                f"{q['question']}\nChoices:\n{choices_text}\n"
                f" Give your answer with \"The answer is X\" where X is the correct letter choice."
            )

            while True:
                try:
                    response = client.chat.completions.create(
                        model="o3-mini",  # Ensure the correct model name
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

def main():
    api_key = "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"  # Replace with your actual API key.
    client = init_gpt_client(api_key)

    # File paths for questions and output results.
    json_file_path = "/home/changhe/MMLU-Pro/walker-v0/question_fc.json"
    result_path = "/home/changhe/MMLU-Pro/walker-v0/results_gpto3mini_api_fc.json"

    questions = load_and_prepare_questions(json_file_path)
    evaluate_multiple_choice(client, questions, result_path)

if __name__ == "__main__":
    main()
