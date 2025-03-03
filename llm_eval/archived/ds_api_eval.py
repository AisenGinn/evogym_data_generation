import json
import re
import time
from tqdm import tqdm
from openai import OpenAI

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

# Initialize DeepSeek API client
def init_deepseek_client(api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    return client

# Load and prepare questions from a JSON file
def load_and_prepare_questions(file_path):
    with open(file_path, 'r') as file:
        questions = json.load(file)

    for q in questions:
        combined_description = (
            f"{q['task description']} "
            f"{q['structure description']} "
            f"{q['actuation description']} "
        )
        q["question"] = q['question'] + combined_description

    return questions

def split_and_extract(text):
    # Try to split the text by the '</think>' tag.
    parts = text.split('</think>', 1)
    
    if len(parts) > 1:
        # If the tag exists, assign the parts accordingly.
        thinking_part = parts[0].strip()
        answer_section = parts[1].strip()
    else:
        # If the tag doesn't exist, use the entire text for both.
        thinking_part = text.strip()
        answer_section = text  # This will allow us to search for the answer sentence in the whole text.
    
    # Look for the sentence "The answer is X." where X is A or B.
    answer_match = re.search(r'The answer is\s+([AB])\.', answer_section)
    final_answer = answer_match.group(1) if answer_match else None
    
    # Convert the extracted answer letter using ANSWER_DICT, if available.
    final_answer = ANSWER_DICT[final_answer] if final_answer and final_answer in ANSWER_DICT else None

    return thinking_part, final_answer

def evaluate_multiple_choice(client, questions, result_path):
    correct_count = 0

    with open(result_path, 'w') as output_file:
        for index, q in tqdm(enumerate(questions), total=len(questions)):
            choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
            
            # Format the prompt for the API
            prompt = (
                f"{q['question']}\nChoices:\n{choices_text}\n"
                " Give your answer with \"The answer is X\" where X is the correct letter choice.[/INST]"
            )

            # Retry loop for API requests in case of the specific error.
            while True:
                try:
                    # Call DeepSeek API
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": "You are a robotics expert. Analyze the question carefully and choose the correct answer."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        stream=False
                    )

                    print(response)
                    raw_output = response.choices[0].message.content
                    thinking_part = response.choices[0].message.reasoning_content

                    # Process the output
                    _, final_answer = split_and_extract(raw_output)
                    correct_answer = q["correct_answer"]

                    is_correct = (final_answer == correct_answer)
                    correct_count += int(is_correct)
                    accuracy = correct_count / (index + 1)

                    result_dict = {
                        "question_id": index + 1,
                        "thinking_part": thinking_part,
                        "final_answer": ANSWER_DICT[final_answer] if final_answer is not None else "Invalid",
                        "correct_answer": ANSWER_DICT[correct_answer],
                        "accuracy": f"{accuracy * 100:.2f}%"
                    }

                    output_file.write(json.dumps(result_dict) + "\n")
                    output_file.flush()
                    
                    # Successfully processed this question, exit the retry loop.
                    break

                except Exception as e:
                    # Check if the error message is the one we want to handle.
                    if "Expecting value: line 1 column 1 (char 0)" in str(e):
                        print("Encountered API error, waiting 3 seconds before retrying...")
                        time.sleep(3)
                        continue  # Retry the request.
                    else:
                        print(f"API Error: {e}")
                        break  # Break out of the retry loop for other errors.

def main():
    # Initialize DeepSeek client
    api_key = "sk-3e7a83ee2f9941fdaca135e27d4b1324"  # Insert your API key here.
    client = init_deepseek_client(api_key)

    # Load questions
    json_file_path = "/home/changhe/MMLU-Pro/walker-v0/question_fc.json"
    result_path = "/home/changhe/MMLU-Pro/walker-v0/results_api_fc.json"
    questions = load_and_prepare_questions(json_file_path)

    # Evaluate questions
    evaluate_multiple_choice(client, questions, result_path)

if __name__ == "__main__":
    main()
