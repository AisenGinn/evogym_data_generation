import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import BitsAndBytesConfig

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


def load_and_prepare_questions(file_path):
    """
    Load questions from a JSON file and combine descriptions into the question field.
    """
    with open(file_path, 'r') as file:
        questions = json.load(file)

    for q in questions:
        combined_description = (
            f"{q['task_description']} "
            f"{q['structure_description']} "
            f"{q['actuation_description']} "
        )
        #q["question"] = "Which of the following robot structure will have worser performance in the evogym Walker task?" + combined_description
        q["question"] = q['question'] + combined_description

    return questions

def evaluate_multiple_choice(questions):
    """
    Evaluate the model on multiple-choice questions and log outputs.
    """
    total_count = 0
    correct_answers = []

    # Open the output file in write mode; each line will be a JSON object.
    for index, q in tqdm(enumerate(questions), total=len(questions)): 
        # Format the input for the mode            
        choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
        input_text = (
            f"{index+1}. {q['question']}\nChoices:\n{choices_text}\n"
            f" Give your answer with \"The answer is X\" where X is the correct letter choice."
        )
        print(input_text)
        correct_answers.append(ANSWER_DICT[q['correct_answer']])
        
        total_count += 1
        if total_count == 20: 
            break
        #print(index+1, input_text)
       
    print(correct_answers)
        
# Main function
def main():
    # Load and prepare questions from JSON file
    json_file_path = "/media/hdd2/users/changhe/saved_questions_fc/Walker-v0_fc_questions_4.json"   
    questions = load_and_prepare_questions(json_file_path)

    # Evaluate questions and log results
    evaluate_multiple_choice(questions)

if __name__ == "__main__":
    main()
