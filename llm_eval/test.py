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
            f"{q['task description']} "
            f"{q['structure description']} "
            f"{q['actuation description']} "
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
            f"[INST]{q['question']}\nChoices:\n{choices_text}\n"
            f" Give your answer with \"The answer is X\" where X is the correct letter choice.[/INST]"
        )
        
        total_count += 1
        if total_count == 201: 
            break
        #print(index+1, input_text)
        correct_answers.append(ANSWER_DICT[q['correct_answer']])
    print(correct_answers)
        
# Main function
def main():
    # # Load and prepare questions from JSON file
    # json_file_path = "/home/changhe/MMLU-Pro/walker-v0/question_fc.json"   
    # questions = load_and_prepare_questions(json_file_path)

    # # Evaluate questions and log results
    # evaluate_multiple_choice( questions)
    import openai

    client = openai.OpenAI(api_key="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")

    models = client.models.list()
    print([model.id for model in models.data])

if __name__ == "__main__":
    main()
