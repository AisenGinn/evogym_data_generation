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

# Load the pre-trained model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    return model, tokenizer

def load_large_model(model_name):
    # Define the quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 offloading to CPU if necessary
    )

    # Load the model with distributed GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically assign model layers across GPUs
        quantization_config=quantization_config
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

# Load and prepare questions from a JSON file
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

def split_and_extract(text):
    """
    Split the text into thinking part and final answer.
    """
    parts = text.split('</think>', 1)  # Split at first occurrence of </think>
    thinking_part = parts[0].strip() if len(parts) > 1 else text.strip()
    answer_section = parts[1].strip() if len(parts) > 1 else ''
    
    # Extract final answer
    answer_match = re.search(r'\b[A-B]\b', answer_section)
    final_answer = answer_match.group(0) if answer_match else None
    final_answer = ANSWER_DICT[final_answer] if final_answer else None

    return thinking_part, final_answer

def evaluate_multiple_choice(model, tokenizer, questions, result_path):
    """
    Evaluate the model on multiple-choice questions and log outputs.
    """
    correct_count = 0
    total_count = 0

    # Open the output file in write mode; each line will be a JSON object.
    with open(result_path, 'w') as output_file:
        for index, q in tqdm(enumerate(questions), total=len(questions)): 
            # Format the input for the mode            
            choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
            input_text = (
                f"[INST]{q['question']}\nChoices:\n{choices_text}\n"
                f" Give your answer with \"The answer is X\" where X is the correct letter choice.[/INST]"
            )
            
            total_count += 1
            if total_count == 40: 
                break
            print(index+1, input_text)

            # Tokenize and generate output
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=5000, temperature=0.6)
            
            # Decode the output
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove repeated input text if necessary
            if raw_output.startswith(input_text):
                raw_output = raw_output[len(input_text):].strip()

            # Process the output
            thinking_part, final_answer = split_and_extract(raw_output)
            correct_answer = q["correct_answer"]

            # Update correct count and calculate accuracy
            is_correct = (final_answer == correct_answer)
            correct_count += int(is_correct)
            accuracy = correct_count / (index + 1)  # Accuracy up to this step

            # Create a result dictionary for this question
            result_dict = {
                "question_id": index + 1,
                "thinking_part": thinking_part,
                "final_answer": ANSWER_DICT[final_answer] if final_answer is not None else "Invalid",
                "correct_answer": ANSWER_DICT[correct_answer],
                "accuracy": f"{accuracy * 100:.2f}%"
            }

            # Write the dictionary as a JSON line and flush immediately
            output_file.write(json.dumps(result_dict) + "\n")
            output_file.flush()

# Main function
def main():
    # Load model and tokenizer
    model_name = "/media/hdd2/local_models/DeepSeek-R1-Distill-Qwen-32B"
    #model, tokenizer = load_model(model_name)
    model, tokenizer = None , None

    # Load and prepare questions from JSON file
    json_file_path = "/home/changhe/MMLU-Pro/walker-v0/question_fc.json" 
    result_path = "/home/changhe/MMLU-Pro/walker-v0/results32B_fc.json"  
    questions = load_and_prepare_questions(json_file_path)

    # Evaluate questions and log results
    evaluate_multiple_choice(model, tokenizer, questions, result_path)

if __name__ == "__main__":
    main()
