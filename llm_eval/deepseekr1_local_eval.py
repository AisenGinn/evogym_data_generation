import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import argparse
import os
import ast 

ANSWER_DICT = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "None": None
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

# def split_and_extract(text):
#     """
#     Split the text into thinking part and final answer.
#     """
#     parts = text.split('</think>', 1)  # Split at first occurrence of </think>
#     thinking_part = parts[0].strip() if len(parts) > 1 else text.strip()
#     answer_section = parts[1].strip() if len(parts) > 1 else ''
    
#     # Extract final answer
#     answer_match = re.search(r'\b[A-B]\b', answer_section)
#     final_answer = answer_match.group(0) if answer_match else None
#     final_answer = ANSWER_DICT[final_answer] if final_answer else None

#     return thinking_part, final_answer

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


def evaluate_multiple_choice(model, tokenizer, questions, result_path):
    """
    Evaluate the model on multiple-choice questions and log outputs.
    """
    # Open the output file in write mode; each line will be a JSON object.
    with open(result_path, 'w') as output_file:
        output_file.write("[\n")  # Start JSON list 
        correct_count = 0
        
        for index, q in tqdm(enumerate(questions), total=len(questions)): 
            if index == 1000: break
            # Format the input for the mode            
            choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
            # Updated prompt to ensure reasoning + final answer format.
            # **Updated Safe Prompt**
            prompt = (
                f"{q['question']}\nChoices:\n{choices_text}\n"
                f"Without any other text, give your answer with \"The answer is X\" where X is the correct letter choice."
            )

            # Tokenize and generate output
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=5000, temperature=0.6)
            
            # Decode the output
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove repeated input text if necessary
            if raw_output.startswith(prompt):
                raw_output = raw_output[len(prompt):].strip()

            #Process the output
            thinking_part, final_answer = extract_thinking_and_answer(raw_output)
            correct_answer = q["correct_answer"]

            # Update correct count and calculate accuracy
            is_correct = (final_answer == correct_answer)
            correct_count += int(is_correct)
            accuracy = correct_count / (index + 1)  # Accuracy up to this step

            # Create a result dictionary for this question
            result_dict = {
                "question_id": index + 1,
                #"thinking_part": thinking_part,
                "final_answer": ANSWER_DICT[final_answer] if final_answer is not None else "Invalid",
                "correct_answer": ANSWER_DICT[correct_answer],
                "accuracy": f"{accuracy * 100:.2f}%"
            }
            #results, total_correct_count, total_count= extract_answers_and_calculate_accuracy(raw_output, questions, total_correct_count, total_count)

            # Write the dictionary as a JSON line and flush immediately
            
            output_file.write(json.dumps(result_dict) + ",\n")
            output_file.flush()

    # **Fix JSON formatting: Remove last comma and add closing bracket**
    with open(result_path, 'rb+') as output_file:
        output_file.seek(0, 2)  # Move to end of file
        output_file.seek(output_file.tell() - 2, 0)  # Move 2 bytes back to remove last comma
        output_file.truncate()  # Remove last comma
        output_file.write(b"\n]")  # Close the JSON list  

def evaluate_multiple_choice_batch(model, tokenizer, questions, result_path, batch_size=20):
    """
    Evaluate the model in batches on multiple-choice questions and log outputs.
    """
    # Open the output file in write mode; each line will be a JSON object.
    with open(result_path, 'w') as output_file:
        output_file.write("[\n")  # Start JSON list 
        correct_count = 0
        total_count = 0
        iter = 0
        for batch_start in tqdm(range(0, len(questions), batch_size), total=iter):
            if iter == 50: break
            batch_questions = questions[batch_start: batch_start + batch_size]
            prompts = None

            for index, q in enumerate(batch_questions):
                choices_text = "\n".join([f"{ANSWER_DICT[i]}: {choice}" for i, choice in enumerate(q["choices"])])
                prompt = prompt + f"{index+1}: {q['question']}\nChoices:\n{choices_text}\n"
                
            prompt = prompt + "Here are 20 questions, answer all of them and return the letter choice answers in a Python list format without any other text."

            
            # total_count += 1
            # if total_count == 40: 
            #     break
            #print(index+1, prompt)

            # Tokenize and generate output
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=5000, temperature=0.6)
            
            # Decode the output
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove repeated input text if necessary
            if raw_output.startswith(prompt):
                raw_output = raw_output[len(prompt):].strip()

            # Process the output
            thinking_part, final_answer = extract_thinking_and_answer(raw_output)
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
            output_file.write(json.dumps(result_dict) + ",\n")
            output_file.flush()

    # **Fix JSON formatting: Remove last comma and add closing bracket**
    with open(result_path, 'rb+') as output_file:
        output_file.seek(0, 2)  # Move to end of file
        output_file.seek(output_file.tell() - 2, 0)  # Move 2 bytes back to remove last comma
        output_file.truncate()  # Remove last comma
        output_file.write(b"\n]")  # Close the JSON list  

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice questions using local DeepSeek 14B.")    
    parser.add_argument("--env_id", type=str, required=True, help="Specify an environment ID to run a single environment, or 'all' for all environments.")
    parser.add_argument("--num_choices", type=int, choices=[2, 4], default=2, help="Number of choices per question (2 or 4).")
    parser.add_argument("--mode", type=str, choices=["easy", "hard"], default="easy", help="difficulty level of the questions. Easy choices will have larger differences in reward values.")
    parser.add_argument("--description", type=str, choices=["better", "worse"], default="better", help="ask LLMs to pick better or worse performance choices.")
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_questions", help="Path to the question folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_answers/deepseek13b", help="Output path for the generated answer JSON.")
    parser.add_argument("--batch", action="store_true", help="Enable batch mode to send 20 questions at once.")
    
    args = parser.parse_args()
    env_list = ["Walker-v0", "BridgeWalker-v0", "Jumper-v0", "Balancer-v0", "UpStepper-v0", "GapJumper-v0",
                "Carrier-v0", "Carrier-v1", "Pusher-v0", "Pusher-v1", "Climber-v0", "Climber-v1"]

    env_names = env_list if args.env_id == "all" else [args.env_id]
    
    # Load model and tokenizer
    model_name = "/media/hdd2/users/changhe/local_models/DeepSeek-R1-Distill-Qwen-14B"
    model, tokenizer = load_model(model_name)

    for env_name in env_names:
        source_QA_path = os.path.join(args.data_dir, f"{env_name}/{env_name}_QA_{args.description}_{args.mode}_{args.num_choices}.json")
        output_env_dir = f"{args.output_dir}/{env_name}"
        os.makedirs(output_env_dir, exist_ok=True)

        # Evaluate questions
        questions = load_and_prepare_questions(source_QA_path)        
        if args.batch:
            output_ANS_path = os.path.join(output_env_dir, f"{env_name}_Batch_ANS_{args.description}_{args.mode}_{args.num_choices}.json")
            evaluate_multiple_choice_batch(model, tokenizer, questions, output_ANS_path)
        else:
            output_ANS_path = os.path.join(output_env_dir, f"{env_name}_ANS_{args.description}_{args.mode}_{args.num_choices}.json")
            evaluate_multiple_choice(model, tokenizer, questions, output_ANS_path)

if __name__ == "__main__":
    main()
