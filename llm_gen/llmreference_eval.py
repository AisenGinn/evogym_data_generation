import os
import shutil
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
import evogym.envs
from google import genai
import gymnasium as gym
import evogym.envs
from tqdm import tqdm
import random
import re
import ast 


import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from ppo.args import add_ppo_args
from ppo.run import run_ppo
from ppo.eval import eval_policy

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

# Initialize GPT O3 Mini High API client.
def init_gemini_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

def data_extraction(data_path):
    all_data = load_json(data_path)
    
    seen_structures = set()
    unique_data_list = []

    # Filter out unique structures
    for data in all_data:
        structure_tuple = tuple(tuple(row) for row in data["structure"])
        if structure_tuple not in seen_structures:
            seen_structures.add(structure_tuple)
            unique_data_list.append(data)

    # Sort by reward
    sorted_data = sorted(unique_data_list, key=lambda x: x["reward"])

    # Keep top 10 and lowest 10 reward dicts
    top_10 = sorted_data[-30:]  # Top 10 highest rewards
    bottom_10 = sorted_data[:30]  # Bottom 10 lowest rewards
    
    return [top_10, bottom_10]  # Concatenating the two lists

def extract_answers_to_array(raw_output):
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
        return np.array(answer_list)

    except (SyntaxError, ValueError, IndexError) as e:
        print(f"Error parsing GPT response: {e}")
        return []  # Return empty results and 0 accuracy on failure

def ask_llm_init(args, client, walker_design, jumper_design, result_path):
    model_save_dir = os.path.join(args.structure_save_path, f"{args.env_name}_saved_model")
    os.makedirs(model_save_dir, exist_ok=True)
    with open(result_path, 'w') as output_file:
        for index in range(50):
            random.shuffle(walker_design[0])
            random.shuffle(walker_design[1])
            random.shuffle(jumper_design[0])
            random.shuffle(jumper_design[1])
            output_file.write("[\n")  # Start JSON list  
            prompt = f"""
                I have a robot designed for both the Walker-v0 and Jumper-v0 environments in EvoGym. The goal in Walker-v0 is for the robot to walk as far as possible on flat terrain, while in Jumper-v0, the goal is to jump as high as possible.

                Here are 20 design-reward pairs for Walker-v0:
                {walker_design[0][:10] + walker_design[1][:10]}

                Each design is represented as a 5x5 material matrix where:
                - 0: Empty voxel
                - 1: Rigid voxel
                - 2: Soft voxel
                - 3: Horizontal Actuator voxel
                - 4: Vertical Actuator voxel

                Each design has an associated reward that reflects its performance in Walker-v0, higher reward means better performance.

                Here are 20 design-reward pairs for Jumper-v0:
                {jumper_design[0][:10] + jumper_design[1][:10]}

                Now, I would like to design a robot for the GapJumper-v0 environment, where the goal is to traverses a series of floating platforms, each spaced 5 units apart, all at the same height.

                Given the previous designs, how would you design a structure optimized for GapJumper-v0? Consider:
                - Structural modifications for stability during jumps.
                - Using a strategic mix of rigid and soft materials for efficiency.

                Return only the 5x5 structure as list format without any other text.
                """
            response = client.models.generate_content(
                model="gemini-1.5-pro",  # Ensure the correct model name
                contents = prompt
            )

            # Extract the model's reply.
            raw_output = response.text

            # Separate reasoning (thinking part) and final answer.
            init_structure = extract_answers_to_array(raw_output)
            
            best_reward = run_ppo(
                args=args,
                body=init_structure,
                env_name="GapJumper-v0",
                model_save_dir=model_save_dir,
                model_save_name=f"{index}_model",
            )
            
            result_dict = {
                "env_name": "GapJumper-v0",
                "improved_structure": init_structure.tolist(),
                "improved_reward": best_reward
            }

            output_file.write(json.dumps(result_dict) + ",\n")
            output_file.flush()
                    
    # **Fix JSON formatting: Remove last comma and add closing bracket**
    with open(result_path, 'rb+') as output_file:
        output_file.seek(0, 2)  # Move to end of file
        output_file.seek(output_file.tell() - 2, 0)  # Move 2 bytes back to remove last comma
        output_file.truncate()  # Remove last comma
        output_file.write(b"\n]")  # Close the JSON list  
    

def main():   
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument(
        "--env_name", default="Walker-v0", type=str, help="Environment name (default: Walker-v0)"
    )
    parser.add_argument(
        "--structure_save_path", default="/media/hdd2/users/changhe/improved_data", type=str, help="Path to save improve strutures"
    )

    add_ppo_args(parser)
    args = parser.parse_args()
    
    data_path_1 = f"/media/hdd2/users/changhe/saved_data/test_ga_{args.env_name}/{args.env_name}_results.json"  
    result_path = f"{args.structure_save_path}/{args.env_name}_improved_comparison.json"
    
    Walker_data_list = data_extraction(data_path_1)
    Jumper_data_list = data_extraction(f"/media/hdd2/users/changhe/saved_data/test_ga_Jumper-v0/Jumper-v0_results.json")
    
    Walker_processed_good_data = []
    Walker_processed_bad_data = []
    
    Jumper_processed_good_data = []
    Jumper_processed_bad_data = []
    
    for i in range(len(Walker_data_list[0])):
        Walker_processed_good_data.append(f"structure: {Walker_data_list[0][i]['structure']}, reward: {Walker_data_list[0][i]['reward']}\n")
        Walker_processed_bad_data.append(f"structure: {Walker_data_list[1][i]['structure']}, reward: {Walker_data_list[1][i]['reward']}\n")
        Jumper_processed_good_data.append(f"structure: {Jumper_data_list[0][i]['structure']}, reward: {Jumper_data_list[0][i]['reward']}\n")
        Jumper_processed_bad_data.append(f"structure: {Jumper_data_list[1][i]['structure']}, reward: {Jumper_data_list[1][i]['reward']}\n")
            
    # initialize gpt api client
    api_key = "YOUR_API_KEY"
    client = init_gemini_client(api_key) 
    
    ask_llm_init(args, client, [Walker_processed_good_data, Walker_processed_bad_data], [Jumper_processed_good_data, Jumper_processed_bad_data], result_path) 

    
if __name__ == "__main__":
    main()
