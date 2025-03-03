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
import time
import re
import ast 


import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from ppo.args import add_ppo_args
from ppo.run import run_ppo
from ppo.eval import eval_policy

from collections import deque
import numpy as np

def is_robot_connected(structure):
    rows, cols = 5, 5  # Fixed size for EvoGym robot
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Find a starting voxel (non-zero)
    start = None
    for r in range(rows):
        for c in range(cols):
            if structure[r][c] != 0:  # Non-empty voxel
                start = (r, c)
                break
        if start:
            break
    
    if not start:
        return False  # No solid voxel in the structure
    
    # BFS/DFS to check connectivity
    queue = deque([start])
    visited[start] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    connected_voxels = 0
    
    while queue:
        r, c = queue.popleft()
        connected_voxels += 1
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and structure[nr][nc] != 0:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Count total non-zero voxels
    total_voxels = sum(1 for r in range(rows) for c in range(cols) if structure[r][c] != 0)
    
    return connected_voxels == total_voxels  # True if all non-zero voxels are connected

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
        if data["reward"] <= -0.5:
            structure_tuple = tuple(tuple(row) for row in data["structure"])
            if structure_tuple not in seen_structures:
                seen_structures.add(structure_tuple)
                unique_data_list.append(data)
    
    return unique_data_list  # Concatenating the two lists

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

def ask_llm_improve(args, client, data_list, result_path):
    model_save_dir = os.path.join(args.structure_save_path, f"{args.env_name}_no_know_saved_model")
    os.makedirs(model_save_dir, exist_ok=True)
    with open(result_path, 'w') as output_file:
        output_file.write("[\n")  # Start JSON list  
        total_eval = 50
        for index in tqdm(range(50)):
            if index == total_eval: break
            prompt = f"""
                I want to design a robot for the 'BridgeWalker-v0' environment in evogym, where the goal is for the robot to walk as far as possible on a soft rope-bridge.
                The robot structure uses a unified multi-material voxel-based representation. It is represented as a 5x5 material matrix. The entries of this matrix correspond to different voxel types:
                - 0: Empty voxel
                - 1: Rigid voxel
                - 2: Soft voxel
                - 3: Horizontal Actuator voxel
                - 4: Vertical Actuator voxel

                All pairs of adjacent voxels are connected to each other. The robot's controller is optimized to maximize task performance by actuating its structure through precise deformation. Given the above robot structure, how would you improve it to enhance its ability to walk further on the soft rope-bridge? Consider:
                - Structural modifications to improve balance and stability.
                - Adjusting the distribution of actuators for better locomotion.
                - Using a better combination of rigid and soft materials for efficiency.

                Design a robot structure matrix while keeping it within a 5x5 grid format, make sure the structure is connected (0 cannot completely seperate two parts). Return only the 5x5 structure as list format without any other text.
            """
            response = client.models.generate_content(
                model="gemini-1.5-pro",  # Ensure the correct model name
                contents = prompt
            )

            # Extract the model's reply.
            raw_output = response.text

            # Separate reasoning (thinking part) and final answer.
            improved_structure = extract_answers_to_array(raw_output)
            improved_structure = improved_structure.reshape(5, 5)
            if not is_robot_connected(improved_structure): 
                result_dict = {
                    "env_name": args.env_name,
                    "improved_structure": "invalid",
                    "improved_reward": -1
                }
                output_file.write(json.dumps(result_dict) + ",\n")
                output_file.flush()
                continue
            
            best_reward = run_ppo(
                args=args,
                body=improved_structure,
                env_name=args.env_name,
                model_save_dir=model_save_dir,
                model_save_name=f"{index}_model",
            )
            
            result_dict = {
                "env_name": args.env_name,
                "improved_structure": improved_structure.tolist(),
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
        "--env_name", default="BridgeWalker-v0", type=str, help="Environment name (default: BridgeWalker-v0)"
    )
    parser.add_argument(
        "--structure_save_path", default="/media/hdd2/users/changhe/improved_data", type=str, help="Path to save improve strutures"
    )

    add_ppo_args(parser)
    args = parser.parse_args()
    
    # low (-1.5, -0.5)
    # middle (-0.5, 0.5)
    # high (0.5, 1.5)
    data_path = f"/media/hdd2/users/changhe/saved_data/test_ga_{args.env_name}/{args.env_name}_results.json"  
    result_path = f"{args.structure_save_path}/{args.env_name}_init_design.json"
    
    data_list = data_extraction(data_path)
            
    # initialize gpt api client
    api_key = "YOUR_API_KEY"
    client = init_gemini_client(api_key) 
    
    ask_llm_improve(args, client, data_list, result_path) 

    
if __name__ == "__main__":
    main()
