import os
import shutil
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
import evogym.envs
from evogym import WorldObject
from evogym import sample_robot

import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from ppo.args import add_ppo_args
from ppo.run import run_ppo
from ppo.eval import eval_policy
    
if __name__ == "__main__":    
    
    # Args
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument(
        "--env-name", default="GapJumper-v0", type=str, help="Environment name (default: Walker-v0)"
    )
    
    add_ppo_args(parser)
    args = parser.parse_args()

    result_path = "/media/hdd2/users/changhe/improved_data/random_init.json"
    with open(result_path, 'w') as output_file:
        for i in range(50):# Train
            body, connections = sample_robot((5,5))
            best_reward = run_ppo(
                args=args,
                body= body,
                env_name=args.env_name,
                model_save_dir="/media/hdd2/users/changhe/improved_data/random_init_model",
                model_save_name=f"{i}_model",
            )
            
            result_dict = {
                "env_name": "GapJumper-v0",
                "improved_structure": body.tolist(),
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