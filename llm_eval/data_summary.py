import os
import json
import numpy as np
import pandas as pd

def process_questions(directory):
    stats_per_env = {}  # Store statistics for each environment
    all_rewards = []  # Store all rewards for global statistics
    all_reward_diffs = []  # Store reward differences globally

    # Iterate through all environment directories
    for env in os.listdir(directory):
        env_path = os.path.join(directory, env)
        if not os.path.isdir(env_path):
            continue  # Skip if not a directory

        # Filter for "better norepeat" JSON files
        better_norepeat_files = [
            f for f in os.listdir(env_path) if "better" in f and "norepeat" in f and f.endswith(".json")
        ]

        env_rewards = []  # Store rewards for this environment
        env_reward_diffs = []  # Store reward differences for this environment
        num_questions = 0

        # Process each JSON file
        for file in better_norepeat_files:
            file_path = os.path.join(env_path, file)
            with open(file_path, "r") as f:
                questions = json.load(f)

            num_questions += len(questions)

            for q in questions:
                rewards = q["reward"]
                reward_diff = abs(rewards[0] - rewards[1])

                env_rewards.extend(rewards)
                env_reward_diffs.append(reward_diff)

        # Compute environment-specific statistics
        if env_rewards:
            stats_per_env[env] = {
                "num_questions": num_questions,
                "reward_diff_range": (min(env_reward_diffs), max(env_reward_diffs)) if env_reward_diffs else (None, None),
                "highest_reward": max(env_rewards),
                "lowest_reward": min(env_rewards),
            }

        # Update global statistics
        all_rewards.extend(env_rewards)
        all_reward_diffs.extend(env_reward_diffs)

    # Compute global statistics
    global_stats = {
        "num_questions": sum(stat["num_questions"] for stat in stats_per_env.values()),
        "reward_diff_range": (min(all_reward_diffs), max(all_reward_diffs)) if all_reward_diffs else (None, None),
        "highest_reward": max(all_rewards) if all_rewards else None,
        "lowest_reward": min(all_rewards) if all_rewards else None,
    }

    return stats_per_env, global_stats

# Set the path to the saved_questions directory
saved_questions_dir = "/media/hdd2/users/changhe/saved_questions"  # Change this to your actual path

# Run the processing function
env_stats, global_stats = process_questions(saved_questions_dir)

# Convert results to DataFrame and display
df_env_stats = pd.DataFrame.from_dict(env_stats, orient="index")
df_global_stats = pd.DataFrame([global_stats], index=["Global"])

# Display the results
print("\nPer-Environment Statistics:")
print(df_env_stats)
print("\nGlobal Statistics:")
print(df_global_stats)

# Optionally, save to CSV
df_env_stats.to_csv("env_stats.csv")
df_global_stats.to_csv("global_stats.csv")