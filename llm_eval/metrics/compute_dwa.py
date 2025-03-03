import os
import json
import pandas as pd
import argparse
import math
from collections import defaultdict

ANSWER_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    "A": 0, "B": 1, "C": 2, "D": 3
}

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def get_reward_diff(question_dir, recur):
    """Calculate max and min reward differences for each environment
    
    Args:
        question_dir (str): Path to directory containing environment folders
        recur (int): Recursion identifier for question files
        
    Returns:
        dict: {
            "env_name": {
                "max_delta_reward": float,
                "min_delta_reward": float
            }, ...
        }
    """
    reward_diff_dict = {}
    
    # Iterate through each environment folder
    for env_name in os.listdir(question_dir):
        env_path = os.path.join(question_dir, env_name)
        
        # Skip non-directory items
        if not os.path.isdir(env_path):
            continue
            
        # Initialize storage for delta rewards
        delta_rewards = []
        
        try:
            # Process both difficulty levels
            for difficulty in ["easy", "hard"]:
                # Build file path
                q_file = os.path.join(
                    env_path, 
                    f"{env_name}_QA_better_{difficulty}_2_{recur}.json"
                )
                
                # Skip missing files
                if not os.path.exists(q_file):
                    continue
                
                # Load question data
                questions = load_json(q_file)
                
                # Process each question
                for q in questions:
                    # Validate data structure
                    if ("correct_answer" not in q) or ("reward" not in q):
                        continue
                        
                    # Extract reward values
                    correct_idx = q["correct_answer"]
                    rewards = q["reward"]
                    
                    # Validate indices
                    if (correct_idx not in {0, 1}) or (len(rewards) != 2):
                        continue
                    
                    # Calculate delta reward
                    delta = abs(rewards[correct_idx] - rewards[1 - correct_idx])
                    delta_rewards.append(delta)
                    
        except Exception as e:
            print(f"Error processing {env_name}: {str(e)}")
            continue
            
        # Calculate statistics
        if delta_rewards:
            reward_diff_dict[env_name] = {
                "max_delta_reward": max(delta_rewards),
                "min_delta_reward": min(delta_rewards)
            }
        else:
            reward_diff_dict[env_name] = {
                "max_delta_reward": None,
                "min_delta_reward": None
            }
            
    return reward_diff_dict

def map_question_to_answer_batches(questions, answers, batch_size=20):
    """
    Maps batch indices from questions to answers, handling skipped batches.

    Args:
        questions (str or list): File path to questions JSON or already loaded data.
        answers (str or list): File path to answers JSON or already loaded data.
        batch_size (int): The number of questions per batch.

    Returns:
        dict: A mapping of question batch_start -> answer batch_start.
        list: List of confirmed batch starts.
        list: List of skipped batch starts in questions.
    """

    # Load JSON data if file paths are given
    if isinstance(questions, str):
        questions = load_json(questions)
    if isinstance(answers, str):
        answers = load_json(answers)

    confirm_batch_start = []  # Stores batch starts that align in both files
    skipped_batches = []       # Tracks batch indices skipped in answers
    question_to_answer_mapping = {}  # Maps question batch_start -> answer batch_start

    question_idx = 0  # Tracks batch_start in questions
    answer_idx = 0    # Tracks batch_start in answers

    while question_idx < len(questions):
        confirm = True
        batch_questions = questions[question_idx: question_idx + batch_size]

        # Ensure we don't go out of bounds in answers
        if answer_idx >= len(answers):  
            #print(f"⚠️ Skipping batch_start {question_idx} - No corresponding batch in answers.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        batch_answer = answers[answer_idx: answer_idx + batch_size]

        if len(batch_questions) != len(batch_answer):  
            print(f"⚠️ Skipping batch_start {question_idx} due to batch size mismatch.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        # Validate answers match in both batches
        for i in range(len(batch_questions)):  
            question_answer = ANSWER_DICT[batch_questions[i]['correct_answer']]
            gt = batch_answer[i]["correct_answer"]

            if question_answer != gt:
                confirm = False
                break  # Skip this batch if any mismatch is found

        if confirm:
            confirm_batch_start.append(question_idx)
            question_to_answer_mapping[question_idx] = answer_idx  # Map question batch_start -> answer batch_start
            answer_idx += batch_size  # Increment answer index **only for valid batches**

        # Always increment question index to check the next batch in questions
        question_idx += batch_size

    return question_to_answer_mapping

def compute_and_save_DWA(question_dir, answer_dir, recur):
    """计算并保存难度加权准确率（DWA）含严格索引对齐"""
    reward_diffs = get_reward_diff(question_dir, recur)
    dwa_results = defaultdict(lambda: {
        "easy": {"weighted": 0, "total": 0},
        "hard": {"weighted": 0, "total": 0},
        "global": {"weighted": 0, "total": 0}
    })

    for env_name in os.listdir(answer_dir):
        env_path = os.path.join(answer_dir, env_name)
        q_env_path = os.path.join(question_dir, env_name)
        if not os.path.isdir(env_path) or not os.path.isdir(q_env_path):
            continue

        # 加载问题数据
        try:
            q_easy = load_json(os.path.join(q_env_path, f"{env_name}_QA_better_easy_2_{recur}.json"))
            q_hard = load_json(os.path.join(q_env_path, f"{env_name}_QA_better_hard_2_{recur}.json"))
        except FileNotFoundError:
            print(f"{env_name}_QA_better_easy_2_{recur}.json not found!" )
            continue

        env_diff = reward_diffs.get(env_name, {})
        if not env_diff or None in env_diff.values():
            print("Error processing reward difference data! Skipping...")
            continue
        delta_min, delta_max = env_diff["min_delta_reward"], env_diff["max_delta_reward"]

        # 处理每个难度级别
        for difficulty, q_data in [("easy", q_easy), ("hard", q_hard)]:
            answer_files = sorted([
                os.path.join(env_path, f) for f in os.listdir(env_path)
                if f.endswith(f"{difficulty }_2_{recur}_1.json") or
                   f.endswith(f"{difficulty}_2_{recur}_2.json") or
                   f.endswith(f"{difficulty}_2_{recur}_3.json")
            ])
            if len(answer_files) != 3:
                print("file number not match. (Should be 3)")
                continue

            # 生成三方映射关系
            mappings = [map_question_to_answer_batches(q_data, load_json(f)) for f in answer_files]
            
            # 计算共同有效批次（严格三方对齐）
            common_batches = set.intersection(
                set(mappings[0].keys()), 
                set(mappings[1].keys()), 
                set(mappings[2].keys())
            )
            
            # 初始化统计量
            total_weight, total_correct = 0.0, 0.0
            
            for batch_start in common_batches:
                # 获取三方答案的对应位置
                a1_idx = mappings[0][batch_start]
                a2_idx = mappings[1][batch_start]
                a3_idx = mappings[2][batch_start]
                
                # 加载三方答案数据
                a1_data = load_json(answer_files[0])[a1_idx : a1_idx+20]
                a2_data = load_json(answer_files[1])[a2_idx : a2_idx+20]
                a3_data = load_json(answer_files[2])[a3_idx : a3_idx+20]
                
                # 处理批次内每个问题
                for q_offset in range(20):
                    q_global_idx = batch_start + q_offset
                    if q_global_idx >= len(q_data):
                        break
                    
                    # 获取问题元数据
                    question = q_data[q_global_idx]
                    correct_answer = ANSWER_DICT[question["correct_answer"]]
                    r_correct = question["reward"][question["correct_answer"]]
                    r_incorrect = question["reward"][1 - question["correct_answer"]]
                    delta = abs(r_correct - r_incorrect)
                    
                    # 计算难度权重
                    weight = 1 - (delta - delta_min)/(delta_max - delta_min) if delta_max != delta_min else 1.0
                    
                    # 获取三方答案
                    try:
                        votes = [
                            a1_data[q_offset]["final_answer"],
                            a2_data[q_offset]["final_answer"],
                            a3_data[q_offset]["final_answer"]
                        ]
                    except (IndexError, KeyError):
                        continue
                    
                    # 多数投票决策
                    pred = max(set(votes), key=votes.count)
                    is_correct = int(pred == correct_answer)
                    
                    # 累加统计
                    total_weight += weight
                    total_correct += is_correct * weight

            # 记录结果
            if total_weight > 0:
                dwa_results[env_name][difficulty]["weighted"] = total_correct
                dwa_results[env_name][difficulty]["total"] = total_weight
                dwa_results[env_name]["global"]["weighted"] += total_correct
                dwa_results[env_name]["global"]["total"] += total_weight

    # 生成最终报告
    generate_dwa_report(dwa_results, answer_dir, recur)

def generate_dwa_report(results, output_dir, recur):
    """生成标准化报告"""
    csv_data = []
    global_stats = {"weighted": 0.0, "total": 0.0}
    
    for env, data in results.items():
        row = [env]
        
        # 计算各难度DWA
        for diff in ["easy", "hard"]:
            if data[diff]["total"] > 0:
                score = (data[diff]["weighted"] / data[diff]["total"]) * 100
                row.append(f"{score:.2f}%")
            else:
                row.append("N/A")
        
        # 计算全局DWA
        if data["global"]["total"] > 0:
            global_score = (data["global"]["weighted"] / data["global"]["total"]) * 100
            row.append(f"{global_score:.2f}%")
            global_stats["weighted"] += data["global"]["weighted"]
            global_stats["total"] += data["global"]["total"]
        else:
            row.append("N/A")
        
        csv_data.append(row)
    
    # 添加全局平均
    if global_stats["total"] > 0:
        global_avg = (global_stats["weighted"] / global_stats["total"]) * 100
        csv_data.append(["Global Average", "N/A", "N/A", f"{global_avg:.2f}%"])
    
    # 保存CSV
    df = pd.DataFrame(csv_data, columns=["Environment", "Easy DWA", "Hard DWA", "Global DWA"])
    output_path = os.path.join(output_dir, f"{recur}_DWA_report.csv")
    df.to_csv(output_path, index=False)
    print(f"DWA report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute consistency for different environments.")
    parser.add_argument("--question_dir", type=str, default="/media/hdd2/users/changhe/saved_questions", help="Path to saved JSON answer files.")
    parser.add_argument("--recur", type=str, choices=["repeat", "norepeat"], default="norepeat", help="Whether include repeated structure in questions")   
    parser.add_argument("--answer_dir", type=str, default="/media/hdd2/users/changhe/saved_answers/grok2", help="Path to saved JSON answer files.") 
    
    args = parser.parse_args()
    compute_and_save_DWA(args.question_dir, args.answer_dir, args.recur)

if __name__ == "__main__":
    main()
