import json
import numpy as np
import random
import matplotlib.pyplot as plt
import re
import os

with open("/home/changhe/MMLU-Pro/walker-v0/question_fc.json", 'r') as file:
    questions = json.load(file)

save_dir = "/home/changhe/evogym/result_analyze"

def draw_diffvresult():
    with open("/home/changhe/MMLU-Pro/walker-v0/results32B_worser.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    final_answers = [re.search(r"final answer[:：]?\s*([A-Z])", line, re.IGNORECASE).group(1) for line in lines if "final answer" in line.lower()]
    correct_answers = [re.search(r"correct answer[:：]?\s*([A-Z])", line, re.IGNORECASE).group(1) for line in lines if "correct answer" in line.lower()]

    prediction_results = [f != c for f, c in zip(final_answers, correct_answers)]

    correct_differences = []
    wrong_differences = []
    for i, result in enumerate(prediction_results):
        reward_diff = abs(questions[i]["reward"][0] - questions[i]["reward"][1])
        if result:
            correct_differences.append(reward_diff)
        else:
            wrong_differences.append(reward_diff)


    plt.figure(figsize=(8, 5))
    plt.scatter([0] * len(correct_differences), correct_differences, color='green', label='Correct Predictions', alpha=0.7,s=10)
    plt.scatter([1] * len(wrong_differences), wrong_differences, color='red', label='Wrong Predictions', alpha=0.7,s=10)
    plt.xticks([0, 1], ['Correct', 'Wrong'])
    plt.ylabel('Reward Difference')
    plt.legend()
    plt.savefig("/home/changhe/MMLU-Pro/result_analyze/reward_prediction_fc_scatter.png", dpi=300, bbox_inches='tight')

def draw_reward_diff():
    differences = [abs(d["reward"][0] - d["reward"][1]) for d in questions]
    differences = differences[:205]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(differences)), differences, color='skyblue')
    plt.xlabel('index')
    plt.ylabel('Reward Difference')
    plt.savefig("/home/changhe/MMLU-Pro/result_analyze/reward_difference_fc.png", dpi=300, bbox_inches='tight')

def get_acc():
    accuracy_dict = {}
    base_folder = "/media/hdd2/users/changhe/saved_answers/gpto3mini"
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        for filename in os.listdir(subfolder_path):
            if "Batch_ANS_better_easy_2_default_repeat" in filename:
                file_path = os.path.join(subfolder_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list) and data:
                            last_item = data[-1]
                            accuracy_str = last_item.get("accuracy")
                            match = re.search(r"\d+", accuracy_str)
                            if match:
                                accuracy = int(match.group())
                                accuracy_dict[subfolder] = accuracy
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    sorted_items = sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)

    print(labels, values)

    plt.figure(figsize=(12, 6))
    bar = plt.bar(labels, values, color='#0076B5')
    plt.bar_label(bar, fmt='%d', padding=3)
    plt.xlabel("Environments")
    plt.ylabel("Accuracy(%)")
    #plt.title("acc_Batch_ANS_better_easy_2_default_norepeat")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/acc_Batch_ANS_better_easy_2_default_repeat.png", dpi=300, bbox_inches='tight')



def main():
    #draw_reward_diff()
    #draw_diffvresult()
    get_acc()

if __name__ == "__main__":
    main()