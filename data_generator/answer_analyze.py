import json
import numpy as np
import random
import matplotlib.pyplot as plt
import re

with open("/home/changhe/MMLU-Pro/walker-v0/question_fc.json", 'r') as file:
    questions = json.load(file)

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

def main():
    draw_reward_diff()
    #draw_diffvresult()

if __name__ == "__main__":
    main()