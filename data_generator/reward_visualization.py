import json
import matplotlib.pyplot as plt

# Load JSON file
result_path = '/media/hdd2/saved_data/test_ppo_0/ppo_data_5/Walker-v0/Walker-v0_results.json'
with open(result_path, 'r') as file:
    data = json.load(file)

# Extract rewards
rewards = [item['mean_reward'] for item in data]

# Plot rewards (Line Plot)
plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Reward Trend')
plt.title('Reward Visualization')
plt.xlabel('Index')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig('reward_trend.png')  # Save the line plot
plt.close()