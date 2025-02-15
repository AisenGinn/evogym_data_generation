import json
import numpy as np
import random
import argparse
import os

# Updated task descriptions
task_description_dict = {
    "Walker-v0": "The robot walks as far as possible on flat terrain.",
    "BridgeWalker-v0": "The robot walks as far as possible on a soft rope-bridge.",
    "BidirectionalWalker-v0": "The robot walks bidirectionally towards changing goals.",
    
    "Carrier-v0": "The robot catches a box initialized above it and carries it as far as possible.",
    "Carrier-v1": "The robot carries a box to a table and places it on the table.",
    "Pusher-v0": "The robot pushes a box initialized in front of it.",
    "Pusher-v1": "The robot pushes or drags a box initialized behind it in the forward direction.",
    # "Thrower-v0": "The robot throws a box initialized on top of it.",
    
    "Climber-v0": "The robot climbs as high as possible on a vertical wall.",
    "Climber-v1": "The robot climbs through a vertical channel made of mixed rigid and soft materials.",
    # "Climber-v2": "The robot climbs through a narrow stepwise channel.",
    
    "UpStepper-v0": "The robot climbs up stairs of varying lengths.",
    # "DownStepper-v0": "The robot climbs down stairs of varying lengths.",
    # "ObstacleTraverser-v0": "The robot walks across terrain that gets increasingly more bumpy.",
    # "ObstacleTraverser-v1": "The robot walks through very bumpy terrain.",
    # "Hurdler-v0": "The robot walks across terrain with tall obstacles.",
    # "PlatformJumper-v0": "The robot traverses a series of floating platforms at different heights.",
    "GapJumper-v0": "The robot traverses a series of spaced-out floating platforms all at the same height.",
    # "Traverser-v0": "The robot traverses a pit of rigid blocks to get to the other side without sinking into the pit.",
    # "CaveCrawler-v0": "The robot squeezes its way through caves and low-hanging obstacles.",
    
    # "AreaMaximizer-v0": "The robot grows to occupy the largest possible surface area.",
    # "AreaMinimizer-v0": "The robot shrinks to occupy the smallest possible surface area.",
    # "WingspanMaximizer-v0": "The robot grows to be as wide as possible.",
    # "HeightMaximizer-v0": "The robot grows to be as tall as possible.",
    
    # "Flipper-v0": "The robot flips counter-clockwise as many times as possible on flat terrain.",
    "Jumper-v0": "The robot jumps as high as possible in place on flat terrain.",
    "Balancer-v0": "The robot is initialized on top of a thin pole and balances on it.",
    #"Balancer-v1": "The robot is initialized next to a thin pole, jumps onto it, and balances."
}

difficulty_dict = {
    # Locomotion Tasks
    "Walker-v0": "easy",
    "BridgeWalker-v0": "medium",
    "BidirectionalWalker-v0": "hard",
    "UpStepper-v0": "medium",
    # "DownStepper-v0": "easy",
    # "ObstacleTraverser-v0": "medium",
    # "ObstacleTraverser-v1": "hard",
    # "Hurdler-v0": "hard",
    # "PlatformJumper-v0": "hard",
    "GapJumper-v0": "hard",
    # "Traverser-v0": "hard",
    # "CaveCrawler-v0": "medium",
    
    # Object Manipulation Tasks
    "Carrier-v0": "medium",
    "Carrier-v1": "hard",
    "Pusher-v0": "easy",
    "Pusher-v1": "medium",
    # "Thrower-v0": "hard",
    
    # Climbing Tasks
    "Climber-v0": "medium",
    "Climber-v1": "hard",
    # "Climber-v2": "very hard",
    
    # Growth Tasks
    # "AreaMaximizer-v0": "medium",
    # "AreaMinimizer-v0": "medium",
    # "WingspanMaximizer-v0": "hard",
    # "HeightMaximizer-v0": "hard",
    
    # Balance and Jumping Tasks
    #"Flipper-v0": "medium",
    "Jumper-v0": "medium",
    "Balancer-v0": "hard",
    #"Balancer-v1": "very hard",
}

def load_data(source_path, num_choices=2):
    """
    Load and preprocess the data.
    
    Args:
        source_path (str): Path to the JSON file.
        num_choices (int): Number of choices per question (default: 2).
    
    Returns:
        list: Preprocessed data sorted by reward.
    """
    with open(source_path, 'r') as file:
        sourcedata = json.load(file)

    # Sort in descending order by reward
    sourcedata.sort(key=lambda x: x["reward"], reverse=True)

    # Ensure the data length is a multiple of num_choices
    sourcedata = sourcedata[:len(sourcedata) - (len(sourcedata) % num_choices)]

    return sourcedata

def create_questions_context(sourcedata, mode="high", num_choices=2):
    """
    Create questions with the specified number of choices (2 or 4).

    Args:
        sourcedata (list): Preprocessed data sorted by reward.
        num_choices (int): Number of choices per question (default: 2).

    Returns:
        list: A list of dictionaries containing questions.
    """
    # Split data into evenly sized sublists
    sub_lists = np.array_split(sourcedata, num_choices)
    sourcedata = [list(sublist) for sublist in sub_lists]
    questiondata = [list(items) for items in zip(*sourcedata)]

    questions = []
    for eachquestion in questiondata:
        question = {"question": None, "choices": None, "correct_answer": None, "reward": None}
        structure, reward = [], []

        # Ensure the number of choices is exactly `num_choices`
        while len(eachquestion) > num_choices:
            eachquestion.pop(random.randint(0, len(eachquestion) - 1))

        # Randomize the order of choices
        random.shuffle(eachquestion)

        # Extract relevant data
        task = eachquestion[0]['env_name']
        for eachoption in eachquestion:
            structure.append(eachoption['structure'])
            reward.append(eachoption['reward'])

        # Construct the question
        question["question"] = f"Which of the following robot structures will perform better in the evogym {task} task?"
        question["choices"] = structure
        question["correct_answer"] = reward.index(max(reward))  # Index of the highest reward (best performance)
        question["reward"] = reward
        
        # Add context
        question["env_name"] = task
        question["difficulty"] = difficulty_dict.get(task, "unknown")
        question["task_description"] = task_description_dict.get(task, "Task description not available.")
        question["structure_description"] = "The robot structure uses a unified multi-material voxel-based representation. " \
                                            "The robot is represented as a 5x5 material matrix. The entries of the material matrix " \
                                            "are integers corresponding to a voxel type from the set {Empty, Rigid, Soft, Horizontal Actuator, Vertical Actuator}. " \
                                            "0 stands for Empty voxel, 1 stands for Rigid voxel, 2 stands for Soft voxel, 3 stands for Horizontal Actuator voxel, 4 stands for Vertical Actuator voxel. " \
                                            "All pairs of adjacent voxels are connected to each other."
        question["actuation_description"] = "A sinusoidal controller is used to generate periodic actuation signals. It controls robot motion with a fixed frequency."

        questions.append(question)

    return questions


def generate_contexts(env_names):
    """ Generate contextual descriptions for environments. """
    contexts = []
    for env in env_names:
        context = {
            "key": env,
            "task description": task_description_dict.get(env, "Task description not available."),
            "structure description": "The robot structure uses a unified multi-material voxel-based representation. "
                                    "The robot is represented as a 5x5 material matrix. The entries of the material matrix "
                                    "are integers corresponding to a voxel type from the set {Empty, Rigid, Soft, Horizontal Actuator, Vertical Actuator}. "
                                    "0 stands for Empty voxel, 1 stands for Rigid voxel, 2 stands for Soft voxel, 3 stands for Horizontal Actuator voxel, 4 stands for Vertical Actuator voxel. "
                                    "All pairs of adjacent voxels are connected to each other.",
            "actuation description": "A sinusoidal controller is used to generate periodic actuation signals. It controls robot motion with a fixed frequency.",                
        }
        contexts.append(context)
    return contexts

def main():
    parser = argparse.ArgumentParser(description="Generate multiple-choice questions for Evolution Gym environments.")
    parser.add_argument("--env_id", type=str, required=True, help="Specify an environment ID to run a single environment, or 'all' for all environments.")
    parser.add_argument("--num_choices", type=int, choices=[2, 4], default=2, help="Number of choices per question (2 or 4).")
    parser.add_argument("--mode", type=str, choices=["high", "low"], default="high", help="high or low difference in reward values. (low difference leads to harder questions)")
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_data", help="Path to the data folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_questions_fc", help="Output path for the generated questions JSON.")

    args = parser.parse_args()

    env_names = list(task_description_dict.keys()) if args.env_id == "all" else [args.env_id]
    
    for env_name in env_names:
        source_path = os.path.join(args.data_dir, f"test_ga_{env_name}/{env_name}_results.json")
        output_path = os.path.join(args.output_dir, f"{env_name}_fc_questions_{args.mode}_{args.num_choices}.json")
        # Load and process data
        sourcedata = load_data(source_path, args.num_choices)
        questions = create_questions_context(sourcedata, args.mode, args.num_choices)
        # Save questions
        with open(output_path, 'w') as file:
            json.dump(questions, file, indent=4)
        print(f"Questions for {env_name} generated successfully! Saved to {output_path}")

if __name__ == "__main__":
    main()
