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
    
    "Climber-v0": "The robot climbs as high as possible on a vertical wall.",
    # "Climber-v1": "The robot climbs through a vertical channel made of mixed rigid and soft materials.",
    
    "UpStepper-v0": "The robot climbs up stairs of varying lengths.",
    "GapJumper-v0": "The robot traverses a series of floating platforms, each spaced 5 units apart, all at the same height.",

    "Jumper-v0": "The robot jumps as high as possible in place on flat terrain.",
    "Balancer-v0": "The robot is initialized on top of a thin pole and balances on it.",

}

difficulty_dict = {
    # Locomotion Tasks
    "Walker-v0": "easy",
    "BridgeWalker-v0": "medium",
    "BidirectionalWalker-v0": "hard",
    "UpStepper-v0": "medium",
    "GapJumper-v0": "hard",

    # Object Manipulation Tasks
    "Carrier-v0": "medium",
    "Carrier-v1": "hard",
    "Pusher-v0": "easy",
    "Pusher-v1": "medium",
    # "Thrower-v0": "hard",
    
    # Climbing Tasks
    "Climber-v0": "medium",
    # "Climber-v1": "hard",
    
    # Balance and Jumping Tasks
    "Jumper-v0": "medium",
    "Balancer-v0": "hard",
}

def load_data(source_path, recur, num_choices=2):
    """
    Load and preprocess the data.
    
    Args:
        source_path (str): Path to the JSON file.
        num_choices (int): Number of choices per question (default: 2).
    
    Returns:
        list: Preprocessed data sorted by reward with duplicate 5x5 structures removed.
    """
    with open(source_path, 'r') as file:
        sourcedata = json.load(file)

    # Sort in descending order by reward
    sourcedata.sort(key=lambda x: x["reward"], reverse=True)

    # Ensure the data length is a multiple of num_choices
    sourcedata = sourcedata[:len(sourcedata) - (len(sourcedata) % num_choices)]
    
    if recur == "norepeat":
        # Remove entries with duplicate 5x5 structures
        seen_structures = set()
        unique_data = []
        for entry in sourcedata:
            # Convert the 5x5 structure (list of lists) to a tuple of tuples (hashable)
            structure_tuple = tuple(tuple(row) for row in entry["structure"])
            if structure_tuple not in seen_structures:
                seen_structures.add(structure_tuple)
                unique_data.append(entry)
        return unique_data

    return sourcedata

def create_questions_context(sourcedata, mode="easy", description="better", num_choices=2):
    """
    Create questions with the specified number of choices (2 or 4).
    If mode is "high", the choices come from largely separated reward buckets.
    If mode is "low", each bucket is further subdivided so that the choices are closer in reward.
    
    Args:
        sourcedata (list): Preprocessed data sorted by reward.
        mode (str): "high" or "low" difference in reward values.
        num_choices (int): Number of choices per question.
    
    Returns:
        list: A list of dictionaries containing questions.
    """
    questions = []
    
    if mode == "hard":
        # First level: split the sorted data into `num_choices` groups.
        top_level_groups = np.array_split(sourcedata, num_choices)
        top_level_groups = [list(group) for group in top_level_groups]
        
        # Second level: for each top-level group, further split into `num_choices` sub-groups.
        subdivided_groups = [np.array_split(group, num_choices) for group in top_level_groups]
        subdivided_groups = [[list(subgroup) for subgroup in groups] for groups in subdivided_groups]

        for top_group in subdivided_groups:  # Iterate over each top-level group separately
            num_subgroups = len(top_group)  # Number of subdivided groups in this top-level group

            # Ensure we have enough different subdivided groups to pick from
            if num_subgroups < num_choices:
                continue  # Skip if we don’t have enough groups to form a question

            num_questions_in_round = min(len(subgroup) for subgroup in top_group)  # Min available data
            for k in range(num_questions_in_round):  # Iterate over available data points
                choices = [top_group[j][k] for j in range(num_choices)]  # Select one from each subdivided group

                if len(choices) != num_choices:
                    continue  # Skip if we don’t have enough choices
                
                # Shuffle the choices randomly
                random.shuffle(choices)

                # Extract task info
                task = choices[0]['env_name']
                structures = [choice['structure'] for choice in choices]
                rewards = [choice['reward'] for choice in choices]

                question = {
                    "question": f"Which of the following robot structures will perform {description} in the evogym {task} task?",
                    "choices": structures,
                    "correct_answer": rewards.index(max(rewards)) if description == "better" else rewards.index(min(rewards)),
                    "reward": rewards,
                    "env_name": task,
                    "difficulty": difficulty_dict.get(task, "unknown"),
                    "task_description": task_description_dict.get(task, "Task description not available."),
                    "structure_description": (
                        "The robot structure uses a unified multi-material voxel-based representation. "
                        "The robot is represented as a 5x5 material matrix. The entries of the material matrix "
                        "are integers corresponding to a voxel type from the set {Empty, Rigid, Soft, Horizontal Actuator, Vertical Actuator}. "
                        "0 stands for Empty voxel, 1 stands for Rigid voxel, 2 stands for Soft voxel, "
                        "3 stands for Horizontal Actuator voxel, 4 stands for Vertical Actuator voxel. "
                        "All pairs of adjacent voxels are connected to each other."
                    ),
                    "actuation_description": (
                        "The robot's controller is optimized to maximize task performance by actuating its structure through precise deformation."
                    )
                }
                questions.append(question)
                
        return questions

    else:  # mode == "easy"
        # Original behavior: split data into num_choices groups and zip them.
        sub_lists = np.array_split(sourcedata, num_choices)
        sub_lists = [list(sublist) for sublist in sub_lists]
        questiondata = [list(items) for items in zip(*sub_lists)]
        
        for eachquestion in questiondata:
            # If there are extra items (should not happen), randomly drop extras.
            while len(eachquestion) > num_choices:
                eachquestion.pop(random.randint(0, len(eachquestion) - 1))
            # Randomize the order.
            random.shuffle(eachquestion)
            
            task = eachquestion[0]['env_name']
            structures = [option['structure'] for option in eachquestion]
            rewards = [option['reward'] for option in eachquestion]
            
            question = {
                "question": f"Which of the following robot structures will perform {description} in the evogym {task} task?",
                "choices": structures,
                "correct_answer": rewards.index(max(rewards)) if description == "better" else rewards.index(min(rewards)),
                "reward": rewards,
                "env_name": task,
                "difficulty": difficulty_dict.get(task, "unknown"),
                "task_description": task_description_dict.get(task, "Task description not available."),
                "structure_description": (
                    "The robot structure uses a unified multi-material voxel-based representation. "
                    "The robot is represented as a 5x5 material matrix. The entries of the material matrix "
                    "are integers corresponding to a voxel type from the set {Empty, Rigid, Soft, Horizontal Actuator, Vertical Actuator}. "
                    "0 stands for Empty voxel, 1 stands for Rigid voxel, 2 stands for Soft voxel, "
                    "3 stands for Horizontal Actuator voxel, 4 stands for Vertical Actuator voxel. "
                    "All pairs of adjacent voxels are connected to each other."
                ),
                "actuation_description": (
                    "The robot's controller is optimized to maximize task performance by actuating its structure through precise deformation."
                )
            }
            questions.append(question)
        return questions

def main():
    parser = argparse.ArgumentParser(description="Generate multiple-choice questions for Evolution Gym environments.")
    parser.add_argument("--env_id", type=str, required=True, help="Specify an environment ID to run a single environment, or 'all' for all environments.")
    parser.add_argument("--num_choices", type=int, choices=[2, 4], default=2, help="Number of choices per question (2 or 4).")
    parser.add_argument("--mode", type=str, choices=["easy", "hard"], default="easy", help="difficulty level of the questions. Easy choices will have larger differences in reward values.")
    parser.add_argument("--description", type=str, choices=["better", "worse"], default="better", help="ask LLMs to pick better or worse performance choices.")
    parser.add_argument("--recur", type=str, choices=["repeat", "norepeat"], default="repeat", help="Whether include repeated structure in questions")
    parser.add_argument("--data_dir", type=str, default="/media/hdd2/users/changhe/saved_data", help="Path to the data folder.")
    parser.add_argument("--output_dir", type=str, default="/media/hdd2/users/changhe/saved_questions", help="Output path for the generated questions JSON.")

    args = parser.parse_args()

    env_names = list(task_description_dict.keys()) if args.env_id == "all" else [args.env_id]
    
    for env_name in env_names:
        source_path = os.path.join(args.data_dir, f"test_ga_{env_name}/{env_name}_results.json")
        output_env_dir = f"{args.output_dir}/{env_name}"
        os.makedirs(output_env_dir, exist_ok=True)
        output_path = os.path.join(output_env_dir, f"{env_name}_QA_{args.description}_{args.mode}_{args.num_choices}_{args.recur}.json")
        # Load and process data
        sourcedata = load_data(source_path, args.recur, args.num_choices)
        questions = create_questions_context(sourcedata, args.mode, args.description, args.num_choices)
        # Save questions
        with open(output_path, 'w') as file:
            json.dump(questions, file, indent=4)
        print(f"Questions for {env_name} generated successfully! Saved to {output_path}")

if __name__ == "__main__":
    main()
