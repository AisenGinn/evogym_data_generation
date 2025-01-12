import os
import numpy as np
import shutil
import random
import math
import json
import sys
import argparse
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from ppo.run import run_ppo
from ppo.args import add_ppo_args
import evogym.envs
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, Structure

# List of environment names
env_names = [
    "Walker-v0",
    "BridgeWalker-v0",
    "CaveCrawler-v0",
    "Jumper-v0",
    "Flipper-v0",
    "Balancer-v0",
    "Balancer-v1",
    "UpStepper-v0",
    "DownStepper-v0",
    "ObstacleTraverser-v0",
    "ObstacleTraverser-v1",
    "Hurdler-v0",
    "GapJumper-v0",
    "PlatformJumper-v0",
    "Traverser-v0",
    "Lifter-v0",
    "Carrier-v0",
    "Carrier-v1",
    "Pusher-v0",
    "Pusher-v1",
    "BeamToppler-v0",
    "BeamSlider-v0",
    "Thrower-v0",
    "Catcher-v0",
    "AreaMaximizer-v0",
    "AreaMinimizer-v0",
    "WingspanMazimizer-v0",
    "HeightMaximizer-v0",
    "Climber-v0",
    "Climber-v1",
    "Climber-v2",
    "BidirectionalWalker-v0",
]

name_dict = {
    "Walker-v0": "walk:WalkingFlat",
    "BridgeWalker-v0": "walk:SoftBridge",
    "CaveCrawler-v0": "walk:Duck",
    "Jumper-v0": "jump:StationaryJump",
    "Flipper-v0": "flip:Flipping",
    "Balancer-v0": "balance:Balance",
    "Balancer-v1": "balance:BalanceJump",
    "UpStepper-v0": "traverse:StepsUp",
    "DownStepper-v0": "traverse:StepsDown",
    "ObstacleTraverser-v0": "traverse:WalkingBumpy",
    "ObstacleTraverser-v1": "traverse:WalkingBumpy2",
    "Hurdler-v0": "traverse:VerticalBarrier",
    "GapJumper-v0": "traverse:Gaps",
    "PlatformJumper-v0": "traverse:FloatingPlatform",
    "Traverser-v0": "traverse:BlockSoup",
    "Lifter-v0": "manipulate:LiftSmallRect",
    "Carrier-v0": "manipulate:CarrySmallRect",
    "Carrier-v1": "manipulate:CarrySmallRectToTable",
    "Pusher-v0": "manipulate:PushSmallRect",
    "Pusher-v1": "manipulate:PushSmallRectOnOppositeSide",
    "BeamToppler-v0": "manipulate:ToppleBeam",
    "BeamSlider-v0": "manipulate:SlideBeam",
    "Thrower-v0": "manipulate:ThrowSmallRect",
    "Catcher-v0": "manipulate:CatchSmallRect",
    "AreaMaximizer-v0": "change_shape:MaximizeShape",
    "AreaMinimizer-v0": "change_shape:MinimizeShape",
    "WingspanMazimizer-v0": "change_shape:MaximizeXShape",
    "HeightMaximizer-v0": "change_shape:MaximizeYShape",
    "Climber-v0": "climb:Climb0",
    "Climber-v1": "climb:Climb1",
    "Climber-v2": "climb:Climb2",
    "BidirectionalWalker-v0": "multi_goal:BiWalk"
}

def run_ga(
    args: argparse.Namespace,
):
    print()
    
    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.structure_shape,
        args.max_evaluations,
        args.num_cores,
    )

    ### MANAGE DIRECTORIES ###
    home_path = os.path.join("/media/hdd2/saved_data", exp_name)
    results_path = os.path.join("/media/hdd2/saved_data", exp_name, f"{name_dict[env_name]}_results.json")
    start_gen = 0

    ### INITIALIZE RESULT STORAGE ###
    results = {f'{env_name}_{name_dict[env_name]}': []}
    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            results = json.load(f)
            start_gen = len(results.get(env_name, []))

    ### DEFINE TERMINATION CONDITION ###

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join("/media/hdd2/saved_data", exp_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join("/media/hdd2/saved_data", exp_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.close()

    else:
        temp_path = os.path.join("/media/hdd2/saved_data", exp_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures: List[Structure] = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = start_gen
    
    #generate a population
    if not is_continuing: 
        for i in range (pop_size):
            
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(*structure_data, i))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:

        ### UPDATE NUM SURVIORS ###			
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(generation), "controller",
                    f"{structure.label}nvidi.zip")
                save_path_controller_part_old = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(generation-1), "controller",
                    f"{structure.prev_gen_label}.zip")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error copying controller for {save_path_controller_part}.\n')
            else:
                ppo_args = (args, structure.body, env_name, save_path_controller, f'{structure.label}', structure.connections)
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
                

        group.run_jobs(num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        generation_data = []
        for structure in structures:
            structure.compute_fitness()
            generation_data.append({
                "structure": structure.body.tolist(),
                "connections": structure.connections.tolist(),
                "reward": structure.fitness
            })

        results[f'{env_name}_{name_dict[env_name]}'].append(generation_data)

        # SAVE RESULTS TO FILE
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join("/media/hdd2/saved_data", exp_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

         ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        generation += 1

def run_all_envs(args: argparse.Namespace):
    """Run GA for all environments in env_names."""
    for env_name in env_names:
        print(f"\nStarting experiment for environment: {env_name}")

        # Create arguments specific to this environment
        args.exp_name = f'test_ga_{env_name}'
        args.env_name = env_name

        # Run GA
        run_ga(args)

        print(f"\nCompleted experiment for environment: {env_name}\n")

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Arguments for GA script")
    parser.add_argument("--exp-name", type=str, default="test_ga", help="Name of the experiment (default: test_ga)")
    parser.add_argument("--env-name", type=str, default=None, help="Name of the environment to run (default: None)")
    parser.add_argument("--pop-size", type=int, default=3, help="Population size (default: 3)")
    parser.add_argument("--structure_shape", type=tuple, default=(5, 5), help="Shape of the structure (default: (5,5))")
    parser.add_argument("--max-evaluations", type=int, default=6, help="Maximum number of robots to evaluate (default: 6)")
    parser.add_argument("--num-cores", type=int, default=3, help="Number of robots to evaluate simultaneously (default: 3)")
    parser.add_argument("--run-all", action="store_true", help="Run GA for all environments in env_names")
    add_ppo_args(parser)

    args = parser.parse_args()

    # Run either a single environment or all environments
    if args.run_all:
        run_all_envs(args)
    else:
        if args.env_name is None:
            print("Please provide an environment name with --env-name or use --run-all to run all environments.")
        else:
            run_ga(args)

