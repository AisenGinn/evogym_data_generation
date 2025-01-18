import random
import numpy as np
import argparse
import time
import evogym.envs
import sys
from pathlib import Path




import json
import os
from re import X
import shutil
import numpy as np
import argparse

from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.core.task.space import Design_space
from GPyOpt.models import GPModel
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.core.evaluators import ThompsonBatch
sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from bo.run import run_bo
from bo.optimizer import Optimization
from ppo.run import run_ppo
from ppo.args import add_ppo_args

from evogym import is_connected, has_actuator, get_full_connectivity


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
    "Walker-v0": "walk_WalkingFlat",
    "BridgeWalker-v0": "walk_SoftBridge",
    "CaveCrawler-v0": "walk_Duck",
    "Jumper-v0": "jump_StationaryJump",
    "Flipper-v0": "flip_Flipping",
    "Balancer-v0": "balance_Balance",
    "Balancer-v1": "balance_BalanceJump",
    "UpStepper-v0": "traverse_StepsUp",
    "DownStepper-v0": "traverse_StepsDown",
    "ObstacleTraverser-v0": "traverse_WalkingBumpy",
    "ObstacleTraverser-v1": "traverse_WalkingBumpy2",
    "Hurdler-v0": "traverse_VerticalBarrier",
    "GapJumper-v0": "traverse_Gaps",
    "PlatformJumper-v0": "traverse_FloatingPlatform",
    "Traverser-v0": "traverse_BlockSoup",
    "Lifter-v0": "manipulate_LiftSmallRect",
    "Carrier-v0": "manipulate_CarrySmallRect",
    "Carrier-v1": "manipulate_CarrySmallRectToTable",
    "Pusher-v0": "manipulate_PushSmallRect",
    "Pusher-v1": "manipulate_PushSmallRectOnOppositeSide",
    "BeamToppler-v0": "manipulate_ToppleBeam",
    "BeamSlider-v0": "manipulate_SlideBeam",
    "Thrower-v0": "manipulate_ThrowSmallRect",
    "Catcher-v0": "manipulate_CatchSmallRect",
    "AreaMaximizer-v0": "change_shape_MaximizeShape",
    "AreaMinimizer-v0": "change_shape_MinimizeShape",
    "WingspanMazimizer-v0": "change_shape_MaximizeXShape",
    "HeightMaximizer-v0": "change_shape_MaximizeYShape",
    "Climber-v0": "climb_Climb0",
    "Climber-v1": "climb_Climb1",
    "Climber-v2": "climb_Climb2",
    "BidirectionalWalker-v0": "multi_goal_BiWalk"
}

class Objective(SingleObjective):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.config = config

        results_path = os.path.join("../saved_data", config['exp_name'], f"{name_dict[config['env_name']]}_results.json")

        ### INITIALIZE RESULT STORAGE ###
        self.results = {f"{config['env_name']}_{name_dict[config['env_name']]}": []}
        if os.path.exists(results_path):
            print(f"Loading existing results from {results_path}")
            with open(results_path, "r") as f:
                results = json.load(f)

    def evaluate(self, x, generation):
        """
        Performs the evaluation of the objective at x.
        """
        if self.n_procs == 1:
            f_evals, cost_evals = self._eval_func(x, generation)
        else:
            try:
                f_evals, cost_evals = self._syncronous_batch_evaluation(x, generation)
            except:
                if not hasattr(self, 'parallel_error'):
                    print('Error in parallel computation. Fall back to single process!')
                else:
                    self.parallel_error = True
                f_evals, cost_evals = self._eval_func(x, generation)

        return f_evals, cost_evals

    def _eval_func(self, x, generation, idx=None, queue=None):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        if idx is None: idx = list(range(x.shape[0]))

        for i in range(x.shape[0]):
            st_time    = time.time()
            rlt = self.func(np.atleast_2d(x[i]), self.config, idx[i], generation,self.results)
            f_evals     = np.vstack([f_evals,rlt])
            cost_evals += [time.time()-st_time]
        if queue is None:
            return f_evals, cost_evals
        else:
            queue.put([idx, f_evals, cost_evals])

    def _syncronous_batch_evaluation(self, x, generation):
        """
        Evaluates the function a x, where x can be a single location or a batch. The evaluation is performed in parallel
        according to the number of accessible cores.
        """
        from multiprocessing import Process, Queue

        # --- parallel evaluation of the function
        divided_samples = [x[i::self.n_procs] for i in range(self.n_procs)]
        divided_idx = [list(range(x.shape[0]))[i::self.n_procs] for i in range(self.n_procs)]
        queue = Queue()
        proc = [Process(target=self._eval_func,args=(k, generation, idx, queue)) for k, idx in zip(divided_samples, divided_idx)]
        [p.start() for p in proc]

        # --- time of evaluation is set to constant (=1). This is one of the hypothesis of synchronous batch methods.
        f_evals = np.zeros((x.shape[0],1))
        cost_evals = np.ones((x.shape[0],1))
        for _ in proc:
            idx, f_eval, _ = queue.get() # throw away costs
            f_evals[idx] = f_eval
        return f_evals, cost_evals

def get_robot_from_genome(genome, config):
    '''
    genome is a 1d vector
    robot is a 2d matrix
    '''
    structure_shape = config['structure_shape']
    robot = genome.reshape(structure_shape)
    return robot

def eval_genome_cost(genome, config, genome_id, generation, results):
    robot = get_robot_from_genome(genome, config)
    args, env_name = config['args'], config['env_name']

    if not (is_connected(robot) and has_actuator(robot)):
        return 10
    else:
        connectivity = get_full_connectivity(robot)
        save_path_generation = os.path.join(config['save_path'], f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        save_path_controller = os.path.join(save_path_generation, 'controller')
        np.savez(save_path_structure, robot, connectivity)
                

        generation_data = []     
        generation_data.append({
            "structure": robot.tolist(),
            "connections": connectivity.tolist(),
            #"reward": fitness
        })
        
        results[f'{env_name}_{name_dict[env_name]}'].append(generation_data) 
        print(results)

        results_path = os.path.join("../saved_data", args.exp_name, f"{name_dict[env_name]}_results.json")
        print(results_path)
        # SAVE RESULTS TO FILE
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        
        
        
        fitness = run_ppo(
            args, robot, env_name, save_path_controller, f'{genome_id}', connectivity
        )
        cost = -fitness
        return cost

def eval_genome_constraint(genomes, config):
    all_violation = []
    for genome in genomes:
        robot = get_robot_from_genome(genome, config)
        violation = not (is_connected(robot) and has_actuator(robot))
        all_violation.append(violation)
    return np.array(all_violation)


def run_bo(
    args: argparse.Namespace,
):
    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.structure_shape,
        args.max_evaluations,
        args.num_cores,
    )
    
    #save_path = os.path.join('saved_data', exp_name)
    save_path = os.path.join("../saved_data", exp_name)

    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n' \
            f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n' \
            f'MAX_EVALUATIONS: {max_evaluations}\n')

    config = {
        'structure_shape': structure_shape,
        'save_path': save_path,
        'args': args, # args for run_ppo
        'env_name': env_name,
        'exp_name':exp_name,
    }
    
    def constraint_func(genome): 
        return eval_genome_constraint(genome, config)

    def before_evaluate(generation):
        save_path = config['save_path']
        save_path_structure = os.path.join(save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def after_evaluate(generation, population_cost):
        save_path = config['save_path']
        save_path_ranking = os.path.join(save_path, f'generation_{generation}', 'output.txt')
        genome_fitness_list = -population_cost
        genome_id_list = np.argsort(population_cost)
        genome_fitness_list = np.array(genome_fitness_list)[genome_id_list]
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome_fitness in zip(genome_id_list, genome_fitness_list):
                out += f'{genome_id}\t\t{genome_fitness}\n'
            f.write(out)

    space = Design_space(
        space=[{'name': 'x', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4), 'dimensionality': np.prod(structure_shape)}], 
        constraints=[{'name': 'const', 'constraint': constraint_func}]
    )

    objective = Objective(eval_genome_cost, config, num_cores=num_cores)

    model = GPModel()

    acquisition = AcquisitionEI(
        model, 
        space, 
        optimizer=AcquisitionOptimizer(space)
    )

    evaluator = ThompsonBatch(acquisition, batch_size=pop_size)
    X_init = initial_design('random', space, pop_size)

    bo = Optimization(model, space, objective, acquisition, evaluator, X_init, de_duplication=True)
    bo.run_optimization(
        max_iter=np.ceil(max_evaluations / pop_size) - 1,
        verbosity=True,
        before_evaluate=before_evaluate,
        after_evaluate=after_evaluate
    )
    best_robot, best_fitness = bo.x_opt, -bo.fx_opt
    return best_robot, best_fitness



if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    parser = argparse.ArgumentParser(description='Arguments for ga script')
    parser.add_argument('--exp-name', type=str, default='test_bo', help='Name of the experiment (default: test_bo)')
    parser.add_argument('--env-name', type=str, default='Walker-v0', help='Name of the environment (default: Walker-v0)')
    parser.add_argument('--pop-size', type=int, default=3, help='Population size (default: 3)')
    parser.add_argument('--structure-shape', type=tuple, default=(5,5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=6, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    add_ppo_args(parser)
    args = parser.parse_args()

    best_robot, best_fitness = run_bo(args)

    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)