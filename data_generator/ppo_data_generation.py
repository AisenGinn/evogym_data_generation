import os
import shutil
import json

import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from evogym import sample_robot, hashable

import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(__file__).parents[1], "examples"))
from ppo.eval import eval_policy
from ppo.args import add_ppo_args

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generator

from ppo_callback import EvalCallback

# List of environment names
env_names_list = [
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

def run_and_save_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo and return the best reward achieved during evaluation.
    """
    
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })
    
    # Eval Callback
    callback = EvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=args.verbose_ppo,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval
    )
    
    return callback.best_model

def run_all_envs(args: argparse.Namespace, body, connections, model_save_dir):
    """Run PPO for all environments in env_names."""
    for env_name in env_names_list:
        print(f"\nStarting experiment for environment: {env_name}")

        # Run GA
        best_model = run_and_save_ppo(args, body=body, env_name=env_name, model_save_dir=model_save_dir, model_save_name=env_name, connections=connections)
        best_model.save(os.path.join(model_save_dir, env_name))

        print(f"\nComplete experiment for environment: {env_name}")

if __name__ == "__main__":    
    
    # Args
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument("--exp_id", type=str, default="0", help="Name of the experiment (default: 0)")
    parser.add_argument("--env-name", type=str, default='None', help="Name of the environment to run (default: None)")
    parser.add_argument("--structure_shape", type=int, default=5, help="Shape of the structure (default: (5,5))")
    parser.add_argument("--run-all", action="store_true", help="Run GA for all environments in env_names")
    
    add_ppo_args(parser)
    args = parser.parse_args()
    
    # modify args with additional parameters
    save_dir = os.path.join("/media/hdd2/saved_data", f"test_ppo_{args.exp_id}/ppo_data_{args.structure_shape}")
    args.save_dir = save_dir
    args.total_timesteps = 1e5
    args.eval_interval = 1
    body, connections = sample_robot((args.structure_shape, args.structure_shape))

    print(args)

    # Run either a single environment or all environments
    if args.run_all:
        run_all_envs(args,body=body,connections=connections,model_save_dir=args.save_dir)
    else:
        if args.env_name is None:
            print("Please provide an environment name with --env-name or use --run-all to run all environments.")
        else:
            run_and_save_ppo(args)