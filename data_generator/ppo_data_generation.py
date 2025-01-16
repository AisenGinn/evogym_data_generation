import os
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from evogym import sample_robot
from multiprocessing import Pool
from pathlib import Path
import sys

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
    # "Carrier-v0",
    # "Carrier-v1",
    # "Pusher-v0",
    # "Pusher-v1",
    # "BeamToppler-v0",
    # "BeamSlider-v0",
    # "Thrower-v0",
    # "Catcher-v0",
    # "AreaMaximizer-v0",
    # "AreaMinimizer-v0",
    # "WingspanMazimizer-v0",
    # "HeightMaximizer-v0",
    # "Climber-v0",
    # "Climber-v1",
    # "Climber-v2",
    # "BidirectionalWalker-v0",
]

def run_and_save_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: np.ndarray = None,
    seed: int = 42,
) -> None:
    """
    Run PPO and save the best model for a given environment.
    """
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })

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

    best_model_path = os.path.join(model_save_dir, model_save_name)
    callback.best_model.save(best_model_path)
    print(f"Saved best model for {env_name} at {best_model_path}")

def run_ppo_parallel(args: argparse.Namespace, body: np.ndarray, connections: np.ndarray, env_name: str):
    """
    Wrapper function for multiprocessing to run PPO on a single environment.
    """
    model_save_dir = os.path.join(args.save_dir, env_name)
    os.makedirs(model_save_dir, exist_ok=True)
    run_and_save_ppo(args, body, env_name, model_save_dir, env_name, connections)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument("--exp_id", type=str, default="0", help="Name of the experiment (default: 0)")
    parser.add_argument("--env-name", type=str, default=None, help="Name of the environment to run (default: None)")
    parser.add_argument("--structure_shape", type=int, default=5, help="Shape of the structure (default: (5,5))")
    parser.add_argument("--run-all", action="store_true", help="Run PPO for all environments in env_names")

    add_ppo_args(parser)
    args = parser.parse_args()

    save_dir = os.path.join("/media/hdd2/saved_data", f"test_ppo_{args.exp_id}/ppo_data_{args.structure_shape}")
    args.save_dir = save_dir
    args.total_timesteps = int(1e5)
    args.eval_interval = 1
    body, connections = sample_robot((args.structure_shape, args.structure_shape))

    print(args)

    if args.run_all:
        with Pool(processes=16) as pool:  # Adjust the number of processes based on your system
            pool.starmap(
                run_ppo_parallel,
                [(args, body, connections, env_name) for env_name in env_names_list]
            )
    else:
        if args.env_name is None:
            print("Please provide an environment name with --env-name or use --run-all to run all environments.")
        else:
            model_save_dir = os.path.join(args.save_dir, args.env_name)
            os.makedirs(model_save_dir, exist_ok=True)
            run_and_save_ppo(args, body, args.env_name, model_save_dir, args.env_name, connections)
