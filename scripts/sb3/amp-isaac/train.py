import argparse
import os
import yaml
from omni.isaac.lab.app import AppLauncher

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'amp.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description="Train an RL agent for amp task.")
parser.add_argument("--num_envs", type=int, default=config['env']['num_envs'], help="Number of environments to simulate.") # type: ignore
parser.add_argument("--task", type=str, default=config['env']['task'], help="Name of the task.") # type: ignore
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=config['training']['checkpoint'], help="Path to model checkpoint.") # type: ignore
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
from datetime import datetime
import torch

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
import numpy as np
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg, get_checkpoint_path
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import socnav_sim.tasks
from socnav_sim.tasks.amp.amp.policy import AMPPPO, PreProcess
from socnav_sim.tasks.amp.amp_utils.utils.discriminator import bias_init
import os


def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed # type: ignore
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs # type: ignore

    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    agent_cfg = process_sb3_cfg(agent_cfg) # type: ignore
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env) # type: ignore
    env.seed(seed=agent_cfg["seed"])

    env.action_space = gym.spaces.Box(
        low = -1,
        high = 1,
        shape= env.action_space.shape,
    )

    if args_cli.checkpoint is None:
        policy_kwargs = dict(
        features_extractor_class=PreProcess,
        net_arch=dict(pi=[1024, 512], vf=[1024, 512]),
        activation_fn=torch.nn.ReLU)
        agent = AMPPPO(policy_arch, env, verbose=2, **agent_cfg, policy_kwargs=policy_kwargs)

    else:
        checkpoint_path = args_cli.checkpoint
        agent = AMPPPO.load(checkpoint_path, env, print_system_info=True)

    agent.policy.apply(bias_init)
    with torch.no_grad():
        agent.policy.log_std.fill_(np.float64(-2.9))
    agent.policy.log_std.requires_grad = False

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)
    print("[INFO]:agents loaded")

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    agent.save(path=os.path.join(log_dir, "model"))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()