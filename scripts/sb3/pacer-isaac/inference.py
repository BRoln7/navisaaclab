"""Script to play a checkpoint."""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from stable_baselines3 import PPO
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg, get_checkpoint_path
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
# from stable_baselines3.common.vec_env import VecNormalize

import socnav_sim.tasks  # noqa: F401

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint

    agent_cfg = process_sb3_cfg(agent_cfg) # type: ignore
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = Sb3VecEnvWrapper(env) # type: ignore
    env.seed(seed=agent_cfg["seed"])

    env.action_space = gym.spaces.Box(
        low = -1,
        high = 1,
        shape= env.action_space.shape,
    )

    agent_pacer = PPO.load(checkpoint_path, env, print_system_info=True)
    print("agents loaded")

    obs = env.reset()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _ = agent_pacer.predict(obs, deterministic=True) # type: ignore
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
