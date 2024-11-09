"""Script to play a checkpoint."""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
# parser.add_argument("--disable_fabric", action="store_true", default=True, help="Disable fabric and use USD I/O operations.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")


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
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
# from stable_baselines3.common.vec_env import VecNormalize

import socnav_sim.tasks  # noqa: F401
from feature_extractor import TaskMLP

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

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

    policy_kwargs = dict(
        features_extractor_class=TaskMLP,
        net_arch=dict(pi=[2048, 1024], vf=[2048, 1024]),
        activation_fn=torch.nn.ReLU)

    pacer_checkpoint = torch.load('models/pacer/path_follower.pth')
    agent_pacer = PPO(policy_arch, env, verbose=2, **agent_cfg, policy_kwargs=policy_kwargs)
    agent_pacer.policy.load_state_dict(pacer_checkpoint)
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
