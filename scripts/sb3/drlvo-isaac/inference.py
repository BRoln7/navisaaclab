"""Script to play a checkpoint."""

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
# parser.add_argument("--disable_fabric", action="store_true", default=True, help="Disable fabric and use USD I/O operations.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import random

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from gymnasium import spaces

from omni.isaac.lab.utils.dict import print_dict

from omni.isaac.lab_tasks.direct.carter.pedestrain_network.pedestrain_noshape import TaskMLP
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

from custom_cnn_full import CustomCNN

obs_nums = 19202
num_ped = 1
device = "cuda:0"

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)
    env.seed(seed=agent_cfg["seed"])

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        print("env is VecNormalized")
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
            # clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )
        env_obs = np.hstack((np.ones(19202), np.full(1401 * num_ped, np.inf)))
        env.observation_space = spaces.Box(
            low=-env_obs,
            high=env_obs,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    robot_checkpoint = torch.load("source/examples/drlvo-isaac/model/drl_vo/policy.pth")
    pacer_checkpoint = torch.load('source/examples/drlvo-isaac/Pacer_noshape.pth')
    
    # robot
    robot_policy_kwargs=dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[128]),
    )
    env_actions = np.ones(2)
    env.action_space = spaces.Box(
        low=-env_actions,
        high=env_actions,
        shape=env_actions.shape,
        dtype=np.float32,
    )
    agent_robot = PPO(policy=policy_arch, env=env, policy_kwargs=robot_policy_kwargs)
    agent_robot.policy.load_state_dict(robot_checkpoint)

    # pedestrain
    pacer_policy_kwargs=dict(
        features_extractor_class=TaskMLP,
        features_extractor_kwargs=dict(features_dim=613),
        net_arch=dict(pi=[2048, 1024], vf=[2048, 1024]),
        activation_fn=torch.nn.ReLU
    )
    env_actions = np.ones(69 * num_ped)
    env.action_space = spaces.Box(
        low=-env_actions,
        high=env_actions,
        shape=env_actions.shape,
        dtype=np.float32,
    )
    agent_pacer = PPO(policy=policy_arch, env=env, policy_kwargs=pacer_policy_kwargs)
    agent_pacer.policy.load_state_dict(pacer_checkpoint)
    print("agents loaded")
    env_actions = np.ones(69 * num_ped + 2)
    env.action_space = spaces.Box(
        low=-env_actions,
        high=env_actions,
        shape=env_actions.shape,
        dtype=np.float32,
    )
    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions_robot, _ = agent_robot.predict(obs, deterministic=True)
            actions_pacer, _ = agent_pacer.predict(obs, deterministic=True)
            # env stepping
            actions = np.concatenate((actions_robot, actions_pacer), axis=-1)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
