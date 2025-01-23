"""Script to play a checkpoint."""

import argparse
import os
import yaml
from omni.isaac.lab.app import AppLauncher

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
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
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=50000, help="Length of the recorded video (in steps).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg, get_checkpoint_path
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from socnav_sim.tasks.amp.pacer.policy import AMPPPO
import socnav_sim.tasks  # noqa: F401

def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    checkpoint_path = args_cli.checkpoint

    agent_cfg = process_sb3_cfg(agent_cfg) # type: ignore
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = Sb3VecEnvWrapper(env) # type: ignore
    env.seed(seed=agent_cfg["seed"])

    env.action_space = gym.spaces.Box(
        low = -1,
        high = 1,
        shape= env.action_space.shape,
    )

    agent = AMPPPO.load(checkpoint_path, env, print_system_info=True)
    print("agents loaded")

    obs = env.reset()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True) # type: ignore
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
