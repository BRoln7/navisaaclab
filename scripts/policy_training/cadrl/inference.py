"""Script to play a checkpoint."""

import argparse
import os

from isaaclab.app import AppLauncher

from socnav_sim.config import load_task_config
config = load_task_config()

parser = argparse.ArgumentParser(description="Run a CADRL checkpoint in an IsaacNav scene.")
parser.add_argument("--num_envs", type=int, default=config["env"]["num_envs"], help="Number of pedestrians.")
parser.add_argument("--num_robots", type=int, default=config["env"]["num_robots"], help="Number of active robots.")
parser.add_argument("--task", type=str, default=config["env"]["task"], help="Registered task name.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=config["training"]["checkpoint"], help="Model checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--realistic_pedestrians",
    action="store_true",
    help="Render clothed TailorNet pedestrian skins (disabled by default).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ["SOCNAV_DEVICE"] = args_cli.device
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import socnav_sim.tasks  # noqa: F401
from socnav_sim.tasks.socnav.nav_utils.policy_base import MRPPO


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not 0 < args_cli.num_robots <= args_cli.num_envs:
        raise ValueError("--num_robots must be positive and no larger than --num_envs.")
    env_cfg.actions.navcmd.active_robot_count = args_cli.num_robots
    env_cfg.events.reset_navigation.params["active_robot_count"] = args_cli.num_robots
    env_cfg.actions.navcmd.realistic_pedestrian_visuals = args_cli.realistic_pedestrians
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    checkpoint_path = args_cli.checkpoint
    if not checkpoint_path:
        raise ValueError("Inference requires --checkpoint /path/to/model.zip")

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
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    env.seed(seed=agent_cfg["seed"])

    env.action_space = gym.spaces.Box(
        low=-1,
        high=1,
        shape=env.action_space.shape,
    )

    agent = MRPPO.load(
        checkpoint_path, env, print_system_info=True, active_envs=args_cli.num_robots,
        device=args_cli.device,
    )
    print(f"[INFO] Playing {checkpoint_path}")

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
    try:
        main()
    finally:
        simulation_app.close()
