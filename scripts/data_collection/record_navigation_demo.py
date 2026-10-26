#!/usr/bin/env python3
"""Record a short third-person SocNav scene driven by the map-path controller."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from socnav_sim.config import PROJECT_ROOT, load_task_config


config = load_task_config()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=config["env"]["task"], help="Registered CADRL task name")
parser.add_argument("--num_envs", type=int, default=config["env"]["num_envs"])
parser.add_argument("--num_robots", type=int, default=config["env"]["num_robots"])
parser.add_argument("--steps", type=int, default=300, help="Number of 30 Hz video frames")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=Path)
parser.add_argument("--lookahead", type=float, default=0.8)
parser.add_argument("--speed", type=float, default=0.42)
parser.add_argument(
    "--realistic_pedestrians",
    action="store_true",
    help="Render clothed TailorNet pedestrian skins (disabled by default).",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True, device="cuda:0")
args_cli = parser.parse_args()
if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if not 0 < args_cli.num_robots <= args_cli.num_envs:
    parser.error("--num_robots must be in [1, --num_envs]")
if not 0.0 < args_cli.speed <= 0.5:
    parser.error("--speed must be in (0, 0.5]")
os.environ["SOCNAV_DEVICE"] = args_cli.device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import socnav_sim.tasks  # noqa: F401
from socnav_sim.tasks.socnav.egonav.controller import path_following_actions


def main() -> None:
    if not args_cli.task.startswith("Isaac-Cadrl-"):
        raise ValueError(f"Expected a CADRL task, got {args_cli.task!r}")
    output_dir = args_cli.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "demos" / args_cli.task / stamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.actions.navcmd.active_robot_count = args_cli.num_robots
    env_cfg.events.reset_navigation.params["active_robot_count"] = args_cli.num_robots
    env_cfg.actions.navcmd.realistic_pedestrian_visuals = args_cli.realistic_pedestrians

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.steps,
        name_prefix="socnav-overview",
        disable_logger=True,
    )
    try:
        env.reset(seed=args_cli.seed)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        nav_term = base_env.action_manager.get_term("navcmd")
        navigation = nav_term.navigation_state
        print(
            f"[SocNav] Recording {args_cli.steps} frames with "
            f"{args_cli.num_robots} robots and {args_cli.num_envs} pedestrians"
        )
        print(f"[SocNav] Output: {output_dir}")

        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = path_following_actions(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    navigation.robot_paths,
                    navigation.robot_path_lengths,
                    active_count=args_cli.num_robots,
                    path_spacing=float(nav_term.cfg.path_spacing),
                    lookahead_distance=args_cli.lookahead,
                    maximum_linear_speed=args_cli.speed,
                )
                env.step(actions)
            if (step + 1) % 100 == 0:
                print(f"[SocNav] {step + 1}/{args_cli.steps} frames")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
