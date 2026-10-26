#!/usr/bin/env python3
"""Record synchronized robot-egocentric RGB-D and state data in a SocNav scene."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from socnav_sim.config import PROJECT_ROOT, load_task_config


config = load_task_config()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=config["env"]["task"], help="Registered EgoNav task name")
parser.add_argument("--num_envs", type=int, default=config["env"]["num_envs"], help="Number of pedestrians")
parser.add_argument("--steps", type=int, default=900, help="Number of 30 Hz control/camera frames")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=Path, help="Dataset directory (a timestamped path is used by default)")
parser.add_argument("--video_only", action="store_true", help="Write WebM and metadata without RGB/depth PNGs")
parser.add_argument("--lookahead", type=float, default=0.8, help="Path look-ahead distance in metres")
parser.add_argument("--speed", type=float, default=0.42, help="Maximum robot linear speed in m/s")
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
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if not 0.0 < args_cli.speed <= 0.5:
    parser.error("--speed must be in (0, 0.5]")
if args_cli.lookahead < 0.0:
    parser.error("--lookahead must be non-negative")
os.environ["SOCNAV_DEVICE"] = args_cli.device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import socnav_sim.tasks  # noqa: F401  register environments after SimulationApp starts
from socnav_sim.tasks.socnav.egonav.controller import path_following_actions
from socnav_sim.tasks.socnav.egonav.recorder import EgoNavCamera, EgoNavDatasetWriter


def _tensor_list(value) -> list[float]:
    return value.detach().cpu().to(torch.float64).tolist()


def main() -> None:
    if not args_cli.task.startswith("Isaac-EgoNav-"):
        raise ValueError(f"Expected an EgoNav task, got {args_cli.task!r}")
    output_dir = args_cli.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "egonav" / args_cli.task / stamp

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.actions.navcmd.active_robot_count = 1
    env_cfg.events.reset_navigation.params["active_robot_count"] = 1
    env_cfg.actions.navcmd.realistic_pedestrian_visuals = args_cli.realistic_pedestrians

    env = None
    writer = None
    camera = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        env.reset(seed=args_cli.seed)
        robot = base_env.scene["robot"]
        nav_term = base_env.action_manager.get_term("navcmd")
        navigation = nav_term.navigation_state
        fps = 1.0 / float(base_env.step_dt)
        writer = EgoNavDatasetWriter(output_dir, fps, save_frames=not args_cli.video_only)
        resolution = (
            int(config["env"].get("camera_width", 640)),
            int(config["env"].get("camera_height", 360)),
        )
        camera = EgoNavCamera(base_env, resolution)
        camera_near = float(config["env"].get("camera_near", 0.1))
        camera_far = float(config["env"].get("camera_far", 30.0))

        writer.write_calibration(camera.intrinsic_matrix, (resolution[1], resolution[0]))
        print(f"[EgoNav] Recording {args_cli.steps} frames at {fps:.1f} Hz")
        print(f"[EgoNav] Output: {writer.output_dir}")

        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = path_following_actions(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    navigation.robot_paths,
                    navigation.robot_path_lengths,
                    active_count=1,
                    path_spacing=float(nav_term.cfg.path_spacing),
                    lookahead_distance=args_cli.lookahead,
                    maximum_linear_speed=args_cli.speed,
                )
                _, _, terminated, truncated, _ = env.step(actions)
                rgb, depth = camera.capture()
                depth[(depth < camera_near) | (depth > camera_far)] = np.inf
                metadata = {
                    "timestamp_seconds": step * float(base_env.step_dt),
                    "action_normalized": _tensor_list(actions[0]),
                    "robot_position_world": _tensor_list(robot.data.root_pos_w[0]),
                    "robot_quaternion_world_wxyz": _tensor_list(robot.data.root_quat_w[0]),
                    "robot_linear_velocity_world": _tensor_list(robot.data.root_lin_vel_w[0]),
                    "robot_angular_velocity_world": _tensor_list(robot.data.root_ang_vel_w[0]),
                    "camera_position_world": camera.position.tolist(),
                    "camera_target_world": camera.target.tolist(),
                    "goal_position_world_xy": _tensor_list(navigation.robot_goals[0]),
                    "episode_reset": bool((terminated[0] | truncated[0]).item()),
                }
                writer.write(rgb, depth, metadata)
            if (step + 1) % 100 == 0:
                print(f"[EgoNav] {step + 1}/{args_cli.steps} frames")
    except KeyboardInterrupt:
        print("[EgoNav] Interrupted; finalizing the partial dataset and video...")
    finally:
        if writer is not None:
            writer.close()
            print(f"[EgoNav] Saved {writer.frames} frames to {writer.output_dir}")
            print(f"[EgoNav] Video: {writer.video_path}")
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
