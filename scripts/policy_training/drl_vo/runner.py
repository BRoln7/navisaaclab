"""Shared DRL-VO training and playback entry point."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path


def main(mode):
    from socnav_sim.config import PROJECT_ROOT, load_task_config

    os.environ.setdefault("SOCNAV_CONFIG", str(PROJECT_ROOT / "data/cfg/drlvo_scene12.yaml"))
    config = load_task_config()
    parser = argparse.ArgumentParser(description=f"{mode.capitalize()} DRL-VO with Jackal and AMP pedestrians.")
    parser.add_argument("--task", default=config["env"]["task"])
    parser.add_argument("--num_envs", type=int, default=config["env"]["num_envs"])
    parser.add_argument("--num_robots", type=int, default=config["env"]["num_robots"])
    parser.add_argument("--checkpoint", default=config["training"].get("checkpoint"))
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--realistic_pedestrians", action="store_true")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 < args.num_robots <= args.num_envs:
        parser.error("Require 0 < num_robots <= num_envs")
    if args.steps <= 0 or args.video_length <= 0:
        parser.error("steps and video_length must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("max_iterations must be positive")
    if args.task not in ("Isaac-Drlvo", "Isaac-Drlvo-Scene12", "Isaac-Drlvo-Scene35"):
        parser.error("Choose a DRL-VO task: Isaac-Drlvo-Scene12 or Isaac-Drlvo-Scene35")
    if mode == "play" and not args.checkpoint:
        parser.error("Playback requires --checkpoint /path/to/model.zip")
    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
    os.environ["SOCNAV_DEVICE"] = args.device
    os.environ["SOCNAV_NUM_ENVS"] = str(args.num_envs)
    os.environ["SOCNAV_TASK"] = args.task
    args.enable_cameras = True  # RTX planar lidar requires rendering even without video.
    app = AppLauncher(args).app
    env = None
    raw = None
    try:
        import gymnasium as gym
        import torch
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.logger import configure
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
        import socnav_sim.tasks  # noqa: F401
        from socnav_sim.tasks.socnav.nav_utils.policy_base import MRPPO
        from custom_cnn_full import CustomCNN

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.actions.navcmd.active_robot_count = args.num_robots
        cfg.events.reset_navigation.params["active_robot_count"] = args.num_robots
        cfg.scene.lidar.active_robot_count = args.num_robots
        cfg.actions.navcmd.realistic_pedestrian_visuals = args.realistic_pedestrians
        agent_cfg = load_cfg_from_registry(args.task, "sb3_cfg_entry_point")
        agent_cfg.update(seed=args.seed, device=args.device)
        if args.max_iterations is not None:
            agent_cfg["n_timesteps"] = args.max_iterations * agent_cfg["n_steps"] * args.num_robots
        out = PROJECT_ROOT / "logs/sb3" / args.task / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dump_yaml(str(out / "params/env.yaml"), cfg)
        dump_yaml(str(out / "params/agent.yaml"), agent_cfg)
        agent_cfg = process_sb3_cfg(agent_cfg, args.num_robots)
        policy = agent_cfg.pop("policy")
        timesteps = agent_cfg.pop("n_timesteps")
        raw = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
        env = raw
        if args.video:
            video_length = args.video_length if mode == "train" else min(args.steps, args.video_length)
            print(f"[DRL-VO] Video directory: {out / 'videos'}", flush=True)
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(out / "videos"), step_trigger=lambda step: step == 0,
                video_length=video_length, disable_logger=True,
            )
        env = Sb3VecEnvWrapper(env, fast_variant=False)
        env.seed(args.seed)
        if checkpoint:
            agent = MRPPO.load(str(checkpoint), env=env, active_envs=args.num_robots, device=args.device)
        else:
            agent = MRPPO(
                policy, env, active_envs=args.num_robots, verbose=1, **agent_cfg,
                policy_kwargs=dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=dict(pi=[256], vf=[128]), normalize_images=False,
                ),
            )
        print(f"[DRL-VO] {args.task}: {args.num_robots} Jackals, {args.num_envs} AMP pedestrians", flush=True)
        if mode == "train":
            agent.set_logger(configure(str(out), ["stdout", "tensorboard"]))
            agent.learn(
                total_timesteps=timesteps,
                callback=CheckpointCallback(save_freq=1000, save_path=str(out), name_prefix="model"),
            )
            agent.save(str(out / "model"))
        else:
            observation = env.reset()
            for _ in range(args.steps):
                if not app.is_running():
                    break
                with torch.inference_mode():
                    action, _ = agent.predict(observation, deterministic=True)
                    observation, _, _, _ = env.step(action)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if raw is not None:
                    raw.unwrapped.scene.sensors["lidar"].close()
            finally:
                app.close()
