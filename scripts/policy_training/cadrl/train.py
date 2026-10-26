import argparse
import os

from isaaclab.app import AppLauncher

from socnav_sim.config import load_task_config
config = load_task_config()

parser = argparse.ArgumentParser(description="Train a CADRL policy in an IsaacNav scene.")
parser.add_argument("--num_envs", type=int, default=config["env"]["num_envs"], help="Number of pedestrians.")
parser.add_argument("--num_robots", type=int, default=config["env"]["num_robots"], help="Number of active robots.")
parser.add_argument("--task", type=str, default=config["env"]["task"], help="Registered task name.")
parser.add_argument("--method", choices=("Cadrl", "Sarl"), default=config["env"]["method"])
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=config["training"]["checkpoint"], help="Optional checkpoint to resume.")
parser.add_argument("--video", action="store_true", help="Record training video in headless mode.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded video length in environment steps.")
parser.add_argument(
    "--realistic_pedestrians",
    action="store_true",
    help="Render clothed TailorNet pedestrian skins (disabled by default).",
)
parser.add_argument(
    "--video_interval", type=int, default=0,
    help="Start another recording every N environment steps; zero records only at startup.",
)
parser.add_argument(
    "--follow_pedestrian_camera",
    action="store_true",
    help="Follow pedestrian zero with a close camera (useful for visual-asset previews).",
)
parser.add_argument(
    "--camera_offset",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(3.2, -4.0, 1.8),
    help="World-space camera offset used with --follow_pedestrian_camera.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ["SOCNAV_DEVICE"] = args_cli.device
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import socnav_sim.tasks  # noqa: F401
from socnav_sim.tasks.socnav.cadrl.policy import CadrlEncoder, SarlEncoder
from socnav_sim.tasks.socnav.nav_utils.policy_base import MRPPO


def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not 0 < args_cli.num_robots <= args_cli.num_envs:
        raise ValueError("--num_robots must be positive and no larger than --num_envs.")
    env_cfg.actions.navcmd.active_robot_count = args_cli.num_robots
    env_cfg.events.reset_navigation.params["active_robot_count"] = args_cli.num_robots
    env_cfg.actions.navcmd.follow_pedestrian_camera = args_cli.follow_pedestrian_camera
    env_cfg.actions.navcmd.pedestrian_camera_offset = tuple(args_cli.camera_offset)
    env_cfg.actions.navcmd.realistic_pedestrian_visuals = args_cli.realistic_pedestrians
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    agent_cfg["device"] = args_cli.device

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * args_cli.num_robots

    run_name = f"{config['env']['log_name']}-{args_cli.num_robots}r-{args_cli.num_envs}p"
    log_dir = os.path.join("logs", "sb3", args_cli.task, run_name)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    agent_cfg = process_sb3_cfg(agent_cfg, args_cli.num_robots)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        if args_cli.video_length <= 0 or args_cli.video_interval < 0:
            raise ValueError("--video_length must be positive and --video_interval must be non-negative.")
        trigger = (
            (lambda step: step == 0)
            if args_cli.video_interval == 0
            else (lambda step: step % args_cli.video_interval == 0)
        )
        video_folder = os.path.join(log_dir, "videos", "train")
        print(
            f"[INFO] Recording {args_cli.video_length} training steps to {video_folder}; "
            + ("startup only" if args_cli.video_interval == 0 else f"interval={args_cli.video_interval}")
        )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=trigger,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = Sb3VecEnvWrapper(env, fast_variant=False)
    env.seed(seed=agent_cfg["seed"])

    env.action_space = gym.spaces.Box(
        low=-1,
        high=1,
        shape=env.action_space.shape,
    )

    encoder = CadrlEncoder if args_cli.method == "Cadrl" else SarlEncoder
    policy_kwargs = dict(
        features_extractor_class=encoder,
        net_arch=dict(pi=[256, 256], vf=[128, 128]),
    )

    if args_cli.checkpoint:
        agent = MRPPO.load(
            args_cli.checkpoint,
            env=env,
            active_envs=args_cli.num_robots,
            device=agent_cfg["device"],
            print_system_info=True,
        )
    else:
        agent = MRPPO(
            policy_arch,
            env,
            verbose=2,
            active_envs=args_cli.num_robots,
            **agent_cfg,
            policy_kwargs=policy_kwargs,
        )
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)
    print(f"[INFO] Navigation policy ready: {args_cli.num_robots} robots, {args_cli.num_envs} pedestrians")

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    agent.save(os.path.join(log_dir, "model"))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
