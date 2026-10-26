"""Run the pinned MimicKit AMP learner with the existing SMPL rig and trajectory task."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import traceback

# Let PyTorch grow CUDA segments instead of reserving several incompatible
# blocks beside PhysX. This changes allocation only, never batches or losses.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
from socnav_sim.mimickit_amp.config import (
    latest_checkpoint, load, resolve_agent_runtime, resolve_paths, run_configurations,
)


def saved_params(checkpoint):
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        folder = parent / "params"
        if (folder / "manifest.json").is_file():
            return folder
        if all((parent / f"{name}_config.yaml").is_file() for name in ("env", "agent", "engine")):
            return parent
    raise FileNotFoundError("Checkpoint needs its original params/ directory from this MimicKit feature")


def saved_env_config(folder):
    for name in ("env.yaml", "env_config.yaml"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No saved environment config in {folder}")


def constrain_waist_sway(action, indices, limit):
    """Limit cumulative Torso/Spine/Chest lateral targets during playback."""
    original = action.index_select(-1, indices).sum(dim=-1)
    if limit is None:
        return action, original, original
    constrained = action.clone()
    lateral = constrained.index_select(-1, indices).clamp(min=-limit, max=limit)
    cumulative = lateral.sum(dim=-1)
    scale = (limit / cumulative.abs().clamp_min(1e-8)).clamp(max=1.0)
    lateral = lateral * scale.unsqueeze(-1)
    constrained[..., indices] = lateral
    return constrained, original, lateral.sum(dim=-1)


def constrain_joint_target(action, index, limit):
    """Clamp one policy position target without mutating the actor output."""
    original = action[..., index]
    if limit is None:
        return action, original, original
    constrained = action.clone()
    constrained[..., index] = original.clamp(min=-limit, max=limit)
    return constrained, original, constrained[..., index]


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "check", "play"), default="train")
    parser.add_argument("--env_config", type=Path, default=ROOT / "data/cfg/mimickit_amp/env.yaml")
    parser.add_argument("--agent_config", type=Path, default=ROOT / "data/cfg/mimickit_amp/agent.yaml")
    parser.add_argument("--engine_config", type=Path, default=ROOT / "data/cfg/mimickit_amp/engine.yaml")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--checkpoint", help="MimicKit .pt weights, or 'latest'; training uses a fresh optimizer")
    parser.add_argument("--motion_file", type=Path, help="Explicit motion-list override, including when loading checkpoint configs")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--video", action="store_true", help="Record four-environment playback as a streamed WebM video")
    parser.add_argument("--amp_only", action="store_true", help="AMP locomotion with symmetry and no trajectory task")
    parser.add_argument(
        "--power_coefficient", type=float,
        help="Override sum(abs(torque * velocity)) coefficient; with --amp_only this enables an isolated energy regularizer",
    )
    parser.add_argument("--finetune_tracking", action="store_true", help="Train from a pure AMP checkpoint using the current tracking configs")
    parser.add_argument("--reset_motion", choices=("training", "standing"), default="training")
    parser.add_argument(
        "--waist_sway_limit", type=float,
        help="Playback-only cumulative Torso/Spine/Chest lateral target limit in radians",
    )
    parser.add_argument(
        "--torso_twist_limit", type=float,
        help="Playback-only absolute Torso_z position-target limit in radians",
    )
    parser.add_argument("--out_dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    args = parser.parse_args()
    lock = enable_mimickit()
    if args.max_iterations <= 0 or args.steps <= 0 or (args.num_envs is not None and args.num_envs <= 0):
        parser.error("Counts must be positive")
    if args.power_coefficient is not None and (
        not math.isfinite(args.power_coefficient) or args.power_coefficient < 0
    ):
        parser.error("power_coefficient must be finite and non-negative")
    if args.waist_sway_limit is not None and (
        not math.isfinite(args.waist_sway_limit) or args.waist_sway_limit <= 0
    ):
        parser.error("waist_sway_limit must be finite and positive")
    if args.waist_sway_limit is not None and args.mode != "play":
        parser.error("waist_sway_limit is a playback-only diagnostic")
    if args.torso_twist_limit is not None and (
        not math.isfinite(args.torso_twist_limit) or args.torso_twist_limit <= 0
    ):
        parser.error("torso_twist_limit must be finite and positive")
    if args.torso_twist_limit is not None and args.mode != "play":
        parser.error("torso_twist_limit is a playback-only diagnostic")
    if args.finetune_tracking and (args.mode != "train" or not args.checkpoint or args.amp_only):
        parser.error("finetune_tracking requires --mode train --checkpoint and cannot be combined with amp_only")
    if args.motion_file is not None and args.reset_motion != "training":
        parser.error("--motion_file cannot be combined with --reset_motion standing")
    checkpoint_params = None
    if args.checkpoint:
        if args.checkpoint == "latest":
            args.checkpoint = latest_checkpoint(ROOT / "logs/mimickit/smpl-amp")
            print(f"[MimicKit] Resolved latest checkpoint: {args.checkpoint}", flush=True)
        else:
            args.checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not args.checkpoint.is_file() or args.checkpoint.suffix != ".pt":
            parser.error(f"Expected a MimicKit .pt checkpoint: {args.checkpoint}")
        checkpoint_params = saved_params(args.checkpoint)
    env_cfg, agent_cfg, engine_cfg = run_configurations(
        (args.env_config, args.agent_config, args.engine_config), checkpoint_params,
        amp_only=args.amp_only, finetune_tracking=args.finetune_tracking,
        motion_file=args.motion_file, power_coefficient=args.power_coefficient)
    if args.mode == "play" and not args.checkpoint:
        parser.error("play requires --checkpoint")
    if args.mode == "train" and args.reset_motion != "training":
        parser.error("Standing reset is a playback/check option; training must retain the demonstration set")
    if args.reset_motion == "standing":
        env_cfg["motion_file"] = str(ROOT / "data/mimickit/official_smpl/standing/motions.yaml")
    env_cfg, engine_cfg = resolve_paths(env_cfg, engine_cfg)
    if args.finetune_tracking:
        import hashlib
        previous = load(saved_env_config(checkpoint_params))
        if previous.get("tracking", {}).get("enabled", False):
            parser.error("finetune_tracking expects a checkpoint trained without trajectory observations")
        manifest_path = checkpoint_params / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            for key, path in (("usd_sha256", engine_cfg["smpl_usd"]), ("skeleton_sha256", env_cfg["char_file"])):
                with Path(path).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[key]:
                        parser.error("Tracking transfer must keep the checkpoint's SMPL model")
    args.num_envs = args.num_envs or (4096 if args.mode == "train" else 4)
    if args.video:
        if args.mode != "play" or args.num_envs != 4:
            parser.error("--video requires --mode play and --num_envs 4")
        args.enable_cameras = True
    agent_cfg = resolve_agent_runtime(agent_cfg, args.num_envs)
    batches = {name: math.ceil(agent_cfg[f"{name}_batch_size"] * args.num_envs)
               for name in ("actor", "critic", "disc")}
    updates = {name: math.ceil(args.num_envs * agent_cfg["steps_per_iter"] / size) * agent_cfg[f"{name}_epochs"]
               for name, size in batches.items()}
    print(f"[MimicKit] Environments: {args.num_envs}; rollout: {args.num_envs * agent_cfg['steps_per_iter']}; "
          f"replay capacity: {agent_cfg['disc_buffer_size']}; logical batches: {batches}; "
          f"microbatch limits: {agent_cfg.get('optimizer_microbatch_size', {})}; optimizer updates: {updates}", flush=True)
    print(f"[MimicKit] Motion list: {env_cfg['motion_file']}; episode: {env_cfg['episode_length']}s; "
          f"global_obs={env_cfg['global_obs']}; tracking={env_cfg['tracking']['enabled']}; "
          f"task/style weights={agent_cfg['task_reward_weight']}/{agent_cfg['disc_reward_weight']}; "
          f"symmetry={agent_cfg['symmetry_weight']} x scale {agent_cfg.get('symmetry_scale', 1.)}", flush=True)
    if env_cfg["tracking"]["enabled"]:
        task = env_cfg["tracking"]
        segment_dt = env_cfg["episode_length"] / (task["num_vertices"] - 1)
        print(
            f"[MimicKit] PACER trajectory: {task['num_vertices']} vertices, dt={segment_dt:g}s, "
            f"future={task['observation_samples']}x{task['observation_timestep']:g}s, "
            f"speed=[{task['speed_min']:g},{task['speed_max']:g}]m/s, "
            f"accel={task['acceleration_max']:g}m/s^2, turn={task['heading_change_max']:g}rad/s, "
            f"sharp_turn={task['sharp_turn_probability']:g}", flush=True)
        print(
            f"[MimicKit] Reward: {agent_cfg['task_reward_weight']:g} * "
            f"(exp(-{task['position_scale']:g} * position_error^2) - "
            f"{task['power_coefficient']:g} * sum(abs(torque * velocity))) + "
            f"{agent_cfg['disc_reward_weight']:g} * AMP; fail_distance={task['max_distance']:g}m",
            flush=True)
    elif float(env_cfg["tracking"]["power_coefficient"]) > 0:
        print(
            f"[MimicKit] Reward: {agent_cfg['disc_reward_weight']:g} * AMP - "
            f"{agent_cfg['task_reward_weight']:g} * {env_cfg['tracking']['power_coefficient']:g} * "
            "sum(abs(torque * velocity)); trajectory task disabled",
            flush=True,
        )
    out = args.out_dir or ROOT / "logs/mimickit/smpl-amp" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # All Isaac Sim-dependent imports follow AppLauncher, including upstream carb.
    launcher = AppLauncher(args)
    app = launcher.app
    agent = None
    video = None
    try:
        import torch
        import yaml
        from util import mp_util, torch_util
        from socnav_sim.mimickit_amp.agent import SymmetricAMPAgent
        from socnav_sim.mimickit_amp.env import SmplTrackingEnv
        from socnav_sim.mimickit_amp.data import SMPL_DFS, sha256

        if args.device.startswith("cuda"):
            torch.cuda.set_device(args.device)
            print(f"[MimicKit] Policy/physics device: {args.device} = {torch.cuda.get_device_name(args.device)}; "
                  f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', 'FASTEST_FIRST (CUDA default)')}; "
                  f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}", flush=True)
        mp_util.init(0, 1, args.device, 29500)
        torch.manual_seed(args.seed)
        import numpy as np
        import random
        np.random.seed(args.seed); random.seed(args.seed)
        env = SmplTrackingEnv(env_cfg, engine_cfg, args.num_envs, args.device, visualize=not args.headless)
        env.reset()
        pose_check = env.validate_reset()
        tracking_check = env.validate_tracking()
        params = out / "params"; params.mkdir()
        for name, cfg in (("env", env_cfg), ("agent", agent_cfg), ("engine", engine_cfg)):
            (params / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        sources = list((ROOT / "exts/socnav_sim/socnav_sim/mimickit_amp").glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
        manifest = dict(upstream=lock, seed=args.seed, num_envs=args.num_envs, mode=args.mode,
                        device=args.device,
                        pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                        checkpoint=str(args.checkpoint) if args.checkpoint else None,
                        finetune_tracking=args.finetune_tracking,
                        power_coefficient=float(env_cfg["tracking"]["power_coefficient"]),
                        control_dt=env._engine.get_timestep(), pose_check=pose_check,
                        tracking_check=tracking_check,
                        usd_sha256=sha256(engine_cfg["smpl_usd"]), skeleton_sha256=sha256(env_cfg["char_file"]),
                        motion_list_sha256=sha256(env_cfg["motion_file"]),
                        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sources})
        (params / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[MimicKit] Reset FK check: {pose_check}\n"
              f"[MimicKit] Trajectory check: {tracking_check}\n[MimicKit] Output: {out}", flush=True)
        if args.mode == "check":
            # Hold the exact sampled joint pose; no policy, no optimization.
            hold = env._engine.get_dof_pos(env._get_char_id()).clone()
            for _ in range(min(args.steps, 120)):
                if not app.is_running():
                    break
                env.step(hold)
            report = env.record_diagnostics()
            report["root_height_min_m"] = env._engine.get_root_pos(env._get_char_id())[:, 2].min().item()
            (out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"[MimicKit] Hold-pose diagnostics: {report}", flush=True)
            return
        agent = SymmetricAMPAgent(agent_cfg, env, args.device)
        if args.checkpoint:
            if args.finetune_tracking:
                agent.load_locomotion_for_tracking(str(args.checkpoint))
                print("[MimicKit] Tracking transfer: preserved locomotion weights; initialized 20 goal inputs. Fresh optimizers/replay.", flush=True)
            else:
                agent.load(str(args.checkpoint))
        if args.mode == "train":
            max_samples = args.max_iterations*agent_cfg["steps_per_iter"]*args.num_envs
            print(f"[MimicKit] Native AMP training: {max_samples} samples, TensorBoard: {out}", flush=True)
            agent.train_model(max_samples=max_samples, out_dir=str(out), save_int_models=True, logger_type="tb")
            agent.save(str(out / "model_final.pt"))
        else:
            from learning.base_agent import AgentMode
            agent.eval(); agent.set_mode(AgentMode.TEST)
            obs, info = env.reset()
            errors, heights = [], []
            resets = np.zeros(args.num_envs, dtype=int)
            completed_steps = 0
            char_id = env._get_char_id()
            upper_body = {
                "Torso", "Spine", "Chest", "Neck", "Head",
                "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
                "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
            }
            # Common DOFs contain three coordinates for every non-root body in
            # SMPL_DFS order.  This split makes a large whole-body power spike
            # attributable to either locomotion or upper-body wobble.
            upper_joint_ids = torch.tensor(
                [i - 1 for i, name in enumerate(SMPL_DFS) if i and name in upper_body],
                device=args.device,
            )
            waist_sway_ids = torch.tensor(
                [3 * (SMPL_DFS.index(name) - 1) for name in ("Torso", "Spine", "Chest")],
                device=args.device,
            )
            waist_dof_names = [
                f"{body}_{axis}"
                for body in ("Torso", "Spine", "Chest")
                for axis in ("x", "y", "z")
            ]
            waist_dof_ids = torch.tensor(
                [3 * (SMPL_DFS.index(body) - 1) + axis
                 for body in ("Torso", "Spine", "Chest") for axis in range(3)],
                device=args.device,
            )
            torso_twist_id = 3 * (SMPL_DFS.index("Torso") - 1) + 2
            torso_id = SMPL_DFS.index("Torso")
            chest_id = SMPL_DFS.index("Chest")
            if args.waist_sway_limit is not None:
                print(
                    f"[MimicKit] Playback waist-sway constraint: Torso_x/Spine_x/Chest_x "
                    f"cumulative target <= {args.waist_sway_limit:g} rad",
                    flush=True,
                )
            if args.torso_twist_limit is not None:
                print(
                    f"[MimicKit] Playback torso-twist constraint: Torso_z target "
                    f"in [-{args.torso_twist_limit:g}, {args.torso_twist_limit:g}] rad",
                    flush=True,
                )
            if args.video:
                from socnav_sim.mimickit_amp.video import FourEnvVideo
                video = FourEnvVideo(env, out / "four-environments.webm")
                print(f"[MimicKit] Recording video: {video.writer.path}", flush=True)
            with torch.no_grad(), (out / "playback.csv").open("w") as stream:
                stream.write(
                    "step,env,root_x,root_y,target_x,target_y,position_error_m,task_reward,"
                    "raw_power_w,upper_body_power_w,lower_body_power_w,power_cost,root_height_m,"
                    "root_roll_rad,root_pitch_rad,root_yaw_rate_rad_s,torso_ang_speed_rad_s,"
                    "chest_ang_speed_rad_s,waist_sway_raw_rad,waist_sway_cmd_rad,"
                    "torso_twist_raw_rad,torso_twist_cmd_rad,"
                    + ",".join(f"target_{name}_rad" for name in waist_dof_names) + ","
                    + ",".join(f"actual_{name}_rad" for name in waist_dof_names) + ","
                    + ",".join(f"velocity_{name}_rad_s" for name in waist_dof_names) + ","
                    + ",".join(f"power_{name}_w" for name in waist_dof_names) + ",done\n"
                )
                for step in range(args.steps):
                    if not app.is_running():
                        break
                    action, _ = agent._decide_action(obs, info)
                    waist_targets = action.index_select(-1, waist_dof_ids).cpu().tolist()
                    action, waist_sway_raw, waist_sway_cmd = constrain_waist_sway(
                        action, waist_sway_ids, args.waist_sway_limit
                    )
                    action, torso_twist_raw, torso_twist_cmd = constrain_joint_target(
                        action, torso_twist_id, args.torso_twist_limit
                    )
                    obs, _, done, info = env.step(action)
                    error = env._position_error.cpu().tolist()
                    roots_tensor = env._engine.get_root_pos(char_id)
                    roots = roots_tensor.cpu().tolist()
                    root_euler = torch_util.quat_to_euler_xyz(env._engine.get_root_rot(char_id)).cpu().tolist()
                    root_ang_vel = env._engine.get_root_ang_vel(char_id)
                    body_ang_speed = torch.linalg.vector_norm(
                        env._engine.get_body_ang_vel(char_id), dim=-1
                    )
                    joint_power = torch.abs(
                        env._engine.get_dof_forces(char_id) * env._engine.get_dof_vel(char_id)
                    ).reshape(args.num_envs, len(SMPL_DFS) - 1, 3).sum(dim=-1)
                    dof_pos = env._engine.get_dof_pos(char_id).index_select(-1, waist_dof_ids).cpu().tolist()
                    dof_vel = env._engine.get_dof_vel(char_id).index_select(-1, waist_dof_ids).cpu().tolist()
                    dof_power = torch.abs(
                        env._engine.get_dof_forces(char_id) * env._engine.get_dof_vel(char_id)
                    ).index_select(-1, waist_dof_ids).cpu().tolist()
                    upper_power = joint_power.index_select(1, upper_joint_ids).sum(dim=-1)
                    lower_power = joint_power.sum(dim=-1) - upper_power
                    if env._tracking:
                        targets = env._trajectory.position(env._env_ids, env._time_buf).cpu().tolist()
                    else:
                        targets = [[float("nan")] * 3 for _ in range(args.num_envs)]
                    task_rewards = env._task_r.cpu().tolist()
                    powers = env._raw_power.cpu().tolist()
                    coefficient = float(env._task["power_coefficient"])
                    height = [root[2] for root in roots]
                    for i, (root, target, e, task_r, power, upper, lower, h, euler, yaw_rate,
                            torso_speed, chest_speed, sway_raw, sway_cmd, dof_target, dof_actual,
                            twist_raw, twist_cmd, dof_velocity, axis_power, d) in enumerate(zip(
                            roots, targets, error, task_rewards, powers, upper_power.cpu().tolist(),
                            lower_power.cpu().tolist(), height, root_euler,
                            root_ang_vel[:, 2].cpu().tolist(), body_ang_speed[:, torso_id].cpu().tolist(),
                            body_ang_speed[:, chest_id].cpu().tolist(), waist_sway_raw.cpu().tolist(),
                            waist_sway_cmd.cpu().tolist(), waist_targets, dof_pos,
                            torso_twist_raw.cpu().tolist(), torso_twist_cmd.cpu().tolist(), dof_vel, dof_power,
                            done.cpu().tolist())):
                        stream.write(
                            f"{step},{i},{root[0]},{root[1]},{target[0]},{target[1]},"
                            f"{e},{task_r},{power},{upper},{lower},{coefficient * power},{h},"
                            f"{euler[0]},{euler[1]},{yaw_rate},{torso_speed},{chest_speed},"
                            f"{sway_raw},{sway_cmd},{twist_raw},{twist_cmd},"
                            + ",".join(map(str, dof_target)) + ","
                            + ",".join(map(str, dof_actual)) + ","
                            + ",".join(map(str, dof_velocity)) + ","
                            + ",".join(map(str, axis_power)) + f",{d}\n"
                        )
                    errors.extend(error); heights.extend(height)
                    resets += (done.cpu().numpy() != 0)
                    completed_steps = step+1
                    if video is not None:
                        video.capture([f"Env {i} | t={(step+1)*env._engine.get_timestep():.1f}s | resets={resets[i]}"
                                       for i in range(args.num_envs)])
                    if completed_steps % 100 == 0:
                        print(f"[MimicKit] Playback {completed_steps}/{args.steps}; resets={resets.tolist()}", flush=True)
                    ids = torch.nonzero(done, as_tuple=False).flatten()
                    # Use upstream reset API; terminal state was recorded above.
                    obs, info = env.reset(ids)
            if video is not None:
                # Finalize encoding before saving metrics. Render-only teardown
                # happens in finally and cannot lose a completed report.
                video.writer.close()
            report = dict(checkpoint=str(args.checkpoint), control_steps=completed_steps,
                          resets=resets.tolist(), reset_motion=args.reset_motion,
                          video=str(video.writer.path) if video is not None else None,
                          video_frames=video.writer.frames if video is not None else 0,
                          mean_position_error_m=float(np.mean(errors)) if errors else None,
                          max_position_error_m=max(errors) if errors else None,
                          min_root_height_m=min(heights) if heights else None, diagnostics=env.record_diagnostics())
            (out / "playback.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"[MimicKit] Playback: {report}", flush=True)
    except KeyboardInterrupt:
        if agent is not None and args.mode == "train":
            agent.save(str(out / "model_interrupted.pt"))
            print(f"[MimicKit] Saved current network/normalizers to {out / 'model_interrupted.pt'}", flush=True)
    except BaseException:
        # Print and persist BEFORE app.close(), which can itself take time.
        failure = traceback.format_exc()
        (out / "failure.txt").write_text(failure)
        print(failure, file=sys.stderr, flush=True)
        print(f"[MimicKit] Failure saved to {out / 'failure.txt'}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            if video is not None:
                video.close()
            if agent is not None and hasattr(agent, "_logger"):
                writer = getattr(agent._logger, "_writer", None)
                if writer:
                    writer.flush(); writer.close()
        finally:
            # Frames are synchronously captured and the encoder was flushed.
            # No asynchronous Replicator workflow needs draining for playback.
            app.close(wait_for_replicator=not args.video)


if __name__ == "__main__":
    main()
