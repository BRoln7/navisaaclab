"""Inspect native reference motions using official MimicKit ViewMotionEnv, without a policy."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "logs/motion-review/amass-pacer-v1/review_plan.json")
    parser.add_argument("--suite", choices=("representative", "all"), default="representative")
    parser.add_argument("--batch", type=int, help="Replay only one zero-based group of four clips")
    parser.add_argument("--full_clips", action="store_true", help="Play complete clips instead of planned excerpts")
    parser.add_argument("--video", action="store_true", help="Record a four-view WebM video")
    parser.add_argument("--out_dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    args = parser.parse_args()
    if args.headless and not args.video:
        parser.error("Use --video for headless reference inspection")
    from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
    from socnav_sim.mimickit_amp.config import load, resolve_paths
    from socnav_sim.mimickit_amp.data import sha256
    enable_mimickit()
    plan = json.loads(args.plan.read_text())
    motion_file = Path(plan["motion_file"])
    if (sha256(motion_file) != plan["motion_list_sha256"] or
            sha256(motion_file.parent / "manifest.json") != plan["motion_manifest_sha256"]):
        parser.error("Reference dataset changed since the CPU audit; regenerate the review plan")
    entries = plan[args.suite]
    batches = [entries[index:index+4] for index in range(0, len(entries), 4)]
    selected_batches = list(enumerate(batches))
    if args.batch is not None:
        if not 0 <= args.batch < len(batches):
            parser.error(f"--batch must be between 0 and {len(batches)-1}")
        selected_batches = [(args.batch, batches[args.batch])]
    if not selected_batches:
        parser.error("Review suite is empty")
    env_cfg = load(ROOT / "third_party/MimicKit/data/envs/view_motion_smpl_env.yaml")
    env_cfg.update(char_file=str(ROOT / "third_party/MimicKit/data/assets/smpl/smpl.xml"),
                   motion_file=str(motion_file), num_disc_obs_steps=10, init_pose=[0., 0., .9])
    engine_cfg = load(ROOT / "data/cfg/mimickit_amp/engine.yaml")
    engine_cfg["control_mode"] = "none"
    # Keep unrelated reference clips out of each other's camera views.
    engine_cfg["env_spacing"] = 1000
    env_cfg, engine_cfg = resolve_paths(env_cfg, engine_cfg)
    if sha256(Path(env_cfg["char_file"])) != plan["skeleton_sha256"]:
        parser.error("Review plan uses a different skeleton")
    out = (args.out_dir or ROOT / "logs/motion-review" / ("video-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    args.enable_cameras = args.video
    launcher = AppLauncher(args)
    app, video = launcher.app, None
    try:
        import numpy as np
        import torch
        from util import mp_util
        from socnav_sim.mimickit_amp.reference_env import SmplReferenceEnv
        from socnav_sim.mimickit_amp.video import FourEnvVideo
        if args.device.startswith("cuda"):
            torch.cuda.set_device(args.device)
        mp_util.init(0, 1, args.device, 29500)
        env = SmplReferenceEnv(env_cfg, engine_cfg, 4, args.device, visualize=not args.headless)
        env.reset()
        records, frames, max_error, max_angle = [], 0, 0., 0.
        fps = round(1/env._engine.get_timestep())
        # Position the first sampled frame before camera warm-up.
        first = selected_batches[0][1]
        padded = first + [first[-1]]*(4-len(first))
        env.show_frame([entry["index"] for entry in padded], [0. if args.full_clips else entry["start_s"] for entry in padded])
        if args.video:
            video = FourEnvVideo(env, out / "reference-motions.webm")
        with torch.no_grad():
            for batch_id, batch in selected_batches:
                if not app.is_running():
                    break
                padded = batch + [batch[-1]]*(4-len(batch))
                for item in batch:
                    if Path(env._motion_lib.get_motion_file(item["index"])).resolve() != Path(item["file"]).resolve():
                        raise RuntimeError("Motion library ordering differs from the review plan")
                    if sha256(Path(item["file"])) != item["sha256"]:
                        raise RuntimeError("Clip differs from the CPU review source")
                starts = np.array([0. if args.full_clips else item["start_s"] for item in padded])
                ends = np.array([item["duration_s"] if args.full_clips else item["end_s"] for item in padded])
                count = int(np.ceil((ends-starts).max()*fps))+1
                record = dict(batch=batch_id, video_start_frame=frames, planned_frames=count,
                              clips=batch, control_dt=1/fps)
                print(f"[Reference] Batch {batch_id+1}/{len(batches)}: " + " | ".join(Path(item["file"]).stem for item in batch), flush=True)
                for item in batch:
                    print(f"  {item['name']} -- {item['reason']}", flush=True)
                completed = 0
                for step in range(count):
                    if not app.is_running():
                        break
                    elapsed = step/fps
                    times = np.minimum(starts+elapsed, ends)
                    error, angle = env.show_frame([item["index"] for item in padded], times)
                    max_error, max_angle = max(max_error, error), max(max_angle, angle)
                    labels = []
                    for tile, (item, time) in enumerate(zip(padded, times)):
                        if tile >= len(batch):
                            labels.append("Unused tile")
                            continue
                        status = " | END (held)" if starts[tile]+elapsed >= ends[tile]-1e-6 else ""
                        labels.append(f"B{batch_id} {Path(item['file']).stem} | {item['group']} | t={time:.2f}s{status}")
                    if video is not None:
                        video.capture(labels)
                    else:
                        env._render()
                    frames += 1
                    completed += 1
                record["rendered_frames"] = completed
                records.append(record)
                (out / "batches.json").write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n")
        if video is not None:
            video.writer.close()
        report = dict(mode="kinematic reference, no policy or physics time stepping", suite=args.suite,
                      full_clips=args.full_clips, batches=len(records), frames=frames, fps=fps,
                      video=str(out / "reference-motions.webm") if args.video else None,
                      max_usd_fk_position_error_m=max_error, max_usd_fk_rotation_error_rad=max_angle)
        (out / "playback.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"[Reference] Complete: {out}", flush=True)
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            app.close(wait_for_replicator=not args.video)


if __name__ == "__main__":
    main()
