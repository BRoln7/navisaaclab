"""CPU diagnostics and bounded reference-motion review plans; no simulator imports."""

from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from .bootstrap import enable_mimickit, PROJECT_ROOT
from .data import sha256
from .sampling import classify_clip

SPECIAL_WALKS = ("catwalk", "one leg", "handrail", "beam", "thinking", "look around", "bag",
                 "backward", "crouch", "limp", "march", "tiptoe", "upstairs", "downstairs")


def rotation_steps(rotvec):
    quats = Rotation.from_rotvec(np.asarray(rotvec).reshape(-1, 3)).as_quat().reshape(*rotvec.shape[:-1], 4)
    cosine = np.abs(np.sum(quats[1:] * quats[:-1], axis=-1))
    return 2 * np.arccos(np.clip(cosine, 0., 1.))


def foot_geometry(skeleton):
    feet = []
    bodies = list(ET.parse(skeleton).getroot().find("worldbody/body").iter("body"))
    for index, body in enumerate(bodies):
        if body.attrib["name"] not in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe"):
            continue
        geom = body.find("geom")
        if geom.attrib["type"] != "box":
            raise ValueError("Expected official SMPL box feet")
        size = np.fromstring(geom.attrib["size"], sep=" ")
        center = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
        quaternion = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" ")[[1, 2, 3, 0]]
        geometry_rot = Rotation.from_quat(quaternion)
        corners = np.array(np.meshgrid(*[[-1, 1]]*3)).reshape(3, -1).T * size
        feet.append((index, geometry_rot.apply(corners)+center,
                     geometry_rot.apply([0., 0., -size[2]])+center))
    if len(feet) != 4:
        raise ValueError("Expected four ankle/toe geometries")
    return feet


def clip_metrics(frames, fps, model, feet):
    from util import torch_util

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 75 or len(frames) < 3 or not np.isfinite(frames).all():
        raise ValueError("Expected finite SMPL frames with shape (T>=3, 75)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    velocity = np.gradient(frames[:, :3].astype(np.float64), axis=0)*fps
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    acceleration = np.linalg.norm(np.diff(velocity, axis=0)*fps, axis=1)
    joint_steps = rotation_steps(frames[:, 6:].reshape(len(frames), 23, 3))
    root_steps = rotation_steps(frames[:, 3:6])
    with torch.no_grad():
        position, rotation = model.forward_kinematics(torch.from_numpy(frames[:, :3]),
            torch_util.exp_map_to_quat(torch.from_numpy(frames[:, 3:6])),
            model.dof_to_rot(torch.from_numpy(frames[:, 6:])))
    positions, rotations = position.numpy(), rotation.numpy()
    heights, slip, slip_times = [], [], []
    for body, corners, marker in feet:
        matrix = Rotation.from_quat(rotations[:, body]).as_matrix()
        world = positions[:, body, None] + np.einsum("tij,kj->tki", matrix, corners)
        height = world[..., 2].min(axis=1)
        heights.append(height)
        point = positions[:, body] + np.einsum("tij,j->ti", matrix, marker)
        horizontal = np.linalg.norm(np.diff(point[:, :2], axis=0)*fps, axis=1)
        vertical = np.abs(np.diff(height)*fps)
        # Low foot height is a proxy, not observed physical contact. Foot rolling
        # and short clips can produce high values without a conversion defect.
        low = (np.maximum(height[1:], height[:-1]) < .06) & (vertical < .2) & (speed[:-1] > .3)
        slip.extend(horizontal[low].tolist())
        slip_times.extend((np.flatnonzero(low)/fps).tolist())
    support_height = np.min(heights, axis=0)
    torso = positions[:, model.get_body_names().index("Head")] - positions[:, 0]
    lean = np.degrees(np.arccos(np.clip(torso[:, 2]/np.maximum(np.linalg.norm(torso, axis=1), 1e-8), -1., 1.)))
    forward = Rotation.from_rotvec(frames[:, 3:6]).apply([1., 0., 0.])[:, :2]
    moving = speed > .3
    alignment = np.sum(forward*velocity[:, :2], axis=1) / np.maximum(np.linalg.norm(forward, axis=1)*speed, 1e-8)
    jump_time = (int(np.argmax(joint_steps.max(axis=1)))+.5)/fps
    result = dict(frames=len(frames), fps=float(fps), duration_s=(len(frames)-1)/fps,
                  mean_speed_m_s=float(speed.mean()), p90_speed_m_s=float(np.quantile(speed, .9)),
                  root_accel_p99_m_s2=float(np.quantile(acceleration, .99)),
                  root_accel_peak_time_s=float((np.argmax(acceleration)+.5)/fps),
                  joint_max_step_rad=float(joint_steps.max()), joint_peak_time_s=jump_time,
                  root_max_step_rad=float(root_steps.max()),
                  root_rotation_peak_time_s=float((np.argmax(root_steps)+.5)/fps),
                  foot_min_height_m=float(np.min(heights)), support_height_p95_m=float(np.quantile(support_height, .95)),
                  support_peak_time_s=float(np.argmax(support_height)/fps),
                  low_foot_slip_p90_m_s=float(np.quantile(slip, .9)) if len(slip) >= 10 else None,
                  slip_peak_time_s=float(slip_times[int(np.argmax(slip))]) if slip else None,
                  low_foot_samples=len(slip), torso_lean_p95_deg=float(np.quantile(lean, .95)),
                  torso_peak_time_s=float(np.argmax(lean)/fps),
                  forward_alignment=float(np.median(alignment[moving])) if moving.any() else None)
    result["group"] = ""  # Filled using the actual clip name outside this function.
    return result


def entry(row, reason, max_seconds, event=None):
    length = row["duration_s"]
    if row["group"] == "walk_run_transition":
        start, end = 0., length  # Preserve every frame of the complete transition.
    else:
        duration = min(length, max_seconds)
        center = length/2 if event is None else float(event)
        start = max(0., min(length-duration, center-duration/2))
        end = start+duration
    return {key: row[key] for key in ("index", "file", "name", "group", "sha256")} | {
        "reason": reason, "start_s": start, "end_s": end, "duration_s": length}


def review_candidates(rows, max_seconds=8.):
    """Cover ordinary gaits and diverse metric extremes with at most twenty clips."""
    chosen, seen = [], set()

    def add(candidates, reason, event_key=None):
        for row in candidates:
            if row["index"] not in seen:
                chosen.append(entry(row, reason, max_seconds, row.get(event_key) if event_key else None))
                seen.add(row["index"])
                return

    walking = [r for r in rows if r["group"] == "walking_label" and not r["special_walk"]]
    for low, high, label in ((.3, .8, "slow walk"), (.8, 1.3, "normal walk"), (1.3, 2., "fast walk")):
        pool = sorted([r for r in walking if low <= r["mean_speed_m_s"] < high],
                      key=lambda r: abs(r["mean_speed_m_s"]-(low+high)/2))
        add(pool, label)
        add(pool, label + " second source")
    running = sorted([r for r in rows if r["group"] == "running_candidate"], key=lambda r: r["mean_speed_m_s"])
    if running:
        for quantile in (0., .33, .67, 1.):
            point = round((len(running)-1)*quantile)
            add(sorted(running, key=lambda r: abs(r["mean_speed_m_s"]-running[point]["mean_speed_m_s"])), "running coverage")
    jogging = sorted([r for r in rows if r["group"] == "jogging_label"], key=lambda r: r["mean_speed_m_s"])
    add(jogging, "jogging")
    add(reversed(jogging), "jogging second speed")
    for row in [r for r in rows if r["group"] == "walk_run_transition"]:
        add([row], "complete walk/run transition")
    for metric, pool, event in (("joint_max_step_rad", rows, "joint_peak_time_s"),
                                ("low_foot_slip_p90_m_s", walking, "slip_peak_time_s"),
                                ("support_height_p95_m", walking, "support_peak_time_s"),
                                ("torso_lean_p95_deg", walking, "torso_peak_time_s"),
                                ("root_accel_p99_m_s2", rows, "root_accel_peak_time_s"),
                                ("root_max_step_rad", rows, "root_rotation_peak_time_s")):
        add(sorted([r for r in pool if r[metric] is not None], key=lambda r: r[metric], reverse=True),
            "inspect extreme: " + metric, event)
    return chosen


def audit(motion_file, output, max_seconds=8.):
    enable_mimickit()
    from anim.mjcf_char_model import MJCFCharModel

    motion_file, output = Path(motion_file).resolve(), Path(output).resolve()
    manifest_path = motion_file.parent / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    skeleton = Path(metadata["skeleton"])
    if sha256(skeleton) != metadata["skeleton_sha256"]:
        raise ValueError("Motion skeleton differs from its source manifest")
    entries = {clip["file"]: clip for clip in metadata["clips"]}
    model = MJCFCharModel("cpu")
    model.load(str(skeleton))
    feet = foot_geometry(skeleton)
    rows = []
    for index, item in enumerate(yaml.safe_load(motion_file.read_text())["motions"]):
        path = Path(item["file"])
        path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
        meta = entries[path.name]
        if path.parent != motion_file.parent or sha256(path) != meta["sha256"]:
            raise ValueError(f"Motion differs from manifest: {path}")
        with path.open("rb") as stream:
            clip = pickle.load(stream)  # Trusted output of the local converter.
        row = clip_metrics(clip["frames"], float(clip["fps"]), model, feet)
        row.update(index=index, file=str(path), name=meta["name"], sha256=meta["sha256"])
        row["group"] = classify_clip(row["name"], row["mean_speed_m_s"])
        row["special_walk"] = any(fragment in row["name"].lower() for fragment in SPECIAL_WALKS)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"[Motion review] CPU audit {len(rows)}/{len(entries)}", flush=True)
    if not rows:
        raise ValueError("Motion list is empty")
    representative = review_candidates(rows, max_seconds)
    complete = [entry(row, "sequential full-list review", max_seconds) for row in rows]
    plan = dict(schema=1, motion_file=str(motion_file), motion_list_sha256=sha256(motion_file),
                motion_manifest_sha256=sha256(manifest_path), skeleton_sha256=metadata["skeleton_sha256"],
                total_clips=len(rows), max_excerpt_seconds=max_seconds,
                audit_code_sha256=sha256(Path(__file__)),
                representative=representative, all=complete,
                note="Kinematic risk indicators only; not contact-force measurements or an automatic naturalness verdict.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "review_plan.json").exists():
        existing = json.loads((output / "review_plan.json").read_text())
        if existing != plan:
            raise FileExistsError("Existing review plan differs; select a new --output")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "review_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    report = dict(total_clips=len(rows), groups=dict(Counter(row["group"] for row in rows)),
                  representative_clips=len(representative), representative_batches=(len(representative)+3)//4,
                  all_batches=(len(rows)+3)//4,
                  metrics=str(output / "metrics.csv"), review_plan=str(output / "review_plan.json"),
                  cuda_initialized=torch.cuda.is_initialized())
    if report["cuda_initialized"]:
        raise RuntimeError("CPU motion audit initialized CUDA")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
