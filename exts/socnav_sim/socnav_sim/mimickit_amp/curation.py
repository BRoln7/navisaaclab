"""CPU curation of continuous walking and native weighted locomotion datasets."""

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation
import torch
import yaml

from .bootstrap import PROJECT_ROOT
from .data import sha256
from .sampling import classify_clip, speed_statistics, _fractions


def continuous_segments(mask, min_frames):
    """Return contiguous [start, end) spans; never concatenate separated motion."""
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(start), int(end)) for start, end in zip(changes[::2], changes[1::2])
            if end-start >= min_frames]


def _odd_frames(seconds, fps):
    size = max(1, round(seconds*fps))
    return size if size % 2 else size+1


def walking_segments(frames, fps, model, settings):
    """Select forward, upright, steadily moving spans using native SMPL FK.

    Smoothed root motion avoids classifying normal pelvis oscillations as sharp
    turns. This screens kinematics, not physical foot contacts or visual realism.
    """
    frames = np.asarray(frames, dtype=np.float32)
    velocity = np.gradient(frames[:, :2], axis=0)*fps
    window = _odd_frames(settings["smoothing_seconds"], fps)
    smooth = uniform_filter1d(velocity, size=window, axis=0, mode="nearest")
    speed = np.linalg.norm(smooth, axis=1)
    forward = Rotation.from_rotvec(frames[:, 3:6]).apply([1., 0., 0.])[:, :2]
    forward = uniform_filter1d(forward, size=window, axis=0, mode="nearest")
    alignment = np.einsum("ij,ij->i", forward, smooth) / np.maximum(np.linalg.norm(forward, axis=1)*speed, 1e-6)
    from util import torch_util
    with torch.no_grad():
        positions, _ = model.forward_kinematics(
            torch.from_numpy(frames[:, :3]), torch_util.exp_map_to_quat(torch.from_numpy(frames[:, 3:6])),
            model.dof_to_rot(torch.from_numpy(frames[:, 6:])))
    torso = (positions[:, model.get_body_names().index("Head")]-positions[:, 0]).numpy()
    upright = torso[:, 2] / np.maximum(np.linalg.norm(torso, axis=1), 1e-6)
    variation_window = _odd_frames(settings["variation_window_seconds"], fps)
    variance = (uniform_filter1d(speed**2, size=variation_window, mode="nearest")
                - uniform_filter1d(speed, size=variation_window, mode="nearest")**2)
    speed_std = np.sqrt(np.maximum(variance, 0.))
    direction = uniform_filter1d(velocity, size=_odd_frames(settings["turn_window_seconds"], fps),
                                 axis=0, mode="nearest")
    heading = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
    turn_rate = np.abs(np.gradient(heading)*fps)
    intervals = np.linalg.norm(np.diff(frames[:, :2], axis=0)*fps, axis=1)
    instant = np.r_[intervals, intervals[-1]]
    instant[1:-1] = np.maximum(intervals[:-1], intervals[1:])
    valid = ((speed >= settings["speed_min"]) & (speed <= settings["speed_max"])
             & (alignment >= settings["forward_alignment_min"]) & (upright >= settings["upright_min"])
             & (speed_std <= settings["speed_std_max"]) & (turn_rate <= settings["turn_rate_max"])
             & (instant <= settings["max_instant_speed"]))
    spans = continuous_segments(valid, int(np.ceil(settings["min_duration_seconds"]*fps))+1)
    return spans


def assign_group_weights(entries, shares, history_seconds):
    """Fixed group mass; balance source clips, then legal time within each source.

    Splitting a long recording into many spans must not multiply its probability.
    """
    if set(shares) != {"walking", "running", "jogging", "transition"}:
        raise ValueError("Expected walking/running/jogging/transition group shares")
    if not all(np.isfinite(value) and value > 0 for value in shares.values()) or not np.isclose(sum(shares.values()), 1.):
        raise ValueError("Positive group shares must sum to one")
    for group, share in shares.items():
        sources = defaultdict(list)
        for entry in entries:
            if entry["group"] == group:
                sources[entry["source_file"]].append(entry)
        if not sources:
            raise ValueError(f"No selected motions for required group: {group}")
        for spans in sources.values():
            durations = np.array([(entry["frames"]-1)/entry["fps"]-history_seconds for entry in spans])
            if np.any(durations <= 0):
                raise ValueError("Selected spans must contain a complete AMP history")
            for entry, duration in zip(spans, durations):
                entry["weight"] = float(share/len(sources)*duration/durations.sum())


def _yaml(path, entries, output):
    path.write_text(yaml.safe_dump({"motions": [dict(file=str(output/e["file"]), weight=e["weight"])
                                               for e in entries]}, sort_keys=False))


def build_curated_dataset(config_path, output_override=None):
    """Preserve original data; atomically write a separately versioned dataset."""
    config_path = Path(config_path).resolve()
    settings = yaml.safe_load(config_path.read_text())
    source = Path(settings["source_motion_file"])
    source = (source if source.is_absolute() else PROJECT_ROOT/source).resolve()
    output = Path(output_override or settings["output_dir"])
    output = (output if output.is_absolute() else PROJECT_ROOT/output).resolve()
    if output == source.parent or source.parent in output.parents:
        raise ValueError("Curated dataset must be a separate directory from original clips")
    source_manifest_path = source.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    skeleton = Path(source_manifest["skeleton"])
    if sha256(skeleton) != source_manifest["skeleton_sha256"]:
        raise ValueError("Source motion skeleton hash changed")
    request = dict(schema=1, source_list_sha256=sha256(source), source_manifest_sha256=sha256(source_manifest_path),
                   config_sha256=sha256(config_path), skeleton_sha256=source_manifest["skeleton_sha256"])
    if output.exists():
        path = output / "manifest.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            if existing.get("build_request") == request:
                for entry in existing["clips"]:
                    if sha256(output/entry["file"]) != entry["sha256"]:
                        raise ValueError(f"Curated clip changed: {entry['file']}")
                for name, digest in existing["list_sha256"].items():
                    if sha256(output/name) != digest:
                        raise ValueError(f"Curated sampling list changed: {name}")
                return existing
        raise FileExistsError(f"Use a new output directory; existing curation differs/incomplete: {output}")
    from anim.mjcf_char_model import MJCFCharModel
    model = MJCFCharModel("cpu")
    model.load(str(skeleton))
    history = (source_manifest["history_steps"]-1)*source_manifest["control_dt"]
    walk_settings = settings["walking"]
    for key, value in walk_settings.items():
        if key != "excluded_name_fragments" and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"Walking threshold {key} must be positive and finite")
    if (walk_settings["speed_min"] >= walk_settings["speed_max"]
            or walk_settings["min_duration_seconds"] <= history
            or walk_settings["forward_alignment_min"] > 1 or walk_settings["upright_min"] > 1):
        raise ValueError("Invalid walking speed, duration or orientation thresholds")
    original_entries = {entry["file"]: entry for entry in source_manifest["clips"]}
    source_rows, selected, seen = [], [], set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".curation-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for item in yaml.safe_load(source.read_text())["motions"]:
            path = Path(item["file"])
            path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
            if path in seen:
                raise ValueError(f"Duplicate source clip: {path}")
            seen.add(path)
            original = original_entries[path.name]
            if path.parent != source.parent or sha256(path) != original["sha256"]:
                raise ValueError(f"Source clip does not match conversion manifest: {path}")
            with path.open("rb") as stream:
                clip = pickle.load(stream)  # Trusted local converter output.
            frames = np.asarray(clip["frames"], dtype=np.float32)
            fps = float(clip["fps"])
            speed, _ = speed_statistics(frames, fps, history)
            category = classify_clip(original["name"], float(speed.mean()))
            row = dict(file=path.name, name=original["name"], category=category, original_frames=len(frames),
                       original_duration_s=(len(frames)-1)/fps, selected_segments=0, selected_duration_s=0., reason="")
            if category == "walking_label":
                normalized_name = original["name"].lower().replace("_", " ")
                excluded = [fragment for fragment in settings["walking"]["excluded_name_fragments"]
                            if fragment in normalized_name]
                if excluded:
                    row["reason"] = "special_walking_label: "+", ".join(excluded)
                    spans = []
                else:
                    spans = walking_segments(frames, fps, model, settings["walking"])
                    row["reason"] = "selected_continuous_walking" if spans else "no_qualifying_continuous_span"
                group = "walking"
            else:
                group = {"running_candidate": "running", "jogging_label": "jogging",
                         "walk_run_transition": "transition"}.get(category)
                spans = [(0, len(frames))] if group else []
                row["reason"] = "preserved_complete_motion" if group else "outside_current_locomotion_scope"
            for start, end in spans:
                filename = f"walk_{path.stem}_{start:06d}_{end:06d}.pkl" if group == "walking" else path.name
                cropped = frames[start:end].copy()
                xy_shift = cropped[0, :2].copy() if group == "walking" else np.zeros(2)
                cropped[:, :2] -= xy_shift
                if group == "walking":
                    # Only trim contiguous original frames and translate XY.
                    # Joint/root rotations, height, FPS and temporal order remain.
                    with (staging/filename).open("wb") as stream:
                        pickle.dump(dict(loop_mode=0, fps=fps, frames=cropped.tolist()), stream)
                else:
                    shutil.copyfile(path, staging/filename)
                _, windows = speed_statistics(cropped, fps, history)
                selected.append(dict(file=filename, name=original["name"], group=group, frames=len(cropped), fps=fps,
                                     source_file=path.name, source_sha256=original["sha256"],
                                     source_start_frame=start, source_end_frame_exclusive=end,
                                     xy_translation=(-xy_shift).tolist(), sha256=sha256(staging/filename),
                                     history_speed_fractions=_fractions(windows)))
                row["selected_segments"] += 1
                row["selected_duration_s"] += (end-start-1)/fps
            source_rows.append(row)
        assign_group_weights(selected, settings["group_shares"], history)
        lists = {"motions.yaml": selected, "motions-walking.yaml": [e for e in selected if e["group"] == "walking"],
                 "motions-running.yaml": [e for e in selected if e["group"] == "running"],
                 "motions-jogging.yaml": [e for e in selected if e["group"] == "jogging"],
                 "motions-transition-starts.yaml": [e for e in selected if e["group"] == "transition"]}
        for filename, entries in lists.items():
            _yaml(staging/filename, entries, output)
        groups = {}
        for group in settings["group_shares"]:
            entries = [e for e in selected if e["group"] == group]
            groups[group] = dict(clips=len(entries), source_clips=len({e["source_file"] for e in entries}),
                                 duration_s=sum((e["frames"]-1)/e["fps"] for e in entries),
                                 sampling_share=sum(e["weight"] for e in entries))
        report = dict(schema=1, build_request=request, source=str(source), skeleton=str(skeleton),
                      skeleton_sha256=source_manifest["skeleton_sha256"], history_steps=source_manifest["history_steps"],
                      control_dt=source_manifest["control_dt"], config=settings, clips=selected, groups=groups,
                      frames=sum(e["frames"] for e in selected), duration_s=sum(g["duration_s"] for g in groups.values()),
                      history_sampling_share={key: sum(e["weight"]*e["history_speed_fractions"][key] for e in selected)
                                              for key in selected[0]["history_speed_fractions"]},
                      motion_yaml_sha256=sha256(staging/"motions.yaml"),
                      list_sha256={name: sha256(staging/name) for name in lists},
                      notes=["Kinematic screening cannot certify natural foot contacts or visual quality.",
                             "Running, jogging and the two real transition clips are preserved byte-for-byte.",
                             "Group probability is balanced by original source; split count cannot multiply its mass.",
                             "Transition-start playback selects initial states; it does not command a gait transition."])
        (staging/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=list(source_rows[0]))
        writer.writeheader(); writer.writerows(source_rows)
        (staging/"selection.csv").write_text(text.getvalue())
        staging.rename(output)
    return report
