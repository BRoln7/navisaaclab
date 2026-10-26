"""Audit trusted native motion clips and reweight running without copying poses."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import re

import numpy as np
import yaml

from .bootstrap import PROJECT_ROOT
from .data import sha256


def classify_clip(name, mean_speed):
    """Conservative filename categories, supported by measured root speed.

    These are candidate groups, not ground-truth contact/gait annotations.
    Dataset directory names like Male2Running must not classify a sidestep.
    """
    action = name.lower().split("_c3d_")[-1]
    if re.search(r"walk[ _-]*to[ _-]*run|run[ _-]*to[ _-]*walk", action):
        return "walk_run_transition"
    if "sidestep" in action or "side step" in action:
        return "lateral_step"
    if "hop" in action or "leap" in action:
        return "walk_hop_leap"
    if "jog" in action:
        return "jogging_label"
    if "walk" in action:
        # walking_run08 is a walk recording/take, not a running label.
        return "walking_label"
    if re.search(r"(?:^|[ _-])run(?:[0-9 _-]|$)", action) and mean_speed >= 2.:
        return "running_candidate"
    return "other_or_uncertain"


def speed_statistics(frames, fps, history_seconds):
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 75 or not np.isfinite(frames).all():
        raise ValueError("Expected finite native 75-coordinate SMPL frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion FPS must be positive and finite")
    intervals = round(history_seconds*fps)
    if intervals < 1 or len(frames)-1 <= intervals:
        raise ValueError("Clip must be longer than the AMP history")
    if not np.isclose(intervals, history_seconds*fps):
        raise ValueError("Audit requires an integer number of source intervals per AMP history")
    speed = np.linalg.norm(np.diff(frames[:, :2], axis=0)*fps, axis=1)
    window_speed = np.convolve(speed, np.ones(intervals)/intervals, mode="valid")
    return speed, window_speed


def _fractions(values):
    bins = (("below_0.2", 0., .2), ("0.2_to_0.8", .2, .8), ("0.8_to_1.4", .8, 1.4),
            ("1.4_to_2.0", 1.4, 2.), ("2.0_to_2.5", 2., 2.5), ("at_least_2.5", 2.5, np.inf))
    return {key: float(np.mean((values >= low) & (values < high))) for key, low, high in bins}


def build_running_sampling(source, output, running_weight=6.):
    """Emit one YAML entry per original clip, plus auditable composition reports."""
    if not np.isfinite(running_weight) or running_weight < 1:
        raise ValueError("running_weight must be finite and at least one")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.parent != source.parent:
        raise ValueError("Use a distinct output YAML alongside the original conversion")
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {entry["file"]: entry for entry in manifest["clips"]}
    history = (manifest["history_steps"]-1)*manifest["control_dt"]
    original = yaml.safe_load(source.read_text())["motions"]
    seen, rows, weighted = set(), [], []
    original_bins, new_bins, time_bins = defaultdict(float), defaultdict(float), defaultdict(float)
    for motion in original:
        path = Path(motion["file"])
        path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
        if path in seen or path.parent != source.parent or path.name not in entries:
            raise ValueError("Expected unique clips from the original converted motion manifest")
        seen.add(path)
        entry = entries[path.name]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Converted clip hash changed: {path}")
        with path.open("rb") as stream:
            clip = pickle.load(stream)  # Trusted local converter output only.
        speed, windows = speed_statistics(clip["frames"], clip["fps"], history)
        group = classify_clip(entry["name"], float(speed.mean()))
        weight = float(motion["weight"])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Motion weights must be finite and nonnegative")
        boosted = group in ("running_candidate", "jogging_label")
        new_weight = weight*(running_weight if boosted else 1.)
        duration = (len(clip["frames"])-1)/clip["fps"]
        row = dict(file=path.name, name=entry["name"], group=group, frames=len(clip["frames"]),
                   duration_s=duration, mean_speed_mps=float(speed.mean()),
                   median_speed_mps=float(np.median(speed)), p90_speed_mps=float(np.quantile(speed, .9)),
                   fast_history_fraction=float(np.mean(windows >= 2.)),
                   low_speed_history_fraction=float(np.mean(windows < .2)),
                   original_weight=weight, new_weight=new_weight, boosted=boosted)
        rows.append(row)
        weighted.append(dict(file=str(path), weight=new_weight))
        for key, fraction in _fractions(windows).items():
            original_bins[key] += weight*fraction
            new_bins[key] += new_weight*fraction
        for key, fraction in _fractions(speed).items():
            time_bins[key] += duration*fraction
    original_total = sum(row["original_weight"] for row in rows)
    new_total = sum(row["new_weight"] for row in rows)
    duration_total = sum(row["duration_s"] for row in rows)
    if not rows or original_total <= 0 or new_total <= 0:
        raise ValueError("Motion list must contain positive total weight")
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        members = [row for row in rows if row["group"] == group]
        groups[group] = dict(clips=len(members), duration_s=sum(row["duration_s"] for row in members),
                             original_sampling_share=sum(row["original_weight"] for row in members)/original_total,
                             new_sampling_share=sum(row["new_weight"] for row in members)/new_total)
    report = dict(schema=1, source=str(source), source_sha256=sha256(source),
                  conversion_manifest_sha256=sha256(manifest_path), output=str(output),
                  running_weight=running_weight, history_seconds=history, clips=len(rows),
                  frames=sum(row["frames"] for row in rows), duration_s=duration_total,
                  groups=groups, boosted_clips=[row["file"] for row in rows if row["boosted"]],
                  speed_bin_time_share={key: value/duration_total for key, value in time_bins.items()},
                  original_history_sampling_share={key: value/original_total for key, value in original_bins.items()},
                  new_history_sampling_share={key: value/new_total for key, value in new_bins.items()},
                  source_datasets=dict(Counter(row["name"].split("-", 1)[-1].split("_")[0] for row in rows)),
                  notes=["Filename/speed candidate categories are not contact-based gait labels.",
                         "History statistics approximate continuous native sampling with legal discrete frame anchors.",
                         "Weights affect both discriminator demonstrations and reference-state initialization.",
                         "No frames are duplicated, cropped or sped up; running diversity is unchanged."])
    text = yaml.safe_dump({"motions": weighted}, sort_keys=False)
    report["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    # Do not silently change a list already referenced by a saved checkpoint.
    if output.exists() and output.read_text() != text:
        raise FileExistsError(f"Existing sampling list differs; use a new output filename: {output}")
    output.write_text(text)
    output.with_suffix(".sampling.json").write_text(json.dumps(report, indent=2)+"\n")
    csv_text = io.StringIO()
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".sampling.csv").write_text(csv_text.getvalue())
    return report
