"""Publish PACER-format stages from a completed AMASS build, without MimicKit conversion."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from socnav_sim.mimickit_amp.data import load_trusted_cpu, sha256

FIELDS = {"pose_quat_global", "pose_quat", "trans_orig", "root_trans_offset", "beta", "gender", "pose_aa", "fps"}


def summary(motions):
    return {"clips": len(motions), "frames": sum(len(clip["pose_aa"]) for clip in motions.values()),
            "seconds": sum((len(clip["pose_aa"])-1)/float(clip["fps"]) for clip in motions.values())}


def export(build, output, reference):
    state = json.loads((build / "build.json").read_text())
    for filename in ("raw_selected.pkl", "processed_smpl.pkl", "pacer_upright.pkl"):
        expected = state["stages"][filename]["sha256"]
        if sha256(build / filename) != expected:
            raise ValueError(f"Stage differs from completed build: {filename}")
    request = {"build_sha256": sha256(build / "build.json"), "reference_sha256": sha256(reference),
               "export_code_sha256": sha256(Path(__file__))}
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["request"] != request:
            raise FileExistsError(f"Existing export differs. Use a new --output: {output}")
        for filename, record in manifest["files"].items():
            if sha256(output / filename) != record["sha256"]:
                raise ValueError(f"Export changed: {filename}")
        return manifest
    complete = load_trusted_cpu(build / "pacer_upright.pkl")
    original = load_trusted_cpu(reference)
    if not set(original) <= set(complete):
        raise ValueError("The rebuilt pool is missing original motions: " + repr(sorted(set(original)-set(complete))))
    for name, clip in complete.items():
        frames = len(clip["pose_aa"])
        if set(clip) != FIELDS:
            raise ValueError(f"Unexpected PACER MotionLib fields: {name}")
        shapes = {"pose_quat_global": (frames, 24, 4), "pose_quat": (frames, 24, 4),
                  "pose_aa": (frames, 72), "trans_orig": (frames, 3), "root_trans_offset": (frames, 3)}
        if not np.isfinite(clip["fps"]) or clip["fps"] <= 0:
            raise ValueError(f"Invalid FPS: {name}")
        for field, shape in shapes.items():
            value = np.asarray(clip[field])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid PACER array {field}: {name}")
    matched = {name: complete[name] for name in original}
    additional = {name: clip for name, clip in complete.items() if name not in original}
    processed = load_trusted_cpu(build / "processed_smpl.pkl")
    if set(processed) != set(complete):
        raise ValueError("Processed SMPL and upright stage motion keys differ")
    # The previous bridge omitted this unused intermediate field. Restore PACER's
    # first-two-matrix-columns 6D convention for inspecting the original schema.
    for clip in processed.values():
        matrices = Rotation.from_rotvec(np.asarray(clip["pose_aa"]).reshape(-1, 3)).as_matrix()
        clip["pose_6d"] = matrices[:, :, :2].transpose(0, 2, 1).reshape(len(clip["pose_aa"]), 24, 6).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pacer-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        shutil.copyfile(build / "raw_selected.pkl", staging / "amass_db_smplh.pt")
        shutil.copyfile(build / "pacer_upright.pkl", staging / "amass_run_isaac.pkl")
        exports = {
            "amass_copycat_take5_all.pkl": processed,
            "amass_copycat_take5_train.pkl": {name: processed[name] for name in original},
            "amass_copycat_take5_additional.pkl": {name: processed[name] for name in additional},
            "amass_isaac_run_upright_slim.pkl": matched,
        }
        for filename, motions in exports.items():
            joblib.dump(motions, staging / filename)
        files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in staging.iterdir()}
        raw_names = ["0-" + name for name in load_trusted_cpu(build / "raw_selected.pkl")]
        manifest = {
            "request": request, "build": str(build), "reference": str(reference),
            "time_mode": state["request"]["time_mode"],
            "raw_stage_clips": len(raw_names), "filtered_out": sorted(set(raw_names)-set(complete)),
            "old": summary(original), "rebuilt_full": summary(complete), "rebuilt_matching_old": summary(matched),
            "additional": summary(additional), "additional_names": sorted(additional),
            "missing_old_names": sorted(set(original)-set(complete)),
            "motionlib_fields": sorted(FIELDS), "files": files,
            "scope": "Raw stage includes the 276 PACER-selected original motions, not every motion in the eight AMASS archives.",
            "matched_subset": "The slim/train files use exactly the reference file's motion-name set; additional is not a declared evaluation split.",
            "pose_6d": "Restored from pose_aa with PACER column convention; not claimed bitwise identical to old TorchGeometry output.",
            "cuda_initialized": torch.cuda.is_initialized(),
        }
        if manifest["cuda_initialized"]:
            raise RuntimeError("Export must not initialize CUDA")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build_dir", type=Path, default=ROOT / "data/mimickit/amass-pacer-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/motion_cap/amass_pacer_rebuilt")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/motion_cap/amass_isaac/amass_isaac_run_upright_slim.pkl")
    args = parser.parse_args()
    result = export(args.build_dir.expanduser().resolve(), args.output.expanduser().resolve(), args.reference.expanduser().resolve())
    print(json.dumps({key: result[key] for key in ("raw_stage_clips", "old", "rebuilt_full", "rebuilt_matching_old", "additional_names", "missing_old_names", "time_mode")}, indent=2))
    print(f"PACER MotionLib PKLs: {args.output}")


if __name__ == "__main__":
    main()
