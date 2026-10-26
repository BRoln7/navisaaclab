"""CPU preparation helpers around the attributed PACER preprocessing functions."""

import ast
import csv
import importlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .bootstrap import PROJECT_ROOT, enable_mimickit
from .data import convert, load_trusted_cpu, sha256, SMPL_DFS

PACER_REVISION = "baeb6241f03fc9d78370a64e3c2dcef9d830d97f"
OCCLUSION_URL = "https://drive.usercontent.google.com/download?id=1rUSjVUYE8bIPZOyhbdQrKZ9-otE3fHpr&export=download&confirm=t"
VENDOR = PROJECT_ROOT / "third_party/pacer_preprocessing"


def selection_names():
    tree = ast.parse((VENDOR / "convert_amass_isaac.py").read_text())
    values = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "amass_run_data" for target in node.targets))
    return sorted(set(values))


def source_index(raw_dir):
    """Index original files by PACER's dataset_subject_action naming convention."""
    selected = set(selection_names())
    found = {}
    for dataset in sorted(raw_dir.iterdir()):
        if not dataset.is_dir():
            continue
        for subject in sorted(dataset.iterdir()):
            if not subject.is_dir():
                continue
            for path in sorted(subject.glob("*.npz")):
                key = f"0-{dataset.name}_{subject.name}_{path.stem}"
                if key in selected:
                    if key in found:
                        raise ValueError(f"Duplicate original motion key: {key}")
                    found[key] = path
    missing = sorted(selected - found.keys())
    if missing:
        raise FileNotFoundError("Missing PACER selections under --raw_dir:\n" + "\n".join(missing))
    return found


def resample_clip(clip, target_fps=30.):
    """Sample rotations with SLERP and translations on the same real-time grid.

    This adapter runs before PACER process_qpos_list: its integer stride becomes
    one. Source arrays/FPS are retained separately in raw_selected.pkl.
    """
    fps = float(np.asarray(clip["mocap_framerate"]).item())
    poses = np.asarray(clip["poses"])
    trans = np.asarray(clip["trans"])
    if (not np.isfinite(fps) or fps <= 0 or not np.isfinite(target_fps) or target_fps <= 0
            or poses.ndim != 2 or poses.shape[1] != 156
            or len(poses) < 2 or trans.shape != (len(poses), 3)
            or not np.isfinite(poses).all() or not np.isfinite(trans).all()):
        raise ValueError("Expected finite original SMPL+H poses/trans and a positive source FPS")
    count = int(np.floor((len(poses)-1) * target_fps / fps + 1e-9)) + 1
    source_times = np.arange(len(poses), dtype=np.float64) / fps
    target_times = np.arange(count, dtype=np.float64) / target_fps
    ratio = fps / target_fps
    stride = round(ratio)
    if stride >= 1 and np.isclose(ratio, stride, atol=1e-10, rtol=0.):
        output_poses = poses[::stride].copy()
        output_trans = trans[::stride].copy()
    else:
        output_poses = np.empty((count, 156), dtype=np.float64)
        for joint in range(52):
            axes = slice(3*joint, 3*joint+3)
            rotations = Rotation.from_rotvec(poses[:, axes])
            output_poses[:, axes] = Slerp(source_times, rotations)(target_times).as_rotvec()
        output_trans = np.column_stack([np.interp(target_times, source_times, trans[:, axis]) for axis in range(3)])
    # These are exactly the fields consumed by the original PACER second stage.
    return {"poses": output_poses, "trans": output_trans,
            "betas": np.asarray(clip["betas"]).copy(), "gender": clip["gender"],
            "mocap_framerate": float(target_fps)}


def ensure_occlusion(path):
    """Fetch only the public annotation linked by PACER download_data.sh."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("[PACER] Downloading official published occlusion v2 annotation ...", flush=True)
        temporary = path.with_name(path.name + ".download")
        with urlopen(OCCLUSION_URL, timeout=120) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024 or not data or data.lstrip().startswith(b"<"):
            raise RuntimeError("Google Drive did not return the annotation. Download file "
                               "1rUSjVUYE8bIPZOyhbdQrKZ9-otE3fHpr from PACER download_data.sh "
                               "and provide its local path with --occlusion.")
        temporary.write_bytes(data)
        try:
            value = load_trusted_cpu(temporary)
            if not isinstance(value, dict) or not all(isinstance(v, dict) and "issue" in v for v in value.values()):
                raise ValueError("Expected the official occlusion dictionary")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(path)
    value = load_trusted_cpu(path)
    if not isinstance(value, dict) or not all(isinstance(v, dict) and "issue" in v for v in value.values()):
        raise ValueError(f"Invalid occlusion dictionary: {path}")
    return value


def _json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _native_conversion(source, skeleton, destination):
    if destination.exists():
        return convert(source, skeleton, destination)
    with tempfile.TemporaryDirectory(prefix=".native-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "native"
        manifest = convert(source, skeleton, staging)
        # The existing converter stores absolute paths. Publish only final paths.
        import yaml
        (staging / "motions.yaml").write_text(yaml.safe_dump({"motions": [
            {"file": str(destination / clip["file"]), "weight": 1.0} for clip in manifest["clips"]
        ]}, sort_keys=False))
        manifest["motion_yaml_sha256"] = sha256(staging / "motions.yaml")
        _json(staging / "manifest.json", manifest)
        staging.rename(destination)
    return manifest


def prepare(raw_dir, output, models, pacer_skeleton, target_skeleton, occlusion_file, time_mode):
    """Run attributed PACER stages and a separate native MimicKit conversion."""
    import sys
    import torch

    indexed = source_index(raw_dir)
    # Only legacy data-processing packages; no task registration or simulator.
    sys.path.insert(0, str(PROJECT_ROOT / "third_party"))
    sys.path.insert(0, str(PROJECT_ROOT / "poselib"))
    raw_module = importlib.import_module("pacer_preprocessing.process_amass_raw")
    db_module = importlib.import_module("pacer_preprocessing.process_amass_db")
    converter = importlib.import_module("pacer_preprocessing.convert_amass_isaac")
    from socnav_sim.uhc.smpllib.smpl_parser import SMPL_Parser
    mimickit_lock = enable_mimickit()
    lock = json.loads((VENDOR / "sources.json").read_text())
    for filename, expected in lock["adapted_sha256"].items():
        if sha256(VENDOR / filename) != expected:
            raise ValueError(f"PACER adapter differs from recorded source: {filename}")
    for gender in ("neutral", "male", "female"):
        if not (models / f"SMPL_{gender.upper()}.pkl").is_file():
            raise FileNotFoundError(f"Missing SMPL model: {gender} under {models}")
    print(f"[PACER] Recording SHA256 for {len(indexed)} sources and three SMPL models ...", flush=True)
    source_hashes = {key: {"file": str(path), "sha256": sha256(path)} for key, path in indexed.items()}
    occlusion = ensure_occlusion(occlusion_file)
    request = {
        "schema": 1, "pacer_revision": PACER_REVISION, "time_mode": time_mode, "target_fps": 30.,
        "raw_dir": str(raw_dir), "sources": source_hashes,
        "model_sha256": {gender: sha256(models / f"SMPL_{gender.upper()}.pkl") for gender in ("neutral", "male", "female")},
        "pacer_skeleton": str(pacer_skeleton), "pacer_skeleton_sha256": sha256(pacer_skeleton),
        "mimickit_skeleton": str(target_skeleton), "mimickit_skeleton_sha256": sha256(target_skeleton),
        "occlusion": str(occlusion_file), "occlusion_sha256": sha256(occlusion_file),
        "pacer_adapters": lock["adapted_sha256"],
        "pipeline_sha256": sha256(Path(__file__)),
        "native_converter_sha256": sha256(Path(__file__).with_name("data.py")),
        "mimickit_revision": mimickit_lock["commit"],
        "smpl_parser_code_sha256": {
            name: sha256(PROJECT_ROOT / "exts/socnav_sim/socnav_sim/uhc/smpllib" / name)
            for name in ("smpl_parser.py", "legacy_smpl.py")
        },
    }
    if output.exists() and not (output / "build.json").is_file():
        raise FileExistsError(f"Output has no matching build manifest. Use a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state_file = output / "build.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["request"] != request:
            raise FileExistsError("Inputs/settings differ from this build. Use a new --output directory.")
    else:
        state = {"request": request, "stages": {}}
        _json(state_file, state)

    def stage(name, builder):
        path = output / name
        record = state["stages"].get(name)
        if record:
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Completed stage changed: {path}")
            print(f"[PACER] Reusing {name}", flush=True)
        else:
            print(f"[PACER] Building {name} ...", flush=True)
            temporary = output / (name + ".partial")
            builder(temporary)
            temporary.replace(path)
            state["stages"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            _json(state_file, state)
        return path

    def read_original(path):
        selected = {key.removeprefix("0-") for key in indexed}
        data = {}
        datasets = sorted({filename.parent.parent.name for filename in indexed.values()})
        for dataset in datasets:
            data.update(raw_module.read_single_sequence(str(raw_dir / dataset), dataset, selected_keys=selected))
        if {"0-" + key for key in data} != set(indexed):
            raise ValueError("PACER reader output did not match the selected source index")
        joblib.dump(data, path)

    raw_file = stage("raw_selected.pkl", read_original)
    original = load_trusted_cpu(raw_file)
    source_info = {"0-" + key: {"source_fps": float(np.asarray(value["mocap_framerate"]).item()),
                               "source_frames": len(value["poses"])} for key, value in original.items()}

    def align_time(path):
        aligned = {}
        for i, (key, clip) in enumerate(original.items(), 1):
            aligned[key] = resample_clip(clip) if time_mode == "exact" else clip
            if i % 25 == 0:
                print(f"[PACER] Time alignment {i}/{len(original)} ({time_mode})", flush=True)
        joblib.dump(aligned, path)

    time_file = stage("time_aligned.pkl", align_time)
    del original
    db_module.target_fr = 30
    db_module.amass_occlusion = occlusion

    def process_database(path):
        print("[PACER] Loading neutral/male/female SMPL parsers on CPU ...", flush=True)
        for gender, suffix in (("neutral", "n"), ("male", "m"), ("female", "f")):
            parser = SMPL_Parser(model_path=str(models), gender=gender, use_pca=False, create_transl=False).cpu()
            setattr(db_module, "smpl_parser_" + suffix, parser)
        # The original converter uses a fixed beta=0 neutral skeleton. Ensure the
        # supplied PACER XML root belongs to the installed neutral SMPL model.
        with torch.no_grad():
            _, joints = db_module.smpl_parser_n.get_joints_verts(torch.zeros(1, 72), th_betas=torch.zeros(1, 10))
        xml_root = ET.parse(pacer_skeleton).getroot().find("worldbody/body")
        offset = np.fromstring(xml_root.attrib["pos"], sep=" ")
        if not np.allclose(joints[0, 0].numpy(), offset, atol=1e-3):
            raise ValueError("PACER XML pelvis offset does not match installed neutral SMPL model")
        names = [body.attrib["name"] for body in xml_root.iter("body")]
        if names != SMPL_DFS:
            raise ValueError("PACER XML must have the original 24-body DFS order")
        data = load_trusted_cpu(time_file)
        # Keep the official processing function: occlusion bounds, hand removal,
        # first-frame mesh height correction, original SMPL fields.
        processed = db_module.process_qpos_list(list(data.items()))
        if not processed:
            raise ValueError("PACER filtering removed every selected clip")
        joblib.dump(processed, path)
        for suffix in ("n", "m", "f"):
            delattr(db_module, "smpl_parser_" + suffix)

    processed_file = stage("processed_smpl.pkl", process_database)
    pacer_file = stage("pacer_upright.pkl", lambda path: converter.run(str(processed_file), str(path), str(pacer_skeleton)))
    native = _native_conversion(pacer_file, target_skeleton, output / "native")
    rows = []
    for clip in native["clips"]:
        rows.append(dict(name=clip["name"], **source_info[clip["name"]], frames=clip["frames"], fps=clip["fps"],
                         duration_seconds=(clip["frames"]-1)/clip["fps"],
                         floor_shift_m=clip["floor_shift"], fk_error_max_m=clip["fk_error_max"]))
    with (output / "clips.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    processed_names = set(load_trusted_cpu(processed_file))
    report = dict(selection_clips=len(indexed), after_pacer_filter=len(processed_names), native_clips=len(rows),
                  filtered_by_pacer=sorted(set(indexed)-processed_names),
                  skipped_short_native=native["skipped_short_clips"], time_mode=time_mode,
                  raw_frames=sum(value["source_frames"] for value in source_info.values()),
                  native_frames=native["frames"], native_seconds=sum(row["duration_seconds"] for row in rows),
                  motion_file=str(output / "native/motions.yaml"), sampling="uniform per clip, weight 1",
                  occlusion_note="Use the actual annotation file/hash in build.json; published download is v2, upstream process_amass_db.py main names v3.",
                  scope="Official selected reference pool across all eight sources; not an AMASS train/test split or the old slim subset.",
                  cuda_initialized=torch.cuda.is_initialized())
    if report["cuda_initialized"]:
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    _json(output / "report.json", report)
    return report
