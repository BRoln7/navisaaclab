"""CPU-only conversion from the project's processed SMPL dictionary to native MimicKit motions."""

import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

SMPL_DFS = ("Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe "
            "Torso Spine Chest Neck Head L_Thorax L_Shoulder L_Elbow L_Wrist L_Hand "
            "R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand").split()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_cpu(path):
    """Offline migration of trusted local pickles, including old CUDA Torch storages."""
    previous = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    try:
        return joblib.load(path)
    finally:
        torch.storage._load_from_bytes = previous


def convert(source, skeleton, output, history_steps=10, control_dt=1/30, clearance=.02):
    from anim.mjcf_char_model import MJCFCharModel
    from anim.motion import Motion, LoopMode
    from util import torch_util

    source, skeleton, output = Path(source).resolve(), Path(skeleton).resolve(), Path(output).resolve()
    metadata = dict(schema=1, source=str(source), source_sha256=sha256(source),
                    skeleton=str(skeleton), skeleton_sha256=sha256(skeleton),
                    history_steps=history_steps, control_dt=control_dt, clearance=clearance)
    manifest = output / "manifest.json"
    if output.exists():
        if manifest.is_file():
            existing = json.loads(manifest.read_text())
            if all(existing.get(k) == v for k, v in metadata.items()):
                for entry in existing["clips"]:
                    if sha256(output / entry["file"]) != entry["sha256"]:
                        raise ValueError(f"Converted motion changed: {entry['file']}")
                return existing
        raise FileExistsError(f"Use a new output directory; existing conversion differs/incomplete: {output}")
    model = MJCFCharModel("cpu")
    model.load(str(skeleton))
    if model.get_body_names() != SMPL_DFS or model.get_dof_size() != 69:
        raise ValueError("Source pose_quat_global uses the project's 24-body DFS SMPL order; skeleton must match")
    xml_bodies = list(ET.parse(skeleton).getroot().find("worldbody/body").iter("body"))
    soles = []
    for i, body in enumerate(xml_bodies):
        if body.attrib["name"] in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe"):
            geom = body.find("geom")
            if geom.attrib["type"] != "box":
                raise ValueError("Expected box foot collision geometry")
            size = np.fromstring(geom.attrib["size"], sep=" ")
            center = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
            quat = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" ")[[1, 2, 3, 0]]
            corners = np.array(np.meshgrid(*[[-1, 1]]*3)).reshape(3, -1).T * size
            soles.append((i, Rotation.from_quat(quat).apply(corners) + center))
    source_data = load_trusted_cpu(source)
    if not isinstance(source_data, dict) or not source_data:
        raise ValueError("Expected nonempty SMPL motion dictionary")
    output.mkdir(parents=True)
    entries, skipped = [], []
    for name, clip in source_data.items():
        root = np.asarray(clip["root_trans_offset"], dtype=np.float32).copy()
        q = np.asarray(clip["pose_quat_global"], dtype=np.float32)
        fps = float(clip["fps"])
        if (root.shape != (len(q), 3) or q.shape[1:] != (24, 4) or not np.isfinite(fps) or fps <= 0
                or not np.isfinite(root).all() or not np.isfinite(q).all() or (np.linalg.norm(q, axis=-1) < 1e-8).any()):
            raise ValueError(f"Invalid shape, FPS or pose: {name}")
        if (len(q)-1)/fps <= (history_steps-1)*control_dt:
            skipped.append(str(name))
            continue
        global_q = torch.from_numpy(q / np.linalg.norm(q, axis=-1, keepdims=True))
        parents = model._parent_indices[1:]
        # Input is already upright in simulator coordinates. Do NOT rotate as raw AMASS again.
        parent_q = global_q[:, parents]
        local_q = torch_util.quat_mul(torch_util.quat_conjugate(parent_q), global_q[:, 1:])
        rest_q = model._local_rotation[1:].expand_as(local_q)
        joint_q = torch_util.quat_mul(torch_util.quat_conjugate(rest_q), local_q)
        root[:, :2] -= root[:1, :2].copy()
        positions, rotations = model.forward_kinematics(torch.from_numpy(root), global_q[:, 0], joint_q)
        # One translation per CLIP preserves flight phases and all vertical velocities.
        pos, rot = positions.numpy(), rotations.numpy()
        floor = min((pos[:, i, None] + np.einsum("tij,kj->tki", Rotation.from_quat(rot[:, i]).as_matrix(), corners))[..., 2].min()
                    for i, corners in soles)
        shift = float(clearance - floor)
        root[:, 2] += shift
        dof = model.rot_to_dof(joint_q).numpy()
        root_exp = torch_util.quat_to_exp_map(global_q[:, 0]).numpy()
        frames = np.concatenate((root, root_exp, dof), axis=-1)
        # Verify native expmap -> quaternion -> FK, including root orientation.
        decoded = model.dof_to_rot(torch.from_numpy(dof))
        reconstructed, _ = model.forward_kinematics(torch.from_numpy(root), torch_util.exp_map_to_quat(torch.from_numpy(root_exp)), decoded)
        expected = pos.copy(); expected[..., 2] += shift
        max_error = float(np.max(np.abs(reconstructed.numpy() - expected)))
        if max_error > 1e-4:
            raise ValueError(f"Motion round trip failed ({max_error} m): {name}")
        filename = f"clip_{len(entries):04d}.pkl"
        Motion(LoopMode.CLAMP, fps, frames).save(str(output / filename))
        entries.append(dict(name=str(name), file=filename, frames=len(q), fps=fps,
                            floor_shift=shift, fk_error_max=max_error, sha256=sha256(output / filename)))
    if not entries:
        raise ValueError("No clips contain a complete AMP history")
    motions = {"motions": [{"file": str(output / e["file"]), "weight": 1.0} for e in entries]}
    (output / "motions.yaml").write_text(yaml.safe_dump(motions, sort_keys=False))
    metadata.update(clips=entries, skipped_short_clips=skipped, body_names=model.get_body_names(), dofs=69,
                    frames=sum(e["frames"] for e in entries), quaternion_input="xyzw/global", output="MimicKit native expmap",
                    motion_yaml_sha256=sha256(output / "motions.yaml"))
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
