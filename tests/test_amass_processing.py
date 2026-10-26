"""CPU preprocessing contracts; run manually, without Isaac Sim or GPU."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
sys.path.insert(0, str(ROOT / "third_party"))
from socnav_sim.mimickit_amp import amass_processing as processing


def sample(fps, seconds=2.):
    time = np.arange(round(fps * seconds) + 1) / fps
    poses = np.zeros((len(time), 156))
    poses[:, 2] = .7 * time
    return dict(poses=poses, trans=np.column_stack([2.3*time, time*0., .9+.1*time]),
                betas=np.zeros(16), gender=np.array("neutral"), mocap_framerate=np.array(fps))


@pytest.mark.parametrize("fps", [60., 100., 120.])
def test_real_time_grid_preserves_linear_and_angular_speed(fps):
    original = sample(fps)
    original_poses = original["poses"].copy()
    actual = processing.resample_clip(original)
    time = np.arange(61)/30
    assert actual["mocap_framerate"] == 30.
    assert len(actual["poses"]) == 61
    np.testing.assert_allclose(actual["trans"][:, 0], 2.3*time, atol=1e-12)
    np.testing.assert_allclose(actual["poses"][:, 2], .7*time, atol=1e-12)
    np.testing.assert_array_equal(original["poses"], original_poses)
    assert float(original["mocap_framerate"]) == fps


def test_rotation_interpolation_across_axis_angle_branch_is_continuous():
    clip = sample(100.)
    clip["poses"][:, 2] += 3.
    actual = processing.resample_clip(clip)
    expected = Rotation.from_rotvec(np.column_stack([np.zeros(61), np.zeros(61), 3.+.7*np.arange(61)/30]))
    difference = expected.inv() * Rotation.from_rotvec(actual["poses"][:, :3])
    np.testing.assert_allclose(difference.magnitude(), 0., atol=1e-12)


def test_pacer_filter_and_height_correction_preserve_relative_root_motion(monkeypatch):
    from pacer_preprocessing import process_amass_db as pacer
    import torch

    class Parser:
        def get_joints_verts(self, pose, betas, th_trans):
            vertices = th_trans[:, None, :].repeat(1, 2, 1)
            vertices[:, :, 2] -= .5
            return vertices, None

    monkeypatch.setattr(pacer, "smpl_parser_n", Parser(), raising=False)
    monkeypatch.setattr(pacer, "amass_occlusion", {
        "0-bad": {"issue": "irrecoverable"},
        "0-shortened": {"issue": "sitting", "idxes": [20]},
    })
    source = sample(30., 1.)
    result = pacer.process_qpos_list([(name, source) for name in ("ok", "bad", "shortened")])
    assert set(result) == {"0-ok", "0-shortened"}
    assert len(result["0-shortened"]["pose_aa"]) == 20
    assert result["0-ok"]["pose_aa"].shape[1] == 72
    np.testing.assert_array_equal(result["0-ok"]["pose_aa"][:, 66:], 0.)
    np.testing.assert_allclose(np.diff(result["0-ok"]["trans"], axis=0), np.diff(source["trans"], axis=0))
    assert result["0-ok"]["trans"][0, 2] == pytest.approx(.5)


def test_atomic_native_publish_contains_final_absolute_paths(tmp_path, monkeypatch):
    def fake_convert(source, skeleton, destination):
        destination.mkdir()
        (destination / "clip.pkl").write_bytes(b"native-placeholder")
        (destination / "motions.yaml").write_text("temporary list")
        manifest = {"clips": [{"file": "clip.pkl"}]}
        (destination / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(processing, "convert", fake_convert)
    destination = tmp_path / "native"
    report = processing._native_conversion(tmp_path / "input.pkl", tmp_path / "skeleton.xml", destination)
    entries = yaml.safe_load((destination / "motions.yaml").read_text())["motions"]
    assert entries == [{"file": str(destination / "clip.pkl"), "weight": 1.0}]
    assert report["motion_yaml_sha256"] == processing.sha256(destination / "motions.yaml")
    assert (destination / "clip.pkl").is_file()
