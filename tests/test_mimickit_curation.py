"""CPU regression contracts for motion curation; no simulator is imported."""

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
enable_mimickit()
from socnav_sim.mimickit_amp.curation import assign_group_weights, build_curated_dataset, continuous_segments
from socnav_sim.mimickit_amp.data import sha256


def test_continuous_spans_do_not_stitch_across_invalid_frames():
    mask = np.array([False, True, True, True, False, True, True, True, True, False])
    assert continuous_segments(mask, 3) == [(1, 4), (5, 9)]
    assert continuous_segments(mask, 5) == []


def test_group_mass_is_invariant_to_splitting_one_source():
    entries = [dict(group="walking", source_file="a", frames=61, fps=30.),
               dict(group="walking", source_file="a", frames=61, fps=30.),
               dict(group="walking", source_file="b", frames=121, fps=30.)]
    entries += [dict(group=group, source_file=group, frames=61, fps=30.)
                for group in ("running", "jogging", "transition")]
    shares = dict(walking=.55, running=.30, jogging=.10, transition=.05)
    assign_group_weights(entries, shares, .3)
    assert entries[0]["weight"]+entries[1]["weight"] == pytest.approx(entries[2]["weight"])
    for group, share in shares.items():
        assert sum(entry["weight"] for entry in entries if entry["group"] == group) == pytest.approx(share)


def test_curated_frames_preserve_pose_timing_and_full_running_clips(tmp_path):
    assert not torch.cuda.is_initialized()
    source = tmp_path/"original"; source.mkdir()
    records, motions, frames_by_file = [], [], {}
    names = ("0-KIT_1_walking_normal01_poses", "0-KIT_1_run01_poses",
             "0-HumanEva_S2_Jog_1_poses", "0-ACCAD_Female1Running_c3d_C5 - walk to run_poses",
             "0-Eyes_Japan_Dataset_aita_walk-21-one leg-aita_poses")
    for index, name in enumerate(names):
        frames = np.zeros((181, 75), dtype=np.float32)
        frames[:, 2] = 1.
        velocity = np.ones(180)*(3. if index == 1 else 1.2)
        if index == 0:
            velocity[:30] = 0.; velocity[-30:] = 0.
        frames[1:, 0] = np.cumsum(velocity/30)
        path = source/f"clip_{index:04d}.pkl"
        with path.open("wb") as stream:
            pickle.dump(dict(loop_mode=0, fps=30., frames=frames.tolist()), stream)
        records.append(dict(file=path.name, name=name, sha256=sha256(path)))
        motions.append(dict(file=str(path), weight=1.))
        frames_by_file[path.name] = frames
    skeleton = ROOT/"data/cfg/smpl_skeletontree.xml"
    (source/"manifest.json").write_text(json.dumps(dict(clips=records, skeleton=str(skeleton),
        skeleton_sha256=sha256(skeleton), history_steps=10, control_dt=1/30)))
    (source/"motions.yaml").write_text(yaml.safe_dump(dict(motions=motions)))
    settings = yaml.safe_load((ROOT/"data/cfg/mimickit_amp/motion_curation.yaml").read_text())
    settings.update(source_motion_file=str(source/"motions.yaml"), output_dir=str(tmp_path/"curated"))
    config = tmp_path/"curation.yaml"; config.write_text(yaml.safe_dump(settings))
    report = build_curated_dataset(config)
    output = tmp_path/"curated"
    assert build_curated_dataset(config) == report
    assert report["groups"]["walking"]["source_clips"] == 1
    assert {entry["source_file"] for entry in report["clips"]} == {f"clip_{i:04d}.pkl" for i in range(4)}
    for entry in report["clips"]:
        with (output/entry["file"]).open("rb") as stream:
            stored = pickle.load(stream)
        if entry["group"] == "walking":
            original = frames_by_file[entry["source_file"]][entry["source_start_frame"]:entry["source_end_frame_exclusive"]]
            actual = np.asarray(stored["frames"], dtype=np.float32)
            np.testing.assert_array_equal(actual[:, 2:], original[:, 2:])
            np.testing.assert_allclose(actual[:, :2], original[:, :2]-original[:1, :2])
            assert stored["fps"] == 30. and stored["loop_mode"] == 0
            assert len(actual) < 181
        else:
            assert sha256(output/entry["file"]) == sha256(source/entry["source_file"])
    assert not torch.cuda.is_initialized()
