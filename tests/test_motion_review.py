"""CPU contracts for reference review; no simulator or policy is constructed."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
enable_mimickit()
from socnav_sim.mimickit_amp.motion_review import clip_metrics, entry, rotation_steps


def test_rotation_jump_uses_physical_rotation_not_expmap_branch():
    vec = np.array([[0., 0., 3.13], [0., 0., -3.13]])
    assert rotation_steps(vec)[0] == pytest.approx(2*np.pi-6.26)


def test_transition_is_kept_complete_and_risk_window_contains_event():
    row = dict(index=2, file="clip.pkl", name="walk to run", group="walk_run_transition", sha256="a", duration_s=17.)
    selected = entry(row, "transition", 8.)
    assert selected["start_s"] == 0.
    assert selected["end_s"] == 17.
    row["group"] = "walking_label"
    selected = entry(row, "inspect start", 8., event=.3)
    assert selected["start_s"] == 0.
    assert selected["end_s"] == 8.


def test_low_foot_slip_metric_detects_known_kinematic_translation():
    class Model:
        def get_body_names(self):
            return ["Pelvis", "Head", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]

        def dof_to_rot(self, dofs):
            return None

        def forward_kinematics(self, root, root_rot, joint_rot):
            positions = root[:, None].repeat(1, 6, 1)
            positions[:, 1, 2] += .8
            positions[:, 2:, 2] = .02
            rotation = torch.zeros((len(root), 6, 4))
            rotation[..., 3] = 1.
            return positions, rotation

    frames = np.zeros((61, 75), dtype=np.float32)
    frames[:, 0] = np.arange(61)/30
    frames[:, 2] = .9
    feet = [(body, np.zeros((1, 3)), np.zeros(3)) for body in range(2, 6)]
    result = clip_metrics(frames, 30., Model(), feet)
    assert result["low_foot_slip_p90_m_s"] == pytest.approx(1., abs=1e-5)
    assert result["support_height_p95_m"] == pytest.approx(.02, abs=1e-6)
    assert result["root_max_step_rad"] == pytest.approx(0.)
