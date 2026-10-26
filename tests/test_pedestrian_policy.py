from __future__ import annotations

from pathlib import Path
import importlib.util

import torch

from socnav_sim.tasks.socnav.nav_utils.pedestrian_policy import (
    MimicKitPedestrianPolicy,
    build_mimickit_observation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exported_policy_matches_checkpoint_actor(tmp_path):
    """Exercise the real exporter with synthetic weights rather than private logs."""
    generator = torch.Generator().manual_seed(11)
    original = {
        "_obs_norm._mean": torch.randn(255, generator=generator),
        "_obs_norm._std": torch.rand(255, generator=generator) + 0.1,
        "_a_norm._mean": torch.randn(69, generator=generator),
        "_a_norm._std": torch.rand(69, generator=generator) + 0.1,
    }
    for prefix, output_size, input_size in (
        ("_model._actor_layers.0", 1024, 255),
        ("_model._actor_layers.2", 512, 1024),
        ("_model._action_dist._mean_net", 69, 512),
    ):
        original[f"{prefix}.weight"] = torch.randn(output_size, input_size, generator=generator) * 0.01
        original[f"{prefix}.bias"] = torch.randn(output_size, generator=generator) * 0.01
    checkpoint, exported = tmp_path / "source.pt", tmp_path / "policy.pth"
    torch.save(original, checkpoint)
    spec = importlib.util.spec_from_file_location("export_policy", ROOT / "scripts/mimickit_amp/export_policy.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.export_policy(checkpoint, exported)
    policy = MimicKitPedestrianPolicy.load(exported, "cpu")
    observation = torch.randn((17, 255), generator=generator)
    normalized = torch.clamp(
        (observation - original["_obs_norm._mean"]) / original["_obs_norm._std"], -10, 10
    )
    hidden = torch.relu(torch.nn.functional.linear(
        normalized, original["_model._actor_layers.0.weight"], original["_model._actor_layers.0.bias"]
    ))
    hidden = torch.relu(torch.nn.functional.linear(
        hidden, original["_model._actor_layers.2.weight"], original["_model._actor_layers.2.bias"]
    ))
    expected = torch.nn.functional.linear(
        hidden,
        original["_model._action_dist._mean_net.weight"],
        original["_model._action_dist._mean_net.bias"],
    )
    expected = expected * original["_a_norm._std"] + original["_a_norm._mean"]
    expected = torch.clamp(
        expected,
        original["_a_norm._mean"] - original["_a_norm._std"],
        original["_a_norm._mean"] + original["_a_norm._std"],
    )
    torch.testing.assert_close(policy(observation), expected)


def test_observation_layout_has_expected_shape_and_is_finite():
    count = 4
    root_quaternion = torch.zeros((count, 4)); root_quaternion[:, 0] = 1
    observation = build_mimickit_observation(
        root_position=torch.zeros((count, 3)),
        root_quaternion_wxyz=root_quaternion,
        root_linear_velocity=torch.zeros((count, 3)),
        root_angular_velocity=torch.zeros((count, 3)),
        joint_position=torch.zeros((count, 69)),
        joint_velocity=torch.zeros((count, 69)),
        key_body_position=torch.zeros((count, 5, 3)),
        trajectory_xy=torch.zeros((count, 10, 2)),
    )
    assert observation.shape == (count, 255)
    assert torch.isfinite(observation).all()
