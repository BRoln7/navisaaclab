#!/usr/bin/env python3
"""Export a MimicKit checkpoint as a portable navigation pedestrian actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = PROJECT_ROOT / "exts/socnav_sim"
sys.path.insert(0, str(EXTENSION_ROOT))

from socnav_sim.tasks.socnav.nav_utils.pedestrian_policy import (  # noqa: E402
    POLICY_FORMAT,
    SMPL_BODY_NAMES,
    SMPL_JOINT_NAMES,
    SMPL_KEY_BODY_NAMES,
    MimicKitPedestrianPolicy,
)


def export_policy(checkpoint: Path, output: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = {
        "_obs_norm._mean", "_obs_norm._std", "_a_norm._mean", "_a_norm._std",
        "_model._actor_layers.0.weight", "_model._actor_layers.0.bias",
        "_model._actor_layers.2.weight", "_model._actor_layers.2.bias",
        "_model._action_dist._mean_net.weight", "_model._action_dist._mean_net.bias",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint lacks MimicKit actor tensors: {sorted(missing)}")
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    resolved_checkpoint = checkpoint.resolve()
    try:
        source_checkpoint = str(resolved_checkpoint.relative_to(PROJECT_ROOT))
    except ValueError:
        source_checkpoint = resolved_checkpoint.name
    bundle = {
        "format": POLICY_FORMAT,
        "observation_size": 255,
        "action_size": 69,
        "observation_mean": state["_obs_norm._mean"].detach().cpu(),
        "observation_std": state["_obs_norm._std"].detach().cpu(),
        "action_mean": state["_a_norm._mean"].detach().cpu(),
        "action_std": state["_a_norm._std"].detach().cpu(),
        "actor": {
            "0.weight": state["_model._actor_layers.0.weight"].detach().cpu(),
            "0.bias": state["_model._actor_layers.0.bias"].detach().cpu(),
            "2.weight": state["_model._actor_layers.2.weight"].detach().cpu(),
            "2.bias": state["_model._actor_layers.2.bias"].detach().cpu(),
            "4.weight": state["_model._action_dist._mean_net.weight"].detach().cpu(),
            "4.bias": state["_model._action_dist._mean_net.bias"].detach().cpu(),
        },
        "metadata": {
            "source_checkpoint": source_checkpoint,
            "source_sha256": source_hash,
            "body_names": SMPL_BODY_NAMES,
            "joint_names": SMPL_JOINT_NAMES,
            "key_body_names": SMPL_KEY_BODY_NAMES,
            "global_observation": True,
            "root_height_observation": True,
            "trajectory_samples": 10,
            "trajectory_interval_s": 0.5,
            "control_frequency_hz": 30,
        },
    }
    # Constructing the runtime model validates all tensor shapes before write.
    MimicKitPedestrianPolicy(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/checkpoints/policy.pth")
    args = parser.parse_args()
    bundle = export_policy(args.checkpoint, args.output)
    print(f"Exported {args.output.resolve()}")
    print(f"Source SHA256: {bundle['metadata']['source_sha256']}")


if __name__ == "__main__":
    main()
