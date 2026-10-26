"""Portable deterministic actor used by the navigation pedestrians.

The file format contains only tensors and plain metadata, so it can be loaded
with ``torch.load(..., weights_only=True)`` without importing MimicKit.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


POLICY_FORMAT = "socnav.mimickit_smpl_actor.v1"
SMPL_BODY_NAMES = (
    "Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe "
    "Torso Spine Chest Neck Head L_Thorax L_Shoulder L_Elbow L_Wrist L_Hand "
    "R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand"
).split()
SMPL_JOINT_NAMES = [f"{body}_{axis}" for body in SMPL_BODY_NAMES[1:] for axis in "xyz"]
SMPL_KEY_BODY_NAMES = ["L_Toe", "R_Toe", "Head", "L_Hand", "R_Hand"]


class MimicKitPedestrianPolicy(nn.Module):
    """MimicKit AMP actor with its frozen observation/action normalizers."""

    def __init__(self, bundle: dict):
        super().__init__()
        if bundle.get("format") != POLICY_FORMAT:
            raise ValueError(f"Unsupported pedestrian policy format: {bundle.get('format')!r}")
        if bundle.get("observation_size") != 255 or bundle.get("action_size") != 69:
            raise ValueError("Expected the 255-observation, 69-action SMPL tracking actor")

        self.actor = nn.Sequential(
            nn.Linear(255, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 69)
        )
        self.actor.load_state_dict(bundle["actor"])
        self.register_buffer("observation_mean", bundle["observation_mean"].float())
        self.register_buffer("observation_std", bundle["observation_std"].float())
        self.register_buffer("action_mean", bundle["action_mean"].float())
        self.register_buffer("action_std", bundle["action_std"].float())
        self.metadata = bundle["metadata"]
        if self.metadata.get("body_names") != SMPL_BODY_NAMES:
            raise ValueError("Policy body order does not match the official MimicKit SMPL asset")
        if self.metadata.get("joint_names") != SMPL_JOINT_NAMES:
            raise ValueError("Policy joint order does not match the official MimicKit SMPL asset")
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device) -> "MimicKitPedestrianPolicy":
        return cls(torch.load(path, map_location="cpu", weights_only=True)).to(device)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.observation_mean) / self.observation_std, -10.0, 10.0
        )
        normalized_action = self.actor(normalized)
        action = normalized_action * self.action_std + self.action_mean
        return torch.clamp(action, self.action_mean - self.action_std, self.action_mean + self.action_std)


def _quat_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    return vector * (2.0 * w.square() - 1.0) + 2.0 * w * torch.cross(xyz, vector, dim=-1) + 2.0 * xyz * (xyz * vector).sum(dim=-1, keepdim=True)


def _quat_to_tangent_normal(quaternion: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(quaternion[..., :3])
    tangent[..., 0] = 1.0
    normal = torch.zeros_like(tangent)
    normal[..., 2] = 1.0
    return torch.cat((_quat_rotate_xyzw(quaternion, tangent), _quat_rotate_xyzw(quaternion, normal)), dim=-1)


def _exp_map_to_quaternion(exp_map: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(exp_map, dim=-1, keepdim=True)
    axis = exp_map / torch.clamp_min(angle, 1e-8)
    xyz = axis * torch.sin(0.5 * angle)
    quaternion = torch.cat((xyz, torch.cos(0.5 * angle)), dim=-1)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(angle > 1e-5, quaternion, identity)


def _inverse_heading(quaternion: torch.Tensor) -> torch.Tensor:
    forward = torch.zeros_like(quaternion[..., :3])
    forward[..., 0] = 1.0
    direction = _quat_rotate_xyzw(quaternion, forward)
    half_heading = -0.5 * torch.atan2(direction[..., 1], direction[..., 0])
    result = torch.zeros_like(quaternion)
    result[..., 2] = torch.sin(half_heading)
    result[..., 3] = torch.cos(half_heading)
    return result


def build_mimickit_observation(
    *,
    root_position: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    key_body_position: torch.Tensor,
    trajectory_xy: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the exact global-observation layout used by the checkpoint."""
    root_quaternion = root_quaternion_wxyz[..., [1, 2, 3, 0]]
    joint_quaternion = _exp_map_to_quaternion(joint_position.reshape(-1, 3))
    joint_rotation = _quat_to_tangent_normal(joint_quaternion).reshape(joint_position.shape[0], -1)
    key_position = (key_body_position - root_position[:, None]).flatten(1)
    character = torch.cat(
        (
            root_position[:, 2:3],
            _quat_to_tangent_normal(root_quaternion),
            root_linear_velocity,
            root_angular_velocity,
            joint_rotation,
            joint_velocity,
            key_position,
        ),
        dim=-1,
    )
    target_delta = torch.cat(
        (trajectory_xy - root_position[:, None, :2], torch.zeros_like(trajectory_xy[..., :1])), dim=-1
    )
    heading = _inverse_heading(root_quaternion)[:, None].expand(-1, trajectory_xy.shape[1], -1)
    task = _quat_rotate_xyzw(heading, target_delta)[..., :2].flatten(1)
    observation = torch.cat((character, task), dim=-1)
    if observation.shape[-1] != 255:
        raise RuntimeError(f"MimicKit pedestrian policy expects 255 observations, got {observation.shape[-1]}")
    return observation
