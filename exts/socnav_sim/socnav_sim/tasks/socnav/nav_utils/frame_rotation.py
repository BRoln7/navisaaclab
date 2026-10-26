"""Small, device-independent frame transformation helpers."""

from __future__ import annotations

import torch


def quat_to_yaw(quaternions: torch.Tensor) -> torch.Tensor:
    """Return yaw from scalar-first ``[w, x, y, z]`` quaternions."""
    w, x, y, z = quaternions.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def world_to_local_2d(vectors: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame x/y vectors into each agent's local frame."""
    yaw = quat_to_yaw(quaternions)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x = vectors[..., 0]
    y = vectors[..., 1]
    while cos_yaw.ndim < x.ndim:
        cos_yaw = cos_yaw.unsqueeze(-1)
        sin_yaw = sin_yaw.unsqueeze(-1)
    return torch.stack((cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y), dim=-1)


def quat_to_rot_matrix(quaternions: torch.Tensor, _num_robots: int | None = None) -> torch.Tensor:
    w, x, y, z = quaternions.unbind(dim=-1)
    matrix = quaternions.new_empty((*quaternions.shape[:-1], 3, 3))
    matrix[..., 0, 0] = 1 - 2 * (y.square() + z.square())
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)
    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x.square() + z.square())
    matrix[..., 1, 2] = 2 * (y * z - w * x)
    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x.square() + y.square())
    return matrix


def rotation_transform(relative_positions: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    return world_to_local_2d(relative_positions[..., :2], quaternions)
