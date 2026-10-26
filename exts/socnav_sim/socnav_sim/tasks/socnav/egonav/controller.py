"""Small path-following controller used by the EgoNav recording demo."""

from __future__ import annotations

import torch


def quaternion_yaw(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return planar yaw for one or more ``(w, x, y, z)`` quaternions."""
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def path_following_actions(
    root_position: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
    paths: torch.Tensor,
    path_lengths: torch.Tensor,
    *,
    active_count: int = 1,
    path_spacing: float = 0.10,
    lookahead_distance: float = 0.80,
    maximum_linear_speed: float = 0.42,
    angular_gain: float = 1.8,
) -> torch.Tensor:
    """Convert global map paths into normalized Jackal actions.

    CADRL's action term maps action 0 from ``[-1, 1]`` to ``[0, 0.5]`` m/s
    and action 1 from ``[-1, 1]`` to ``[-2, 2]`` rad/s.  The controller uses
    the closest global-path waypoint plus a fixed look-ahead, so it follows
    obstacle-safe paths rather than steering directly toward the final goal.
    """
    count = root_position.shape[0]
    if not 0 <= active_count <= count:
        raise ValueError(f"active_count must be in [0, {count}], got {active_count}")
    if path_spacing <= 0 or lookahead_distance < 0:
        raise ValueError("path_spacing must be positive and lookahead_distance non-negative")

    actions = torch.empty((count, 2), dtype=root_position.dtype, device=root_position.device)
    actions[:, 0] = -1.0  # zero physical linear velocity
    actions[:, 1] = 0.0
    yaw = quaternion_yaw(root_quaternion_wxyz)
    lookahead_steps = max(1, round(lookahead_distance / path_spacing))

    for index in range(active_count):
        length = int(path_lengths[index].item())
        if length <= 0:
            continue
        path = paths[index, :length]
        closest = int(torch.argmin(torch.linalg.vector_norm(path - root_position[index, :2], dim=-1)).item())
        target = path[min(closest + lookahead_steps, length - 1)]
        direction = target - root_position[index, :2]
        desired_yaw = torch.atan2(direction[1], direction[0])
        yaw_error = torch.atan2(torch.sin(desired_yaw - yaw[index]), torch.cos(desired_yaw - yaw[index]))

        # Slow down through sharp turns and rotate in place if the target is
        # behind the camera.  This avoids the oscillatory stop-go behavior of
        # a constant-speed heading controller.
        alignment = torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)
        linear_speed = maximum_linear_speed * alignment.square()
        angular_speed = torch.clamp(angular_gain * yaw_error, min=-2.0, max=2.0)
        actions[index, 0] = torch.clamp(linear_speed / 0.25 - 1.0, min=-1.0, max=1.0)
        actions[index, 1] = angular_speed / 2.0

    return actions
