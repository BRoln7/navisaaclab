"""Device-independent DRL-VO observation encoding and per-environment history."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from socnav_sim.tasks.socnav.nav_utils.frame_rotation import world_to_local_2d


def pedestrian_velocity_grid(robot_position, robot_quaternion, robot_velocity,
                             pedestrian_position, pedestrian_velocity, size=80, extent=10.0):
    """Encode relative velocities in a centered, robot-frame 20 m square.

    When multiple pedestrians occupy a cell, retain the closest one. Ignore
    out-of-view pedestrians instead of clamping them onto the image boundary.
    """
    relative = world_to_local_2d(
        pedestrian_position[None, :, :2] - robot_position[:, None, :2], robot_quaternion
    )
    velocity = world_to_local_2d(
        pedestrian_velocity[None, :, :2] - robot_velocity[:, None, :2], robot_quaternion
    )
    cell = torch.floor((relative + extent) * (size / (2 * extent))).long()
    valid = ((cell >= 0) & (cell < size)).all(dim=-1)
    order = relative.square().sum(dim=-1).argsort(dim=1, descending=True)
    grid = relative.new_zeros((len(robot_position), 2, size, size))
    # Far-to-near assignment makes duplicate cells deterministic, including CUDA.
    for rank in range(len(pedestrian_position)):
        ped = order[:, rank]
        robots = torch.arange(len(robot_position), device=relative.device)
        mask = valid[robots, ped]
        selected = cell[robots, ped]
        values = velocity[robots, ped].clamp(-2, 2) / 2
        grid[robots[mask], :, selected[mask, 0], selected[mask, 1]] = values[mask]
    return grid.flatten(1)


def resample_scan(ranges, beams=720, max_range=30.0):
    """Convert a full flat scan to the fixed beam count used by the CNN."""
    if ranges.numel() == 0:
        return ranges.new_full((beams,), max_range)
    ranges = torch.nan_to_num(ranges.flatten(), nan=max_range, posinf=max_range, neginf=max_range)
    ranges = torch.where(ranges > 0, ranges.clamp(max=max_range), max_range)
    if ranges.numel() == beams:
        return ranges
    # Conservative min-pooling retains thin obstacles when reducing resolution.
    return -F.adaptive_max_pool1d(-ranges[None, None], beams).flatten()


class ScanHistory:
    """Ten 720-beam scans, with independent episode resets and step caching."""

    def __init__(self, count, device):
        self.scans = torch.full((count, 10, 720), 30.0, device=device)
        self.previous_episode_steps = torch.full((count,), -1, device=device, dtype=torch.long)
        self.last_step = None

    def observe(self, ranges, step, episode_steps):
        reset = (episode_steps < self.previous_episode_steps) | (self.previous_episode_steps < 0)
        if step != self.last_step:
            self.scans[:, :-1] = self.scans[:, 1:].clone()
            self.scans[:, -1] = ranges
            self.last_step = step
        self.scans[reset] = ranges[reset, None]
        self.previous_episode_steps.copy_(episode_steps)
        groups = self.scans.reshape(len(ranges), 10, 80, 9)
        pooled = torch.cat((groups.amin(dim=-1), groups.mean(dim=-1)), dim=1)
        return (pooled.repeat(1, 4, 1).flatten(1) / 15.0 - 1.0).clamp(-1, 1)
