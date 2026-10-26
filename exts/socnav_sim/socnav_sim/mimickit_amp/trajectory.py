"""PACER-compatible stochastic trajectory generation and observations."""

import math

import torch
from util import torch_util


class Trajectory:
    """Vectorized port of PACER's ``TrajGenerator``.

    The sampling order and interpolation convention intentionally match PACER.
    In particular, the first heading is sampled independently of the character
    heading and ``duration`` is reported as ``num_vertices * segment_dt`` by
    PACER even though the vertices span one fewer segment.
    """

    def __init__(
        self,
        count,
        episode_duration,
        device,
        num_vertices=101,
        heading_change_max=2.0,
        speed_min=0.0,
        speed_max=3.0,
        acceleration_max=2.0,
        sharp_turn_probability=0.02,
        observation_samples=10,
        observation_timestep=0.5,
    ):
        if count <= 0 or episode_duration <= 0 or num_vertices < 2:
            raise ValueError("Trajectory count, duration, and vertex count must be positive")
        if not 0 <= speed_min <= speed_max:
            raise ValueError("Invalid trajectory speed range")
        if heading_change_max < 0 or acceleration_max < 0:
            raise ValueError("Trajectory turn and acceleration limits must be nonnegative")
        if not 0 <= sharp_turn_probability <= 1:
            raise ValueError("sharp_turn_probability must be in [0, 1]")
        if observation_samples <= 0 or observation_timestep <= 0:
            raise ValueError("Trajectory observation settings must be positive")

        self.segment_dt = float(episode_duration) / (int(num_vertices) - 1)
        self.duration = int(num_vertices) * self.segment_dt
        self.count = int(num_vertices)
        self.paths = torch.zeros(count, self.count, 3, dtype=torch.float32, device=device)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.acceleration_max = float(acceleration_max)
        self.heading_change_max = float(heading_change_max)
        self.sharp_turn_probability = float(sharp_turn_probability)
        self.observation_samples = int(observation_samples)
        self.observation_timestep = float(observation_timestep)

    @torch.no_grad()
    def reset(self, ids, positions, rotations=None):
        """Generate new paths for ``ids`` using PACER's random draw order."""
        del rotations  # PACER samples the initial direction independently.
        n = len(ids)
        if n == 0:
            return
        device = self.paths.device
        segments = self.count - 1

        heading_delta = 2 * torch.rand((n, segments), device=device) - 1
        heading_delta *= self.heading_change_max * self.segment_dt
        sharp_heading = math.pi * (2 * torch.rand((n, segments), device=device) - 1)
        sharp_mask = torch.bernoulli(torch.full_like(heading_delta, self.sharp_turn_probability)).bool()
        heading_delta[sharp_mask] = sharp_heading[sharp_mask]
        heading_delta[:, 0] = math.pi * (2 * torch.rand(n, device=device) - 1)

        speed_delta = 2 * torch.rand((n, segments), device=device) - 1
        speed_delta *= self.acceleration_max * self.segment_dt
        speed_delta[:, 0] = (
            (self.speed_max - self.speed_min) * torch.rand(n, device=device) + self.speed_min
        )
        speed = torch.zeros_like(speed_delta)
        speed[:, 0] = speed_delta[:, 0]
        for i in range(1, segments):
            speed[:, i] = torch.clamp(
                speed[:, i - 1] + speed_delta[:, i], self.speed_min, self.speed_max
            )

        heading = torch.cumsum(heading_delta, dim=-1)
        segment_length = speed * self.segment_dt
        displacement = torch.stack(
            (torch.cos(heading), -torch.sin(heading), torch.zeros_like(heading)), dim=-1
        )
        displacement *= segment_length.unsqueeze(-1)
        displacement[:, 0, :2] += positions[:, :2]
        vertices = torch.cumsum(displacement, dim=1)

        # PACER stores a ground-plane trajectory and only consumes XY.
        self.paths[ids, 0, :2] = positions[:, :2]
        self.paths[ids, 0, 2] = 0
        self.paths[ids, 1:] = vertices

    def position(self, ids, times):
        phase = torch.clamp(times / self.duration, 0.0, 1.0)
        segment_index = phase * (self.count - 1)
        low = torch.floor(segment_index).long()
        high = torch.ceil(segment_index).long()
        factor = (segment_index - low).unsqueeze(-1)
        flat = self.paths.reshape(-1, 3)
        offset = ids * self.count
        return (1 - factor) * flat[offset + low] + factor * flat[offset + high]

    def samples(self, ids, times):
        """Return the moving PACER target window used by the policy.

        The first point is the target at the current environment time.  The
        remaining points look ahead at ``observation_timestep`` intervals, so
        the whole marker window advances on every control step.
        """
        offsets = torch.arange(self.observation_samples, device=times.device, dtype=times.dtype)
        future = times[:, None] + offsets * self.observation_timestep
        tiled_ids = ids[:, None].expand_as(future)
        return self.position(tiled_ids, future)

    def observation(self, ids, positions, rotations, times):
        targets = self.samples(ids, times)
        inverse_heading = torch_util.calc_heading_quat_inv(rotations)
        inverse_heading = inverse_heading[:, None].expand(-1, self.observation_samples, -1)
        local = torch_util.quat_rotate(inverse_heading, targets - positions[:, None])
        return local[..., :2].flatten(1)
