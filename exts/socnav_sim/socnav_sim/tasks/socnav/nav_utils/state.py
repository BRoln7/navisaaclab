"""Per-environment state for map-based social navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .map_manager import MapInfo, load_map
from .planning import (
    astar_path,
    pad_paths,
    resample_polyline,
    sample_paths,
    smooth_grid_path,
    time_parameterize_path,
)


@dataclass(frozen=True)
class NavigationRuntimeCfg:
    active_robot_count: int = 1
    pedestrian_speed: float = 0.85
    pedestrian_speed_variation: float = 0.15
    fast_pedestrian_probability: float = 0.10
    fast_pedestrian_speed: float = 1.20
    fast_pedestrian_speed_variation: float = 0.15
    pedestrian_acceleration_max: float = 2.0
    pedestrian_turn_rate_max: float = 2.0
    path_spacing: float = 0.10
    robot_goal_distance: tuple[float, float] = (4.0, 8.0)
    pedestrian_goal_distance: tuple[float, float] = (6.0, 10.0)
    map_scale: float = 0.25
    map_dilation: int = 7
    map_image_path: str = ""
    map_yaml_path: str = ""
    parking_origin: tuple[float, float] = (100.0, 100.0)
    minimum_spawn_separation: float = 1.2


class NavigationState:
    """Mutable map and path state owned by one Isaac Lab environment instance."""

    def __init__(self, env, cfg: NavigationRuntimeCfg):
        self.cfg = cfg
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        if not cfg.map_image_path or not cfg.map_yaml_path:
            raise ValueError("NavigationRuntimeCfg requires map_image_path and map_yaml_path")
        self.map: MapInfo = load_map(
            cfg.map_image_path, cfg.map_yaml_path, scale=cfg.map_scale, dilation_size=cfg.map_dilation
        )
        self._occupancy_grid = torch.as_tensor(
            self.map.occupancy, dtype=torch.bool, device=self.device
        )
        active_count = cfg.active_robot_count
        if not 0 < active_count <= self.num_envs:
            raise ValueError(
                f"active_robot_count must be in [1, {self.num_envs}], got {active_count}."
            )
        self.active_robot_count = active_count
        self.active_mask = torch.arange(self.num_envs, device=self.device) < active_count
        self.active_ids = torch.arange(active_count, device=self.device, dtype=torch.long)
        self.rng = np.random.default_rng(getattr(env.cfg, "seed", None))

        self.robot_spawns = torch.zeros((self.num_envs, 2), device=self.device)
        self.robot_goals = torch.zeros_like(self.robot_spawns)
        self.pedestrian_spawns = torch.zeros_like(self.robot_spawns)
        self.pedestrian_goals = torch.zeros_like(self.robot_spawns)
        self.robot_paths = torch.zeros((self.num_envs, 1, 2), device=self.device)
        self.pedestrian_paths = torch.zeros((self.num_envs, 1, 2), device=self.device)
        self.robot_path_lengths = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.pedestrian_path_lengths = torch.ones_like(self.robot_path_lengths)
        self.pedestrian_path_times = torch.zeros((self.num_envs, 1), device=self.device)
        self.pedestrian_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.pedestrian_progress = torch.zeros(self.num_envs, device=self.device)
        self.pedestrian_speeds = torch.full(
            (self.num_envs,), cfg.pedestrian_speed, dtype=torch.float32, device=self.device
        )
        self.last_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.nearest_pedestrian_clearance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.initialized = False

        if cfg.pedestrian_speed <= cfg.pedestrian_speed_variation:
            raise ValueError("pedestrian_speed must be greater than pedestrian_speed_variation")
        if cfg.fast_pedestrian_speed <= cfg.fast_pedestrian_speed_variation:
            raise ValueError(
                "fast_pedestrian_speed must be greater than fast_pedestrian_speed_variation"
            )
        if not 0.0 <= cfg.fast_pedestrian_probability <= 1.0:
            raise ValueError("fast_pedestrian_probability must be in [0, 1]")
        if cfg.pedestrian_acceleration_max <= 0 or cfg.pedestrian_turn_rate_max <= 0:
            raise ValueError("Pedestrian acceleration and turn-rate limits must be positive")

    @property
    def parking_positions(self) -> torch.Tensor:
        positions = torch.empty((self.num_envs, 2), device=self.device)
        positions[:, 0] = self.cfg.parking_origin[0] + torch.arange(self.num_envs, device=self.device) * 2.0
        positions[:, 1] = self.cfg.parking_origin[1]
        return positions

    def _plan_one(self, start_world: np.ndarray, distance_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        start_cell = self.map.world_to_map_numpy(start_world)
        for _ in range(64):
            goal_world = self.map.sample_goal(start_world, *distance_range, self.rng)
            goal_cell = self.map.world_to_map_numpy(goal_world)
            grid_path = astar_path(self.map.occupancy, start_cell, goal_cell)
            if grid_path is not None and len(grid_path) > 1:
                smooth_path = smooth_grid_path(grid_path, self.map.occupancy)
                world_path = self.map.map_to_world_numpy(smooth_path)
                return goal_world, resample_polyline(world_path, self.cfg.path_spacing)
        raise RuntimeError(f"Unable to find a path from {start_world.tolist()} after 64 attempts.")

    def _sample_spawns(self, count: int, occupied: np.ndarray | None = None) -> np.ndarray:
        occupied_points = [] if occupied is None else [point for point in np.asarray(occupied)]
        selected: list[np.ndarray] = []
        free = self.map.free_cells_xy
        for _ in range(count):
            choice = None
            for _ in range(256):
                cell = free[int(self.rng.integers(free.shape[0]))]
                candidate = self.map.map_to_world_numpy(cell)
                references = occupied_points + selected
                if not references or np.min(np.linalg.norm(np.asarray(references) - candidate, axis=-1)) >= self.cfg.minimum_spawn_separation:
                    choice = candidate
                    break
            if choice is None:
                choice = self.map.map_to_world_numpy(free[int(self.rng.integers(free.shape[0]))])
            selected.append(choice)
        return np.asarray(selected, dtype=np.float32)

    def _replace_paths(
        self,
        current_paths: torch.Tensor,
        current_lengths: torch.Tensor,
        env_ids: torch.Tensor,
        new_paths: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_padded, new_lengths = pad_paths(new_paths, device=self.device)
        final_length = max(current_paths.shape[1], new_padded.shape[1])
        combined = torch.empty((self.num_envs, final_length, 2), dtype=torch.float32, device=self.device)
        combined[:] = current_paths[:, -1:].expand(-1, final_length, -1)
        combined[:, : current_paths.shape[1]] = current_paths
        combined[env_ids] = new_padded[:, -1:].expand(-1, final_length, -1)
        combined[env_ids, : new_padded.shape[1]] = new_padded
        lengths = current_lengths.clone()
        lengths[env_ids] = new_lengths
        return combined, lengths

    def _replace_path_times(
        self, env_ids: torch.Tensor, new_times: list[np.ndarray]
    ) -> torch.Tensor:
        final_length = max(self.pedestrian_path_times.shape[1], max(map(len, new_times)))
        combined = torch.empty((self.num_envs, final_length), dtype=torch.float32, device=self.device)
        combined[:] = self.pedestrian_path_times[:, -1:].expand(-1, final_length)
        combined[:, : self.pedestrian_path_times.shape[1]] = self.pedestrian_path_times
        for row, times in zip(env_ids.tolist(), new_times):
            tensor = torch.as_tensor(times, dtype=torch.float32, device=self.device)
            combined[row] = tensor[-1]
            combined[row, : tensor.shape[0]] = tensor
        return combined

    def initialize(self) -> None:
        if self.initialized:
            return
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_pedestrians(all_ids, sample_spawn=True)
        self.reset_robots(all_ids, sample_spawn=True)
        self.initialized = True

    def reset_robots(self, env_ids: torch.Tensor, *, sample_spawn: bool = True, current_positions=None) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        active_ids = env_ids[self.active_mask[env_ids]]
        inactive_ids = env_ids[~self.active_mask[env_ids]]
        if inactive_ids.numel():
            parking = self.parking_positions[inactive_ids]
            self.robot_spawns[inactive_ids] = parking
            self.robot_goals[inactive_ids] = parking
        if not active_ids.numel():
            return

        if sample_spawn:
            occupied = self.pedestrian_spawns.detach().cpu().numpy()
            if self.initialized:
                keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                keep[active_ids] = False
                occupied = np.concatenate((occupied, self.robot_spawns[keep].detach().cpu().numpy()))
            starts = self._sample_spawns(active_ids.numel(), occupied)
        else:
            starts = current_positions[active_ids].detach().cpu().numpy()

        paths: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        for start in starts:
            goal, path = self._plan_one(start, self.cfg.robot_goal_distance)
            goals.append(goal)
            paths.append(path)
        self.robot_spawns[active_ids] = torch.as_tensor(starts, device=self.device)
        self.robot_goals[active_ids] = torch.as_tensor(np.stack(goals), device=self.device)
        self.robot_paths, self.robot_path_lengths = self._replace_paths(
            self.robot_paths, self.robot_path_lengths, active_ids, paths
        )
        self.last_goal_distance[active_ids] = 0.0

    def reset_pedestrians(self, env_ids: torch.Tensor, *, sample_spawn: bool = True, current_positions=None) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if not env_ids.numel():
            return
        if sample_spawn:
            occupied = None
            if self.initialized:
                keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                keep[env_ids] = False
                occupied = np.concatenate(
                    (self.pedestrian_spawns[keep].detach().cpu().numpy(),
                     self.robot_spawns[self.active_mask].detach().cpu().numpy())
                )
            starts = self._sample_spawns(env_ids.numel(), occupied)
        else:
            starts = current_positions[env_ids].detach().cpu().numpy()

        paths: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        for start in starts:
            goal, path = self._plan_one(start, self.cfg.pedestrian_goal_distance)
            goals.append(goal)
            paths.append(path)
        self.pedestrian_spawns[env_ids] = torch.as_tensor(starts, device=self.device)
        self.pedestrian_goals[env_ids] = torch.as_tensor(np.stack(goals), device=self.device)
        self.pedestrian_paths, self.pedestrian_path_lengths = self._replace_paths(
            self.pedestrian_paths, self.pedestrian_path_lengths, env_ids, paths
        )
        self._resample_pedestrian_speeds(env_ids)
        cruise_speeds = self.pedestrian_speeds[env_ids].detach().cpu().numpy()
        path_times = [
            time_parameterize_path(
                path,
                float(speed),
                acceleration_max=self.cfg.pedestrian_acceleration_max,
                turn_rate_max=self.cfg.pedestrian_turn_rate_max,
            )
            for path, speed in zip(paths, cruise_speeds)
        ]
        self.pedestrian_path_times = self._replace_path_times(env_ids, path_times)
        self.pedestrian_elapsed[env_ids] = 0.0
        self.pedestrian_progress[env_ids] = 0.0

    def _resample_pedestrian_speeds(self, env_ids: torch.Tensor) -> None:
        """Sample mostly normal indoor walking speeds with a small fast-walker group."""
        count = int(env_ids.numel())
        speeds = self.rng.uniform(
            self.cfg.pedestrian_speed - self.cfg.pedestrian_speed_variation,
            self.cfg.pedestrian_speed + self.cfg.pedestrian_speed_variation,
            size=count,
        ).astype(np.float32)
        fast = self.rng.random(count) < self.cfg.fast_pedestrian_probability
        if np.any(fast):
            speeds[fast] = self.rng.uniform(
                self.cfg.fast_pedestrian_speed - self.cfg.fast_pedestrian_speed_variation,
                self.cfg.fast_pedestrian_speed + self.cfg.fast_pedestrian_speed_variation,
                size=int(np.count_nonzero(fast)),
            )
        self.pedestrian_speeds[env_ids] = torch.as_tensor(speeds, device=self.device)

    def _sample_pedestrian_times(self, query_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = query_times.ndim == 1
        if squeeze:
            query_times = query_times[:, None]
        high = torch.searchsorted(
            self.pedestrian_path_times.contiguous(), query_times.contiguous(), right=True
        )
        maximum = (self.pedestrian_path_lengths - 1)[:, None]
        high = torch.minimum(torch.clamp_min(high, 1), maximum)
        low = high - 1
        rows = torch.arange(self.num_envs, device=self.device)[:, None].expand_as(low)
        low_time = self.pedestrian_path_times[rows, low]
        high_time = self.pedestrian_path_times[rows, high]
        blend = torch.clamp(
            (query_times - low_time) / torch.clamp_min(high_time - low_time, 1e-6), 0.0, 1.0
        )
        low_position = sample_paths(self.pedestrian_paths, self.pedestrian_path_lengths, low)
        high_position = sample_paths(self.pedestrian_paths, self.pedestrian_path_lengths, high)
        position = torch.lerp(low_position, high_position, blend.unsqueeze(-1))
        progress = low + blend
        if squeeze:
            return position[:, 0], progress[:, 0]
        return position, progress

    def pedestrian_targets(self, sample_count: int = 10, sample_interval: float = 0.5) -> torch.Tensor:
        offsets = torch.arange(sample_count, device=self.device)[None, :] * sample_interval
        targets, _ = self._sample_pedestrian_times(self.pedestrian_elapsed[:, None] + offsets)
        return targets

    def advance_pedestrians(self, control_dt: float) -> torch.Tensor:
        self.pedestrian_elapsed += control_dt
        _, self.pedestrian_progress[:] = self._sample_pedestrian_times(self.pedestrian_elapsed)
        return self.pedestrian_progress >= self.pedestrian_path_lengths - 2

    def robot_subgoals(self, lookahead_points: int = 20) -> torch.Tensor:
        if self.active_robot_count == 0:
            return torch.empty((0, 2), device=self.device)
        positions = self._robot_positions_for_subgoal
        paths = self.robot_paths[: self.active_robot_count]
        lengths = self.robot_path_lengths[: self.active_robot_count]
        distances = torch.linalg.norm(paths - positions[:, None], dim=-1)
        mask = torch.arange(paths.shape[1], device=self.device)[None, :] >= lengths[:, None]
        distances.masked_fill_(mask, torch.inf)
        nearest = torch.argmin(distances, dim=1)
        return sample_paths(paths, lengths, nearest + lookahead_points)

    def set_robot_positions_for_subgoal(self, positions: torch.Tensor) -> None:
        self._robot_positions_for_subgoal = positions[: self.active_robot_count, :2]

    def map_collision_mask(self, positions_world: torch.Tensor) -> torch.Tensor:
        """Return whether each world-space position is occupied or outside the map."""
        positions = positions_world.to(device=self.device)
        origin = positions.new_tensor(self.map.origin[:2])
        cells = torch.round((positions[..., :2] - origin) / self.map.resolution).to(torch.long)
        x, y = cells.unbind(dim=-1)
        rows, columns = self._occupancy_grid.shape
        inside = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
        collision = torch.ones_like(inside)
        collision[inside] = self._occupancy_grid[y[inside], x[inside]]
        return collision


def get_navigation_state(env, cfg: NavigationRuntimeCfg | None = None) -> NavigationState:
    state = getattr(env, "_socnav_navigation_state", None)
    if state is None:
        if cfg is None:
            raise RuntimeError("Navigation state must be initialized with a runtime configuration")
        state = NavigationState(env, cfg)
        setattr(env, "_socnav_navigation_state", state)
    elif cfg is not None and state.cfg != cfg:
        raise RuntimeError(f"Inconsistent navigation configuration: {state.cfg!r} != {cfg!r}")
    return state
