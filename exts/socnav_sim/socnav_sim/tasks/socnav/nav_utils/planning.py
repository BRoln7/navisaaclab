"""CPU path planning utilities for the CADRL examples."""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch


_NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)


def astar_path(occupancy: np.ndarray, start_xy, goal_xy) -> np.ndarray | None:
    """Return an ``[L, 2]`` x/y grid path, or ``None`` when unreachable."""
    start = tuple(np.asarray(start_xy, dtype=np.int64).tolist())
    goal = tuple(np.asarray(goal_xy, dtype=np.int64).tolist())
    rows, cols = occupancy.shape

    def free(node: tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < cols and 0 <= y < rows and occupancy[y, x] == 0

    if not free(start) or not free(goal):
        return None
    if start == goal:
        return np.asarray([start], dtype=np.float32)

    queue = [(0.0, 0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = parent[current]
                path.append(current)
            path.reverse()
            return np.asarray(path, dtype=np.float32)
        closed.add(current)

        cx, cy = current
        for dx, dy, step_cost in _NEIGHBORS:
            neighbor = (cx + dx, cy + dy)
            if not free(neighbor):
                continue
            # Prevent diagonal paths from cutting through obstacle corners.
            if dx and dy and (not free((cx + dx, cy)) or not free((cx, cy + dy))):
                continue
            new_cost = current_cost + step_cost
            if new_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = new_cost
            parent[neighbor] = current
            heuristic = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
            heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
    return None


def _segment_is_free(occupancy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
    """Check every grid cell touched by a continuous segment."""
    # Grid points denote cell centers, so shifting by half a cell lets the
    # standard voxel traversal operate on integer cell boundaries.
    start = np.asarray(start_xy, dtype=np.float64) + 0.5
    end = np.asarray(end_xy, dtype=np.float64) + 0.5
    delta = end - start
    cell = np.floor(start).astype(np.int64)
    final = np.floor(end).astype(np.int64)
    step = np.sign(delta).astype(np.int64)
    inverse = np.full(2, np.inf, dtype=np.float64)
    moving = np.abs(delta) > 1e-12
    inverse[moving] = 1.0 / np.abs(delta[moving])
    boundary = np.where(step > 0, cell + 1.0, cell.astype(np.float64))
    maximum = np.full(2, np.inf, dtype=np.float64)
    maximum[moving] = np.abs(boundary[moving] - start[moving]) * inverse[moving]

    rows, cols = occupancy.shape

    def free(candidate: np.ndarray) -> bool:
        x, y = candidate.tolist()
        return 0 <= x < cols and 0 <= y < rows and occupancy[y, x] == 0

    if not free(cell):
        return False
    while not np.array_equal(cell, final):
        if maximum[0] < maximum[1]:
            cell[0] += step[0]
            maximum[0] += inverse[0]
        elif maximum[1] < maximum[0]:
            cell[1] += step[1]
            maximum[1] += inverse[1]
        else:
            # A corner crossing touches both side cells as well as the
            # diagonal one; rejecting occupied side cells prevents clipping.
            side_x = cell + np.array((step[0], 0), dtype=np.int64)
            side_y = cell + np.array((0, step[1]), dtype=np.int64)
            if not free(side_x) or not free(side_y):
                return False
            cell += step
            maximum += inverse
        if not free(cell):
            return False
    return True


def simplify_grid_path(path: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Remove A* stair steps while retaining collision-free line segments."""
    points = np.asarray(path, dtype=np.float32)
    if points.shape[0] <= 2:
        return points.copy()
    simplified = [points[0]]
    anchor = 0
    while anchor < points.shape[0] - 1:
        candidate = points.shape[0] - 1
        while candidate > anchor + 1:
            if _segment_is_free(occupancy, points[anchor], points[candidate]):
                break
            candidate -= 1
        simplified.append(points[candidate])
        anchor = candidate
    return np.asarray(simplified, dtype=np.float32)


def smooth_grid_path(path: np.ndarray, occupancy: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Round polyline corners with collision-checked Chaikin subdivisions."""
    current = simplify_grid_path(path, occupancy)
    for _ in range(max(0, iterations)):
        if current.shape[0] <= 2:
            break
        candidate = [current[0]]
        for start, end in zip(current[:-1], current[1:]):
            candidate.extend((0.75 * start + 0.25 * end, 0.25 * start + 0.75 * end))
        candidate.append(current[-1])
        candidate = np.asarray(candidate, dtype=np.float32)
        if not all(
            _segment_is_free(occupancy, start, end)
            for start, end in zip(candidate[:-1], candidate[1:])
        ):
            break
        current = candidate
    return current


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at approximately uniform metric spacing."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return points.copy()
    segments = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    length = float(cumulative[-1])
    if length <= 1e-6:
        return points[:1].copy()
    distances = np.arange(0.0, length, spacing, dtype=np.float32)
    if distances.size == 0 or distances[-1] < length:
        distances = np.append(distances, length)
    return np.column_stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in range(2)]
    ).astype(np.float32)


def time_parameterize_path(
    points: np.ndarray,
    cruise_speed: float,
    *,
    acceleration_max: float = 2.0,
    turn_rate_max: float = 2.0,
    minimum_speed: float = 0.10,
) -> np.ndarray:
    """Compute arrival times with PACER-compatible acceleration/turn limits.

    The geometry remains on the collision-checked map path. Speed is reduced
    before corners, then recovered with a forward/backward acceleration pass.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.float32)
    if cruise_speed <= 0 or acceleration_max <= 0 or turn_rate_max <= 0:
        raise ValueError("Trajectory speed, acceleration and turn-rate limits must be positive")

    displacement = np.diff(points, axis=0)
    segment_length = np.linalg.norm(displacement, axis=-1)
    segment_length = np.maximum(segment_length, 1e-6)
    heading = np.arctan2(displacement[:, 1], displacement[:, 0])
    turn = np.abs((np.diff(heading) + np.pi) % (2.0 * np.pi) - np.pi)
    local_distance = 0.5 * (segment_length[:-1] + segment_length[1:])
    curvature = turn / np.maximum(local_distance, 1e-6)

    speed = np.full(points.shape[0], float(cruise_speed), dtype=np.float64)
    if curvature.size:
        corner_limit = turn_rate_max / np.maximum(curvature, 1e-6)
        speed[1:-1] = np.minimum(speed[1:-1], corner_limit)
    speed = np.maximum(speed, min(float(minimum_speed), float(cruise_speed)))

    # Enforce |dv/dt| <= acceleration_max in both travel directions.
    for index, distance in enumerate(segment_length, start=1):
        reachable = math.sqrt(speed[index - 1] ** 2 + 2.0 * acceleration_max * distance)
        speed[index] = min(speed[index], reachable)
    for index in range(points.shape[0] - 2, -1, -1):
        reachable = math.sqrt(speed[index + 1] ** 2 + 2.0 * acceleration_max * segment_length[index])
        speed[index] = min(speed[index], reachable)

    segment_dt = 2.0 * segment_length / np.maximum(speed[:-1] + speed[1:], 1e-6)
    return np.concatenate(([0.0], np.cumsum(segment_dt))).astype(np.float32)


def pad_paths(paths: list[np.ndarray], *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    if not paths:
        return torch.empty((0, 1, 2), device=device), torch.empty((0,), dtype=torch.long, device=device)
    lengths = torch.tensor([len(path) for path in paths], dtype=torch.long, device=device)
    padded = torch.empty((len(paths), int(lengths.max().item()), 2), dtype=torch.float32, device=device)
    for index, path in enumerate(paths):
        tensor = torch.as_tensor(path, dtype=torch.float32, device=device)
        padded[index] = tensor[-1]
        padded[index, : tensor.shape[0]] = tensor
    return padded, lengths


def sample_paths(paths: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Sample each padded path at integer indices shaped ``[N]`` or ``[N, K]``."""
    if indices.ndim == 1:
        clamped = torch.minimum(indices, lengths - 1)
        return paths[torch.arange(paths.shape[0], device=paths.device), clamped]
    clamped = torch.minimum(indices, (lengths - 1).unsqueeze(-1))
    rows = torch.arange(paths.shape[0], device=paths.device).unsqueeze(-1).expand_as(clamped)
    return paths[rows, clamped]


# Compatibility wrapper for tools that used the former function name.
def get_dijkstra_path(grid, _image, start, goal, viz=False, traj_optim=False):
    del viz, traj_optim
    return astar_path(grid, start, goal)


def path_align_length(path: np.ndarray, trajectory_length: int, *, device="cpu") -> torch.Tensor:
    tensor = torch.as_tensor(path, dtype=torch.float32, device=device)
    if tensor.shape[0] >= trajectory_length:
        return tensor[:trajectory_length]
    padding = tensor[-1].expand(trajectory_length - tensor.shape[0], -1)
    return torch.cat((tensor, padding), dim=0)
