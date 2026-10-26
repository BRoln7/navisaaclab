from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
import yaml

from socnav_sim.tasks.socnav.nav_utils.frame_rotation import world_to_local_2d
from socnav_sim.tasks.socnav.nav_utils.map_manager import get_navigation_scene_assets, load_map
from socnav_sim.tasks.socnav.nav_utils.planning import (
    astar_path,
    resample_polyline,
    smooth_grid_path,
    time_parameterize_path,
)
from socnav_sim.tasks.socnav.nav_utils.state import NavigationRuntimeCfg, NavigationState


class _Cfg:
    seed = 7


class _Env:
    num_envs = 8
    device = "cpu"
    cfg = _Cfg()


@pytest.fixture
def synthetic_map(tmp_path):
    """Small connected map for CI; no private scene assets are required."""
    from types import SimpleNamespace
    image = np.full((240, 240), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    image[100:140, 100:140] = 0
    png = tmp_path / "map.png"
    metadata = tmp_path / "map.yaml"
    Image.fromarray(image).save(png)
    metadata.write_text(yaml.safe_dump(dict(
        image="map.png", resolution=0.1, origin=[-12.0, -12.0, 0.0],
        negate=0, occupied_thresh=0.65, free_thresh=0.196,
    )))
    return SimpleNamespace(map_image=png, map_yaml=metadata)


def map_assets(scene_id, synthetic_map):
    if scene_id is None:
        return synthetic_map
    try:
        return get_navigation_scene_assets(scene_id)
    except FileNotFoundError:
        pytest.skip(f"Optional scene {scene_id} data bundle is not installed")


@pytest.mark.parametrize("scene_id", [None, 12, 27, 34, 35])
def test_scene_map_coordinate_roundtrip_and_path(scene_id, synthetic_map):
    assets = map_assets(scene_id, synthetic_map)
    scene_map = load_map(assets.map_image, assets.map_yaml, scale=0.25, dilation_size=4)
    rng = np.random.default_rng(3)
    start = scene_map.sample_free_cells(1, rng)[0]
    start_world = scene_map.map_to_world_numpy(start)
    goal_world = scene_map.sample_goal(start_world, 4.0, 8.0, rng)
    goal = scene_map.world_to_map_numpy(goal_world)

    np.testing.assert_array_equal(scene_map.world_to_map_numpy(start_world), start)
    path = astar_path(scene_map.occupancy, start, goal)
    assert path is not None and path.shape[0] > 1
    smooth = smooth_grid_path(path, scene_map.occupancy)
    sampled = resample_polyline(scene_map.map_to_world_numpy(smooth), spacing=0.1)
    assert np.linalg.norm(sampled[0] - start_world) < 1e-5
    assert np.linalg.norm(sampled[-1] - goal_world) < 0.3
    sampled_cells = scene_map.world_to_map_numpy(sampled)
    assert np.all(scene_map.occupancy[sampled_cells[:, 1], sampled_cells[:, 0]] == 0)


def test_path_time_parameterization_slows_for_corner_and_is_monotonic():
    path = resample_polyline(
        np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float32), spacing=0.1
    )
    times = time_parameterize_path(path, 1.0, acceleration_max=2.0, turn_rate_max=2.0)
    assert times.shape == (path.shape[0],)
    assert np.all(np.diff(times) > 0)
    segment_speed = np.linalg.norm(np.diff(path, axis=0), axis=-1) / np.diff(times)
    corner = np.argmin(np.linalg.norm(path - np.asarray([2.0, 0.0]), axis=-1))
    assert segment_speed[max(0, corner - 1)] < 0.5
    assert segment_speed[0] > segment_speed[max(0, corner - 1)]


@pytest.mark.parametrize("scene_id", [None, 12, 27, 34, 35])
def test_navigation_state_masks_inactive_robots_and_builds_paths(scene_id, synthetic_map):
    assets = map_assets(scene_id, synthetic_map)
    cfg = NavigationRuntimeCfg(
        active_robot_count=3,
        map_image_path=str(assets.map_image),
        map_yaml_path=str(assets.map_yaml),
    )
    state = NavigationState(_Env(), cfg)
    state.initialize()

    assert state.active_mask.tolist() == [True, True, True, False, False, False, False, False]
    assert torch.all(state.robot_path_lengths[:3] > 1)
    assert torch.all(state.pedestrian_path_lengths > 1)
    assert torch.allclose(state.robot_spawns[3:], state.parking_positions[3:])
    targets = state.pedestrian_targets()
    assert targets.shape == (8, 10, 2)
    assert torch.isfinite(targets).all()
    assert not torch.any(
        state.map_collision_mask(state.robot_spawns[state.active_mask])
    )

    occupied_yx = np.argwhere(state.map.occupancy != 0)[0]
    occupied_xy = occupied_yx[::-1].copy()
    occupied_world = state.map.map_to_world_numpy(occupied_xy)
    occupied_position = torch.as_tensor(occupied_world).unsqueeze(0)
    assert state.map_collision_mask(occupied_position).item()
    assert state.map_collision_mask(torch.tensor([[1.0e6, 1.0e6]])).item()


def test_world_to_local_uses_scalar_first_quaternions():
    half_sqrt = 2.0**-0.5
    quaternion = torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]])
    world_forward = torch.tensor([[0.0, 1.0]])
    local = world_to_local_2d(world_forward, quaternion)
    torch.testing.assert_close(local, torch.tensor([[1.0, 0.0]]), atol=1e-6, rtol=0.0)


