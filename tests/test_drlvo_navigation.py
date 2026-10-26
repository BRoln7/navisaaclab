"""CPU regression checks for DRL-VO input frames and episode history."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import gymnasium as gym

from socnav_sim.tasks.socnav.drl_vo.features import ScanHistory, pedestrian_velocity_grid, resample_scan
from socnav_sim.tasks.socnav.drl_vo.mdp import observations


def test_velocity_grid_uses_each_robots_local_frame_and_ignores_outside_pedestrians():
    position = torch.zeros(2, 3)
    quaternion = torch.tensor([[1., 0, 0, 0], [2**-.5, 0, 0, 2**-.5]])
    pedestrian = torch.tensor([[1., 0, 0], [99., 0, 0]])
    velocity = torch.tensor([[2., 0, 0], [2., 0, 0]])
    result = pedestrian_velocity_grid(position, quaternion, torch.zeros(2, 3), pedestrian, velocity)
    grid = result.reshape(2, 2, 80, 80)
    assert grid[0, 0, 44, 40] == 1
    assert grid[1, 1].min() == pytest.approx(-1.)
    assert grid[0].count_nonzero() == 1
    assert grid[1].abs().sum() == pytest.approx(1., abs=1e-6)


def test_nearest_pedestrian_wins_cell_collision():
    result = pedestrian_velocity_grid(
        torch.zeros(1, 3), torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3),
        torch.tensor([[1.01, 0, 0], [1.02, 0, 0]]),
        torch.tensor([[1., 0, 0], [-1., 0, 0]]),
    )
    assert result.reshape(1, 2, 80, 80)[0, 0, 44, 40] == .5


def test_empty_scan_and_invalid_returns_mean_no_hit_and_pooling_preserves_obstacles():
    assert torch.all(resample_scan(torch.tensor([])) == 30)
    assert resample_scan(torch.tensor([0., float('nan'), float('inf'), -1.]), beams=4).tolist() == [30.] * 4
    scan = torch.full((1440,), 30.)
    scan[15] = 1
    assert resample_scan(scan).min() == 1


def test_history_is_per_environment_resets_selected_rows_and_does_not_append_twice():
    a, b = ScanHistory(2, 'cpu'), ScanHistory(2, 'cpu')
    episode = torch.tensor([0, 0])
    first = torch.full((2, 720), 3.)
    assert a.observe(first, 0, episode).shape == (2, 6400)
    a.observe(torch.full_like(first, 9.), 1, torch.tensor([1, 1]))
    before = a.scans.clone()
    a.observe(torch.full_like(first, 12.), 1, torch.tensor([1, 1]))
    torch.testing.assert_close(a.scans, before)
    a.observe(torch.full_like(first, 6.), 1, torch.tensor([0, 1]))
    assert torch.all(a.scans[0] == 6)
    torch.testing.assert_close(a.scans[1], before[1])
    assert torch.all(b.scans == 30)


def test_zero_subgoal_and_inactive_robot_observations_are_finite(monkeypatch):
    state = SimpleNamespace(
        active_robot_count=1, active_mask=torch.tensor([True, False]),
        set_robot_positions_for_subgoal=lambda position: None,
        robot_subgoals=lambda **kwargs: torch.zeros(1, 2),
    )
    monkeypatch.setattr(observations, 'get_navigation_state', lambda env: state)
    data = SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=torch.tensor([[1.,0,0,0]]*2),
                           root_lin_vel_w=torch.zeros(2, 3))
    env = SimpleNamespace(num_envs=2, device='cpu', scene={'robot':SimpleNamespace(data=data),
                         'ped':SimpleNamespace(data=data)})
    assert torch.isfinite(observations.compute_subgoal_obs(env)).all()
    assert torch.count_nonzero(observations.compute_subgoal_obs(env)[1]) == 0
    assert torch.count_nonzero(observations.compute_peds_obs(env)[1]) == 0


def test_drlvo_registry_points_to_new_scene_configs():
    for scene in (12, 35):
        spec = gym.spec(f'Isaac-Drlvo-Scene{scene}')
        assert spec.kwargs['env_cfg_entry_point'].endswith(f':DrlvoScene{scene}EnvCfg')


def test_existing_cnn_accepts_flattened_observations_and_backpropagates():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts/policy_training/drl_vo'))
    from custom_cnn_full import CustomCNN
    network = CustomCNN(gym.spaces.Box(-1, 1, shape=(19202,), dtype=np.float32), features_dim=32)
    result = network(torch.zeros(2, 19202))
    assert result.shape == (2, 32)
    result.square().sum().backward()
    assert network.conv1.weight.grad is not None
