"""Termination predicates for map-based navigation.

Termination functions are intentionally side-effect free.  Isaac Lab invokes
the reset event after a termination; all pose writes and replanning happen
there.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .state import get_navigation_state


def time_runout(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def reach_goal(
    env: ManagerBasedRLEnv,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    state = get_navigation_state(env)
    robot: Articulation = env.scene[asset_cfg.name]
    distance = torch.linalg.norm(robot.data.root_pos_w[:, :2] - state.robot_goals, dim=-1)
    return state.active_mask & (distance <= goal_radius)


def robot_map_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate active robots that enter the inflated occupancy map."""
    state = get_navigation_state(env)
    robot: Articulation = env.scene[asset_cfg.name]
    collision = state.map_collision_mask(robot.data.root_pos_w[:, :2])
    return state.active_mask & collision


def pedestrian_fall(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
) -> torch.Tensor:
    pedestrian: Articulation = env.scene[asset_cfg.name]
    return pedestrian.data.root_pos_w[:, 2] < minimum_height


# Migration alias for older task configs.
ped_reset = pedestrian_fall
