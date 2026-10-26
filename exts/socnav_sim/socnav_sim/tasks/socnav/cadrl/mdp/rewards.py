"""Rewards for CADRL navigation."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from socnav_sim.tasks.socnav.nav_utils.state import get_navigation_state


def compute_goal_reward(
    env: ManagerBasedEnv,
    goal_radius: float = 0.45,
    progress_scale: float = 3.2,
    success_reward: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    state = get_navigation_state(env)
    robot: Articulation = env.scene[asset_cfg.name]
    distance = torch.linalg.norm(robot.data.root_pos_w[:, :2] - state.robot_goals, dim=-1)
    initialized = state.last_goal_distance > 0.0
    progress = torch.where(initialized, state.last_goal_distance - distance, torch.zeros_like(distance))
    reward = progress_scale * progress + success_reward * (distance <= goal_radius)
    reward *= state.active_mask
    state.last_goal_distance.copy_(distance)
    state.last_goal_distance[~state.active_mask] = 0.0
    return reward


def compute_collision_reward(
    env: ManagerBasedEnv,
    collision_penalty: float = -0.25,
    danger_distance: float = 0.2,
    danger_penalty: float = -0.1,
    robot_radius: float = 0.5,
    pedestrian_radius: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pedestrian_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
) -> torch.Tensor:
    state = get_navigation_state(env)
    robot: Articulation = env.scene[robot_cfg.name]
    pedestrian: Articulation = env.scene[pedestrian_cfg.name]
    active = state.active_robot_count
    reward = torch.zeros(env.num_envs, device=env.device)
    if active == 0:
        return reward
    pairwise_distance = torch.cdist(
        robot.data.root_pos_w[:active, :2], pedestrian.data.root_pos_w[:, :2]
    )
    clearance = pairwise_distance.min(dim=1).values - robot_radius - pedestrian_radius
    state.nearest_pedestrian_clearance[:active] = clearance
    collision = clearance <= 0.0
    danger = (clearance > 0.0) & (clearance <= danger_distance)
    active_reward = reward[:active]
    active_reward[collision] = collision_penalty
    active_reward[danger] = danger_penalty * (1.0 - clearance[danger] / danger_distance)
    return reward
