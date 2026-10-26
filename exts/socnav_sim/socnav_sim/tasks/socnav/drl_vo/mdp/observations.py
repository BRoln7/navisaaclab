"""DRL-VO observations backed by the shared navigation runtime."""

from __future__ import annotations

import torch

from socnav_sim.tasks.socnav.nav_utils.frame_rotation import world_to_local_2d
from socnav_sim.tasks.socnav.nav_utils.state import get_navigation_state
from ..features import ScanHistory, pedestrian_velocity_grid


def compute_peds_obs(env):
    state = get_navigation_state(env)
    robot, ped = env.scene["robot"], env.scene["ped"]
    active = state.active_robot_count
    result = torch.zeros((env.num_envs, 12800), device=env.device)
    result[:active] = pedestrian_velocity_grid(
        robot.data.root_pos_w[:active], robot.data.root_quat_w[:active],
        robot.data.root_lin_vel_w[:active], ped.data.root_pos_w, ped.data.root_lin_vel_w,
    )
    return result


def compute_lidar_obs(env):
    state = get_navigation_state(env)
    history = getattr(env, "_drlvo_scan_history", None)
    if history is None:
        history = env._drlvo_scan_history = ScanHistory(env.num_envs, env.device)
    result = history.observe(
        env.scene.sensors["lidar"].data, env.common_step_counter, env.episode_length_buf
    )
    result[~state.active_mask] = 0
    return result


def compute_subgoal_obs(env):
    state = get_navigation_state(env)
    robot = env.scene["robot"]
    active = state.active_robot_count
    position = robot.data.root_pos_w[:active, :2]
    state.set_robot_positions_for_subgoal(robot.data.root_pos_w[:active])
    local = world_to_local_2d(
        state.robot_subgoals(lookahead_points=20) - position, robot.data.root_quat_w[:active]
    )
    result = torch.zeros((env.num_envs, 2), device=env.device)
    result[:active] = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True).clamp_min(1e-6)
    return result
