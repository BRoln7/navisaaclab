"""Policy observations for CADRL navigation."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from socnav_sim.tasks.socnav.nav_utils.frame_rotation import world_to_local_2d
from socnav_sim.tasks.socnav.nav_utils.state import get_navigation_state


def compute_joint_states_obs(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pedestrian_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
    nearest_pedestrians: int = 4,
    robot_radius: float = 0.5,
    pedestrian_radius: float = 0.5,
    subgoal_lookahead: int = 20,
) -> torch.Tensor:
    """Build 7 robot features plus 7 features for each nearby pedestrian.

    Inactive robot slots are explicitly zeroed.  They remain in the vectorized
    simulation to host pedestrians but are discarded by ``MRPPO``.
    """
    state = get_navigation_state(env)
    robot: Articulation = env.scene[robot_cfg.name]
    pedestrian: Articulation = env.scene[pedestrian_cfg.name]
    active = state.active_robot_count
    feature_count = 7 + nearest_pedestrians * 7
    observation = torch.zeros((env.num_envs, feature_count), device=env.device)
    if active == 0:
        return observation

    robot_position = robot.data.root_pos_w[:active]
    robot_quaternion = robot.data.root_quat_w[:active]
    robot_velocity = robot.data.root_lin_vel_w[:active]
    local_robot_velocity = world_to_local_2d(robot_velocity[:, :2], robot_quaternion)

    state.set_robot_positions_for_subgoal(robot_position)
    subgoal_world = state.robot_subgoals(lookahead_points=subgoal_lookahead)
    subgoal_local = world_to_local_2d(subgoal_world - robot_position[:, :2], robot_quaternion)
    robot_features = torch.cat(
        (
            local_robot_velocity[:, :1],
            robot.data.root_ang_vel_w[:active, 2:3],
            torch.full((active, 1), robot_radius, device=env.device),
            robot_position[:, :2],
            subgoal_local,
        ),
        dim=-1,
    )

    relative_position_world = pedestrian.data.root_pos_w[None, :, :2] - robot_position[:, None, :2]
    relative_velocity_world = pedestrian.data.root_lin_vel_w[None, :, :2] - robot_velocity[:, None, :2]
    relative_position = world_to_local_2d(relative_position_world, robot_quaternion)
    relative_velocity = world_to_local_2d(relative_velocity_world, robot_quaternion)
    distance = torch.linalg.norm(relative_position, dim=-1)
    count = min(nearest_pedestrians, env.num_envs)
    nearest_distance, nearest_index = torch.topk(distance, k=count, dim=1, largest=False)
    gather_index = nearest_index[..., None].expand(-1, -1, 2)
    nearest_position = torch.gather(relative_position, 1, gather_index)
    nearest_velocity = torch.gather(relative_velocity, 1, gather_index)
    radius = torch.full((active, count, 1), pedestrian_radius, device=env.device)
    pedestrian_features = torch.cat(
        (
            nearest_position,
            nearest_velocity,
            radius,
            nearest_distance[..., None],
            torch.full((active, count, 1), robot_radius + pedestrian_radius, device=env.device),
        ),
        dim=-1,
    )
    if count < nearest_pedestrians:
        padding = torch.zeros((active, nearest_pedestrians - count, 7), device=env.device)
        pedestrian_features = torch.cat((pedestrian_features, padding), dim=1)

    observation[:active] = torch.cat((robot_features, pedestrian_features.flatten(1)), dim=-1)
    state.nearest_pedestrian_clearance[:active] = nearest_distance[:, 0] - robot_radius - pedestrian_radius
    return observation
