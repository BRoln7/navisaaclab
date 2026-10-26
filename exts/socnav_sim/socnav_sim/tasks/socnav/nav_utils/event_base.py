"""Reset events for map-based social navigation."""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .pedestrian_policy import SMPL_JOINT_NAMES
from .state import NavigationRuntimeCfg, get_navigation_state


def reset_pedestrian_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    replan: bool = True,
    pedestrian_asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
) -> None:
    """Reset only selected humanoids without ending their paired robot episodes."""
    state = get_navigation_state(env)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if replan:
        state.reset_pedestrians(env_ids, sample_spawn=True)
    pedestrian: Articulation = env.scene[pedestrian_asset_cfg.name]
    count = env_ids.numel()
    root_pose = torch.empty((count, 7), device=env.device)
    root_pose[:, :2] = state.pedestrian_spawns[env_ids]
    root_pose[:, 2] = 0.9
    root_pose[:, 3:] = 0.0
    root_pose[:, 3] = 1.0
    root_velocity = torch.zeros((count, 6), device=env.device)
    joint_ids, names = pedestrian.find_joints(
        [f"^{name}$" for name in SMPL_JOINT_NAMES], preserve_order=True
    )
    if names != SMPL_JOINT_NAMES:
        raise RuntimeError("Navigation SMPL USD joint order differs from MimicKit")
    joint_position = torch.zeros((count, len(joint_ids)), device=env.device)
    # Match MimicKit's neutral pose: arms hang along the body's sides.
    joint_position[:, SMPL_JOINT_NAMES.index("L_Shoulder_x")] = -1.5708
    joint_position[:, SMPL_JOINT_NAMES.index("R_Shoulder_x")] = 1.5708
    joint_velocity = torch.zeros_like(joint_position)
    pedestrian.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pedestrian.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    pedestrian.write_joint_state_to_sim(
        joint_position, joint_velocity, joint_ids=joint_ids, env_ids=env_ids
    )
    pedestrian.set_joint_position_target(joint_position, joint_ids=joint_ids, env_ids=env_ids)


def reset_navigation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    active_robot_count: int,
    pedestrian_speed: float = 0.85,
    pedestrian_speed_variation: float = 0.15,
    fast_pedestrian_probability: float = 0.10,
    fast_pedestrian_speed: float = 1.20,
    fast_pedestrian_speed_variation: float = 0.15,
    pedestrian_acceleration_max: float = 2.0,
    pedestrian_turn_rate_max: float = 2.0,
    path_spacing: float = 0.10,
    robot_goal_min: float = 4.0,
    robot_goal_max: float = 8.0,
    pedestrian_goal_min: float = 6.0,
    pedestrian_goal_max: float = 10.0,
    map_scale: float = 0.25,
    map_dilation: int = 7,
    map_image_path: str = "",
    map_yaml_path: str = "",
    visualize_paths: bool = False,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pedestrian_asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
) -> None:
    """Reset requested agents and rebuild their collision-aware global paths."""
    runtime_cfg = NavigationRuntimeCfg(
        active_robot_count=active_robot_count,
        pedestrian_speed=pedestrian_speed,
        pedestrian_speed_variation=pedestrian_speed_variation,
        fast_pedestrian_probability=fast_pedestrian_probability,
        fast_pedestrian_speed=fast_pedestrian_speed,
        fast_pedestrian_speed_variation=fast_pedestrian_speed_variation,
        pedestrian_acceleration_max=pedestrian_acceleration_max,
        pedestrian_turn_rate_max=pedestrian_turn_rate_max,
        path_spacing=path_spacing,
        robot_goal_distance=(robot_goal_min, robot_goal_max),
        pedestrian_goal_distance=(pedestrian_goal_min, pedestrian_goal_max),
        map_scale=map_scale,
        map_dilation=map_dilation,
        map_image_path=map_image_path,
        map_yaml_path=map_yaml_path,
    )
    state = get_navigation_state(env, runtime_cfg)
    first_reset = not state.initialized
    state.initialize()
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if not first_reset:
        state.reset_robots(env_ids, sample_spawn=True)
        state.reset_pedestrians(env_ids, sample_spawn=True)

    robot: Articulation = env.scene[robot_asset_cfg.name]
    robot_root = robot.data.default_root_state[env_ids].clone()
    robot_root[:, :2] = state.robot_spawns[env_ids]
    robot_root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(robot_root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids
    )

    reset_pedestrian_assets(
        env, env_ids, replan=False, pedestrian_asset_cfg=pedestrian_asset_cfg
    )
    if visualize_paths:
        _visualize_robot_paths(env, state)


def _visualize_robot_paths(env: ManagerBasedEnv, state) -> None:
    marker = getattr(env, "_socnav_robot_path_marker", None)
    if marker is None:
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/SocNav/RobotPaths",
                markers={
                    "path": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 1.0)),
                    )
                },
            )
        )
        setattr(env, "_socnav_robot_path_marker", marker)
    points = [
        state.robot_paths[index, : int(state.robot_path_lengths[index].item())]
        for index in range(state.active_robot_count)
    ]
    if points:
        points_xy = torch.cat(points, dim=0)
        marker.visualize(
            translations=torch.cat((points_xy, torch.full_like(points_xy[:, :1], 0.05)), dim=-1)
        )
