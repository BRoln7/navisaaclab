from __future__ import annotations

import torch
import numpy as np
import os
import yaml
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.sensors.sensors.ray_caster import RTXRayCaster
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
num_envs = config['env']['num_envs'] # type: ignore

if config['env']['task'] == 'Isaac-Drlvo': # type: ignore
    lidar_visual_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/LidarMarkers",
        markers={
            "lidar": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        }
    )
    lidar_marker = VisualizationMarkers(lidar_visual_cfg)

    num_robots = config['env']['num_robots'] # type: ignore

lidar_history_data = None

############################ peds ############################
def compute_peds_obs(env: ManagerBasedEnv, 
                     asset_cfg1: SceneEntityCfg = SceneEntityCfg("robot"),
                     asset_cfg2: SceneEntityCfg = SceneEntityCfg("ped")
                    ) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg1.name]
    ped: Articulation = env.scene[asset_cfg2.name]

    # get kinematic info of peds in world frame
    ped_pos_w = ped.data.root_pos_w.unsqueeze(0).repeat(num_robots, 1, 1)
    ped_vel_w = ped.data.root_lin_vel_w.unsqueeze(0).repeat(num_robots, 1, 1)
    
    # transform kinematic info into robot frame
    robot_pos_w = robot.data.root_pos_w[:num_robots].unsqueeze(1).repeat(1, env.num_envs, 1)
    robot_quat_w = robot.data.root_quat_w[:num_robots]
    robot_vel_w = robot.data.root_lin_vel_w[:num_robots].unsqueeze(1).repeat(1, env.num_envs, 1)
    # for pos
    ped_pos_r = ped_pos_w - robot_pos_w # translation
    ped_pos_r = rotation_transform(ped_pos_r, robot_quat_w) # rotation
    ped_pos_r = torch.clamp(ped_pos_r, max=10, min=-10)
    # for vel
    ped_vel_r = ped_vel_w - robot_vel_w
    # peds info encode
    peds_encode = torch.zeros((num_robots, 2, 80, 80), device='cuda:0')
    collumn = torch.floor(-(ped_pos_r[...,1]-10)/0.25).to(torch.int)
    row = torch.floor(ped_pos_r[...,0]/0.25).to(torch.int)
    
    collumn = torch.clamp(collumn, max=79, min=0)
    row = torch.clamp(row, max=79, min=0)

    peds_encode[:, 0, row, collumn] = ped_vel_r[..., 0]
    peds_encode[:, 1, row, collumn] = ped_vel_r[..., 1]

    if (num_envs-num_robots) > 0:
        peds_encode = torch.cat([peds_encode, 
                                 torch.zeros((num_envs-num_robots, 2, 80, 80), device='cuda:0')], 
                                 dim=0)

    # MaxAbsScaler
    v_min = -2
    v_max = 2
    peds_encode = peds_encode.reshape(env.num_envs, -1)
    peds_encode = 2 * (peds_encode - v_min) / (v_max - v_min) + (-1)

    return peds_encode

def quat_to_rot_matrix(quats, _num_robots):
    qw, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    rot_matrix = torch.zeros((_num_robots, 3, 3), device='cuda:0', dtype=torch.float32)
    rot_matrix[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    rot_matrix[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot_matrix[:, 0, 2] = 2 * (qx * qz + qw * qy)

    rot_matrix[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot_matrix[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    rot_matrix[:, 1, 2] = 2 * (qy * qz - qw * qx)

    rot_matrix[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot_matrix[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot_matrix[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    return rot_matrix

def rotation_transform(relative_positions, quats):
    m, n, _ = relative_positions.shape
    rot_matrices = quat_to_rot_matrix(quats, m) # (num_robots, 3, 3)
    relative_positions = relative_positions.unsqueeze(-1)  # (num_robots, num_peds, 3, 1)
    transformed_positions = torch.matmul(rot_matrices.unsqueeze(1).repeat(1, n, 1, 1), 
                                         relative_positions)  # (m, n, 3, 1)
    transformed_positions = transformed_positions[:, :, :2, 0]

    return transformed_positions

############################ lidar ############################
def compute_lidar_obs(env: ManagerBasedEnv, 
                      sensor_cfg: SceneEntityCfg,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    lidar_sensor: RTXRayCaster = env.scene.sensors[sensor_cfg.name] # type: ignore
    if lidar_sensor.data[list(lidar_sensor.data.keys())[0]][0].data.numel() != 0: # type: ignore
        lidar_data, lidar_ranges = raw_lidar_process(lidar_sensor)

        # for lidar visualization
        lidar_marker_pose = (lidar_data + asset.data.root_pos_w[:num_robots].unsqueeze(1).repeat(1, 720, 1))
        Rotation_visual = quat_to_rot_matrix(asset.data.root_quat_w[:num_robots], num_robots)
        lidar_marker_pose = torch.matmul(Rotation_visual.unsqueeze(1).repeat(1, 720, 1, 1),
                                         lidar_marker_pose.unsqueeze(-1)).squeeze(-1).reshape(-1, 3)

        lidar_marker.visualize(translations=lidar_marker_pose) # type: ignore
    else:
        lidar_ranges = torch.zeros((num_robots, 720), device='cuda:0')

    lidar_obs = lidar_pooling(lidar_ranges, num_robots)

    if (num_envs-num_robots) > 0:
        lidar_obs = torch.cat([lidar_obs, torch.zeros((num_envs-num_robots, 80*80),device='cuda:0')], dim=0)

    return lidar_obs


def raw_lidar_process(_lidar:RTXRayCaster) -> tuple[torch.Tensor, torch.Tensor]:
    """
    load raw lidar data and extract the 720 points needed
    """

    lidar = _lidar
    lidar_data_info = []
    lidar_ranges_info = []
    robot_index = 0

    for key in lidar.data:
        # collect raw data
        raw_lidar_data = lidar.data[key][0].data.to('cuda:0') # type: ignore
        raw_lidar_ranges = lidar.data[key][0].distance.to('cuda:0') # type: ignore
        # choose channel
        z_min = -0.02
        z_max = 0.02
        z_data = raw_lidar_data[:, 2]
        mask_z = (z_data >= z_min) & (z_data <= z_max)
        filtered_lidar_data = raw_lidar_data[mask_z]
        filtered_lidar_ranges = raw_lidar_ranges[mask_z]
        # set range
        x_min = -0.1
        x_data = filtered_lidar_data[:, 0]
        mask_x = (x_data >= x_min)
        filtered_lidar_data = filtered_lidar_data[mask_x]
        filtered_lidar_ranges = filtered_lidar_ranges[mask_x]
        # sample the num_of_points
        num_samples = 720
        indices = torch.linspace(0, filtered_lidar_ranges.size(0) - 1, num_samples).long()
        filtered_lidar_data = filtered_lidar_data[indices]
        filtered_lidar_ranges = filtered_lidar_ranges[indices]    
        # sort by yaw
        yaw = torch.atan2(filtered_lidar_data[:, 1], filtered_lidar_data[:, 0])
        sorted_indices = torch.argsort(yaw)
        filtered_lidar_data = filtered_lidar_data[sorted_indices]
        filtered_lidar_ranges = filtered_lidar_ranges[sorted_indices]

        lidar_data_info.append(filtered_lidar_data)
        lidar_ranges_info.append(filtered_lidar_ranges)

        robot_index += 1

        if robot_index == num_robots:
            break

    final_lidar_data = torch.stack(lidar_data_info)
    final_lidar_ranges = torch.stack(lidar_ranges_info)

    return final_lidar_data, final_lidar_ranges

def lidar_pooling(_lidar_current_data:torch.Tensor,
                  _num_robots)->torch.Tensor:
    """
    MIN-AVG Pooling
    """
    # collect history data
    global lidar_history_data
    if lidar_history_data is None:
        lidar_history_data = _lidar_current_data.unsqueeze(1)
    else:
        lidar_history_data = torch.cat([lidar_history_data, _lidar_current_data.unsqueeze(1)],dim=1)
    
    # pooling    
    if lidar_history_data.size(1) == 11:
        lidar_history_data = lidar_history_data[:,1:,:]
        lidar_data_input = lidar_history_data.reshape(_num_robots, -1)
        
        # MIN-AVG
        lidar_data_input = lidar_data_input.reshape(_num_robots, 800, 9)
        scan_min = lidar_data_input.min(dim=2).values.reshape(_num_robots, 10, -1)
        scan_avg = lidar_data_input.mean(dim=2).reshape(_num_robots, 10, -1)
        scan_min_avg = torch.cat([scan_min, scan_avg], dim=1)
        scan_min_avg = scan_min_avg.repeat(1, 4, 1)

        # Normalize
        s_min = 0
        s_max = 30
        scan_min_avg = scan_min_avg.reshape(_num_robots, -1)
        scan_min_avg = 2 * (scan_min_avg - s_min) / (s_max - s_min) + (-1)

        return scan_min_avg

    else:
        return torch.ones((_num_robots, 80*80), device='cuda:0') * -1


def compute_subgoal_obs(env: ManagerBasedEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    
    # map and global path-finding to-do
    
    return torch.ones((env.num_envs, 2), device='cuda:0') * np.sqrt(2)