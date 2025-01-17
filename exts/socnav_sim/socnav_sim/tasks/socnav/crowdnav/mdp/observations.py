from __future__ import annotations

import torch
import numpy as np
import os
import yaml
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils
import socnav_sim.tasks.socnav.nav_utils.event_base as event
from socnav_sim.tasks.socnav.nav_utils.frame_rotation import * # type: ignore

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def config_init():
    motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
    with open(motion_file, 'r') as file:
        config = yaml.safe_load(file)
    if config['env']['task'] == 'Isaac-Crowdnav': # type: ignore
        global num_envs, num_robots
        num_envs = config['env']['num_envs'] # type: ignore
        num_robots = config['env']['num_robots'] # type: ignore
config_init()

def compute_joint_states_obs(env: ManagerBasedEnv, 
                     asset_cfg1: SceneEntityCfg = SceneEntityCfg("robot"),
                     asset_cfg2: SceneEntityCfg = SceneEntityCfg("ped")
                    ) -> torch.Tensor:
    """
    robot_state:            0.dg  1.vpref  2. alpha  3.radius  4.px  5.py  
    vicinity agent state:   6.px^  7.py^  8.vx^  9.vy^  10.radius^ 11.da^ 12.radius~+radius(in robot local frame)
    observations size:      [env_nums, ped_nums*joint_state]
    """
    robot: Articulation = env.scene[asset_cfg1.name]
    ped: Articulation = env.scene[asset_cfg2.name]

    w_pos_r = robot.data.root_pos_w[:num_robots]
    w_vel_r = robot.data.root_vel_w[:num_robots]
    
    # robot state
    goal_r = event.goals_r_w - w_pos_r[..., :2]
    dg = torch.sqrt(goal_r[:,0]**2 + goal_r[:,1]**2).unsqueeze(-1)
    v_pref = w_vel_r[:,0].unsqueeze(-1)
    alpha = w_vel_r[:,-1].unsqueeze(-1)
    radius_r = torch.ones(num_robots, device='cuda:0').unsqueeze(-1)*0.5
    px, py = w_pos_r[:,0].unsqueeze(-1), w_pos_r[:,1].unsqueeze(-1)

    robot_state = torch.cat([dg, v_pref, alpha, radius_r, px, py],dim=-1)

    # ped state
    ped_pos_w = ped.data.root_pos_w.unsqueeze(0).repeat(num_robots, 1, 1)
    ped_vel_w = ped.data.root_lin_vel_w.unsqueeze(0).repeat(num_robots, 1, 1)
    
    robot_pos_w = robot.data.root_pos_w[:num_robots].unsqueeze(1).repeat(1, env.num_envs, 1)
    robot_quat_w = robot.data.root_quat_w[:num_robots]
    robot_vel_w = robot.data.root_lin_vel_w[:num_robots].unsqueeze(1).repeat(1, env.num_envs, 1)
    
    ped_pos_r = ped_pos_w - robot_pos_w # translation
    ped_pos_r = rotation_transform(ped_pos_r, robot_quat_w) # rotation

    ped_vel_r = (ped_vel_w - robot_vel_w)[..., :2]

    radius_p = (torch.ones(num_robots, num_envs, 1, device='cuda:0')*0.5)

    da = torch.sqrt(ped_pos_r[...,0]**2 + ped_pos_r[...,1]**2)
    da = da.unsqueeze(-1)

    # reserve the nearest 4 pedestrains
    ped_state = torch.cat([ped_pos_r, ped_vel_r, radius_p, da, 
                           (radius_r.unsqueeze(1).repeat(1,num_envs,1)+radius_p)], dim=-1)
    key_values = ped_state[..., 5]
    sorted_key_values, sorted_indices = torch.sort(key_values, dim=1, descending=True)
    sorted_ped_state = ped_state[torch.arange(ped_state.size(0)).unsqueeze(1), sorted_indices]
    nearest_ped_state = sorted_ped_state[:, :4].reshape(num_robots, -1)

    joint_state = torch.cat([robot_state, nearest_ped_state], dim=-1)

    global d_min
    d_min = joint_state[:, 11]

    _, N = joint_state.shape
    joint_state = torch.cat([joint_state, torch.ones(num_envs-num_robots, N, device='cuda:0')], dim=0)

    return joint_state



