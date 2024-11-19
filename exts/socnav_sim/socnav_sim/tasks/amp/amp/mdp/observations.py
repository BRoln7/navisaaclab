# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    root_states_wxyz_to_xyzw,
    compute_location_observations,
    key_body_names,
    build_amp_observations_smpl,
    dof_sim_to_gym,
    body_sim_to_gym,
    body_flip_left_to_right_index
)
import socnav_sim.tasks.amp.amp_utils.isaac_utils.traj_generator as traj_generator
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
from socnav_sim.tasks.amp.amp_utils.utils.discriminator import Discriminator
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils
import socnav_sim.tasks.amp.pacer.mdp.reset as Reset

# config params
dof_subset = torch.tensor([ 
    0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
    46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 
    59, 60, 61, 62, 63, 64, 65]).to("cuda:0")

amp_obs_shape = 1950
amp_obs_shape_step = 195
running_mean_obs_shape = 1401
reward_shape = 1
flip_humanoid_obs = None


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

trajectory_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/TrajectoryMarkers",
    markers={
        "marker": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.2)),
        ),
    }
)
trajectory_marker = VisualizationMarkers(trajectory_cfg)

tar_pos: torch.Tensor = None # type: ignore

#########################################################################################
##################################### function for obs ################################## 
#########################################################################################
class ObsComputer():
    def __init__(self) -> None:
        self._speed_min = 0.0
        self._speed_max = 3.0
        self._accel_max = 2.0
        self._sharp_turn_prob = 0.02
        self._num_traj_samples = 10
        self._traj_sample_timestep = 0.5
        self._traj_gen_init = False
        self._traj_gen = None
        self.progress_buf = None
        self._has_upright_start = True
        self.height_meas_scale = 5
        self.num_height_points = 1024
        self.num_center_height_points = 9
        self._local_root_obs = True
        self._amp_root_height_obs = False
        self._has_dof_subset = True
        self._has_shape_obs_disc = False
        self.dof_subset = dof_subset
        self.amp_hist_buff = None

        checkpoint_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Humanoid.pth')
        running_mean_std_params = torch.load(checkpoint_path)
        self.reward_mean_std = RunningMeanStd(reward_shape).to("cuda:0")
        self.reward_mean_std.load_state_dict(running_mean_std_params['reward_mean_std'])
        self.reward_mean_std.eval()
        self.discriminator = Discriminator().to("cuda:0")
        self.discriminator.eval()

    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w[:, body_sim_to_gym]
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w)[:, body_sim_to_gym]
        body_vel = asset.data.body_lin_vel_w[:, body_sim_to_gym]
        body_ang_vel = asset.data.body_ang_vel_w[:, body_sim_to_gym]

        obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, 
                                                    local_root_obs_config, local_height_obs_config, 
                                                    upright_config)
        
        global flip_humanoid_obs
        flip_humanoid_obs = self.compute_flip_humanoid_obs(env, asset_cfg)

        return obs
    
    def compute_flip_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w[:, body_sim_to_gym]
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w)[:, body_sim_to_gym]
        body_vel = asset.data.body_lin_vel_w[:, body_sim_to_gym]
        body_ang_vel = asset.data.body_ang_vel_w[:, body_sim_to_gym]

        body_pos[..., 1] *= -1 # position
        body_pos = body_pos[..., body_flip_left_to_right_index, :]

        body_rot[..., 0] *= -1 # angular rotation, global
        body_rot[..., 2] *= -1
        body_rot = body_rot[..., body_flip_left_to_right_index, :]

        body_vel[..., 1] *= -1 # linear velocity
        body_vel = body_vel[..., body_flip_left_to_right_index, :]

        body_ang_vel[..., 0] *= -1 # angular velocity
        body_ang_vel[..., 2] *= -1
        body_ang_vel = body_ang_vel[..., body_flip_left_to_right_index, :]

        obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, 
                                                    local_root_obs_config, local_height_obs_config, 
                                                    upright_config)
        return obs
    
    def compute_amp_reward(self,
                        env: ManagerBasedEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        key_body_ids = asset.find_bodies(key_body_names)
        key_body_ids = [8, 7, 21, 20]
        key_body_pos = asset.data.body_pos_w[:, key_body_ids]
        
        spine_id = asset.find_bodies("Spine")
        spine_body_pos = asset.data.body_pos_w[:, spine_id[0], :].squeeze(1)
        spine_body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w[:, spine_id[0], :]).squeeze(1)
        spine_body_vel = asset.data.body_lin_vel_w[:, spine_id[0], :].squeeze(1)
        spine_body_ang_vel = asset.data.body_ang_vel_w[:, spine_id[0], :].squeeze(1)

        joint_pos = asset.data.joint_pos[:, dof_sim_to_gym]
        joint_vel = asset.data.joint_vel[:, dof_sim_to_gym]

        curr_amp_obs = build_amp_observations_smpl(
            spine_body_pos, spine_body_rot,
            spine_body_vel, spine_body_ang_vel,
            joint_pos, joint_vel, key_body_pos,
            self.dof_subset, 
            self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_upright_start)

        if self.amp_hist_buff is None:
            self.amp_hist_buff = curr_amp_obs
        else:
            self.amp_hist_buff = torch.cat((self.amp_hist_buff.clone(), curr_amp_obs), dim=1)

        if self.amp_hist_buff.size(1) == amp_obs_shape:
            amp_obs = self.amp_hist_buff
            amp_reward = self.discriminator(amp_obs)
            self.amp_hist_buff = self.amp_hist_buff[:, 1*amp_obs_shape_step:]
            amp_reward = amp_reward.squeeze(-1)
            reward = amp_reward.clone().detach()

            return reward
            # return torch.zeros(env.num_envs).to("cuda:0")
        
        else:
            return torch.zeros(env.num_envs).to("cuda:0")