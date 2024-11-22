# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    key_body_names,
    build_amp_observations_smpl,
    dof_sim_to_gym,
    body_sim_to_gym,
    dof_subset
)
import socnav_sim.tasks.amp.amp.policy as policy
import omni.isaac.lab.sim as sim_utils

# config params

amp_obs_shape = 1950
amp_obs_shape_step = 195
flip_humanoid_obs = None


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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
            amp_reward = policy.discriminator(amp_obs)
            self.amp_hist_buff = self.amp_hist_buff[:, 1*amp_obs_shape_step:]
            amp_reward = amp_reward.squeeze(-1)
            reward = amp_reward.clone().detach()

            return reward
            # return torch.zeros(env.num_envs).to("cuda:0")
        
        else:
            return torch.zeros(env.num_envs).to("cuda:0")