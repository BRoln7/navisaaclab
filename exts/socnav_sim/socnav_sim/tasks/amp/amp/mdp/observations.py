# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
import yaml
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    root_wxyz_to_xyzw,
    build_amp_observations_smpl,
    dof_sim_to_gym,
    body_sim_to_gym,
    dof_subset
)
import socnav_sim.tasks.amp.amp.policy as policy
import socnav_sim.tasks.amp.amp.mdp.events as events
import socnav_sim.tasks.amp.amp.mdp.reset as reset
import omni.isaac.lab.sim as sim_utils

# config params

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'amp.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
num_envs = config['env']['num_envs'] # type: ignore

amp_obs_shape = 1950
amp_obs_shape_step = 195
amp_replay_obs = torch.tensor([]).to("cuda:0")

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

#########################################################################################
##################################### function for obs ################################## 
#########################################################################################
class ObsComputer():
    def __init__(self) -> None:
        self._has_upright_start = True
        self._local_root_obs = True
        self._amp_root_height_obs = False
        self._has_dof_subset = True
        self._has_shape_obs_disc = False
        self.dof_subset = dof_subset

        self.amp_obs_buf = torch.zeros(
            (num_envs, 10, amp_obs_shape_step), device="cuda:0", dtype=torch.float)
        self.curr_amp_obs_buf = self.amp_obs_buf[:, 0]
        self.hist_amp_obs_buf = self.amp_obs_buf[:, 1:]


    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") # type: ignore
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
        # calculate current amp observation
        asset: Articulation = env.scene[asset_cfg.name]
        # ankles, wrists
        key_body_ids = [8, 7, 21, 20]
        key_body_pos = asset.data.body_pos_w[:, key_body_ids]
        # root pos
        spine_body_pos = asset.data.root_pos_w
        spine_body_rot = root_wxyz_to_xyzw(asset.data.root_quat_w).squeeze(1)
        spine_body_vel = asset.data.root_lin_vel_w.squeeze(1)
        spine_body_ang_vel = asset.data.root_ang_vel_w.squeeze(1)

        joint_pos = asset.data.joint_pos[:, dof_sim_to_gym]
        joint_vel = asset.data.joint_vel[:, dof_sim_to_gym]

        curr_amp_obs = build_amp_observations_smpl(
            spine_body_pos, spine_body_rot,
            spine_body_vel, spine_body_ang_vel,
            joint_pos, joint_vel, key_body_pos,
            self.dof_subset, 
            self._local_root_obs, self._amp_root_height_obs, self._has_dof_subset, self._has_upright_start)

        # events reset
        if events.amp_obs_reset is True:
            self.reset_amp_obs(events.amp_reset_env_ids,
                               events.amp_motion_ids,
                               events.amp_motion_times)
            events.amp_obs_reset = False
            
        # update history amp observation
        self.amp_obs_buf = torch.cat((curr_amp_obs.reshape(env.num_envs, 1, -1), 
                                      self.hist_amp_obs_buf), dim=1)
        self.hist_amp_obs_buf = self.amp_obs_buf[:, 0:9]

        # calculate amp reward
        norm_amp_obs = policy.amp_input_mean_std(self.amp_obs_buf.reshape(num_envs, -1))
        amp_reward = policy.discriminator(norm_amp_obs).squeeze(1)
        reward = amp_reward.clone().detach()

        # record amp_replay_obs
        global amp_replay_obs
        amp_replay_obs = self.amp_obs_buf.reshape(num_envs, -1).unsqueeze(0)

        return reward
        
    def reset_amp_obs(self, env_ids, motion_ids, motion_times):
        dt = 1 / 30

        motion_ids = motion_ids[env_ids]
        motion_times = motion_times[env_ids]
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, 9])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -dt * (torch.arange(0, 9, device='cuda:0') + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        motion_res = policy.motion_load.get_smpl_state_from_motionlib_cache(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, \
            key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
                motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["motion_bodies"], \
                    motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], \
                        motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        amp_obs_demo = build_amp_observations_smpl(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
            key_pos, dof_subset, self._local_root_obs,
            self._amp_root_height_obs, self._has_dof_subset, self._has_upright_start)

        self.hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self.hist_amp_obs_buf[env_ids].shape)
        