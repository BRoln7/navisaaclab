# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    root_states_wxyz_to_xyzw,
    compute_location_observations,
)
import socnav_sim.tasks.amp.smpl_humanoid.mdp.isaac_utils.traj_generator as traj_generator
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils import running_mean_std

# config params
body_gym_to_sim = [
    10, 9, 11, 0, 12, 14, 19, 1, 5, 
    13, 15, 20, 2, 6, 16, 21, 3, 7, 17, 
    22, 4, 8, 18, 23]

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

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
        self._num_envs = None
        self._device = None
        self._dt = None
        self.progress_buf = None
        self._has_upright_start = True
        self.height_meas_scale = 5
        self.num_height_points = 1024
        self.num_center_height_points = 9


    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w)
        body_vel = asset.data.body_lin_vel_w
        body_ang_vel = asset.data.body_ang_vel_w
        # gym to sim
        body_pos = body_pos.clone()[:, body_gym_to_sim]
        body_rot = body_rot.clone()[:, body_gym_to_sim]
        body_vel = body_vel.clone()[:, body_gym_to_sim]
        body_ang_vel = body_ang_vel.clone()[:, body_gym_to_sim]

        obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, 
                                                    local_root_obs_config, local_height_obs_config, 
                                                    upright_config)
        
        if self._traj_gen_init == False:
            self._traj_gen_init = True
            self._build_traj_generator(env, asset_cfg)

        return obs
    
    def _build_traj_generator(self, 
                              env: ManagerBasedEnv, 
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        asset: Articulation = env.scene[asset_cfg.name]
        self._num_envs = env.num_envs
        self._device = env.device
        self._dt = env.physics_dt
        episode_dur = env.max_episode_length * env.physics_dt # type: ignore
        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(self._num_envs, episode_dur, num_verts,
                                                      env.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob)

        env_ids = torch.arange(self._num_envs, device=env.device, dtype=torch.long)
        root_pos = asset.data.root_pos_w[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos)
        self.progress_buf = torch.zeros(self._num_envs, device=env.device, dtype=torch.long)

        print(f"INFO: traj_gen init")

        return


    def _fetch_traj_samples(self):
        # 5 seconds with 0.5 second intervals, 10 samples.
        env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long) # type: ignore

        timestep_beg = self.progress_buf[env_ids] * self._dt # type: ignore
        self.progress_buf += 1 # type: ignore
        timesteps = torch.arange(self._num_traj_samples, device=self._device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples


    def compute_task_obs(self, 
                         env: ManagerBasedEnv, 
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                         ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        if self._traj_gen_init:
            traj_samples = self._fetch_traj_samples()
            root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w)
            obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)
        else:
            return torch.zeros(env.num_envs, 20).to(env.device)

        return obs
    

    def compute_height_obs(self,
                           env: ManagerBasedEnv, 
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        # asset: Articulation = env.scene[asset_cfg.name]
        # head_index = asset.find_bodies("Head")
        # head_pos = asset.data.body_pos_w[:, head_index]
        # head_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w[:, head_index])
        # head_pose = torch.cat((head_pos, head_rot), dim=1)
        if self._traj_gen_init:
            self.measured_heights = torch.zeros(self._num_envs, self.num_height_points,  # type: ignore
                                                device=self._device, requires_grad=False)
            
            center_heights = torch.zeros(self._num_envs, self.num_center_height_points, # type: ignore
                                device=self._device, requires_grad=False)
            center_heights = center_heights.mean(dim=-1, keepdim=True)
            heights = center_heights - self.measured_heights
            heights = torch.clip(heights, -3, 3.) * self.height_meas_scale
        else:
            return torch.zeros(env.num_envs, 1024).to(env.device)

        return heights