# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import os
import yaml
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    compute_humanoid_observations_smpl_max,
    rot_wxyz_to_xyzw,
    root_wxyz_to_xyzw,
    root_states_wxyz_to_xyzw,
    compute_location_observations,
    build_amp_observations_smpl,
    get_height_maps_jit,
    dof_sim_to_gym,
    body_sim_to_gym,
    dof_subset,
    body_flip_left_to_right_index
)
import socnav_sim.tasks.amp.amp_utils.utils.traj_generator as traj_generator
# from socnav_sim.tasks.amp.amp_utils.utils.traj_socnav import TrajSocnav
import socnav_sim.tasks.amp.pacer.policy as policy
import socnav_sim.tasks.amp.pacer.mdp.events as events
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

trajectory_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/TrajectoryMarkers",
    markers={
        "marker": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5)),
        ),
    }
)
trajectory_marker = VisualizationMarkers(trajectory_cfg)

# config params
amp_obs_shape = 1950
amp_obs_shape_step = 195
amp_replay_obs = torch.tensor([]).to("cuda:0")
flip_humanoid_obs = None
tar_pos: torch.Tensor = None # type: ignore

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'amp.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
num_envs = config['env']['num_envs'] # type: ignore

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
        self.height_sample_width = 1.0
        self.num_samples_per_axis = 16
        self.height_sample_init_flag = False
    
        # self.traj_socnav = TrajSocnav(num_envs)

        self.amp_obs_buf = torch.zeros(
            (num_envs, 10, amp_obs_shape_step), device="cuda:0", dtype=torch.float)
        self.curr_amp_obs_buf = self.amp_obs_buf[:, 0]
        self.hist_amp_obs_buf = self.amp_obs_buf[:, 1:]

    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w.clone()[:, body_sim_to_gym]
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w.clone())[:, body_sim_to_gym]
        body_vel = asset.data.body_lin_vel_w.clone()[:, body_sim_to_gym]
        body_ang_vel = asset.data.body_ang_vel_w.clone()[:, body_sim_to_gym]

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

        body_pos = asset.data.body_pos_w.clone()[:, body_sim_to_gym]
        body_rot = rot_wxyz_to_xyzw(asset.data.body_quat_w.clone())[:, body_sim_to_gym]
        body_vel = asset.data.body_lin_vel_w.clone()[:, body_sim_to_gym]
        body_ang_vel = asset.data.body_ang_vel_w.clone()[:, body_sim_to_gym]

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
    
    def _build_traj_generator(self, 
                              env: ManagerBasedEnv, 
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        asset: Articulation = env.scene[asset_cfg.name]
        
        episode_dur = env.max_episode_length * env.physics_dt # type: ignore
        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(env.num_envs, episode_dur, num_verts,
                                                      env.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob)

        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        root_pos = asset.data.root_pos_w[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos)
        self.progress_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        print(f"INFO: traj_gen init")

        return

    def _fetch_traj_samples(self,
                            env: ManagerBasedEnv):
        # 5 seconds with 0.5 second intervals, 10 samples.
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long) # type: ignore

        timestep_beg = self.progress_buf[env_ids] * env.physics_dt # type: ignore
        
        # for local reward
        global tar_pos
        tar_pos = self._traj_gen.calc_pos(env_ids, timestep_beg) # type: ignore
        
        self.progress_buf += 1 # type: ignore
        timesteps = torch.arange(self._num_traj_samples, device=env.device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten()) # type: ignore
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples

    def compute_task_obs(self, 
                         env: ManagerBasedEnv, 
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                         ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        if self._traj_gen_init == False:
            self._traj_gen_init = True
            self._build_traj_generator(env, asset_cfg)

        # for traj_reset
        root_pos = asset.data.root_pos_w[:, 0:3]
        if events.amp_obs_reset is True:
            root_pos_reset = root_pos[events.amp_reset_env_ids, ...]
            self._traj_gen.reset(events.amp_reset_env_ids, root_pos_reset) # type: ignore
            self.progress_buf[events.amp_reset_env_ids] = 0 # type: ignore

        traj_samples = self._fetch_traj_samples(env)
        root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w)
        obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)

        trajectory_marker.visualize(translations=traj_samples.reshape(-1, 3))
        
        # root_pos = asset.data.root_pos_w[:, 0:3]
        # root_rot = asset.data.root_quat_w
        # if events.amp_obs_reset is True:
        #     self.traj_socnav.reset_traj(root_rot, root_pos, events.amp_reset_env_ids)

        # traj_samples = self.traj_socnav.traj_pos.unsqueeze(1).repeat(1, 10, 1)
        # root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w)
        # obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)        

        # global tar_pos
        # tar_pos = traj_samples[:, 0, :]

        # self.traj_socnav.update()
        # trajectory_marker.visualize(translations=self.traj_socnav.traj_pos.reshape(-1, 3))

        return obs
    
    def compute_height_obs(self,
                           env: ManagerBasedEnv, 
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        if self.height_sample_init_flag is False:
            self.height_points = self.init_height_points()
            self.height_sample_init_flag = True

        asset: Articulation = env.scene[asset_cfg.name]
        root_states = root_states_wxyz_to_xyzw(asset.data.root_state_w.clone())[..., :7]
        base_pos = root_states[..., :3]

        self.height_samples = torch.zeros((2800, 1400), device='cuda:0')

        height_obs = get_height_maps_jit(
            root_states=root_states,
            base_pos=base_pos,
            env_ids=torch.arange(0, env.num_envs, 1).to('cuda:0'),
            num_envs=num_envs,
            height_points=self.height_points,
            height_samples=self.height_samples,
            num_height_points=self.num_height_points,
            terrain_horizontal_scale=0.1,
            w_last=True,
            return_all_dims=False,
        )

        return height_obs
    
    def init_height_points(self):
        """
        Pre-defines the grid for the height-map observation.
        """
        y = torch.tensor(
            np.linspace(
                -self.height_sample_width,
                self.height_sample_width,
                self.num_samples_per_axis,
            ),
            device="cuda:0",
            requires_grad=False,
        )
        x = torch.tensor(
            np.linspace(
                -self.height_sample_width,
                self.height_sample_width,
                self.num_samples_per_axis,
            ),
            device="cuda:0",
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            num_envs,
            self.num_height_points,
            3,
            device='cuda:0',
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    

    def compute_amp_reward(self,
                        env: ManagerBasedEnv, 
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        # calculate current amp observation
        asset: Articulation = env.scene[asset_cfg.name]
        # ankles, wrists
        key_body_ids = [8, 7, 21, 20]
        key_body_pos = asset.data.body_pos_w.clone()[:, key_body_ids]
        # root pos
        spine_body_pos = asset.data.body_pos_w[:, 0, :]
        spine_body_rot = root_wxyz_to_xyzw(asset.data.body_quat_w[:, 0, :].clone()).squeeze(1)
        spine_body_vel = asset.data.body_lin_vel_w[:, 0, :].squeeze(1)
        spine_body_ang_vel = asset.data.body_ang_vel_w[:, 0, :].squeeze(1)

        joint_pos = asset.data.joint_pos.clone()[:, dof_sim_to_gym]
        joint_vel = asset.data.joint_vel.clone()[:, dof_sim_to_gym]

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
                               events.amp_motion_times,
                               env)
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
        
    def reset_amp_obs(self, env_ids, motion_ids, motion_times, env: ManagerBasedEnv):
        dt = 1 / 120

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

        origin_offsets = env.scene.env_origins[env_ids].repeat_interleave(9, dim=0)
        root_pos += origin_offsets

        amp_obs_demo = build_amp_observations_smpl(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
            key_pos, dof_subset, self._local_root_obs,
            self._amp_root_height_obs, self._has_dof_subset, self._has_upright_start)

        self.hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self.hist_amp_obs_buf[env_ids].shape)
        