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
    obs_wxyz_to_xyzw,
)

# config params
height_conditioned = False
num_height_points = 256
terrain_horizontal_scale = 0.1

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
        pass
    def compute_humanoid_obs(self, 
                             env: ManagerBasedEnv, 
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        upright_config = True
        local_root_obs_config = True
        local_height_obs_config = False

        body_pos = asset.data.body_pos_w
        body_rot = obs_wxyz_to_xyzw(asset.data.body_quat_w)
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

        return obs


    # def compute_height_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    #                     ) -> torch.Tensor:
    #     """
    #     Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
    #     Each sample is the billinear interpolation between adjacent points.
    #     """
    #     asset: Articulation = env.scene[asset_cfg.name]

    #     env_ids = torch.arange(env.num_envs, device="cuda:0").long()
    #     num_envs = len(env_ids)

    #     root_states = asset.data.root_state_w
    #     base_pos = root_states[:, :3]
    #     height_points = init_height_points(num_envs)
    #     height_samples = torch.zeros(2800, 1400).to("cuda:0")

    #     return get_height_maps_jit(
    #         root_states=root_states,
    #         base_pos=base_pos,
    #         env_ids=env_ids,
    #         num_envs=num_envs,
    #         height_points=height_points,
    #         height_samples=height_samples,
    #         num_height_points=num_height_points,
    #         terrain_horizontal_scale=terrain_horizontal_scale,
    #         w_last=True,
    #         return_all_dims=False,
    #     )


    # def compute_task_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    #                     ) -> torch.Tensor:
    #     asset: Articulation = env.scene[asset_cfg.name]

    #     traj_samples = self._fetch_traj_samples(env_ids)