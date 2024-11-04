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
    get_heights_with_scene,
    compute_humanoid_observations_max,
    init_height_points,
    get_height_maps_jit,
    compute_path_observations
)

# config params
height_conditioned = False
num_height_points = 256
terrain_horizontal_scale = 0.1

traj_tmp = torch.tensor([[[0.2, 0.2, 0.8],
                         [0.4, 0.4, 0.8],
                         [0.6, 0.6, 0.8],
                         [0.8, 0.8, 0.8],
                         [1.0, 1.0, 0.8],
                         [1.2, 1.2, 0.8],
                         [1.4, 1.4, 0.8],
                         [1.6, 1.6, 0.8],
                         [1.8, 1.8, 0.8],
                         [2.0, 2.0, 0.8],],]).to("cuda:0")

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

#########################################################################################
###load_obs----->compute_obs(self, path, terrain)----->normalize_self_obs------>concat### 
#########################################################################################
def compute_humanoid_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    body_pos = asset.data.body_pos_w
    body_rot = asset.data.body_quat_w
    body_vel = asset.data.body_lin_vel_w
    body_ang_vel = asset.data.body_ang_vel_w
    root_state = asset.data.root_state_w

    ground_heights = get_heights_with_scene(root_state[..., :2])

    obs = compute_humanoid_observations_max(
        body_pos,
        body_rot,
        body_vel,
        body_ang_vel,
        ground_heights,
        True,
        True,
        True,
    )

    return obs


def get_height_maps(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    """
    Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
    Each sample is the billinear interpolation between adjacent points.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    env_ids = torch.arange(env.num_envs, device="cuda:0").long()
    num_envs = len(env_ids)

    root_states = asset.data.root_state_w
    base_pos = root_states[:, :3]
    height_points = init_height_points(num_envs)
    height_samples = torch.zeros(2800, 1400).to("cuda:0")

    return get_height_maps_jit(
        root_states=root_states,
        base_pos=base_pos,
        env_ids=env_ids,
        num_envs=num_envs,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=num_height_points,
        terrain_horizontal_scale=terrain_horizontal_scale,
        w_last=True,
        return_all_dims=False,
    )


def compute_task_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    bodies_positions = asset.data.body_pos_w
    root_states = asset.data.root_state_w[..., :7]
    head_position = bodies_positions[:, asset.find_bodies('Head')[0], :]
    ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]

    traj_samples = traj_tmp + root_states[:, :3]

    flat_head_position = head_position.view(-1, 3)
    flat_head_position[..., 2] -= ground_below_head.view(-1)

    obs = compute_path_observations(
        root_states,
        flat_head_position,
        traj_samples,
        True,
        height_conditioned,
    )

    path_obs = obs

    return path_obs