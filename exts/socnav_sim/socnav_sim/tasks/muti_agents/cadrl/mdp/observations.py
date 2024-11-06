# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

#########################################################################################
###load_obs----->compute_obs(self, path, terrain)----->normalize_self_obs------>concat### 
#########################################################################################
def compute_cadrl_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    obs = torch.zeros(env.num_envs, env.num_envs, 7)

    goal = torch.rand(env.num_envs, 2).to("cuda:0")
    goal = goal.clone() * 15

    p = asset.data.root_pos_w[:, :2]
    dg_tmp = p - goal
    dg = torch.sqrt(dg_tmp[:, 0]**2 + dg_tmp[:, 1]**2).reshape(env.num_envs, -1)
    v_pref = asset.data.root_lin_vel_b[:, 0].reshape(env.num_envs, -1)
    alpha = torch.atan2(asset.data.root_lin_vel_w[:, 1], 
                        asset.data.root_lin_vel_w[:, 0]).reshape(env.num_envs, -1)
    r = torch.ones(env.num_envs, 1).to("cuda:0") * 0.5

    s = torch.concat((dg, v_pref, alpha, r, p), dim=1)

    return  s
