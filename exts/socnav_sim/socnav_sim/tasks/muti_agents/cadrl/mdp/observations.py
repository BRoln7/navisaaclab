# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

joint_states = torch.zeros(6, 6, 7).to("cuda:0")
turtlebot_radius = 0.2


def compute_cadrl_obs(env: ManagerBasedEnv, goal: torch.tensor, 
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                      ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    obs = torch.zeros(env.num_envs, env.num_envs, 13)
    global joint_states

    # calculate self states
    p = asset.data.root_pos_w[:, :2]
    dg_tmp = p - goal
    dg = torch.sqrt(dg_tmp[:, 0]**2 + dg_tmp[:, 1]**2).reshape(env.num_envs, -1)
    v_pref = asset.data.root_lin_vel_b[:, 0].reshape(env.num_envs, -1)
    alpha = torch.atan2(asset.data.root_lin_vel_w[:, 1], 
                        asset.data.root_lin_vel_w[:, 0]).reshape(env.num_envs, -1)
    r = torch.ones(env.num_envs, 1).to("cuda:0") * turtlebot_radius
    s = torch.concat((dg, v_pref, alpha, r, p), dim=1)

    # calculate joint states
    px, py = s[:, 4], s[:, 5]
    px_i = px.view(env.num_envs, 1)
    py_i = py.view(env.num_envs, 1)
    dx = px.view(1, env.num_envs) - px_i
    dy = py.view(1, env.num_envs) - py_i

    cos_alpha = torch.cos(-alpha).view(env.num_envs, 1)
    sin_alpha = torch.sin(-alpha).view(env.num_envs, 1)

    px_prime = dx * cos_alpha - dy * sin_alpha
    py_prime = dx * sin_alpha + dy * cos_alpha

    cos_delta_alpha = torch.cos(alpha.view(1, env.num_envs) - alpha.view(env.num_envs, 1))
    sin_delta_alpha = torch.sin(alpha.view(1, env.num_envs) - alpha.view(env.num_envs, 1))

    vx_prime = v_pref.view(1, env.num_envs) * cos_delta_alpha
    vy_prime = v_pref.view(1, env.num_envs) * sin_delta_alpha

    relative_states = torch.stack([px_prime, py_prime, vx_prime, vy_prime], dim=-1)
    mask = ~torch.eye(env.num_envs, dtype=torch.bool)
    relative_states = relative_states[mask].view(env.num_envs, env.num_envs-1, 4)

    r_prim = torch.ones(env.num_envs, env.num_envs-1, 1).to("cuda:0") * turtlebot_radius
    da = torch.sqrt(relative_states[:, :, 0]**2 + relative_states[:, :, 1]**2).reshape(env.num_envs, -1, 1)
    joint_state = torch.concat((relative_states, r_prim, da, 2*r_prim), dim=2)

    # concat self_obs and joint_obs
    s_obs = torch.concat((s, torch.zeros(env.num_envs, 1).to("cuda:0")), dim=1).reshape(env.num_envs, -1, 7)
    obs = torch.concat((s_obs, joint_state), dim=1)

    joint_states = obs

    return obs
