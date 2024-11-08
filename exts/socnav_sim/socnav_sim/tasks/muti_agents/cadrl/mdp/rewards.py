# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

reach_goal_threshold = 0.1

def reach_goal(env: ManagerBasedRLEnv, 
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    goal_rewards = (obs.joint_states[:, 0, 0] < reach_goal_threshold).float()
    
    return goal_rewards

def cadrl_collision(env: ManagerBasedRLEnv, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    d_min = torch.min(obs.joint_states[:, 1:, -1])
    collision_rewards = torch.where(d_min <= obs.turtlebot_radius*2,
                                    torch.full_like(d_min, -0.25),
                                    torch.where((d_min>0.4) & (d_min<0.5), 
                                                -0.1+0.05*d_min,
                                                torch.zeros_like(d_min)))

    return collision_rewards