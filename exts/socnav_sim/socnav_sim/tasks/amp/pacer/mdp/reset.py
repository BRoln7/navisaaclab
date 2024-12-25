from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
import socnav_sim.tasks.amp.pacer.mdp.observations as obs

def root_state_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[:, 2] < minimum_height

def far_from_traj(
    env: ManagerBasedRLEnv, maximum_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    if obs.tar_pos is not None:
        root_pos = asset.data.root_pos_w
        diff_pos = root_pos - obs.tar_pos
        distance = diff_pos[:, 0]**2 + diff_pos[:, 1]**2

    return distance > (maximum_distance**2) # type: ignore