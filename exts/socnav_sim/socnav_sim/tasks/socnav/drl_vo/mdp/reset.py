from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv

def root_state_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[:, 2] < minimum_height