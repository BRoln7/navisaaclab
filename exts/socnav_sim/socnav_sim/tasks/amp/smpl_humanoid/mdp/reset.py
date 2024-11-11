from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv

traj_reset_flag = None

def trajectory_end(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]



    return env.episode_length_buf >= env.max_episode_length

def root_state_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    idx = torch.nonzero(asset.data.root_pos_w[:, 2] < minimum_height).squeeze()
    global traj_reset_flag
    traj_reset_flag = idx

    return asset.data.root_pos_w[:, 2] < minimum_height