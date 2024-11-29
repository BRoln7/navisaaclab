from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv

# time_out_env_ids = None
# fall_reset_env_ids = None

def amp_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""

    # idx = torch.nonzero(env.episode_length_buf >= env.max_episode_length).squeeze()
    # global time_out_env_ids
    # time_out_env_ids = idx

    return env.episode_length_buf >= env.max_episode_length


def root_state_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # idx = torch.nonzero(asset.data.root_pos_w[:, 2] < minimum_height).squeeze()
    # global fall_reset_env_ids
    # fall_reset_env_ids = idx

    return asset.data.root_pos_w[:, 2] < minimum_height