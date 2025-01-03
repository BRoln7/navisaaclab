from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
import socnav_sim.tasks.socnav.drl_vo.mdp.events as events

def time_runout(env: ManagerBasedRLEnv) -> torch.Tensor:

    reset_ids = env.episode_length_buf >= env.max_episode_length

    return reset_ids

def reach_goal(
        env: ManagerBasedRLEnv, goal_radius: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    N, _ = events.goals.shape
    diff_r = asset.data.root_pos_w[:N, :2] - events.goals
    distance = torch.sqrt(diff_r[:,0]**2 + diff_r[:,1]**2)
    reset_ids = distance < goal_radius 

    reset_ids = torch.cat([reset_ids, torch.tensor([False], device='cuda:0').repeat(env.num_envs-N)])

    return reset_ids

