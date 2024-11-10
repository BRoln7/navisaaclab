import torch
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.rewards_utils import (
    compute_location_rew,
)

power_coefficient = 0.0005

def compute_location_reward(env: ManagerBasedEnv, 
                            tar_pos: torch.tensor,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos = asset.data.root_pos_w[:, :3]
    if tar_pos is None:
        return torch.zeros(env.num_envs).to("cuda:0")    
    location_reward = compute_location_rew(root_pos, tar_pos)

    return location_reward
    
def compute_power_reward(env: ManagerBasedEnv, 
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_force = asset.data.applied_torque / asset.data.joint_damping
    joint_vel = asset.data.joint_vel
    power = torch.abs(torch.multiply(joint_force, joint_vel)).sum(dim = -1)
    power_reward = -power_coefficient * power

    return power_reward

