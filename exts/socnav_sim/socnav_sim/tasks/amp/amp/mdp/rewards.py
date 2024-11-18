import torch
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
power_coefficient = 0.0005


def compute_power_reward(env: ManagerBasedEnv, 
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_force = asset.data.applied_torque / asset.data.joint_damping
    joint_vel = asset.data.joint_vel
    power = torch.abs(torch.multiply(joint_force, joint_vel)).sum(dim = -1)
    power_reward = -power_coefficient * power

    return power_reward

