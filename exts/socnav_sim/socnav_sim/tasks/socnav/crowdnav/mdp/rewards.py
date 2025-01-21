import torch
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
import socnav_sim.tasks.socnav.crowdnav.mdp.observations as obs
import socnav_sim.tasks.socnav.nav_utils.event_base as event

dist_coeff = 1.0
subgoal_coeff = 10.0
r_waypoint = 3.2

def compute_goal_reward(env: ManagerBasedEnv,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    ##### final goal reward #####
    num_robot, _ = event.goals_r_w.shape
    root_pos = asset.data.root_pos_w[:num_robot, :2]
    dist_2d = event.goals_r_w - root_pos
    distance_goal = torch.sqrt(dist_2d[..., 0]**2 + dist_2d[..., 1]**2)

    rewards = torch.zeros(env.num_envs, device='cuda:0')
    rew_ids = torch.where(distance_goal <= 0.45)
    rewards[rew_ids] = 1

    B, _ = obs.subgoal.shape

    #### getting closer to goal ####
    dist_reward = r_waypoint * obs.dist_diff
    rewards[:B] += dist_reward

    return rewards

def compute_collision_reward(env: ManagerBasedEnv
                    ) -> torch.Tensor:

    rewards = torch.zeros(env.num_envs, device='cuda:0')

    collision_ids = torch.where(obs.d_min <= 0.1)
    rewards[collision_ids] = -0.25

    danger_ids = torch.where((obs.d_min > 0.1) & (obs.d_min <= 0.2))
    rewards[danger_ids] = -0.1 + 0.05 * obs.d_min[danger_ids]

    return rewards
