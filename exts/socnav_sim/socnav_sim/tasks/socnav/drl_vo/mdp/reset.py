# from __future__ import annotations

# import torch
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.assets import Articulation, RigidObject
# from omni.isaac.lab.envs import ManagerBasedRLEnv
# from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
# import socnav_sim.tasks.socnav.drl_vo.mdp.events as events

# ped_fall_reset, reset_env_ids = None, None

# def time_runout(env: ManagerBasedRLEnv) -> torch.Tensor:

#     reset_ids = env.episode_length_buf >= env.max_episode_length

#     return reset_ids

# def reach_goal(
#         env: ManagerBasedRLEnv, goal_radius: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: RigidObject = env.scene[asset_cfg.name]

#     N, _ = events.goals.shape
#     diff_r = asset.data.root_pos_w[:N, :2] - events.goals
#     distance = torch.sqrt(diff_r[:,0]**2 + diff_r[:,1]**2)
#     reset_ids = distance < goal_radius 

#     reset_ids = torch.cat([reset_ids, torch.tensor([False], device='cuda:0').repeat(env.num_envs-N)])

#     return reset_ids

# def ped_fall(
#         env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
# ) -> torch.Tensor:
#     ped_asset: RigidObject = env.scene[asset_cfg.name]

#     ped_reset_ids = ped_asset.data.root_state_w[:, 2] < 0.4
#     ped_reset_ids = torch.nonzero(ped_reset_ids).squeeze(-1)

#     if ped_reset_ids.numel() != 0:
#         wxyz_to_xyzw = [3, 0, 1, 2]
#         motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
#         root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = events.motion_load.sample_ref_state(env.num_envs)
#         root_pos += env.scene.env_origins
#         ped_asset.write_root_pose_to_sim(torch.cat([root_pos[ped_reset_ids], 
#                                                     root_rot[:, wxyz_to_xyzw][ped_reset_ids]], dim=-1), 
#                                                     env_ids=ped_reset_ids) # type: ignore
#         ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[ped_reset_ids], 
#                                                         root_ang_vel[ped_reset_ids]], dim=-1), 
#                                                         env_ids=ped_reset_ids) # type: ignore
#         ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][ped_reset_ids],  # type: ignore
#                                         velocity=dof_vel[:,dof_gym_to_sim][ped_reset_ids], 
#                                         env_ids=ped_reset_ids) # type: ignore
#         global ped_fall_reset
#         ped_fall_reset = True
#         global reset_env_ids
#         reset_env_ids = ped_reset_ids

#     return  torch.tensor([False], device='cuda:0').repeat(env.num_envs)

        