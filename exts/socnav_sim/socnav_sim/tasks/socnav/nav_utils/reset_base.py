from __future__ import annotations

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    dof_gym_to_sim,
    root_pos_init,
    root_rot_init,
    root_vel_init,
    root_ang_vel_init,
    dof_pos_init,
    dof_vel_init)
import socnav_sim.tasks.socnav.nav_utils.event_base as events
import socnav_sim.tasks.socnav.nav_utils.action_base as action
import socnav_sim.tasks.socnav.nav_utils.map_manager as map_nanager
import socnav_sim.tasks.socnav.nav_utils.planning as planning
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils
import os
import yaml

ped_reset_flag, fall_ids, reach_ids = False, torch.tensor([], device='cuda:0'), torch.tensor([], device='cuda:0')

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
fixed_pose = config['env']['fixed_pose'] # type: ignore

if config['env']['task'] != 'Isaac-Pacer': # type: ignore
    traj_p_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/TrajPedMarkers",
        markers={
            "traj": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)))})
    traj_p_marker = VisualizationMarkers(traj_p_cfg)
    goal_p_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/GoalPedMarkers",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)))})
    goal_p_marker = VisualizationMarkers(goal_p_cfg)

def time_runout(env: ManagerBasedRLEnv) -> torch.Tensor:

    reset_ids = env.episode_length_buf >= env.max_episode_length

    return reset_ids

def reach_goal(
        env: ManagerBasedRLEnv, goal_radius: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    N, _ = events.goals_r_w.shape
    diff_r = asset.data.root_pos_w[:N, :2] - events.goals_r_w
    distance = torch.sqrt(diff_r[:,0]**2 + diff_r[:,1]**2)
    reset_ids = distance < goal_radius 

    reset_ids = torch.cat([reset_ids, torch.tensor([False], device='cuda:0').repeat(env.num_envs-N)])

    return reset_ids

def ped_reset(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ped")
) -> torch.Tensor:
    
    global ped_reset_flag, fall_ids, reach_ids
    ped_asset: RigidObject = env.scene[asset_cfg.name]

    wxyz_to_xyzw = [3, 0, 1, 2]

    if fixed_pose:
        root_pos, root_rot, dof_pos, \
            root_vel, root_ang_vel, dof_vel = root_pos_init, root_rot_init, dof_pos_init,\
            root_vel_init, root_ang_vel_init, dof_vel_init
        root_pos = root_pos.repeat(env.num_envs, 1)
        root_rot = root_rot.repeat(env.num_envs, 1)
        dof_pos = dof_pos.repeat(env.num_envs, 1)
        root_vel = root_vel.repeat(env.num_envs, 1)
        root_ang_vel = root_ang_vel.repeat(env.num_envs, 1)
        dof_vel = dof_vel.repeat(env.num_envs, 1)
    else:
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
        root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = events.motion_load.sample_ref_state(env.num_envs)

    # for fall-reset
    ped_fall_ids = torch.nonzero(ped_asset.data.root_state_w[:, 2] < 0.4).squeeze(-1)

    if ped_fall_ids.numel() != 0:
        
        ped_reset_flag, fall_ids = True, ped_fall_ids

        # root_pos += env.scene.env_origins
        ped_asset.write_root_pose_to_sim(torch.cat([events.spawn_p_w_3d[ped_fall_ids], 
                                                    root_rot[:, wxyz_to_xyzw][ped_fall_ids]], dim=-1), 
                                                    env_ids=ped_fall_ids) # type: ignore
        ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[ped_fall_ids], 
                                                        root_ang_vel[ped_fall_ids]], dim=-1), 
                                                        env_ids=ped_fall_ids) # type: ignore
        ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][ped_fall_ids],  # type: ignore
                                           velocity=dof_vel[:,dof_gym_to_sim][ped_fall_ids], 
                                           env_ids=ped_fall_ids) # type: ignore

    # for reach goal
    goal_dist_2d = ped_asset.data.root_state_w[:, :2] - events.goals_p_w
    goal_dist = torch.sqrt(goal_dist_2d[:,0]**2 + goal_dist_2d[:,1]**2)
    reach_id = torch.nonzero(goal_dist < 0.15).squeeze(-1)

    traj_runout_id = torch.nonzero(action.traj_sample_index[:, -1] >= (events.traj_length_p-100)).squeeze(-1)

    goal_reach_id = torch.cat([reach_id, traj_runout_id])
    goal_reach_id = torch.unique(goal_reach_id)

    if goal_reach_id.numel() != 0:

        ped_reset_flag, reach_ids = True, goal_reach_id

        small_scene_p_id = goal_reach_id[goal_reach_id < (events.num_p_store + events.num_p_garden)]
        station_p_id = goal_reach_id[goal_reach_id >= (events.num_p_store + events.num_p_garden)]

        if small_scene_p_id.numel() != 0:
            # small scene, reset pos to spawn
            ped_asset.write_root_pose_to_sim(torch.cat([events.spawn_p_w_3d[small_scene_p_id], 
                                                        root_rot[:, wxyz_to_xyzw][small_scene_p_id]], dim=-1), 
                                                        env_ids=small_scene_p_id) # type: ignore
            ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[small_scene_p_id], 
                                                            root_ang_vel[small_scene_p_id]], dim=-1), 
                                                            env_ids=small_scene_p_id) # type: ignore
            ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][small_scene_p_id],  # type: ignore
                                            velocity=dof_vel[:,dof_gym_to_sim][small_scene_p_id], 
                                            env_ids=small_scene_p_id) # type: ignore        

        if station_p_id.numel() != 0:
            # station, publish a new goal, edit spawn-pos and trajs
            reach_goal_pos_w = ped_asset.data.root_pos_w[station_p_id, :2]
            reach_goal_pos_m = map_nanager.transform_world_to_map(reach_goal_pos_w, events.map3_info)

            events.spawn_p_w[station_p_id], events.spawn_p_m[station_p_id] = reach_goal_pos_w, reach_goal_pos_m
            events.spawn_p_w_3d[station_p_id] = ped_asset.data.root_pos_w[station_p_id, :3]
            events.goals_p_m[station_p_id], events.goals_p_w[station_p_id] = map_nanager.publish_random_goal(
                                                                                events.spawn_p_w[station_p_id], 
                                                                                events.map3_info, 6.0, 7.0)
            for id in station_p_id:
                path = planning.get_dijkstra_path(events.map3_info.dilated_img, 
                                                events.map3_info.binary_img, 
                                                tuple(events.spawn_p_m[id].cpu().numpy()), 
                                                tuple(events.goals_p_m[id].cpu().numpy()),
                                                traj_optim=True)
                path = planning.path_align_length(path, events.traj_length_p)
                path = map_nanager.transform_map_to_world(path, events.map3_info)
                events.trajs_p[id] = path        


    # C, _ = events.trajs_p.reshape(-1, 2).shape
    # goal_p_marker.visualize(translations=torch.cat([events.goals_p_w, torch.zeros((env.num_envs,1),device='cuda:0')],
    #                                                dim=-1).reshape(-1,3))
    # traj_p_marker.visualize(translations=torch.cat([events.trajs_p.reshape(-1, 2),
    #                                               torch.zeros((C, 1),device='cuda:0')], dim=-1))    

    return  torch.tensor([False], device='cuda:0').repeat(env.num_envs)
