from omni.isaac.lab.envs import ManagerBasedEnv
import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    dof_gym_to_sim,
    root_pos_init,
    root_rot_init,
    root_vel_init,
    root_ang_vel_init,
    dof_pos_init,
    dof_vel_init)
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
import omni.isaac.lab.sim as sim_utils
import socnav_sim.tasks.socnav.nav_utils.map_manager as map_nanager
import socnav_sim.tasks.socnav.nav_utils.planning as planning
import os
import yaml
import socnav_sim.tasks.socnav.crowdnav.mdp.observations as obs

# fixed positions for small scenes
num_r_store = 2
num_p_store = 3
# num_r_store = 0
# num_p_store = 0
joint_r_store = torch.tensor([[30, 25, 60, 25], [40, 25, 60, 30], 
                              [50, 25, 58, 45], [100, 90, 70, 88]], device='cuda:0', dtype=torch.int)
joint_p_store = torch.tensor([[33, 90, 60, 90], [60, 90, 33, 90], 
                              [55, 20, 55, 45]], device='cuda:0', dtype=torch.int)
num_r_garden = 3
num_p_garden = 6
# num_r_garden = 0
# num_p_garden = 0
joint_r_garden = torch.tensor([[30, 62, 48, 60], [70, 140, 85, 125], 
                               [10, 40, 10, 60]], device='cuda:0', dtype=torch.int)
joint_p_garden = torch.tensor([[20, 20, 50, 20], [20, 30, 50, 20], [20, 40, 50, 40], 
                               [80, 20, 50, 20], [80, 30, 50, 20], [80, 40, 50, 40], 
                               [10, 110, 40, 120], [10, 130, 40, 140], 
                               [34, 82, 70, 82]], device='cuda:0', dtype=torch.int)

traj_length_r = 400
traj_length_p = 700

def config_init():
    motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
    with open(motion_file, 'r') as file:
        config = yaml.safe_load(file)
        global num_envs, num_robots,  num_r_station, num_p_station, fixed_pose
        num_envs = config['env']['num_envs'] # type: ignore
        # num_envs = 16
        num_robots = config['env']['num_robots'] # type: ignore
        num_r_station = max(num_robots - num_r_store - num_r_garden, 1)
        num_p_station = max(num_envs - num_p_store - num_p_garden, 2)

    if config['env']['task'] != 'Isaac-Pacer': # type: ignore
        global motion_load, goal_r_marker, traj_r_marker, ped_init, trajs_r, trajs_p

        fixed_pose = config['env']['fixed_pose'] # type: ignore
        if not fixed_pose:
            motion_load = MotionLoad(num_envs)
        ped_init = False

        goal_r_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/GoalRobotMarkers",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))})
        goal_r_marker = VisualizationMarkers(goal_r_cfg)
        traj_r_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TrajRobotMarkers",
            markers={
                "traj": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))})
        traj_r_marker = VisualizationMarkers(traj_r_cfg)

        trajs_r = torch.zeros(num_robots, traj_length_r, 2, device='cuda:0')
        trajs_p = torch.zeros(num_envs, traj_length_p, 2, device='cuda:0')
        
config_init()

dilated_img1, binary_img1, res1, origin1 = map_nanager.load_image_scaled("data/usd/scenes/scene_crowdnav/store_map.png",
                                                                         "data/usd/scenes/scene_crowdnav/store_map.yaml",
                                                                     1/4, 4)
dilated_img2, binary_img2, res2, origin2 = map_nanager.load_image_scaled("data/usd/scenes/scene_crowdnav/garden_map.png",
                                                                         "data/usd/scenes/scene_crowdnav/garden_map.yaml",
                                                                     1/4, 4)
dilated_img3, binary_img3, res3, origin3 = map_nanager.load_image_scaled("data/usd/scenes/scene_crowdnav/station_map.png",
                                                                         "data/usd/scenes/scene_crowdnav/station_map.yaml",
                                                                     1/4, 4)
map1_info = map_nanager.MapInfo(dilated_img=dilated_img1, binary_img=binary_img1, res=res1, origin=origin1)
map2_info = map_nanager.MapInfo(dilated_img=dilated_img2, binary_img=binary_img2, res=res2, origin=origin2)
map3_info = map_nanager.MapInfo(dilated_img=dilated_img3, binary_img=binary_img3, res=res3, origin=origin3)

# goals of station should be published randomly in the events
spawn_r_m, spawn_r_w, goals_r_w, goals_r_m = map_nanager.get_spawn_and_goal(num_robots, num_r_store, num_r_garden, 
                                                                            num_r_station, joint_r_store, joint_r_garden, 
                                                                            map1_info, map2_info, map3_info)
spawn_p_m, spawn_p_w, goals_p_w, goals_p_m = map_nanager.get_spawn_and_goal(num_envs, num_p_store, num_p_garden, 
                                                                            num_p_station, joint_p_store, joint_p_garden, 
                                                                            map1_info, map2_info, map3_info)

# spawn_p_m, spawn_p_w = map_nanager.random_spawn(map3_info, 2)
# goals_p_w, goals_p_m = torch.zeros_like(spawn_p_m, device='cuda:0', dtype=torch.float), torch.zeros_like(spawn_p_w, device='cuda:0', dtype=torch.int)
# spawn_r_m, spawn_r_w = map_nanager.random_spawn(map3_info, 1)
# goals_r_w, goals_r_m = torch.zeros_like(spawn_r_m, device='cuda:0', dtype=torch.float), torch.zeros_like(spawn_r_w, device='cuda:0', dtype=torch.int)

spawn_p_w_3d = torch.cat([spawn_p_w, torch.ones(num_envs, 1, device='cuda:0')*0.9556],dim=-1)

# import matplotlib.pyplot as plt
# plt.imshow(dilated_img, cmap='gray')

def reset_init_state_amp_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ped_asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
):
    '''
    peds would continuously walk following trajs. Here peds only reset once just at the initial phrase
    robots would reset to the default places
    '''

    ##################### for peds init, just reset and planning once #######################
    ped_asset: RigidObject | Articulation = env.scene[ped_asset_cfg.name]

    store_r_id = env_ids[env_ids < num_r_store]
    garden_r_id = env_ids[(env_ids >= num_r_store) & (env_ids < (num_r_store + num_r_garden))]
    station_r_id = env_ids[(env_ids >= (num_r_store + num_r_garden)) & (env_ids < num_robots)]
    
    global ped_init, goals_r_w, goals_r_m, trajs_r
    if not ped_init:
        ped_init = True
        ped_init_ids = torch.arange(env.num_envs, device='cuda:0')
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
            root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = motion_load.sample_ref_state(env.num_envs)

        ped_asset.write_root_pose_to_sim(torch.cat([spawn_p_w_3d[ped_init_ids], 
                                                    root_rot[:, wxyz_to_xyzw][ped_init_ids]], dim=-1), 
                                                    env_ids=ped_init_ids) # type: ignore
        ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[ped_init_ids], 
                                                        root_ang_vel[ped_init_ids]], dim=-1), 
                                                        env_ids=ped_init_ids) # type: ignore
        ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][ped_init_ids],  # type: ignore
                                           velocity=dof_vel[:,dof_gym_to_sim][ped_init_ids], 
                                           env_ids=ped_init_ids) # type: ignore
        
        global goals_p_w, goals_p_m
        store_p_id = env_ids[env_ids < num_p_store]
        garden_p_id = env_ids[(env_ids >= num_p_store) & (env_ids < (num_p_store + num_p_garden))]
        station_p_id = env_ids[env_ids >= (num_p_store + num_p_garden)]
        goals_p_m[station_p_id], goals_p_w[station_p_id] = map_nanager.publish_random_goal(spawn_p_w[station_p_id], map3_info, 6.0, 7.0)

        for id in store_p_id:
            path = planning.get_dijkstra_path(map1_info.dilated_img, 
                                              map1_info.binary_img, 
                                              tuple(spawn_p_m[id].cpu().numpy()), 
                                              tuple(goals_p_m[id].cpu().numpy()),
                                              traj_optim=True)
            path = planning.path_align_length(path, traj_length_p)
            path = map_nanager.transform_map_to_world(path, map1_info)
            trajs_p[id] = path
        for id in garden_p_id:
            path = planning.get_dijkstra_path(map2_info.dilated_img, 
                                              map2_info.binary_img, 
                                              tuple(spawn_p_m[id].cpu().numpy()), 
                                              tuple(goals_p_m[id].cpu().numpy()),
                                              traj_optim=True)
            path = planning.path_align_length(path, traj_length_p)
            path = map_nanager.transform_map_to_world(path, map2_info)
            trajs_p[id] = path
        for id in station_p_id:
            path = planning.get_dijkstra_path(map3_info.dilated_img, 
                                              map3_info.binary_img, 
                                              tuple(spawn_p_m[id].cpu().numpy()), 
                                              tuple(goals_p_m[id].cpu().numpy()),
                                              traj_optim=True)
            path = planning.path_align_length(path, traj_length_p)
            path = map_nanager.transform_map_to_world(path, map3_info)
            trajs_p[id] = path

    ###################### for robots reset ########################
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    
    root_pose_default = robot_asset.data.default_root_state.clone()[:, :7]
    root_pose_default[num_robots:, :2] += 100
    root_pose_default[..., :3] += env.scene.env_origins
    root_pose_default[:num_robots, :2] = spawn_r_w

    robot_asset.write_root_pose_to_sim(root_pose_default[env_ids], env_ids=env_ids) # type: ignore
    
    ############################## path planning for robots ###############################
    if spawn_r_w[station_r_id].numel() != 0:
        goals_r_m[station_r_id], goals_r_w[station_r_id] = map_nanager.publish_random_goal(spawn_r_w[station_r_id], 
                                                                                           map3_info, 4.0, 5.0)
    # path planning and lenght alignment
    for id in store_r_id:
        path = planning.get_dijkstra_path(map1_info.dilated_img, 
                                          map1_info.binary_img, 
                                          tuple(spawn_r_m[id].cpu().numpy()), 
                                          tuple(goals_r_m[id].cpu().numpy()),
                                          traj_optim=True)
        path = planning.path_align_length(path, traj_length_r)
        path = map_nanager.transform_map_to_world(path, map1_info)
        trajs_r[id] = path
    for id in garden_r_id:
        path = planning.get_dijkstra_path(map2_info.dilated_img, 
                                          map2_info.binary_img, 
                                          tuple(spawn_r_m[id].cpu().numpy()), 
                                          tuple(goals_r_m[id].cpu().numpy()),
                                          traj_optim=True)
        path = planning.path_align_length(path, traj_length_r)
        path = map_nanager.transform_map_to_world(path, map2_info)
        trajs_r[id] = path
    for id in station_r_id:
        path = planning.get_dijkstra_path(map3_info.dilated_img,
                                          map3_info.binary_img,
                                          tuple(spawn_r_m[id].cpu().numpy()), 
                                          tuple(goals_r_m[id].cpu().numpy()),
                                          traj_optim=True)
        path = planning.path_align_length(path, traj_length_r)
        path = map_nanager.transform_map_to_world(path, map3_info)
        trajs_r[id] = path

    # for visualize
    B, _ = trajs_r.reshape(-1, 2).shape
    goal_r_marker.visualize(translations=torch.cat([goals_r_w, torch.zeros((num_robots,1),device='cuda:0')],
                                                 dim=-1).reshape(-1,3))
    traj_r_marker.visualize(translations=torch.cat([trajs_r.reshape(-1, 2),
                                                  torch.zeros((B, 1),device='cuda:0')], dim=-1))
    
    robot_id = env_ids[env_ids < num_robots]
    obs.last_dg[robot_id] = 0.0