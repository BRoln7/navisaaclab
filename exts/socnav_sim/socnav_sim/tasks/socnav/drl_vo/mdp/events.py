# from omni.isaac.lab.envs import ManagerBasedEnv
# import torch
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.assets import Articulation, RigidObject
# from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
# from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
# from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
# import omni.isaac.lab.sim as sim_utils
# import socnav_sim.tasks.socnav.nav_utils.map_manager as map_nanager
# import os
# import yaml

# def config_init():
#     motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
#     with open(motion_file, 'r') as file:
#         config = yaml.safe_load(file)
#     if config['env']['task'] == 'Isaac-Drlvo': # type: ignore
#         global num_envs, motion_load, num_robots, goal_marker, robot_init, ped_init, map_path, yaml_path, goals

#         num_envs = config['env']['num_envs'] # type: ignore
#         motion_load = MotionLoad(num_envs)
#         num_robots = config['env']['num_robots'] # type: ignore
#         robot_init = False
#         ped_init = False

#         goal_cfg = VisualizationMarkersCfg(
#             prim_path="/World/Visuals/GoalMarkers",
#             markers={
#                 "goal": sim_utils.SphereCfg(
#                     radius=0.25,
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.5)))})
#         goal_marker = VisualizationMarkers(goal_cfg)
#         goals = torch.ones((num_robots, 2),device='cuda:0')
        
# config_init()

# map_pic, res, origin = map_nanager.load_map('data/usd/scenes/warehouse/warehouse.png',
#                                             'data/usd/scenes/warehouse/warehouse.yaml')
# map_info = map_nanager.MapInfo(binary_map=map_pic, res=res, origin=origin)

# def reset_init_state_amp_motion(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ped_asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
# ):
#     '''
#     peds would continuously walk following trajs
#     robots would reset to the default places
#     '''
#     #################### for peds init #######################
#     ped_asset: RigidObject | Articulation = env.scene[ped_asset_cfg.name]
    
#     global ped_init
#     if not ped_init:
#         ped_init = True
#         ped_init_ids = torch.arange(0, env.num_envs, 1, device='cuda:0')
#         wxyz_to_xyzw = [3, 0, 1, 2]
#         motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
#         root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = motion_load.sample_ref_state(env.num_envs)
#         root_pos += env.scene.env_origins
#         ped_asset.write_root_pose_to_sim(torch.cat([root_pos[ped_init_ids], 
#                                                     root_rot[:, wxyz_to_xyzw][ped_init_ids]], dim=-1), 
#                                                     env_ids=ped_init_ids) # type: ignore
#         ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[ped_init_ids], 
#                                                         root_ang_vel[ped_init_ids]], dim=-1), 
#                                                         env_ids=ped_init_ids) # type: ignore
#         ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][ped_init_ids],  # type: ignore
#                                         velocity=dof_vel[:,dof_gym_to_sim][ped_init_ids], 
#                                         env_ids=ped_init_ids) # type: ignore
    
#     ###################### for robots ########################
#     robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]

#     global robot_init
#     if not robot_init:
#         robot_init = True

#         root_pose_default = robot_asset.data.default_root_state.clone()[:, :7]
#         root_pose_default[num_robots:, :2] += 150
#         root_pose_default[..., :3] += env.scene.env_origins
#         robot_asset.write_root_pose_to_sim(root_pose_default)

#         start_points = robot_asset.data.root_pos_w[:num_robots, :2]
#         global goals
#         goals = map_nanager.publish_random_goal(start_points, map_info)
#         goal_marker.visualize(translations=torch.cat([goals, 
#                                                       torch.zeros((num_robots,1),device='cuda:0')],
#                                                       dim=-1).reshape(-1,3))
#     else:
#         root_pose_default = robot_asset.data.default_root_state.clone()[:, :7]
#         root_pose_default[num_robots:, :2] += 150
#         root_pose_default[..., :3] += env.scene.env_origins
#         robot_asset.write_root_pose_to_sim(root_pose_default[env_ids], env_ids=env_ids) # type: ignore
        
#         start_points = robot_asset.data.root_pos_w[env_ids, :2]
#         goals[env_ids] = map_nanager.publish_random_goal(start_points, map_info)
#         goal_marker.visualize(translations=torch.cat([goals, 
#                                                       torch.zeros((num_robots,1),device='cuda:0')],
#                                                       dim=-1).reshape(-1,3))
