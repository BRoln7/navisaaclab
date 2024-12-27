from omni.isaac.lab.envs import ManagerBasedEnv
import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
import omni.isaac.lab.utils.math as math_utils
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
import socnav_sim.tasks.amp.pacer.policy as policy
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
import os
import yaml

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
num_envs = config['env']['num_envs'] # type: ignore

if config['env']['task'] == 'Isaac-Drlvo': # type: ignore
    motion_load = MotionLoad(num_envs)

    num_robots = config['env']['num_robots'] # type: ignore

robots_init = False

############## for peds ################
def reset_init_state_amp_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ped_asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
):
    ###################### for peds #########################
    ped_asset: RigidObject | Articulation = env.scene[ped_asset_cfg.name]
    wxyz_to_xyzw = [3, 0, 1, 2]
    motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
      root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = motion_load.sample_ref_state(env.num_envs)
    root_pos += env.scene.env_origins
    ped_asset.write_root_pose_to_sim(torch.cat([root_pos[env_ids], root_rot[:, wxyz_to_xyzw][env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    ped_asset.write_root_velocity_to_sim(torch.cat([root_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    ped_asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][env_ids], velocity=dof_vel[:,dof_gym_to_sim][env_ids], env_ids=env_ids) # type: ignore

    global amp_obs_reset
    amp_obs_reset = True
    global reset_env_ids
    reset_env_ids = env_ids

    ##################### for robots ########################
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    global robots_init
    if not robots_init:
        robots_init = True
        root_pose_default = robot_asset.data.default_root_state.clone()[:, :7]
        root_pose_default[num_robots:, :2] += 150
        root_pose_default[..., :3] += env.scene.env_origins
        robot_asset.write_root_pose_to_sim(root_pose_default)
