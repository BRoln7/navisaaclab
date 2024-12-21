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

############## for peds ################
def reset_init_state_amp_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ped"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    wxyz_to_xyzw = [3, 0, 1, 2]
    motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
      root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = motion_load.sample_ref_state(env.num_envs)
    root_pos += env.scene.env_origins
    asset.write_root_pose_to_sim(torch.cat([root_pos[env_ids], root_rot[:, wxyz_to_xyzw][env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_root_velocity_to_sim(torch.cat([root_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][env_ids], velocity=dof_vel[:,dof_gym_to_sim][env_ids], env_ids=env_ids) # type: ignore

    global amp_obs_reset
    amp_obs_reset = True
    global reset_env_ids
    reset_env_ids = env_ids

############# for robots ###############
def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids) # type: ignore