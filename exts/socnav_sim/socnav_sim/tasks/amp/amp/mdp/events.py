from omni.isaac.lab.envs import ManagerBasedEnv
import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
import omni.isaac.lab.utils.math as math_utils
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
import socnav_sim.tasks.amp.amp.policy as policy

amp_obs_reset = False

def reset_init_state_amp_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    wxyz_to_xyzw = [3, 0, 1, 2]
    motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
      root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = policy.motion_load.sample_ref_state(env.num_envs)
    root_pos += env.scene.env_origins
    asset.write_root_pose_to_sim(torch.cat([root_pos[env_ids], root_rot[:, wxyz_to_xyzw][env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_root_velocity_to_sim(torch.cat([root_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids) # type: ignore
    asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim][env_ids], velocity=dof_vel[:,dof_gym_to_sim][env_ids], env_ids=env_ids) # type: ignore

    global amp_motion_ids
    amp_motion_ids = motion_ids
    global amp_motion_times
    amp_motion_times = motion_times
    global amp_obs_reset
    amp_obs_reset = True
    global amp_reset_env_ids
    amp_reset_env_ids = env_ids

