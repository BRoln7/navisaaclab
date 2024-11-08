from easydict import EasyDict
import torch
from torch import Tensor
import numpy as np
from socnav_sim.tasks.amp.smpl_humanoid.mdp.isaac_utils import torch_utils, gym_utils

# config params
height_conditioned = False
num_height_points = 256
terrain_horizontal_scale = 0.1
sample_width = 1
num_samples_per_axis = 16

dof_offset_indices_isaac_gym_to_sim = torch.tensor(
    [0,  1,  2, 12, 13, 14, 24, 25, 26,  3,  4,  5, 15, 16, 17, 27, 28, 29,
     6,  7,  8, 18, 19, 20, 30, 31, 32,  9, 10, 11, 21, 22, 23, 33, 34, 35,
     39, 40, 41, 54, 55, 56, 36, 37, 38, 42, 43, 44, 57, 58, 59, 45, 46, 47,
     60, 61, 62, 48, 49, 50, 63, 64, 65, 51, 52, 53, 66, 67, 68]).to("cuda:0")

dof_pos_init = torch.tensor(
    [[-0.0140, -0.4742, -0.1278,  0.0699,  0.1536,  0.0104,  0.0312,  0.1021,
      0.0937, -0.0229,  0.2237, -0.1033, -0.0254,  0.2446,  0.1620, -0.0252,
      0.1331, -0.0717,  0.0398, -0.0280,  0.2981, -0.0367, -0.3991, -0.2295,
      -0.0168, -0.0687,  0.0338,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
      0.0000, -0.1267,  0.1423, -0.0479, -0.6345, -0.0646,  0.0377,  0.5659,
      -0.1481, -0.0872,  0.2225, -0.2570,  0.0845, -0.6625,  0.1895, -0.0651,
      0.7634,  0.0903,  0.0430,  0.1243,  0.1841, -0.4212, -0.0301,  0.1777,
      0.4103, -0.1374,  0.3663, -0.2161,  0.1501,  0.4028,  0.1223,  0.0000,
      0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]).to("cuda:0")
dof_vel_init = torch.tensor(
    [[-6.4920e-02, -4.6257e-02, -6.7547e-02, -8.7669e-02,  6.0045e-01,
      -2.1030e-01,  7.6774e-03, -2.3904e-01,  1.2162e-01, -1.1249e-01,
      4.7980e-01, -1.1856e-01, -1.6505e-01, -6.1049e-01,  6.7768e-02,
      -5.3376e-02,  1.0962e-01, -3.9592e-02,  9.0892e-04, -4.1263e-01,
      9.1939e-02, -1.2719e-01, -5.0340e-02, -7.0638e-02, -5.4677e-02,
      6.9695e-03,  1.2052e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00,
      0.0000e+00,  0.0000e+00,  0.0000e+00, -1.2525e-02,  8.9889e-02,
      -1.1416e-01, -1.3892e-01,  3.5504e-03,  1.1212e-01, -3.3329e-04,
      -3.8711e-02, -1.2987e-01, -7.0509e-02, -9.5197e-02, -3.7028e-02,
      -6.5643e-02,  8.4898e-03,  8.9316e-02,  1.3922e-01, -7.2215e-02,
      -2.5090e-01, -2.1959e-02,  2.0733e-01,  2.3881e-01,  6.3994e-02,
      9.8222e-02, -4.2778e-01, -2.1922e-01,  3.4660e-01,  3.6390e-02,
      2.7800e-01,  1.3214e-01, -1.9794e-02,  0.0000e+00,  0.0000e+00,
      0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]).to("cuda:0")


@torch.jit.script
def obs_wxyz_to_xyzw(rot_obs):
    wxyz_to_xyzw = [3, 0, 1, 2]
    rot_obs[:, :, wxyz_to_xyzw] = rot_obs.clone()
    return rot_obs


@torch.jit.script
def remove_base_rot(quat):
    # ZL: removing the base rotation for SMPL model
    base_rot = gym_utils.quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    shape = quat.shape[0]
    return gym_utils.quat_mul(quat, base_rot.repeat(shape, 1))


@torch.jit.script
def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel,
                                           local_root_obs:bool, root_height_obs:bool, upright:bool):
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = gym_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0],
        body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs): ##### ZL Here: it should be "not local_root_obs". If local, it shouldn't be overriden
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(
        flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs: obs_list.append(root_h_obs)
    obs_list += [
        local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel
    ]

    obs = torch.cat(obs_list, dim=-1)
    return obs