from easydict import EasyDict
import torch
from torch import Tensor
import numpy as np
from socnav_sim.tasks.amp.smpl_humanoid.mdp.isaac_utils import rotations, torch_utils

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

dof_body_indices_isaac_gym_to_sim = torch.tensor(
    [ 0,  0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11, 13, 18, 12, 14, 19, 
     15, 20, 16, 21, 17, 22]).to('cuda:0')

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


def init_height_points(_num_envs):
    """
    Pre-defines the grid for the height-map observation.
    """
    num_envs = _num_envs

    y = torch.tensor(
        np.linspace(
            -sample_width,
            sample_width,
            num_samples_per_axis,
        ),
        device="cuda:0",
        requires_grad=False,
    )
    x = torch.tensor(
        np.linspace(
            -sample_width,
            sample_width,
            num_samples_per_axis,
        ),
        device="cuda:0",
        requires_grad=False,
    )
    grid_x, grid_y = torch.meshgrid(x, y)

    num_height_points = grid_x.numel()
    points = torch.zeros(
        num_envs,
        num_height_points,
        3,
        device="cuda:0",
        requires_grad=False,
    )
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points


def get_heights_with_scene(root_states):
    """
    This provides the height-map projection that a sensor would see.
    This takes into account objects in the scene, such as chairs, tables, etc...
    Use this function to provide a heightmap representation for the character.
    """
    height_samples = torch.zeros(2800, 1400).to("cuda:0")
    horizontal_scale = 0.1

    return get_heights(
        root_states=root_states,
        height_samples=height_samples,
        horizontal_scale=horizontal_scale,
    )


@torch.jit.script
def get_heights(
    root_states: Tensor,
    height_samples: Tensor,
    horizontal_scale: float,
):
    num_envs = root_states.shape[0]

    points = root_states[..., :2].clone().reshape(num_envs, 1, 2)
    points = (points / horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    heights1 = height_samples[px, py]
    heights2 = height_samples[px + 1, py + 1]
    heights = torch.max(heights1, heights2)

    return heights.view(num_envs, -1)


@torch.jit.script
def compute_humanoid_observations_max(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    ground_height: Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
    w_last: bool,
) -> Tensor:
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_local_body_pos, w_last
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = rotations.quat_mul(flat_heading_rot, flat_body_rot, w_last)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot, w_last)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot, w_last)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = rotations.quat_rotate(flat_heading_rot, flat_body_vel, w_last)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = rotations.quat_rotate(
        flat_heading_rot, flat_body_ang_vel, w_last
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_path_observations(
    root_states: Tensor,
    head_states: Tensor,
    traj_samples: Tensor,
    w_last: bool,
    height_conditioned: bool,
) -> Tensor:
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, w_last
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


@torch.jit.script_if_tracing
def get_height_maps_jit(
    root_states: Tensor,
    base_pos: Tensor,
    env_ids: Tensor,
    num_envs: int,
    height_points: Tensor,
    height_samples: Tensor,
    num_height_points: int,
    terrain_horizontal_scale: float,
    w_last: bool,
    return_all_dims: bool,
):
    base_quat = root_states[:, 3:7]

    points = rotations.quat_apply_yaw(
        base_quat.repeat(1, num_height_points), height_points[env_ids], w_last
    ) + (base_pos[:, :3]).unsqueeze(1)

    points = points / terrain_horizontal_scale
    floored_points = points.long()
    # this encompases 4 possible points.
    # points are the top left corner of the 4 points
    # we will interpolate between the 4 points to get the height
    px = floored_points[:, :, 0].view(-1)
    py = floored_points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    # Calculate the fractional part of the points' positions
    fx = points[:, :, 0].view(-1) - px.float()
    fy = points[:, :, 1].view(-1) - py.float()

    # Get the heights of the four surrounding points
    h_tl = height_samples[px, py]  # Top-left
    h_tr = height_samples[px + 1, py]  # Top-right
    h_bl = height_samples[px, py + 1]  # Bottom-left
    h_br = height_samples[px + 1, py + 1]  # Bottom-right

    # Perform bilinear interpolation
    h_t = h_tl + (h_tr - h_tl) * fx  # Top interpolation
    h_b = h_bl + (h_br - h_bl) * fx  # Bottom interpolation
    interpolated_heights = h_t + (h_b - h_t) * fy  # Final interpolation

    # heights = torch.min(heights1, heights2).view(num_envs, -1)
    heights = base_pos[:, 2:3] - interpolated_heights.view(num_envs, -1)

    if return_all_dims:
        # This is only for visualization purposes, plotting the height map the humanoid sees
        points = rotations.quat_apply_yaw(
            base_quat.repeat(1, num_height_points), height_points[env_ids], w_last
        ) + (base_pos[:, :3]).unsqueeze(1)
        heights = interpolated_heights.view(num_envs, -1, 1)
        return torch.cat(
            [points[..., :2].view(num_envs, -1, 2), heights], dim=-1
        ).clone()

    return heights.view(num_envs, -1).clone()


