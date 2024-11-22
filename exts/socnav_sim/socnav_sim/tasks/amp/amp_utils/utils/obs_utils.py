import torch
from socnav_sim.tasks.amp.amp_utils.isaac_utils import torch_utils, gym_utils

dof_subset = torch.tensor([ 
    0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
    46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 
    59, 60, 61, 62, 63, 64, 65]).to("cuda:0")

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

body_flip_left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]

key_body_names = ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]

gym_dof_dict = {'L_Hip_x': 0, 'L_Hip_y': 1, 'L_Hip_z': 2, 'L_Knee_x': 3, 'L_Knee_y': 4, 'L_Knee_z': 5,
                'L_Ankle_x': 6, 'L_Ankle_y': 7, 'L_Ankle_z': 8, 'L_Toe_x': 9, 'L_Toe_y': 10, 'L_Toe_z': 11, 
                'R_Hip_x': 12, 'R_Hip_y': 13, 'R_Hip_z': 14, 'R_Knee_x': 15, 'R_Knee_y': 16, 'R_Knee_z': 17,
                'R_Ankle_x': 18, 'R_Ankle_y': 19, 'R_Ankle_z': 20, 'R_Toe_x': 21, 'R_Toe_y': 22, 'R_Toe_z' : 23,
                'Torso_x' : 24, 'Torso_y' : 25, 'Torso_z' : 26, 'Spine_x' : 27, 'Spine_y' : 28, 'Spine_z' : 29,
                'Chest_x': 30, 'Chest_y': 31, 'Chest_z': 32, 'Neck_x': 33, 'Neck_y': 34, 'Neck_z': 35, 
                'Head_x': 36, 'Head_y': 37, 'Head_z': 38, 'L_Thorax_x': 39, 'L_Thorax_y': 40, 'L_Thorax_z': 41, 
                'L_Shoulder_x': 42, 'L_Shoulder_y': 43, 'L_Shoulder_z': 44, 'L_Elbow_x': 45, 'L_Elbow_y': 46, 'L_Elbow_z': 47, 
                'L_Wrist_x': 48, 'L_Wrist_y': 49, 'L_Wrist_z': 50, 'L_Hand_x': 51, 'L_Hand_y': 52, 'L_Hand_z': 53,   
                'R_Thorax_x': 54, 'R_Thorax_y': 55, 'R_Thorax_z': 56, 'R_Shoulder_x': 57, 'R_Shoulder_y': 58, 'R_Shoulder_z': 59,
                'R_Elbow_x': 60, 'R_Elbow_y': 61, 'R_Elbow_z': 62, 'R_Wrist_x' : 63, 'R_Wrist_y' : 64, 'R_Wrist_z' : 65, 
                'R_Hand_x': 66, 'R_Hand_y': 67, 'R_Hand_z': 68}

sim_joint_dict = {'L_Hip_x': 0, 'L_Hip_y': 1, 'L_Hip_z': 2, 'R_Hip_x': 3, 'R_Hip_y': 4, 'R_Hip_z': 5, 
                'Torso_x': 6, 'Torso_y': 7, 'Torso_z': 8, 'L_Knee_x': 9, 'L_Knee_y': 10, 'L_Knee_z': 11, 
                'R_Knee_x': 12, 'R_Knee_y': 13, 'R_Knee_z': 14, 'Spine_x': 15, 'Spine_y': 16, 'Spine_z': 17, 
                'L_Ankle_x': 18, 'L_Ankle_y': 19, 'L_Ankle_z': 20, 'R_Ankle_x': 21, 'R_Ankle_y': 22, 'R_Ankle_z': 23, 
                'Chest_x': 24, 'Chest_y': 25, 'Chest_z': 26, 'L_Toe_x': 27, 'L_Toe_y': 28, 'L_Toe_z': 29, 
                'R_Toe_x': 30, 'R_Toe_y': 31, 'R_Toe_z': 32, 'Neck_x': 33, 'Neck_y': 34, 'Neck_z': 35, 
                'L_Thorax_x': 36, 'L_Thorax_y': 37, 'L_Thorax_z': 38, 'R_Thorax_x': 39, 'R_Thorax_y': 40, 'R_Thorax_z': 41, 
                'Head_x': 42, 'Head_y': 43, 'Head_z': 44, 'L_Shoulder_x': 45, 'L_Shoulder_y': 46, 'L_Shoulder_z': 47, 
                'R_Shoulder_x': 48, 'R_Shoulder_y': 49, 'R_Shoulder_z': 50, 'L_Elbow_x': 51, 'L_Elbow_y': 52, 'L_Elbow_z': 53, 
                'R_Elbow_x': 54, 'R_Elbow_y': 55, 'R_Elbow_z': 56, 'L_Wrist_x': 57, 'L_Wrist_y': 58, 'L_Wrist_z' : 59,
                'R_Wrist_x': 60, 'R_Wrist_y':61, 'R_Wrist_z':62, 'L_Hand_x':63, 'L_Hand_y':64, 'L_Hand_z':65,
                'R_Hand_x':66, 'R_Hand_y':67, 'R_Hand_z':68}

gym_rigid_body_dict = {'Pelvis': 0, 'L_Hip': 1, 'L_Knee': 2, 'L_Ankle': 3, 'L_Toe': 4, 'R_Hip': 5, 'R_Knee': 6,
                       'R_Ankle': 7, 'R_Toe': 8, 'Torso': 9,'Spine': 10,'Chest': 11,'Neck': 12, 'Head': 13, 
                       'L_Thorax': 14,'L_Shoulder': 15, 'L_Elbow': 16,'L_Wrist': 17,'L_Hand': 18,'R_Thorax': 19,
                       'R_Shoulder': 20,  'R_Elbow': 21,'R_Wrist': 22, 'R_Hand': 23}

sim_rigid_body_dict = {'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Torso': 3, 'L_Knee': 4, 'R_Knee': 5, 'Spine': 6, 
                       'L_Ankle': 7, 'R_Ankle': 8, 'Chest': 9, 'L_Toe': 10, 'R_Toe': 11, 'Neck': 12, 'L_Thorax': 13, 
                       'R_Thorax': 14, 'Head': 15, 'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19, 
                       'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23}

dof_sim_to_gym = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 
                  6, 7, 8, 18, 19, 20, 30, 31, 32, 9, 10, 11, 21, 22, 23, 33, 34, 35, 
                  39, 40, 41, 54, 55, 56, 36, 37, 38, 42, 43, 44, 57, 58, 59, 45, 46, 
                  47, 60, 61, 62, 48, 49, 50, 63, 64, 65, 51, 52, 53, 66, 67, 68]
body_sim_to_gym = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]
dof_gym_to_sim = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 
                  21, 22, 23, 30, 31, 32, 6, 7, 8, 15, 16, 17, 24, 25, 26, 33, 34, 
                  35, 42, 43, 44, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 
                  63, 64, 65, 39, 40, 41, 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68]
body_gym_to_sim = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]

@torch.jit.script
def rot_wxyz_to_xyzw(rot_obs):
    wxyz_to_xyzw = [3, 0, 1, 2]
    rot_obs[:, :, wxyz_to_xyzw] = rot_obs.clone()
    return rot_obs


@torch.jit.script
def root_states_wxyz_to_xyzw(rot_obs):
    wxyz_to_xyzw = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    rot_obs[:, wxyz_to_xyzw] = rot_obs.clone()
    return rot_obs


@torch.jit.script
def remove_base_rot(quat):
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

@torch.jit.script
def compute_location_observations(root_states, traj_samples, upright:bool):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]))
    heading_rot_exp = torch.reshape(
        heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1],
                          heading_rot_exp.shape[2]))
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
         traj_samples_delta.shape[2]))

    local_traj_pos = torch_utils.my_quat_rotate(heading_rot_exp,
                                                traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(local_traj_pos,
                        (traj_samples.shape[0],
                         traj_samples.shape[1] * local_traj_pos.shape[1]))
    return obs


@torch.jit.script
def dof_to_obs_smpl(pose):
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = torch_utils.quat_to_tan_norm(
        torch_utils.exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert ((num_joints * joint_obs_size) == joint_dof_obs.shape[1])

    return joint_dof_obs


@torch.jit.script
def build_amp_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel,
                                dof_pos, dof_vel, key_body_pos,
                                dof_subset, 
                                local_root_obs:bool, root_height_obs:bool, has_dof_subset:bool, upright:bool):
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = gym_utils.quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat(
        (1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2])
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs: obs_list.append(root_h)
    obs_list += [
        root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel,
        flat_local_key_pos
    ]
    obs = torch.cat(obs_list, dim=-1)
    return obs