import torch

def quat_to_rot_matrix(quats, _num_robots):
    qw, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    rot_matrix = torch.zeros((_num_robots, 3, 3), device='cuda:0', dtype=torch.float32)
    rot_matrix[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    rot_matrix[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot_matrix[:, 0, 2] = 2 * (qx * qz + qw * qy)

    rot_matrix[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot_matrix[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    rot_matrix[:, 1, 2] = 2 * (qy * qz - qw * qx)

    rot_matrix[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot_matrix[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot_matrix[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    return rot_matrix

def rotation_transform(relative_positions, quats):
    m, n, _ = relative_positions.shape
    w_Rot_r = quat_to_rot_matrix(quats, m) # (num_robots, 3, 3)
    r_Rot_w = torch.inverse(w_Rot_r)
    relative_positions = relative_positions.unsqueeze(-1)  # (num_robots, num_peds, 3, 1)
    transformed_positions = torch.matmul(r_Rot_w.unsqueeze(1).repeat(1, n, 1, 1), 
                                         relative_positions)  # (m, n, 3, 1)
    transformed_positions = transformed_positions[:, :, :2, 0]

    return transformed_positions