import torch

@torch.jit.script
def compute_location_rew(root_pos, tar_pos):
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward