"""Reflection for MimicKit's native SMPL observations and exponential-map actions."""

import torch


def counterpart(name):
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("R_"):
        return "L_" + name[2:]
    return name


class SmplReflection:
    """Reflect across sagittal (XZ) plane, swapping corresponding left/right bodies.

    Applied in physical coordinates BEFORE observation normalization. Polar vectors
    use (x,-y,z); axial vectors/rotation vectors use (-x,y,-z). Rotation 6D is
    MimicKit's concatenated rotated X and Z axes. No discriminator augmentation.
    """

    def __init__(self, body_names, key_names, future_points=10, device="cpu"):
        joints = list(body_names[1:])
        keys = list(key_names)
        self.joints = torch.tensor([joints.index(counterpart(n)) for n in joints], device=device)
        self.keys = torch.tensor([keys.index(counterpart(n)) for n in keys], device=device)
        self.polar = torch.tensor([1., -1., 1.], device=device)
        self.axial = -self.polar
        self.rotation = self.polar.repeat(2)
        self.future_points = future_points
        self.num_joints = len(joints)
        self.num_keys = len(keys)
        self.action_size = self.num_joints * 3
        self.obs_size = 1 + 6 + 3 + 3 + self.num_joints*6 + self.action_size + self.num_keys*3 + future_points*2

    def action(self, action):
        if action.shape[-1] != self.action_size:
            raise ValueError("Symmetry requires three spherical DOFs per SMPL joint")
        return (action.reshape(*action.shape[:-1], self.num_joints, 3)[..., self.joints, :] * self.axial).flatten(-2)

    def observation(self, obs):
        if obs.shape[-1] != self.obs_size:
            raise ValueError(f"Unexpected MimicKit observation layout: {obs.shape[-1]} != {self.obs_size}")
        h, rot, vel, omega, joints, dofs, keys, future = torch.split(
            obs, [1, 6, 3, 3, self.num_joints*6, self.action_size, self.num_keys*3, self.future_points*2], dim=-1)
        shape = obs.shape[:-1]
        joints = (joints.reshape(*shape, self.num_joints, 6)[..., self.joints, :] * self.rotation).flatten(-2)
        keys = (keys.reshape(*shape, self.num_keys, 3)[..., self.keys, :] * self.polar).flatten(-2)
        future = future.reshape(*shape, self.future_points, 2) * self.polar[:2]
        return torch.cat((h, rot*self.rotation, vel*self.polar, omega*self.axial,
                          joints, self.action(dofs), keys, future.flatten(-2)), dim=-1)
