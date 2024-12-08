from scipy.spatial.transform import Rotation as R
import torch

class TrajSocnav():
    def __init__(self, num_envs):
        self.traj_pos = torch.zeros(num_envs, 3).to('cuda:0')
        self.yaw = torch.zeros(num_envs).to('cuda:0')
        self.speed_mode = torch.tensor([0.03, 0.03, 0.03], device='cuda:0')
        self.process_buf = torch.zeros(num_envs).to('cuda:0')
        self.current_speed = torch.full((num_envs,), self.speed_mode[0], device='cuda:0') # type: ignore

    def reset_traj(self, q:torch.tensor, pos, env_id):
        wxyz_to_xyzw = [1, 2, 3, 0]
        q = q[:, wxyz_to_xyzw]
        rotation = R.from_quat(q.cpu())
        self.yaw[env_id] = torch.tensor(rotation.as_euler('xyz', degrees=False), 
                                        device='cuda:0').to(torch.float32)[env_id, 2]
        self.traj_pos[env_id] = pos[env_id]
        self.process_buf[env_id] = 0

    def update(self):
        self.traj_pos += torch.stack([
            self.current_speed * torch.cos(self.yaw).squeeze(),
            self.current_speed * torch.sin(self.yaw).squeeze(),
            torch.zeros(self.traj_pos.shape[0], device='cuda:0')
        ], dim=1)    

        env_id = (self.process_buf >= 50)
        self.current_speed[env_id] = self.speed_mode[torch.randint(0, len(self.speed_mode), (env_id.sum(),), device='cuda:0')] # type: ignore
        self.process_buf[env_id] = 0
