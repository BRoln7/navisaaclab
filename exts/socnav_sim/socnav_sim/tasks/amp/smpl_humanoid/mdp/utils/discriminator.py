import torch
import torch.nn as nn
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.running_mean_std import RunningMeanStd
import os

amp_input_shape = 1950

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc_mlp = nn.Sequential(
            nn.Linear(1950, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
        checkpoint_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Humanoid.pth')
        discriminator_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Discriminator.pth')
        amp_mean_std_params = torch.load(checkpoint_path)
        discriminator_params = torch.load(discriminator_path)
        self.amp_input_mean_std = RunningMeanStd(amp_input_shape).to("cuda:0")
        self.amp_input_mean_std.load_state_dict(amp_mean_std_params['amp_input_mean_std'])
        self.amp_input_mean_std.eval()
        self.disc_mlp.load_state_dict(discriminator_params)
        self.disc_mlp.eval()

    def forward(self, obs):
        norm_obs = self.amp_input_mean_std(obs)
        out = self.disc_mlp(norm_obs)
        return out