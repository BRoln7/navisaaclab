import torch
import torch.nn as nn
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
import os
import yaml

amp_input_shape = 1950
motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'amp.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc_mlp = nn.Sequential(
            nn.Linear(1950, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        self.amp_input_mean_std = RunningMeanStd(amp_input_shape).to("cuda:0")
        if config['training']['train']: # type: ignore
            self.amp_input_mean_std.train()
        else:
            self.amp_input_mean_std.eval()

    def forward(self, obs):
        norm_obs = self.amp_input_mean_std(obs)
        disc_logits = self.disc_mlp(norm_obs)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_r = -torch.log(
            torch.maximum(1 - prob,
                            torch.tensor(0.0001, device="cuda:0")))
        return disc_r
    
    
def bias_init(module):
    mlp_init = torch.nn.Identity()
    for m in module.modules():
        if isinstance(m, nn.Linear):
            mlp_init(m.weight)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)
                