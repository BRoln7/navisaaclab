import torch
import torch.nn as nn
from torch.optim import Adam
import os
import yaml

amp_input_shape = 1950
motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)

DISC_LOGIT_INIT_SCALE = 1.0

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.disc_mlp = nn.Sequential(
            nn.Linear(1950, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU())
        
        self.disc_logit = nn.Linear(512, 1)

        torch.nn.init.uniform_(self.disc_logit.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        torch.nn.init.zeros_(self.disc_logit.bias)

        self.optimizer = Adam(self.parameters(), lr=2e-5, eps=1e-08, weight_decay=0.0)

    def forward(self, obs):
        disc_obs = self.disc_mlp(obs)
        disc_logits = self.disc_logit(disc_obs)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_r = -torch.log(
            torch.maximum(1 - prob,
                            torch.tensor(0.0001, device="cuda:0")))
        return disc_r
    
    def get_logit(self, obs):
        disc_obs = self.disc_mlp(obs)
        disc_logits = self.disc_logit(disc_obs)

        return disc_logits
    
def bias_init(module):
    mlp_init = torch.nn.Identity()
    for m in module.modules():
        if isinstance(m, nn.Linear):
            mlp_init(m.weight)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)
                