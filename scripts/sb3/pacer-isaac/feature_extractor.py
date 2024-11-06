import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


norm_shape = 358
device = "cuda:0"

class RunningMeanStd(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 634):
        super(RunningMeanStd, self).__init__(observation_space, features_dim)
        self.mean = nn.Parameter(
            torch.zeros(norm_shape, dtype=torch.float32, device=device), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.ones(norm_shape, dtype=torch.float32, device=device), requires_grad=False
        )

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        else:
            result = arr * torch.sqrt(self.var + 1e-5) + self.mean

        return result
    
    def forward(self, obs):
        humanoid_obs = obs[:, :norm_shape]
        humanoid_obs = self.normalize(humanoid_obs)
        obs_norm = torch.concat((humanoid_obs, obs[:, norm_shape:]), dim=1)
        return obs_norm