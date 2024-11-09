import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np
import random
import os

SEED = 4801
seed = 0
random.seed(seed)
np.random.seed(SEED)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class TaskMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 613):
      super(TaskMLP, self).__init__(observation_space, features_dim)
      self._task_mlp = nn.Sequential(
          nn.Linear(1044, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU())
    
    def forward(self, obs):
      self.self_obs_size = 357
      self.task_obs_size = 1044
      self_obs = obs[:, :self.self_obs_size]
      task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
      task_out = self._task_mlp(task_obs)
      task_features = torch.cat([self_obs, task_out], dim = -1)
      assert(task_features.size(1) == self.features_dim)
      return task_features
        

"""
Network(
  (actor_cnn): Sequential()
  (critic_cnn): Sequential()
  (actor_mlp): Sequential(
    (0): Linear(in_features=613, out_features=2048, bias=True)
    (1): ReLU()
    (2): Linear(in_features=2048, out_features=1024, bias=True)
    (3): ReLU()
  )
  (critic_mlp): Sequential(
    (0): Linear(in_features=613, out_features=2048, bias=True)
    (1): ReLU()
    (2): Linear(in_features=2048, out_features=1024, bias=True)
    (3): ReLU()
  )
  (value): Linear(in_features=1024, out_features=1, bias=True)
  (value_act): Identity()
  (mu): Linear(in_features=1024, out_features=69, bias=True)
  (mu_act): Identity()
  (sigma_act): Identity()
  (_disc_mlp): Sequential(
    (0): Linear(in_features=1950, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=512, bias=True)
    (3): ReLU()
  )
  (_disc_logits): Linear(in_features=512, out_features=1, bias=True)
  (_task_mlp): Sequential(
    (0): Linear(in_features=1044, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
  )
)
"""