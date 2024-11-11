import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np
import random
import os
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.running_mean_std import RunningMeanStd

obs_shape = 1401
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
      self.task_mlp = nn.Sequential(
          nn.Linear(1044, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU())
      
      checkpoint_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Humanoid.pth')
      running_mean_std_params = torch.load(checkpoint_path)
      self.running_mean_std = RunningMeanStd(obs_shape).to("cuda:0")
      self.running_mean_std.load_state_dict(running_mean_std_params['running_mean_std'])
      self.running_mean_std.eval()
      
    
    def forward(self, obs):
      obs_norm = self.running_mean_std(obs)
      self.self_obs_size = 357
      self.task_obs_size = 1044
      self_obs = obs_norm[:, :self.self_obs_size]
      task_obs = obs_norm[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
      task_out = self.task_mlp(task_obs)
      task_features = torch.cat([self_obs, task_out], dim = -1)
      assert(task_features.size(1) == self.features_dim)
      return task_features
    