import gym.spaces
import torch
from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

lstm_hidden_dim = 50
robot_state_dim = 6
joint_state_dim = 13

"""
robot_state:            0.dg  1.vpref  2. alpha  3.radius  4.px  5.py
vicinity agent state:   6.px~  7.py~  8.vx~  9.vy~  10.radius~ 11.da~ 12.radius~+radius(in robot local frame)
"""

class CadrlEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim = 256):
        super(CadrlEncoder, self).__init__(observation_space, features_dim) # type: ignore
        self.robot_state_dim = robot_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.joint_state_dim = joint_state_dim
        self.lstm = nn.LSTM(input_dim=joint_state_dim, hidden_dim=lstm_hidden_dim, batch_first=True)

    def forward(self, observations: torch.Tensor)->torch.Tensor:
        """
        observation: (num_envs(num_agents), num_agents(top one is self), joint_dim(top one only have self states))
        """
        shape = observations.shape
        robot_state = observations[:, 0, :self.robot_state_dim]
        #   [num_layers * num_directions, batch_size, hidden_dim]
        h0 = torch.zeros(1, shape[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, shape[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(observations[:, 1:, :], (h0, c0))
        features = torch.cat([robot_state, hn], dim=2)
        return features