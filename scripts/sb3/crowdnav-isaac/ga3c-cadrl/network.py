import torch
from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

lstm_hidden_dim = 50
robot_state_dim = 6
joint_state_dim = 13
MAX_PED_NUM = 8

"""
joint_state = (robot_state, ped_state)
for LSTM Encoder, input joint state
finally, concat self_state and output of LSTM
"""


class PedsEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 256):
        super(PedsEncoder, self).__init__(observation_space, features_dim)
        self.robot_state_dim = robot_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.joint_state_dim = joint_state_dim
        self.lstm = nn.LSTM(input_dim=joint_state_dim, hidden_dim=lstm_hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim+robot_state_dim, features_dim),
            nn.ReLU()
        )
    """
        robot_state:            0.dg  1.vpref  2. alpha  3.radius  4.px  5.py  
        vicinity agent state:   6.px~  7.py~  8.vx~  9.vy~  10.radius~ 11.da~ 12.radius~+radius(in robot local frame)
        observations size:      [env_nums, ped_nums*joint_state]
    """
    def forward(self, observations: torch.Tensor)->torch.Tensor:
        observations = observations.reshape(-1, MAX_PED_NUM, joint_state_dim)
        size = observations.shape
        robot_state = observations[:, 0, :self.robot_state_dim]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(observations, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([robot_state, hn], dim=1)
        features = self.mlp(joint_state)
        return features
