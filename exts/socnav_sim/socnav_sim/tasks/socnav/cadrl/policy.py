"""Stable-Baselines3 feature extractors for the 35-D CADRL observation."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CadrlEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 39):
        super().__init__(observation_space, features_dim)
        self.hidden_dim = features_dim - 7
        self.lstm = nn.LSTM(7, self.hidden_dim, batch_first=True)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self_state = observation[:, :7]
        pedestrian_state = observation[:, 7:].reshape(observation.shape[0], -1, 7)
        _, (hidden, _) = self.lstm(pedestrian_state)
        return torch.cat((self_state, hidden.squeeze(0)), dim=-1)


class SarlEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 57):
        super().__init__(observation_space, features_dim)
        self.interaction = nn.Sequential(nn.Linear(14, 150), nn.ReLU(), nn.Linear(150, 100), nn.ReLU())
        self.embedding = nn.Linear(100, features_dim - 7)
        self.attention = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self_state = observation[:, :7]
        pedestrian_state = observation[:, 7:].reshape(observation.shape[0], -1, 7)
        repeated_self = self_state[:, None, :].expand(-1, pedestrian_state.shape[1], -1)
        interaction = self.interaction(torch.cat((repeated_self, pedestrian_state), dim=-1))
        weights = torch.softmax(self.attention(interaction), dim=1)
        crowd_feature = torch.sum(weights * self.embedding(interaction), dim=1)
        return torch.cat((self_state, crowd_feature), dim=-1)
