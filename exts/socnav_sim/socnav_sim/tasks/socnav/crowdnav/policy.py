###################### feature extractor ###########################
import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import os
import yaml

def config_init():
    motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
    with open(motion_file, 'r') as file:
        config = yaml.safe_load(file)
    global num_envs
    num_envs = config['env']['num_envs'] # type: ignore
config_init()

# ga3c-cadrl
class CadrlEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 39):
        super(CadrlEncoder, self).__init__(observation_space, features_dim)
        self.input_dim = 7
        self.hidden_dim = 32
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, device='cuda:0')
    
    def forward(self, input_obs):
        B, _ = input_obs.shape
        self_obs = input_obs[..., :7]
        agent_obs = input_obs[..., 7:].reshape(B, -1, 7)

        h0 = torch.zeros(1, B, self.hidden_dim, device='cuda:0')
        c0 = torch.zeros(1, B, self.hidden_dim, device='cuda:0')
        output, (hn, cn) = self.lstm(agent_obs, (h0, c0))

        hn = hn.squeeze(0)
        joint_obs = torch.cat([self_obs, hn], dim=-1)

        return joint_obs

# sarl
class SarlEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(SarlEncoder, self).__init__(observation_space)
        self.mlp1 = nn.Sequential(
            nn.Linear(13, 150),
            nn.ReLU(),
            nn.Linear(150, 100)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(100, 50)
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, input_obs):
        agent_state = input_obs[:, 13:].reshape(num_envs, -1, 7)
        _, num_agents, _ = agent_state.shape
        self_state = input_obs[:, :13].unsqueeze(1).repeat(1, num_agents, 1)
        interaction_input = torch.cat([self_state, agent_state], dim=-1)
        
        e_n = self.mlp1(interaction_input)
        h_n = self.mlp2(e_n)

        e_m = torch.mean(e_n, dim=-1).unsqueeze(1).repeat(1, num_agents, 1)
        attention_input = torch.cat([e_n, e_m], dim=-1)
        attention_output = self.attention_mlp(attention_input)
        alpha_n = nn.Softmax(attention_output)

        h_n = h_n * alpha_n
        final_feature = torch.sum(h_n, dim=-1)

        return final_feature




    
