# put the desired policy.pth in data/checkpoints, then run command "python script/tools convert_sb3_policy.py"

import torch
import os

actions_path = os.path.join(os.getcwd(), 'data', 'checkpoints', 'policy.pth')
origin_policy = torch.load(actions_path)

network_weights = {}

network_weights['running_mean_std.running_mean'] = origin_policy["features_extractor.running_mean_std.running_mean"]
network_weights['running_mean_std.running_var'] = origin_policy["features_extractor.running_mean_std.running_var"]
network_weights['running_mean_std.count'] = origin_policy["features_extractor.running_mean_std.count"]

network_weights['disc_mlp.0.weight'] = origin_policy["mlp_extractor.policy_net.0.weight"]
network_weights['disc_mlp.0.bias'] = origin_policy["mlp_extractor.policy_net.0.bias"]
network_weights['disc_mlp.2.weight'] = origin_policy["mlp_extractor.policy_net.2.weight"]
network_weights['disc_mlp.2.bias'] = origin_policy["mlp_extractor.policy_net.2.bias"]
network_weights['disc_mlp.4.weight'] = origin_policy["action_net.weight"]
network_weights['disc_mlp.4.bias'] = origin_policy["action_net.bias"]

torch.save(network_weights, os.path.join(os.getcwd(), 'data', 'checkpoints', 'actions_policy.pth'))