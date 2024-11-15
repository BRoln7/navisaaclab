import torch
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np
import random
import os
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.running_mean_std import RunningMeanStd
from stable_baselines3 import PPO
from socnav_sim.tasks.amp.smpl_humanoid.mdp.utils.obs_utils import flip_left_to_right_index
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
import socnav_sim.tasks.amp.smpl_humanoid.mdp.observations as mdpobs

obs_shape = 1401
SEED = 4801
seed = 0
random.seed(seed)
np.random.seed(SEED)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

checkpoint_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Humanoid.pth')
running_mean_std_params = torch.load(checkpoint_path)
running_mean_std = RunningMeanStd(obs_shape).to("cuda:0")
running_mean_std.load_state_dict(running_mean_std_params['running_mean_std'])
running_mean_std.eval()
print("[INFO]:running_mean_std init")

origin_observation = None
flip_observation = None

class TaskMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 613):
      super(TaskMLP, self).__init__(observation_space, features_dim)
      self.task_mlp = nn.Sequential(
          nn.Linear(1044, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU())
      self.self_obs_size = 357
      self.task_obs_size = 1044
      # self.running_mean_std = RunningMeanStd(obs_shape).to("cuda:0")
      # self.running_mean_std.eval()
    
    def forward(self, obs):
      # compute for loss
      global origin_observation
      origin_observation = obs
      global flip_observation
      norm_task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
      flip_task_obs = self.compute_flip_task_obs(norm_task_obs)
      flip_observation = torch.cat((mdpobs.flip_humanoid_obs, flip_task_obs), dim=1)

      obs_norm = running_mean_std(obs)
      self_obs = obs_norm[:, :self.self_obs_size]
      task_obs = obs_norm[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
      task_out = self.task_mlp(task_obs)
      task_features = torch.cat([self_obs, task_out], dim = -1)
      assert(task_features.size(1) == self.features_dim)
      return task_features
    
    def compute_flip_task_obs(self, normal_task_obs):
        B, D = normal_task_obs.shape
        traj_samples_dim = 20
        num_height_points = 1024
        obs_acc = []
        normal_task_obs = normal_task_obs.clone()
        traj_samples = normal_task_obs[:, :traj_samples_dim].view(B, 10, 2)
        traj_samples[..., 1] *= -1
        obs_acc.append(traj_samples.view(B, -1))

        height_samples = normal_task_obs[..., traj_samples_dim: 
                                         traj_samples_dim + num_height_points].view(B, 
                                         int(np.sqrt(num_height_points)), int(np.sqrt(num_height_points)))
        height_samples = height_samples.flip(2)
        obs_acc.append(height_samples.view(B, -1))
        obs = torch.cat(obs_acc, dim=1)
        del obs_acc
        return obs
    
########################################################
######## extend from ppo adding symloss item############
########################################################
class SymPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(SymPPO, self).__init__(*args, **kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf # type: ignore
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                #############################################################################
                ############################ here add flip_obs loss #########################
                #############################################################################
                sym_loss = self.get_sym_loss()
                loss += sym_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


    def get_sym_loss(self):
        global flip_observation, origin_observation
        flip_obs_norm = self._preproc_obs(flip_observation)
        orig_obs_norm = self._preproc_obs(origin_observation)
        s_loss = self._sym_loss(flip_obs_norm, orig_obs_norm)
        s_loss = torch.mean(s_loss)
        return s_loss

    def _preproc_obs(self, obs):
        obs_batch_out = running_mean_std(obs)
        return obs_batch_out

    def _sym_loss(self, flip_obs, orig_obs):
        B, _ = flip_obs.shape
        
        with torch.no_grad():
          flip_a, _ = self.policy.predict(flip_obs, deterministic=True)
          orig_a, _ = self.policy.predict(orig_obs, deterministic=True)

        orig_a = orig_a.view(B, -1, 3)
        orig_a[..., 0] *= -1
        orig_a[..., 2] *= -1
        orig_a = orig_a[..., flip_left_to_right_index, :]

        sym_loss = orig_a.view(B, -1) - flip_a

        sym_loss = sym_loss.pow(2).mean(dim = -1) * 50
        return sym_loss