import torch
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
import socnav_sim.tasks.amp.pacer.mdp.observations as mdpobs

obs_shape = 357
self_humanoid_obs_size = 357
SEED = 4801
seed = 0
random.seed(seed)
np.random.seed(SEED)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# checkpoint_path = os.path.join(os.getcwd(), 'models', 'pacer', 'Humanoid.pth')
# running_mean_std_params = torch.load(checkpoint_path)
running_mean_std = RunningMeanStd(obs_shape).to("cuda:0")
# running_mean_std.load_state_dict(running_mean_std_params['running_mean_std'])
running_mean_std.eval()
print("[INFO]:running_mean_std init")

flip_left_to_right_index = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17]

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
      obs_norm = running_mean_std(obs)
      self_obs = obs_norm[:, :self.self_obs_size]
      task_obs = obs_norm[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
      task_out = self.task_mlp(task_obs)
      task_features = torch.cat([self_obs, task_out], dim = -1)
      assert(task_features.size(1) == self.features_dim)
      return task_features
    
##########################################################
######## extended from ppo adding symloss item ###########
##########################################################
class SymPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(SymPPO, self).__init__(*args, **kwargs)
        self.flip_obs = None


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            #################################
            ####### flip_obs record #########
            #################################
            task_origin_obs = torch.tensor(self._last_obs[:, self_humanoid_obs_size:]).to("cuda:0") # type: ignore
            flip_task_obs = self.compute_flip_task_obs(task_origin_obs)
            flip_obs = torch.cat((mdpobs.flip_humanoid_obs, flip_task_obs), dim=1)
            if self.flip_obs is None:
                self.flip_obs = flip_obs
            else:
                self.flip_obs = torch.cat((self.flip_obs, flip_obs), dim=0)


        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


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
            flip_observations = self.flip_obs # type: ignore
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
                flip_observations_batch = flip_observations[:self.batch_size, :] # type: ignore
                flip_observations = flip_observations[self.batch_size:, :] # type: ignore
                sym_loss = self.get_sym_loss(flip_observations_batch, rollout_data.observations)
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

        self.flip_obs = None


    def get_sym_loss(self, flip_observations, origin_observations):
        # flip_obs_norm = self._preproc_obs(flip_observations)
        # orig_obs_norm = self._preproc_obs(origin_observations)
        flip_obs_norm = flip_observations
        orig_obs_norm = origin_observations       
        s_loss = self._sym_loss(flip_obs_norm.cpu(), orig_obs_norm.cpu())
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

        orig_a = torch.tensor(orig_a).to("cuda:0")
        flip_a = torch.tensor(flip_a).to("cuda:0")
        orig_a = orig_a.view(B, -1, 3)
        orig_a[..., 0] *= -1
        orig_a[..., 2] *= -1
        orig_a = orig_a[..., flip_left_to_right_index, :]

        sym_loss = orig_a.view(B, -1) - flip_a

        sym_loss = sym_loss.pow(2).mean(dim = -1) * 50
        return sym_loss