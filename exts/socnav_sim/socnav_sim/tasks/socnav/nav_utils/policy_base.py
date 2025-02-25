import torch
import numpy as np
import os
import yaml

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor
from typing import Iterable, Optional, Union
import io
import pathlib
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from gymnasium import spaces
import time

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
if config['env']['task'] != 'Isaac-Pacer': # type: ignore
    num_robots = config['env']['num_robots'] # type: ignore
    log_name = config['env']['log_name'] # type: ignore


class MRPPO(PPO):
    '''
    Multi-Robots PPO (MRPPO)
    num of robots to be used now is not equal with num of envs, as some of robots are abandoned
    so in this case, we should just collect trajs of envs used
    '''

    def __init__(self, *args, **kwargs):
        super(MRPPO, self).__init__(*args, **kwargs)
        # self.reward_time_record = torch.zeros(num_robots, 10, 2, device='cuda:0') # include reward and training time
        # self.curr_episode_rew = torch.zeros(num_robots, device='cuda:0')
        # self.init_time = None
        # self.policy_update_num = 0

    def _setup_model(self) -> None:

        super()._setup_model()

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=num_robots,
            **self.rollout_buffer_kwargs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device) # type: ignore
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            # episode reward record
            # rewards_tensor = torch.tensor(rewards, device='cuda:0')
            # self.curr_episode_rew += rewards_tensor[:num_robots]

            # dones_tensor = torch.tensor(dones, device='cuda:0')[:num_robots]
            # dones_ids = torch.nonzero(dones_tensor).squeeze(-1)

            # if dones_ids.numel() != 0:
            #     training_time = torch.tensor(time.time() - self.init_time, device='cuda:0').repeat(dones_ids.numel())
            #     rew_time = torch.cat([self.curr_episode_rew[dones_ids].unsqueeze(-1), 
            #                           training_time.unsqueeze(-1)], dim=-1)
            #     self.reward_time_record[dones_ids] = torch.cat([self.reward_time_record[dones_ids][:, 1:], 
            #                                                     rew_time.unsqueeze(1)], dim=1)
            #     self.curr_episode_rew[dones_ids] = 0

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

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
                self._last_obs[:num_robots],  # type: ignore[arg-type]
                actions[:num_robots],
                rewards[:num_robots],
                self._last_episode_starts[:num_robots],  # type: ignore[arg-type]
                values[:num_robots],
                log_probs[:num_robots],
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))[:num_robots] # type: ignore[arg-type]
        dones = dones[:num_robots]
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        super().save(path, exclude, include)


    def train(self):
        super().train()

        # self.logger.record("train/rew_mean", self.rollout_buffer.rewards.mean().item())
        # self.policy_update_num += 1
        # print('[INFO]: policy update num: ', self.policy_update_num)
        # if self.policy_update_num == 500:
        #     torch.save(self.reward_time_record, log_name+'-500.pt')
        #     print('[INFO]: policy update finished')
        # if (time.time() - self.init_time >= 900) and (time.time() - self.init_time <= 930):
        #     torch.save(self.reward_time_record, log_name+'-15min.pt')
        #     print('[INFO]: training time fininshed')


    # def train(self) -> None:
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """

    #     if self.init_time == None:
    #         self.init_time = time.time()

    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(True)
    #     # Update optimizer learning rate
    #     self._update_learning_rate(self.policy.optimizer)
    #     # Compute current clip range
    #     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    #     # Optional: clip range for the value function
    #     if self.clip_range_vf is not None:
    #         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    #     entropy_losses = []
    #     pg_losses, value_losses = [], []
    #     clip_fractions = []

    #     continue_training = True
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         approx_kl_divs = []
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             actions = rollout_data.actions
    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 actions = rollout_data.actions.long().flatten()

    #             # Re-sample the noise matrix because the log_std has changed
    #             if self.use_sde:
    #                 self.policy.reset_noise(self.batch_size)

    #             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
    #             values = values.flatten()
    #             # Normalize advantage
    #             advantages = rollout_data.advantages
    #             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    #             if self.normalize_advantage and len(advantages) > 1:
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #             # ratio between old and new policy, should be one at the first iteration
    #             ratio = torch.exp(log_prob - rollout_data.old_log_prob)

    #             # clipped surrogate loss
    #             policy_loss_1 = advantages * ratio
    #             policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    #             policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    #             # Logging
    #             pg_losses.append(policy_loss.item())
    #             clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
    #             clip_fractions.append(clip_fraction)

    #             if self.clip_range_vf is None:
    #                 # No clipping
    #                 values_pred = values
    #             else:
    #                 # Clip the difference between old and new value
    #                 # NOTE: this depends on the reward scaling
    #                 values_pred = rollout_data.old_values + torch.clamp(
    #                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf # type: ignore
    #                 )
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())

    #             # Entropy loss favor exploration
    #             if entropy is None:
    #                 # Approximate entropy when no analytical form
    #                 entropy_loss = -torch.mean(-log_prob)
    #             else:
    #                 entropy_loss = -torch.mean(entropy)

    #             entropy_losses.append(entropy_loss.item())

    #             loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    #             # Calculate approximate form of reverse KL Divergence for early stopping
    #             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
    #             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
    #             # and Schulman blog: http://joschu.net/blog/kl-approx.html
    #             with torch.no_grad():
    #                 log_ratio = log_prob - rollout_data.old_log_prob
    #                 approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
    #                 approx_kl_divs.append(approx_kl_div)

    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 continue_training = False
    #                 if self.verbose >= 1:
    #                     print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
    #                 break

    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward(retain_graph=False)
    #             # Clip grad norm
    #             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #             self.policy.optimizer.step()

    #             self.policy_update_num += 1
    #             if self.policy_update_num == 5000:
    #                 training_time = time.time() - self.init_time
    #                 with open('training_time.txt', 'w') as file:
    #                     file.write(f"{training_time}")

    #         self._n_updates += 1
    #         if not continue_training:
    #             break

    #     explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    #     # Logs
    #     self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    #     self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    #     self.logger.record("train/value_loss", np.mean(value_losses))
    #     self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    #     self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    #     self.logger.record("train/loss", loss.item())
    #     self.logger.record("train/explained_variance", explained_var)
    #     if hasattr(self.policy, "log_std"):
    #         self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
    #     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    #     self.logger.record("train/clip_range", clip_range)
    #     if self.clip_range_vf is not None:
    #         self.logger.record("train/clip_range_vf", clip_range_vf)