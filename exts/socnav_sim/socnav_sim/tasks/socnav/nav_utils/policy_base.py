"""PPO adapter that trains only the active robot slots."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from socnav_sim.config import load_task_config


class MRPPO(PPO):
    """Run all simulated pedestrians while storing only active robot rollouts."""

    def __init__(self, *args, active_envs: int | None = None, **kwargs):
        if active_envs is None:
            active_envs = int(load_task_config()["env"]["num_robots"])
        self.active_envs = active_envs
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.env is not None and not 0 < self.active_envs <= self.env.num_envs:
            raise ValueError(f"active_envs must be in [1, {self.env.num_envs}], got {self.active_envs}.")
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.active_envs,
            **self.rollout_buffer_kwargs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        for step in range(n_rollout_steps):
            if self.use_sde and self.sde_sample_freq > 0 and step % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)
            with torch.no_grad():
                actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device))
            actions_numpy = actions.cpu().numpy()
            clipped_actions = actions_numpy
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = (
                    self.policy.unscale_action(actions_numpy)
                    if self.policy.squash_output
                    else np.clip(actions_numpy, self.action_space.low, self.action_space.high)
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += self.active_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos[: self.active_envs], dones[: self.active_envs])

            stored_actions = actions_numpy
            if isinstance(self.action_space, spaces.Discrete):
                stored_actions = stored_actions.reshape(-1, 1)
            for index in range(self.active_envs):
                if (
                    dones[index]
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with torch.no_grad():
                        rewards[index] += self.gamma * self.policy.predict_values(terminal_obs)[0].item()

            rollout_buffer.add(
                self._last_obs[: self.active_envs],
                stored_actions[: self.active_envs],
                rewards[: self.active_envs],
                self._last_episode_starts[: self.active_envs],
                values[: self.active_envs],
                log_probs[: self.active_envs],
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))[: self.active_envs]
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=dones[: self.active_envs]
        )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
