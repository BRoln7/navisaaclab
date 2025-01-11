import torch
import numpy as np
import os
import yaml

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor

from gymnasium import spaces

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)
if config['env']['task'] != 'Isaac-Pacer': # type: ignore
    num_robots = config['env']['num_robots'] # type: ignore


class MRPPO(PPO):
    '''
    Multi-Robots PPO (MRPPO)
    num of robots to be used now is not equal with num of envs, as some of robots are abandoned
    so in this case, we should just collect trajs of envs used
    '''

    def __init__(self, *args, **kwargs):
        super(MRPPO, self).__init__(*args, **kwargs)

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
            values = self.policy.predict_values(obs_as_tensor(new_obs[:num_robots], self.device))  # type: ignore[arg-type]
        dones = dones[:num_robots]
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

