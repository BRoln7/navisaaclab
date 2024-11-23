import torch
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import random
import os
import yaml
from gymnasium import spaces
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    build_amp_observations_smpl,
    dof_subset
)
from socnav_sim.tasks.amp.amp_utils.utils.replay_buffer import ReplayBuffer
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
from socnav_sim.tasks.amp.amp_utils.utils.discriminator import Discriminator, bias_init

import importlib

obs_shape = 357
amp_demo_buffer_size = 200000
amp_batch_size = 512

SEED = 4801
seed = 0
random.seed(seed)
np.random.seed(SEED)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'amp.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)

num_envs = config['env']['num_envs'] # type: ignore
global motion_load
motion_load = MotionLoad(num_envs)

###################################################
################# demo buffer #####################
###################################################
class AMPDemoBuffer():
    def __init__(self):
        self._amp_obs_demo_buffer = ReplayBuffer(amp_demo_buffer_size, "cuda:0")
        self.amp_obs_demo_buf = None
        self.amp_batch_size = amp_batch_size
        self.num_amp_obs_steps = 10
        self.num_amp_obs_per_step = 195

        self._amp_root_height_obs = False
        self._has_dof_subset = True
        self._has_upright_start = True
        self._local_root_obs = True

    def init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self.amp_batch_size))

        for i in range(num_batches):
            curr_samples = self.fetch_amp_obs_demo(self.amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return

    def update_amp_demos(self):
        new_amp_obs_demo = self.fetch_amp_obs_demo(self.amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return
    
    def fetch_amp_obs_demo(self, num_samples):
        # Creates the reference motion amp obs. For discrinminiator

        if (self.amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert (self._amp_obs_demo_buf.shape[0] == num_samples)

        motion_ids = motion_load.motion_lib.sample_motions(num_samples)
        motion_times0 = motion_load.motion_lib.sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(
            self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(
            -1, self.get_num_amp_obs())

        return amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # Compute observation for the motion starting point
        dt = 1 / 120
        motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, self.num_amp_obs_steps])

        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self.num_amp_obs_steps, device="cuda:0")
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        motion_res = motion_load.get_smpl_state_from_motionlib_cache(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        amp_obs_demo = build_amp_observations_smpl(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
            key_pos, dof_subset, self._local_root_obs,
            self._amp_root_height_obs, self._has_dof_subset, self._has_upright_start)

        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self.num_amp_obs_steps, self.num_amp_obs_per_step),
            device="cuda:0",
            dtype=torch.float32)
        return

    def get_num_amp_obs(self):
        return self.num_amp_obs_steps * self.num_amp_obs_per_step


################################################################
################### discriminator network ######################
################################################################
global discriminator
discriminator = Discriminator()
bias_init(discriminator)
discriminator.to("cuda:0")
if config['training']['train']: # type: ignore
    discriminator.train()
else:
    discriminator.eval()
disc_optimizer = Adam(discriminator.parameters(), lr=2e-5)
disc_epoches = 10

################################################################
########################## AMP PPO #############################
################################################################

class AMPPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(AMPPPO, self).__init__(*args, **kwargs)
        # demo buffer
        self.amp_demo_buffer = AMPDemoBuffer()
        self.amp_demo_buffer.init_amp_demo_buf()
        # replay buffer
        self.amp_replay_buffer = torch.tensor([]).to("cuda:0")


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        ########################### import amp obs ##############################
        obs = importlib.import_module("socnav_sim.tasks.amp.amp.mdp.observations")

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        ################### clear buffer ######################
        self.amp_replay_buffer = torch.tensor([]).to("cuda:0")

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

            ######################################################
            ############## collect extra info here ###############
            ######################################################
            if self.amp_replay_buffer.numel() == 0:
                self.amp_replay_buffer = obs.amp_replay_obs
            else:
                self.amp_replay_buffer = torch.cat((self.amp_replay_buffer, obs.amp_replay_obs), dim=0)


        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


    def train(self) -> None:
        super().train()
        self.amp_demo_buffer.update_amp_demos()



    def save(
        self,
        path,
        exclude = None,
        include = None,
    ) -> None:
        super().save(path, exclude, include)
        torch.save(discriminator, f='logs/sb3/Isaac-AMP/discriminator.pth')