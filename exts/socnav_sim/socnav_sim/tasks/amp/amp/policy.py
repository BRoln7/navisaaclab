import gc
import torch
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
import numpy as np
import gymnasium as gym
import math
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
from socnav_sim.tasks.amp.amp_utils.isaac_utils.gym_utils import to_torch
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
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

disc_coef = config['training']['disc_coef'] # type: ignore

###################################################
################## preprocess #####################
###################################################
class PreProcess(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 357):
      super(PreProcess, self).__init__(observation_space, features_dim)
      self.running_mean_std = RunningMeanStd(obs_shape).to("cuda:0")
      if config['training']['train']: # type: ignore
          self.running_mean_std.train()
      else:
          self.running_mean_std.eval()

    def forward(self, obs):
      obs_norm = self.running_mean_std(obs)
      assert(obs_norm.size(1) == self.features_dim)
      return obs_norm

###################################################
################# demo buffer #####################
###################################################
class AMPDemoBuffer():
    def __init__(self):
        self._amp_obs_demo_buffer = ReplayBuffer(amp_demo_buffer_size, "cuda:0")
        self._amp_obs_demo = None
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

        if (self._amp_obs_demo is None):
            self._build_amp_obs_demo(num_samples)
        else:
            assert (self._amp_obs_demo.shape[0] == num_samples)

        motion_ids = motion_load.motion_lib.sample_motions(num_samples)
        motion_times0 = motion_load.motion_lib.sample_time(motion_ids)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo[:] = amp_obs_demo.view(self._amp_obs_demo.shape) # type: ignore
        amp_obs_demo_flat = self._amp_obs_demo.view(-1, self.get_num_amp_obs()) # type: ignore

        return amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0):
        # Compute observation for the motion starting point
        dt = 1 / 30
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

    def _build_amp_obs_demo(self, num_samples):
        self._amp_obs_demo = torch.zeros(
            (num_samples, self.num_amp_obs_steps, self.num_amp_obs_per_step),
            device="cuda:0",
            dtype=torch.float32)
        return

    def get_num_amp_obs(self):
        return self.num_amp_obs_steps * self.num_amp_obs_per_step
    
    def sample(self, num_samples):
        return self._amp_obs_demo_buffer.sample(num_samples)


################################################################
################### discriminator network ######################
################################################################
global discriminator
discriminator = Discriminator()
bias_init(discriminator)
discriminator.to("cuda:0")
discriminator.train()

global amp_input_mean_std
amp_input_mean_std = RunningMeanStd(1950).to("cuda:0")
amp_input_mean_std.train()
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
        self.amp_buffer = torch.tensor([]).to("cuda:0")
        self.amp_replay_buffer = ReplayBuffer(amp_demo_buffer_size, "cuda:0")

        self._amp_replay_keep_prob = 0.01
        self._disc_logit_reg = 0.01
        self._disc_grad_penalty = 5
        self._disc_weight_decay = 0.0001
        self.bounds_loss_coef = 10

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self.amp_replay_buffer.get_buffer_size()
        buf_total_count = self.amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] *
                                           amp_obs.shape[0]), device="cuda:0")
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self.amp_replay_buffer.store({'amp_obs': amp_obs})
        return

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
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        ################### clear buffer ######################
        self.amp_buffer = torch.tensor([]).to("cuda:0")

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
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
            if self.amp_buffer.numel() == 0:
                self.amp_buffer = obs.amp_replay_obs
            else:
                self.amp_buffer = torch.cat((self.amp_buffer, obs.amp_replay_obs), dim=0)

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        super().train()
        
        ##################################################################
        ######################## from here amp ###########################
        ##################################################################
        amp_obs = self.amp_buffer.reshape(-1, 1950)
        amp_obs = amp_input_mean_std(amp_obs)

        num_obs_samples = self.amp_buffer.shape[0] * self.amp_buffer.shape[1]
        self._store_replay_amp_obs(self.amp_buffer.reshape(-1, self.amp_buffer.shape[2]))
        amp_obs_replay = self.amp_demo_buffer.sample(num_obs_samples)['amp_obs']
        amp_obs_replay = amp_input_mean_std(amp_obs_replay)

        self.amp_demo_buffer.update_amp_demos()
        amp_obs_demo = self.amp_demo_buffer.sample(num_obs_samples)['amp_obs']
        amp_obs_demo = amp_input_mean_std(amp_obs_demo)

        disc_loss_total = torch.zeros(1).to("cuda:0").requires_grad_(False)
        times = 0
        for epoch in range(self.n_epochs):
            optim_times = math.ceil(self.amp_buffer.shape[0] * self.amp_buffer.shape[1] / self.batch_size)
            for mini_batch in range(optim_times):
                if (mini_batch+1) * self.batch_size < self.amp_buffer.shape[0] * self.amp_buffer.shape[1]:
                    amp_obs_batch = amp_obs[mini_batch*self.batch_size : 
                                            (mini_batch+1)*self.batch_size]
                    amp_obs_replay_batch = amp_obs_replay[mini_batch*self.batch_size : 
                                                          (mini_batch+1)*self.batch_size]
                    amp_obs_demo_batch = amp_obs_demo[mini_batch*self.batch_size : 
                                                      (mini_batch+1)*self.batch_size]
                else:
                    amp_obs_batch = amp_obs[mini_batch*self.batch_size:]
                    amp_obs_replay_batch = amp_obs_replay[mini_batch*self.batch_size:]
                    amp_obs_demo_batch = amp_obs_demo[mini_batch*self.batch_size:]

                amp_obs_demo_batch.requires_grad_(True)
                disc_agent_logit = discriminator.get_logit(amp_obs_batch)
                disc_agent_replay_logit = discriminator.get_logit(amp_obs_replay_batch)
                disc_demo_logit = discriminator.get_logit(amp_obs_demo_batch)

                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                disc_loss = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo_batch)
                disc_loss =  disc_coef * disc_loss

                discriminator.optimizer.zero_grad()
                disc_loss.backward(retain_graph=False)
                discriminator.optimizer.step()

                disc_loss_total += disc_loss.detach()
                times += 1

        del amp_obs, amp_obs_replay, amp_obs_demo
        torch.cuda.empty_cache()
        gc.collect()
        disc_loss_avg = disc_loss_total / times
        self.logger.record("train/disc_loss", disc_loss_avg.item())

    def save(
        self,
        path,
        exclude = None,
        include = None,
    ) -> None:
        super().save(path, exclude, include)
        torch.save(discriminator, f='logs/sb3/Isaac-AMP/discriminator.pth')
        torch.save(amp_input_mean_std, f='logs/sb3/Isaac-AMP/amp_input_mean_std.pth')


    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss
    
    def get_disc_weights(self, model):
        weights = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        return weights

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        '''
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit 
        obs_demo: gradient penalty demo obs (real examples)
        '''
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = torch.flatten(discriminator.disc_logit.weight)
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]

        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.get_disc_weights(discriminator.disc_mlp)
            disc_weights.append(logit_weights)
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        return disc_loss

    def bound_loss(self, mu):
        soft_bound = 1.0
        mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device="cuda:0"))**2
        mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device="cuda:0"))**2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1) # type: ignore

        return b_loss