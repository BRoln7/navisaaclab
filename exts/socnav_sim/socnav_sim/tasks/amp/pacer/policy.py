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
import os
import yaml
from gymnasium import spaces
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    build_amp_observations_smpl,
    dof_subset)
from socnav_sim.tasks.amp.amp_utils.utils.replay_buffer import ReplayBuffer
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
from socnav_sim.tasks.amp.amp_utils.utils.discriminator import Discriminator, bias_init
from socnav_sim.tasks.amp.amp_utils.isaac_utils.gym_utils import to_torch
from socnav_sim.tasks.amp.amp_utils.utils.running_mean_std import RunningMeanStd
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import flip_loss_left_to_right_index
import importlib

self_obs_shape = 357
obs_shape = 357+20+256
amp_demo_buffer_size = 200000
amp_batch_size = 512

motion_file = os.path.join(os.getcwd(), 'data', 'cfg', 'task.yaml')
with open(motion_file, 'r') as file:
    config = yaml.safe_load(file)

num_envs = config['env']['num_envs'] # type: ignore

if config['env']['task'] == 'Isaac-Pacer': # type: ignore
    global motion_load
    motion_load = MotionLoad(num_envs)

###################################################
################## preprocess #####################
###################################################
class PreProcess(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = obs_shape):
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
global amp_input_mean_std
if config['training']['checkpoint'] is None: # type: ignore
    discriminator = Discriminator()
    bias_init(discriminator)
    amp_input_mean_std = RunningMeanStd(1950).to("cuda:0")

elif config['training']['train'] == True: # type: ignore
    discriminator = torch.load(config['training']['disc_checkpoint']) # type: ignore
    amp_input_mean_std = torch.load(config['training']['std_checkpoint']) # type: ignore

else: 
    discriminator = Discriminator()
    bias_init(discriminator)
    amp_input_mean_std = RunningMeanStd(1950).to("cuda:0")

discriminator.to("cuda:0")
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
        # flip obs buffer
        self.flip_obs_buffer = []
        self.previous_flip_obs = None

        self._amp_replay_keep_prob = 0.01
        self._disc_logit_reg = 0.01
        self._disc_grad_penalty = 5
        self._disc_weight_decay = 0.0001
        self.bounds_loss_coef = 10
        self.disc_coef = 5

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
        obs = importlib.import_module("socnav_sim.tasks.amp.pacer.mdp.observations")

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        ################### init buffer ######################
        self.amp_buffer = torch.tensor([]).to("cuda:0")
        
        self.flip_obs_buffer = []
        if self.previous_flip_obs is None:
            self.previous_flip_obs = obs.flip_humanoid_obs

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

            self.flip_obs_buffer.append(self.previous_flip_obs)
            self.previous_flip_obs = obs.flip_humanoid_obs

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        add sym_loss and loss of discriminator
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        a_losses = []
        clip_fractions = []
        sym_loss = []

        ############################## reshape flip_obs_buffer for training ##########################
        self.flip_obs_buffer = torch.stack(self.flip_obs_buffer, dim=0) # type: ignore
        self.flip_obs_buffer = self.flip_obs_buffer.reshape(self.flip_obs_buffer.size(0)*
                                                            self.flip_obs_buffer.size(1), 
                                                            -1)

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            flip_batch_index = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
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

                ##################################################################
                ###################### from here sym loss ########################
                ##################################################################
                flip_obs_batch = self.flip_obs_buffer[flip_batch_index * self.batch_size:
                                                      (flip_batch_index+1) * self.batch_size]
                orig_obs_batch = torch.tensor(
                    self.rollout_buffer.observations[flip_batch_index * self.batch_size:
                                                     (flip_batch_index+1) * self.batch_size]).to("cuda:0")
                flip_obs_batch = torch.cat([flip_obs_batch, orig_obs_batch[:, self_obs_shape:]], dim=1)

                flip_batch_index += 1

                self.sym_loss_coef = 1
                s_info = self.sym_loss(flip_obs_batch, orig_obs_batch)
                s_loss = s_info['sym_loss']
                s_loss = torch.mean(s_loss)
                sym_loss.append(s_loss.cpu().detach().numpy())
                ##################################################################
                ##################### from here bound loss #######################
                ##################################################################
                mu = self.policy(rollout_data.observations)
                b_loss = self.bound_loss(mu[0])
                b_loss = torch.mean(b_loss)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss +\
                      self.sym_loss_coef * s_loss + self.bounds_loss_coef * b_loss

                a_losses.append(policy_loss.cpu().detach().numpy())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/a_loss", np.mean(a_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/sym_loss", np.mean(sym_loss))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        
        ##################################################################
        ######################## from here amp ###########################
        ##################################################################
        amp_input_mean_std.train()
        discriminator.train()
        # amp_obs
        amp_obs = self.amp_buffer.reshape(-1, 1950)
        amp_obs = amp_input_mean_std(amp_obs)
        # amp_replay_obs
        num_obs_samples = self.amp_buffer.shape[0] * self.amp_buffer.shape[1]
        self._store_replay_amp_obs(self.amp_buffer.reshape(-1, self.amp_buffer.shape[2]))
        amp_obs_replay = self.amp_replay_buffer.sample(num_obs_samples)['amp_obs']
        amp_obs_replay = amp_input_mean_std(amp_obs_replay)
        # amp_demo_obs
        self.amp_demo_buffer.update_amp_demos()
        amp_obs_demo = self.amp_demo_buffer.sample(num_obs_samples)['amp_obs']
        amp_obs_demo = amp_input_mean_std(amp_obs_demo)
        # drop out
        dropout_mask = self.get_dropout_mask(amp_obs)
        amp_obs = self.dropout_amp_obs(amp_obs, dropout_mask[..., 0])
        amp_obs_replay = self.dropout_amp_obs(amp_obs_replay, dropout_mask[..., 1])
        amp_obs_demo = self.dropout_amp_obs(amp_obs_demo, dropout_mask[..., 2])
        del dropout_mask
        torch.cuda.empty_cache()

        disc_loss_record = []
        disc_agent_logit_record = []
        disc_agent_replay_record = []
        disc_demo_logit_record = []
        disc_grad_penalty_record = []
        disc_logit_loss_record = []
        for epoch in range(self.n_epochs):
            optim_times = math.ceil(self.amp_buffer.shape[0] * self.amp_buffer.shape[1] / self.batch_size)
            for mini_batch in range(optim_times):
                amp_obs_demo_batch = amp_obs_demo[torch.randperm(amp_obs_demo.size(0))[:self.batch_size]]
                amp_obs_replay_batch = amp_obs_replay[torch.randperm(amp_obs_replay.size(0))[:self.batch_size]]
                amp_obs_batch = amp_obs[torch.randperm(amp_obs.size(0))[:self.batch_size]]

                amp_obs_demo_batch.requires_grad_(True)
                disc_agent_logit = discriminator.get_logit(amp_obs_batch)
                disc_agent_replay_logit = discriminator.get_logit(amp_obs_replay_batch)
                disc_demo_logit = discriminator.get_logit(amp_obs_demo_batch)

                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo_batch)
                disc_loss = self.disc_coef * disc_info['disc_loss']

                discriminator.optimizer.zero_grad()
                disc_loss.backward(retain_graph=False)
                discriminator.optimizer.step()

                # record
                disc_loss_record.append(disc_loss.cpu().detach().numpy())
                disc_grad_penalty_record.append(disc_info['disc_grad_penalty'])
                disc_logit_loss_record.append(disc_info['disc_logit_loss'])
                disc_demo_logit_record.append(np.mean(disc_demo_logit.cpu().detach().numpy()))
                disc_agent_logit_record.append(np.mean(disc_agent_logit.cpu().detach().numpy()))
                disc_agent_replay_record.append(np.mean(disc_agent_replay_logit.cpu().detach().numpy()))

        del amp_obs, amp_obs_replay, amp_obs_demo
        self.logger.record("train/disc_loss", np.mean(disc_loss_record))
        self.logger.record("train/disc_demo_logit", np.mean(disc_demo_logit_record))
        self.logger.record("train/disc_agent_logit", np.mean(disc_agent_logit_record))  
        self.logger.record("train/disc_replay_logit", np.mean(disc_agent_replay_record)) 
        self.logger.record("train/disc_grad_penalty", np.mean(disc_grad_penalty_record))  
        self.logger.record("train/disc_logit_loss", np.mean(disc_logit_loss_record))  
        del disc_loss_record, disc_demo_logit_record, disc_agent_logit_record, \
            disc_grad_penalty_record, disc_logit_loss_record, disc_agent_replay_record
        torch.cuda.empty_cache()
        gc.collect()

    def save(
        self,
        path,
        exclude = ['amp_buffer', 'amp_demo_buffer', 'amp_replay_buffer', 'flip_obs_buffer'],
        include = None,
    ) -> None:
        super().save(path, exclude, include)

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

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.cpu().detach().numpy(),
            'disc_logit_loss': disc_logit_loss.cpu().detach().numpy(),
            'disc_agent_logit': disc_agent_logit.cpu().detach().numpy(),
            'disc_demo_logit': disc_demo_logit.cpu().detach().numpy()
        }
        return disc_info

    def bound_loss(self, mu):
        soft_bound = 1.0
        mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device="cuda:0"))**2
        mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device="cuda:0"))**2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1) # type: ignore

        return b_loss
    
    def sym_loss(self, flip_obs_batch, orig_obs_batch):
        B, _ = flip_obs_batch.shape

        left_to_right_index = flip_loss_left_to_right_index
        flip_a, flip_s, _ = self.policy.forward(flip_obs_batch, deterministic=True)
        orig_a, orig_s, _ = self.policy.forward(orig_obs_batch, deterministic=True)

        orig_a = orig_a.view(B, -1, 3)
        orig_a[..., 0] *= -1
        orig_a[..., 2] *= -1
        orig_a = orig_a[..., left_to_right_index, :]

        sym_loss = orig_a.view(B, -1) - flip_a
        sym_loss = sym_loss.pow(2).mean(dim = -1) * 50
        return {'sym_loss': sym_loss}

    
    def dropout_amp_obs(self, amp_obs, dropout_mask):
        return amp_obs * dropout_mask

    def get_dropout_mask(self,
                            amp_obs,
                            steps=10,
                            num_masks=3,
                            dropout_rate=0.3):
        # ZL Hack: amp_obs_dims, should drop out whole joints
        # [root_rot 6, root_vel 3, root_ang_vel 3, dof_pos 23 * 6 - 4 * 6, dof_vel 69 - 12, key_body_pos 3 * 4, shape_obs_disc 11]
        # [root_rot 6, root_vel 3, root_ang_vel 3, dof_pos 23 * 6 - 4 * 6, dof_vel 69 - 12, key_body_pos 3 * 4, shape_obs_disc 47]
        # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 11 = 206 # normal with betas
        # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 10 = 205 # concise limb weight
        # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 59 = 254 # masterfoot
        B, F = amp_obs.shape
        B, _, amp_f = amp_obs.view(B, steps, -1).shape
        masterfoot, remove_neck = False, False
        try:
            assert (F / steps == 205 or F / steps == 254 or F / steps == 242 or F / steps == 206 or F / steps == 197 or F / steps == 188 or F / steps == 187 or F / steps == 195 or F / steps == 196 or F / steps == 207)
        except:
            print(F/steps)
            import ipdb; ipdb.set_trace()
            print(F/steps)

        dof_joints_offset = 12  # 6 + 3 + 3
        num_joints = 19

        if F / steps == 197:  # Remove neck
            num_joints = 18
        elif F / steps == 188:  # Remove hands
            num_joints = 17
        elif F / steps == 196 or F / steps == 207:
            dof_joints_offset = 13  # 1 + 6 + 3 + 3

        dof_vel_offsets = dof_joints_offset + num_joints * 6  # 12 + 19 * 6

        dropout_mask = torch.ones([B, amp_f, num_masks])

        for idx_joint in range(num_joints):
            has_drop_out = torch.rand(B, num_masks) > dropout_rate
            dropout_mask[:, dof_joints_offset + idx_joint * 6:dof_joints_offset + idx_joint * 6 + 6, :] = has_drop_out[:, None]
            dropout_mask[:, dof_vel_offsets + idx_joint * 3:dof_vel_offsets + idx_joint * 3 + 3, :] = has_drop_out[:, None]
        return dropout_mask.repeat(1, steps, 1).to(amp_obs)