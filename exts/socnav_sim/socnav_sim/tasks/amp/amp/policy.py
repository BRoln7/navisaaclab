import torch
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import yaml
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import (
    build_amp_observations_smpl,
    dof_subset
)
from socnav_sim.tasks.amp.amp_utils.utils.replay_buffer import ReplayBuffer
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
from socnav_sim.tasks.amp.amp_utils.utils.discriminator import Discriminator, bias_init

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
################################################################
########################## AMP PPO #############################
################################################################

class AMPPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(AMPPPO, self).__init__(*args, **kwargs)
        self.amp_demo_buffer = AMPDemoBuffer()
        self.amp_demo_buffer.init_amp_demo_buf()
        self._amp_replay_buffer = torch.tensor([]).to("cuda:0")

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

                ###############################################################################
                ############################# here add loss for amp ###########################
                ###############################################################################



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
        # self.logger.record("train/disc_loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self.flip_obs = None

    # def save(
    #     self,
    #     path,
    #     exclude = None,
    #     include = None,
    # ) -> None:
    #     super().save(path, exclude, include)

    #     torch.save(discriminator, f='amp_discriminator.pth')