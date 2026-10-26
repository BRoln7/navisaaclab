"""Native AMP with actor symmetry, bounded-memory data staging and progress logs."""

from contextlib import nullcontext
from functools import wraps
from pathlib import Path
import time
import torch

from learning.amp_agent import AMPAgent
from .buffers import StagedExperienceBuffer
from .optimizer import MicrobatchMPOptimizer
from util import torch_util


def training_stage(name):
    """Instrument delegated upstream calls; keep its optimizer/update code intact."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            start = time.monotonic()
            self._report_progress(f"{name}: start")
            result = method(self, *args, **kwargs)
            self._report_progress(f"{name}: complete ({time.monotonic()-start:.1f}s)")
            return result
        return wrapped
    return decorate


class SymmetricAMPAgent(AMPAgent):
    def _build_optimizer(self, config):
        # Same parameters, optimizer classes/settings and sync as upstream.
        # The extension only separates backward chunking from optimizer.step.
        for role in ("actor", "critic", "disc"):
            parameters = getattr(self._model, f"get_{role}_params")()
            parameters = [p for p in parameters if p.requires_grad]
            optimizer = MicrobatchMPOptimizer(config[f"{role}_optimizer"], parameters)
            setattr(self, f"_{role}_optimizer", optimizer)

    def _build_exp_buffer(self, config):
        # Native rollout/GAE/PPO data stay on GPU. Demonstrations and historical
        # replay are staged from CPU; float32 contents and capacities stay intact.
        self._exp_buffer = StagedExperienceBuffer(
            self._get_exp_buffer_length(), self.get_num_envs(), self._device,
            host_keys=("disc_obs_demo",))
        self._disc_buffer = StagedExperienceBuffer(
            config["disc_buffer_size"], 1, "cpu", sample_device=self._device)

    def _build_logger(self, logger_type, log_file, config):
        self._progress_file = Path(log_file).with_name("progress.log")
        if logger_type != "tb":
            return super()._build_logger(logger_type, log_file, config)
        from .logging import IterationTBLogger
        logger = IterationTBLogger()
        logger.set_step_key("Samples")
        logger.configure_output_file(log_file)
        return logger

    def _report_progress(self, message):
        line = f"[MimicKit][iteration={self._iter}] {message}"
        if torch.device(self._device).type == "cuda" and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(self._device)/1024**3
            reserved = torch.cuda.memory_reserved(self._device)/1024**3
            line += f" | CUDA allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"
        print(line, flush=True)
        path = getattr(self, "_progress_file", None)
        if path is not None:
            with path.open("a") as stream:
                stream.write(line+"\n")
        writer = getattr(getattr(self, "_logger", None), "_writer", None)
        if writer is not None:
            writer.add_text("Progress/Stage", message, self._sample_count)
            writer.flush()

    def _reset_envs(self, env_ids=None):
        initial = self._curr_obs is None
        if initial:
            self._report_progress("Initial environment reset: start")
        result = super()._reset_envs(env_ids)
        if initial:
            self._report_progress("Initial environment reset: complete")
        return result

    @training_stage("Initialize rollout buffers")
    def _init_train(self):
        result = super()._init_train()
        feature_bytes = self._disc_obs_norm.get_mean().numel() * self._disc_obs_norm.get_mean().element_size()
        host_bytes = feature_bytes * (self._exp_buffer.get_capacity() + self._disc_buffer.get_capacity())
        self._report_progress(f"Demo/replay storage: CPU (~{host_bytes/1024**3:.2f} GiB when allocated); "
                              f"data chunks <= {self._data_batch_size} samples")
        return result

    @training_stage("Rollout")
    def _rollout_train(self, num_steps):
        self._rollout_progress = 0
        return super()._rollout_train(num_steps)

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._rollout_progress = getattr(self, "_rollout_progress", 0) + 1
        if self._rollout_progress == 1 or self._rollout_progress % 8 == 0:
            self._report_progress(f"Rollout {self._rollout_progress}/{self._steps_per_iter} steps x {self.get_num_envs()} environments")

    @training_stage("Demo sampling and normalization statistics")
    @torch.no_grad()
    def _record_disc_demo_data(self):
        obs = self._exp_buffer.get_data_flat("disc_obs")
        if "disc_obs_demo" not in self._exp_buffer._buffers:
            self._exp_buffer.add_buffer("disc_obs_demo", obs.shape[1:], obs.dtype)
        target = self._exp_buffer.get_data_flat("disc_obs_demo")
        n = obs.shape[0]
        for start in range(0, n, self._data_batch_size):
            end = min(start + self._data_batch_size, n)
            # The native motion sampler, FK and ten-frame features are unchanged.
            demo = self._env.fetch_disc_obs_demo(end-start)
            target[start:end].copy_(demo)
            self._disc_obs_norm.record(demo)
            if end == n or (start // self._data_batch_size + 1) % 16 == 0:
                self._report_progress(f"Demo samples {end}/{n} (chunk <= {self._data_batch_size})")
            del demo

    @training_stage("Replay insertion")
    @torch.no_grad()
    def _store_disc_replay_data(self):
        obs = self._exp_buffer.get_data_flat("disc_obs")
        n = obs.shape[0]
        indices = torch.randperm(n, device=self._device, dtype=torch.long)
        count = min(n, self._disc_replay_samples) if self._disc_buffer.is_full() else n
        # Same native selection and circular insertion, with bounded GPU gathers.
        for start in range(0, count, self._data_batch_size):
            selected = obs[indices[start:min(start+self._data_batch_size, count)]]
            self._disc_buffer.push({"disc_obs": selected.unsqueeze(1)})

    @training_stage("AMP rewards")
    @torch.no_grad()
    def _compute_rewards(self):
        obs = self._exp_buffer.get_data_flat("disc_obs")
        task_reward = self._exp_buffer.get_data_flat("reward")
        disc_reward = torch.empty_like(task_reward)
        # Upstream chunks network evaluation only AFTER full-rollout normalization.
        # Chunk normalization too, keeping the native reward formula/statistics.
        for start in range(0, obs.shape[0], self._data_batch_size):
            end = min(start+self._data_batch_size, obs.shape[0])
            normalized = self._disc_obs_norm.normalize(obs[start:end])
            disc_reward[start:end] = self._calc_disc_rewards(normalized)
        std, mean = torch.std_mean(disc_reward)
        reward = self._task_reward_weight*task_reward + self._disc_reward_weight*disc_reward
        self._exp_buffer.set_data_flat("reward", reward)
        return {"disc_reward_mean": mean, "disc_reward_std": std}

    @training_stage("Critic update")
    def _update_critic(self, batch_size, steps):
        with self._exp_buffer.sampling_fields(("obs", "tar_val")):
            if self._optimizer_microbatch_size.get("critic", batch_size) >= batch_size:
                return super()._update_critic(batch_size, steps)
            return self._update_microbatched("critic", batch_size, steps)

    @training_stage("Actor and symmetry update")
    def _update_actor(self, batch_size, steps):
        with self._exp_buffer.sampling_fields(("obs", "action", "a_logp", "adv", "rand_action_mask")):
            if self._optimizer_microbatch_size.get("actor", batch_size) >= batch_size:
                return super()._update_actor(batch_size, steps)
            return self._update_microbatched("actor", batch_size, steps)

    @training_stage("Discriminator update")
    def _update_disc(self, batch_size, steps):
        with self._exp_buffer.sampling_fields(("disc_obs", "disc_obs_demo")):
            if self._optimizer_microbatch_size.get("disc", batch_size) >= batch_size:
                return super()._update_disc(batch_size, steps)
            return self._update_microbatched("disc", batch_size, steps)

    def _update_microbatched(self, role, batch_size, steps):
        """Keep native logical batches and update counts; stage only small chunks."""
        microbatch = self._optimizer_microbatch_size[role]
        optimizer = getattr(self, f"_{role}_optimizer")
        compute_loss = getattr(self, f"_compute_{role}_loss")
        loss_key = f"{role}_loss"
        info = {}
        self._report_progress(f"{role}: {steps} optimizer updates; logical batch={batch_size}, microbatch<={microbatch}")
        for _ in range(steps):
            indices = self._exp_buffer.sample_indices(batch_size)
            random_mask = None
            if role == "actor":
                # One small host copy per logical batch, not one CUDA sync per
                # chunk. PPO averages over random actions; symmetry uses all.
                random_mask = (self._exp_buffer.get_data_flat("rand_action_mask")[indices] == 1).cpu()
                random_count = int(random_mask.sum())
                if random_count == 0:
                    raise ValueError("Native PPO requires random actions in each logical actor batch")

            def losses():
                for start in range(0, batch_size, microbatch):
                    end = min(start + microbatch, batch_size)
                    batch = self._exp_buffer.gather(indices[start:end])
                    fraction = (end-start) / batch_size
                    with torch.amp.autocast(device_type=torch.device(self._device).type,
                                            enabled=self._use_mixed_precision, dtype=torch.bfloat16):
                        if role == "actor":
                            random_fraction = int(random_mask[start:end].sum()) / random_count
                            chunk_info = self._compute_weighted_actor_loss(batch, fraction, random_fraction)
                        else:
                            chunk_info = compute_loss(batch)
                            chunk_info = {key: value * fraction for key, value in chunk_info.items()}
                    loss = chunk_info[loss_key]
                    torch_util.add_torch_dict({key: value.detach() for key, value in chunk_info.items()}, info)
                    yield loss
                    del batch, chunk_info, loss

            # Reserve replay indices once, just as a native full-batch loss does.
            # Its unmodified loss can then fetch each replay microbatch itself.
            replay_plan = self._disc_buffer.sampling_plan(batch_size) if role == "disc" else nullcontext()
            with replay_plan:
                optimizer.step_accumulated(losses())
        torch_util.scale_torch_dict(1. / steps, info)
        return info

    def _compute_weighted_actor_loss(self, batch, fraction, random_fraction):
        if random_fraction:
            info = super()._compute_actor_loss(batch)
            info = {key: value * random_fraction for key, value in info.items()}
        else:
            # A chunk can contain no random actions although the logical batch
            # has some. Avoid upstream mean(empty); it contributes only symmetry.
            info = {"actor_loss": torch.zeros((), device=self._device)}
        if self._symmetry_weight:
            symmetry = self._compute_symmetry_loss(batch["obs"])
            info["actor_loss"] = info["actor_loss"] + self._symmetry_weight * fraction * symmetry
            info["sym_loss"] = fraction * symmetry.detach()
        return info

    @training_stage("Policy evaluation")
    def test_model(self, *args):
        return super().test_model(*args)

    @torch.no_grad()
    def initialize_tracking_inputs(self, goal_features=20):
        """Initialize previously zero goal inputs while preserving the locomotion policy.

        The baseline reserves the same observation layout. Its goal RMS can have
        std=1e-4 and the unused input weights remain random. Enabling targets
        directly would change the loaded policy before its first task update.
        """
        for layers in (self._model._actor_layers, self._model._critic_layers):
            first = next(layer for layer in layers.modules() if isinstance(layer, torch.nn.Linear))
            if first.in_features != self._obs_norm.get_mean().numel():
                raise ValueError("Tracking transfer requires the native fully connected observation input")
            first.weight[:, -goal_features:] = 0.
        self._obs_norm.get_mean()[-goal_features:] = 0.
        self._obs_norm.get_std()[-goal_features:] = 1.
        # Rebuild native running second moments from the updated mean/std.
        self._obs_norm._mean_sq = None

    @torch.no_grad()
    def load_locomotion_for_tracking(self, in_file, goal_features=20):
        """Load pure AMP weights and append zero-initialized trajectory inputs."""
        state = torch.load(in_file, map_location=self._device, weights_only=False)
        current = self.state_dict()
        mean_key = "_obs_norm._mean"
        std_key = "_obs_norm._std"
        input_keys = (
            "_model._actor_layers.0.weight",
            "_model._critic_layers.0.weight",
        )
        source_features = state[mean_key].numel()
        target_features = current[mean_key].numel()
        if source_features == target_features:
            self.load_state_dict(state)
            self.initialize_tracking_inputs(goal_features)
        elif source_features + goal_features == target_features:
            adapted = {}
            for key, target in current.items():
                source = state.get(key)
                if source is None:
                    raise ValueError(f"Locomotion checkpoint is missing {key}")
                if key in input_keys:
                    if source.shape != (target.shape[0], source_features):
                        raise ValueError(f"Incompatible locomotion input tensor {key}: {tuple(source.shape)}")
                    value = torch.zeros_like(target)
                    value[:, :source_features] = source
                elif key == mean_key:
                    value = torch.zeros_like(target)
                    value[:source_features] = source
                elif key == std_key:
                    value = torch.ones_like(target)
                    value[:source_features] = source
                else:
                    if source.shape != target.shape:
                        raise ValueError(
                            f"Incompatible locomotion tensor {key}: {tuple(source.shape)} != {tuple(target.shape)}"
                        )
                    value = source
                adapted[key] = value
            self.load_state_dict(adapted)
            self._obs_norm._mean_sq = None
        else:
            raise ValueError(
                f"Tracking transfer expected {target_features - goal_features} or {target_features} policy inputs, "
                f"checkpoint has {source_features}"
            )
        self._sync_optimizer()
        print(
            f"[MimicKit] Loaded locomotion checkpoint with {source_features} policy inputs; "
            f"initialized {goal_features} trajectory inputs to zero.",
            flush=True,
        )

    def _load_params(self, config):
        super()._load_params(config)
        self._symmetry_weight = float(config.get("symmetry_weight", 0.))
        self._symmetry_scale = float(config.get("symmetry_scale", 1.))
        self._data_batch_size = int(config.get("data_batch_size", 4096))
        self._optimizer_microbatch_size = config.get("optimizer_microbatch_size", {}).copy()
        for role, size in self._optimizer_microbatch_size.items():
            if role not in ("actor", "critic", "disc") or type(size) is not int or size <= 0:
                raise ValueError("optimizer_microbatch_size requires positive integer actor/critic/disc sizes")
        if self._data_batch_size <= 0:
            raise ValueError("data_batch_size must be positive")
        if self._symmetry_weight < 0 or self._symmetry_scale <= 0:
            raise ValueError("symmetry_weight must be nonnegative and symmetry_scale must be positive")

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        if self._symmetry_weight:
            loss = self._compute_symmetry_loss(batch["obs"])
            info["actor_loss"] = info["actor_loss"] + self._symmetry_weight * loss
            info["sym_loss"] = loss.detach()
        return info

    def _compute_symmetry_loss(self, obs):
        # Mirror the current raw observations, before RMS normalization. Both
        # actor branches receive gradients, as in the equivariance objective.
        mirrored = self._env.reflect_observation(obs)
        mean = self._model.eval_actor(self._obs_norm.normalize(obs)).mean
        reflected_mean = self._model.eval_actor(self._obs_norm.normalize(mirrored)).mean
        physical_mean = self._a_norm.unnormalize(mean)
        target = self._a_norm.normalize(self._env.reflect_action(physical_mean))
        return self._symmetry_scale * torch.mean(torch.square(reflected_mean - target))
