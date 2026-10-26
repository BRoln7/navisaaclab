"""Trajectory/power task layered onto upstream MimicKit's SMPL AMP environment."""

import torch
from envs.amp_env import AMPEnv
from envs.base_env import DoneFlags
from util import torch_util

from .symmetry import SmplReflection
from .trajectory import Trajectory


def sample_history_times(library, n, history_dt, random=True):
    ids = library.sample_motions(n)
    lengths = library.get_motion_length(ids)
    if torch.any(lengths <= history_dt):
        raise ValueError("Convert/filter clips with the configured AMP history first")
    phase = torch.rand_like(lengths) if random else torch.zeros_like(lengths)
    return ids, history_dt + phase*(lengths-history_dt)


def trajectory_failure(position_error, max_distance, enabled=True):
    """PACER terminates as soon as planar trajectory error exceeds the limit."""
    if not enabled:
        return torch.zeros_like(position_error, dtype=torch.bool)
    return position_error > float(max_distance)


def mechanical_power(applied_torque, joint_velocity):
    """PACER's instantaneous absolute joint power, without a hidden dt."""
    return torch.sum(torch.abs(applied_torque * joint_velocity), dim=-1)


class SmplTrackingEnv(AMPEnv):
    def __init__(self, env_config, *args, **kwargs):
        self._task = env_config["tracking"].copy()
        self._tracking = bool(self._task.get("enabled", True))
        # Older pure-AMP checkpoints reserved the same ten trajectory samples
        # without storing this now-explicit PACER parameter.
        self._task.setdefault("observation_samples", 10)
        # Reflection assumes the unmodified native character observation layout.
        if env_config.get("enable_phase_obs") or env_config.get("enable_tar_obs"):
            raise ValueError("SMPL task expects native AMP observations without phase/reference targets")
        if env_config.get("root_height_obs", True) is not True:
            raise ValueError("SMPL symmetry requires root height in native observations")
        super().__init__(env_config, *args, **kwargs)
        samples = int(self._task["observation_samples"])
        self._reflection = SmplReflection(
            self._kin_char_model.get_body_names(), env_config["key_bodies"],
            future_points=samples, device=self._device)
        common = self._engine.get_obj_body_names(self._get_char_id())
        if list(common) != self._kin_char_model.get_body_names():
            raise ValueError(f"USD/XML body order differs: {common}")

    def _build_engine(self, engine_config, num_envs, device, visualize, record_video=False):
        from .engine import SmplIsaacLabEngine
        return SmplIsaacLabEngine(engine_config, num_envs, device, visualize, record_video=record_video)

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        self._trajectory = None
        if self._tracking:
            keys = (
                "num_vertices", "heading_change_max", "speed_min", "speed_max",
                "acceleration_max", "sharp_turn_probability", "observation_samples",
                "observation_timestep",
            )
            try:
                settings = {key: self._task[key] for key in keys}
            except KeyError as error:
                raise ValueError(
                    f"Tracking config predates the PACER trajectory port and lacks {error.args[0]!r}"
                ) from error
            self._trajectory = Trajectory(
                self.get_num_envs(), self._episode_length, self._device, **settings)
        self._position_error = torch.zeros(self.get_num_envs(), device=self._device)
        self._raw_power = torch.zeros_like(self._position_error)
        self._task_r = torch.zeros_like(self._position_error)
        self._traj_fail = torch.zeros_like(self._position_error, dtype=torch.bool)
        self._fall_fail = torch.zeros_like(self._traj_fail)
        self._stats = torch.zeros(6, device=self._device)
        self._stat_steps = 0
        self._trajectory_marker = None

    def _sample_motion_times(self, n):
        history = (self._num_disc_obs_steps-1)*self._engine.get_timestep()
        return sample_history_times(self._motion_lib, n, history, self._rand_reset)

    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        # Preserve sampled root link velocity after upstream clears body caches.
        char_id = self._get_char_id()
        self._engine.set_root_vel(env_ids, char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, char_id, self._ref_root_ang_vel[env_ids])

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids):
            char_id = self._get_char_id()
            self._engine.flush_resets(char_id, env_ids)
            if self._tracking:
                self._trajectory.reset(env_ids, self._engine.get_root_pos(char_id)[env_ids], self._engine.get_root_rot(char_id)[env_ids])
            self._position_error[env_ids] = 0
            self._traj_fail[env_ids] = False
            self._fall_fail[env_ids] = False

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        if not self._tracking:
            # Reserve transfer inputs without computing an unused trajectory or
            # drawing trajectory RNG in a native pure-AMP experiment.
            task_size = 2 * int(self._task["observation_samples"])
            return torch.cat((obs, obs.new_zeros((obs.shape[0], task_size))), dim=-1)
        ids = self._env_ids if env_ids is None else env_ids
        char_id = self._get_char_id()
        task_obs = self._trajectory.observation(ids, self._engine.get_root_pos(char_id)[ids],
                                               self._engine.get_root_rot(char_id)[ids], self._time_buf[ids])
        return torch.cat((obs, task_obs), dim=-1)

    def _update_reward(self):
        char_id = self._get_char_id()
        self._raw_power[:] = mechanical_power(
            self._engine.get_dof_forces(char_id), self._engine.get_dof_vel(char_id)
        )
        if not self._tracking:
            self._position_error.zero_()
            self._task_r.zero_()
            # Native pure AMP uses coefficient zero.  An explicit non-zero
            # coefficient creates the isolated AMP - power ablation without
            # constructing or observing a trajectory task.
            self._reward_buf[:] = -float(self._task["power_coefficient"])*self._raw_power
            return
        root = self._engine.get_root_pos(char_id)
        target = self._trajectory.position(self._env_ids, self._time_buf)
        self._position_error[:] = torch.linalg.vector_norm(target[:, :2]-root[:, :2], dim=-1)
        self._task_r[:] = torch.exp(-float(self._task["position_scale"])*self._position_error.square())
        # Environment emits task + small power cost. Native AMPAgent mixes in style.
        self._reward_buf[:] = self._task_r - float(self._task["power_coefficient"])*self._raw_power

    def _update_done(self):
        super()._update_done()
        self._fall_fail[:] = self._done_buf == DoneFlags.FAIL.value
        self._traj_fail[:] = trajectory_failure(
            self._position_error, self._task["max_distance"], self._tracking)
        self._done_buf[self._traj_fail] = DoneFlags.FAIL.value
        self._stats += torch.stack((self._position_error.mean(), self._task_r.mean(), self._raw_power.mean(),
                                    self._traj_fail.float().sum(), self._fall_fail.float().sum(),
                                    (self._done_buf == DoneFlags.TIME.value).float().sum()))
        self._stat_steps += 1

    def record_diagnostics(self):
        diag = super().record_diagnostics()
        values = self._stats / max(self._stat_steps, 1)
        power_cost = values[2]*float(self._task["power_coefficient"])
        diag.update(position_error_m=values[0].item(), task_reward=values[1].item(),
                    raw_power_w=values[2].item(), power_cost=power_cost.item(),
                    net_task_reward=(values[1]-power_cost).item(),
                    traj_fail_per_step=values[3].item(), fall_fail_per_step=values[4].item(), timeouts_per_step=values[5].item())
        self._stats.zero_(); self._stat_steps = 0
        return diag

    def reflect_observation(self, obs):
        return self._reflection.observation(obs)

    def reflect_action(self, action):
        return self._reflection.action(action)

    def update_trajectory_markers(self, env_ids=None):
        """Draw the ten moving trajectory targets consumed by the policy."""
        if not self._tracking:
            return
        if env_ids is None:
            env_ids = self._env_ids[:min(self.get_num_envs(), 4)]
        if self._trajectory_marker is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/MimicKitTrajectoryMarkers",
                markers={
                    "future_waypoint": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.55, 1.0)),
                    ),
                    "current_target": sim_utils.SphereCfg(
                        radius=0.10,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.15)),
                    ),
                },
            )
            self._trajectory_marker = VisualizationMarkers(marker_cfg)

        points = self._trajectory.samples(env_ids, self._time_buf[env_ids]).clone()
        points[:, :, :2] += self._engine._env_offsets[env_ids, None, :]
        points[:, :, 2] = 0.12
        # One green current target plus nine blue future targets per environment.
        marker_indices = torch.zeros(points.shape[:2], dtype=torch.long, device=points.device)
        marker_indices[:, 0] = 1
        self._trajectory_marker.visualize(
            translations=points.flatten(0, 1), marker_indices=marker_indices.flatten()
        )

    def _render_scene(self):
        if self._tracking:
            self.update_trajectory_markers()
        super()._render_scene()

    def validate_reset(self, tolerance=.05):
        """Fail before training if USD joint representation does not match native FK."""
        char_id = self._get_char_id()
        dof = self._engine.get_dof_pos(char_id)
        root = self._engine.get_root_pos(char_id)
        rot = self._engine.get_root_rot(char_id)
        expected, expected_rot = self._kin_char_model.forward_kinematics(root, rot, self._kin_char_model.dof_to_rot(dof))
        actual = self._engine.get_body_pos(char_id)
        actual_rot = self._engine.get_body_rot(char_id)
        if not all(torch.isfinite(x).all() for x in (expected, expected_rot, actual, actual_rot)):
            raise RuntimeError("Non-finite USD/native SMPL state after reset")
        distance = torch.linalg.vector_norm(expected-actual, dim=-1).max().item()
        cosine = torch.nn.functional.cosine_similarity(expected_rot, actual_rot, dim=-1).abs().clamp(max=1)
        angle = (2*torch.acos(cosine)).max().item()
        if distance > tolerance or angle > .15:
            raise RuntimeError(f"USD/native SMPL FK mismatch: {distance:.4f} m, {angle:.4f} rad. Do not train this rig.")
        return {"max_body_position_error_m": distance, "max_body_rotation_error_rad": angle}

    def validate_tracking(self):
        """Verify that reset produced finite, moving paths and nonzero goal observations."""
        if not self._tracking:
            return {"enabled": False}
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        paths = self._trajectory.paths
        start_error = torch.linalg.vector_norm(paths[:, 0, :2] - root_pos[:, :2], dim=-1)
        segment_speed = torch.linalg.vector_norm(paths[:, 1:, :2] - paths[:, :-1, :2], dim=-1)
        segment_speed = segment_speed / self._trajectory.segment_dt
        task_obs = self._trajectory.observation(
            self._env_ids, root_pos, root_rot, self._time_buf
        )
        tensors = (paths, start_error, segment_speed, task_obs)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError("Non-finite PACER trajectory state after reset")
        if torch.count_nonzero(segment_speed).item() == 0 or torch.count_nonzero(task_obs).item() == 0:
            raise RuntimeError("PACER trajectory reset produced no motion or no policy goal signal")
        return {
            "enabled": True,
            "max_start_error_m": start_error.max().item(),
            "mean_path_speed_mps": segment_speed.mean().item(),
            "max_path_speed_mps": segment_speed.max().item(),
            "mean_abs_goal_observation": task_obs.abs().mean().item(),
        }
