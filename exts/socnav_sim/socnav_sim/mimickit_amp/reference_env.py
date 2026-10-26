"""Official MimicKit ViewMotionEnv with the project's paired SMPL backend.

Import after AppLauncher. Frames are written and rendered without advancing
physics, evaluating a policy, constructing AMP histories, or optimizing a loss.
"""

import torch
from envs.view_motion_env import ViewMotionEnv

from .engine import SmplIsaacLabEngine


class SmplReferenceEnv(ViewMotionEnv):
    def _build_engine(self, engine_config, num_envs, device, visualize, record_video=False):
        return SmplIsaacLabEngine(engine_config, num_envs, device, visualize, record_video=record_video)

    def _get_env_motion_ids(self):
        if hasattr(self, "_review_motion_ids"):
            return self._review_motion_ids
        return super()._get_env_motion_ids()

    def show_frame(self, motion_ids, times):
        self._review_motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self._device)
        self._time_buf[:] = torch.as_tensor(times, dtype=self._time_buf.dtype, device=self._device)
        # Reuse upstream motion interpolation, joint mapping and state assignment.
        super()._sync_motion()
        expected = self._ref_body_pos.clone()
        char = self._get_char_id()
        self._engine.flush_resets(char, self._env_ids)
        actual = self._engine.get_body_pos(char)
        root_pos, root_rot, _, _, joint_rot, _ = self._motion_lib.calc_motion_frame(self._review_motion_ids, self._time_buf)
        _, expected_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        actual_rot = self._engine.get_body_rot(char)
        if not torch.isfinite(actual).all() or not torch.isfinite(actual_rot).all():
            raise RuntimeError("Non-finite USD body positions in reference playback")
        error = torch.linalg.vector_norm(actual-expected, dim=-1).amax().item()
        cosine = torch.nn.functional.cosine_similarity(actual_rot, expected_rot, dim=-1).abs().clamp(max=1.)
        angle = (2*torch.acos(cosine)).amax().item()
        if error > .05 or angle > .15:
            raise RuntimeError(f"Reference/USD FK differs by {error:.4f} m, {angle:.4f} rad; inspect the rig mapping before judging motions")
        return error, angle
