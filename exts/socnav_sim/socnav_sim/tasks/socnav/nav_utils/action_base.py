"""Combined differential-drive navigation and AMP pedestrian action term."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from socnav_sim.config import PROJECT_ROOT
from .event_base import reset_pedestrian_assets
from .pedestrian_policy import (
    SMPL_BODY_NAMES,
    SMPL_JOINT_NAMES,
    SMPL_KEY_BODY_NAMES,
    MimicKitPedestrianPolicy,
    build_mimickit_observation,
)
from .state import NavigationRuntimeCfg, get_navigation_state


DEFAULT_PEDESTRIAN_POLICY = PROJECT_ROOT / "data/checkpoints/policy.pth"


class NavWithPedAction(ActionTerm):
    """Map a 2-D navigation command to a differential-drive robot and a pedestrian policy."""

    cfg: "NavWithPedActionCfg"

    def __init__(self, cfg: "NavWithPedActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation

        self._right_wheel_joint_ids, _ = self._asset.find_joints(
            cfg.right_wheel_joint_names, preserve_order=True
        )
        self._left_wheel_joint_ids, _ = self._asset.find_joints(
            cfg.left_wheel_joint_names, preserve_order=True
        )
        if not self._right_wheel_joint_ids or not self._left_wheel_joint_ids:
            raise RuntimeError("The differential-drive wheel joints could not be resolved.")

        self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._right_wheel_actions = torch.zeros(
            (self.num_envs, len(self._right_wheel_joint_ids)), device=self.device
        )
        self._left_wheel_actions = torch.zeros(
            (self.num_envs, len(self._left_wheel_joint_ids)), device=self.device
        )
        runtime_cfg = NavigationRuntimeCfg(
            active_robot_count=cfg.active_robot_count,
            pedestrian_speed=cfg.pedestrian_speed,
            pedestrian_speed_variation=cfg.pedestrian_speed_variation,
            fast_pedestrian_probability=cfg.fast_pedestrian_probability,
            fast_pedestrian_speed=cfg.fast_pedestrian_speed,
            fast_pedestrian_speed_variation=cfg.fast_pedestrian_speed_variation,
            pedestrian_acceleration_max=cfg.pedestrian_acceleration_max,
            pedestrian_turn_rate_max=cfg.pedestrian_turn_rate_max,
            path_spacing=cfg.path_spacing,
            robot_goal_distance=(cfg.robot_goal_min, cfg.robot_goal_max),
            pedestrian_goal_distance=(cfg.pedestrian_goal_min, cfg.pedestrian_goal_max),
            map_scale=cfg.map_scale,
            map_dilation=cfg.map_dilation,
            map_image_path=cfg.map_image_path,
            map_yaml_path=cfg.map_yaml_path,
        )
        self._navigation = get_navigation_state(env, runtime_cfg)
        self._pedestrian: Articulation = env.scene[cfg.pedestrian_asset_name]

        checkpoint = Path(cfg.pedestrian_policy_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pedestrian AMP policy does not exist: {checkpoint}")
        self._animation_net = MimicKitPedestrianPolicy.load(checkpoint, self.device)

        self._pedestrian_actions = torch.zeros((self.num_envs, 69), device=self.device)
        self._trajectory_marker = self._build_marker() if cfg.visualize_pedestrian_trajectory else None

        self._body_ids, resolved_body_names = self._pedestrian.find_bodies(
            [f"^{name}$" for name in SMPL_BODY_NAMES], preserve_order=True
        )
        self._policy_joint_ids, resolved_joints = self._pedestrian.find_joints(
            [f"^{name}$" for name in SMPL_JOINT_NAMES], preserve_order=True
        )
        self._key_body_ids, resolved_bodies = self._pedestrian.find_bodies(
            [f"^{name}$" for name in SMPL_KEY_BODY_NAMES], preserve_order=True
        )
        if (
            resolved_body_names != SMPL_BODY_NAMES
            or resolved_joints != SMPL_JOINT_NAMES
            or resolved_bodies != SMPL_KEY_BODY_NAMES
        ):
            raise RuntimeError("Navigation SMPL USD does not match the exported MimicKit policy")

        self._pedestrian_visuals = None
        if cfg.realistic_pedestrian_visuals:
            from .tailornet_visual import TailorNetPedestrianVisuals

            visual_asset = Path(cfg.pedestrian_visual_asset).expanduser()
            if not visual_asset.is_absolute():
                visual_asset = PROJECT_ROOT / visual_asset
            self._pedestrian_visuals = TailorNetPedestrianVisuals(
                asset_path=visual_asset,
                num_envs=self.num_envs,
                body_ids=self._body_ids,
                update_interval=cfg.pedestrian_visual_update_interval,
                hide_physics_visuals=cfg.hide_pedestrian_physics_visuals,
            )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def navigation_state(self):
        """Map paths and goals used by this action term.

        The state is exposed read-only so data-collection and visualization
        tools can follow the same global paths as the training task without
        reaching into a private implementation attribute.
        """
        return self._navigation

    def _build_marker(self) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/SocNav/PedestrianTargets",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.65)),
                )
            },
        )
        return VisualizationMarkers(marker_cfg)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        linear = (actions[:, 0] + 1.0) * 0.25
        angular = actions[:, 1] * 2.0
        linear = linear * self._navigation.active_mask
        angular = angular * self._navigation.active_mask
        half_width = 0.5 * self.cfg.wheel_base
        right_speed = (linear - half_width * angular) / self.cfg.wheel_radius
        left_speed = (linear + half_width * angular) / self.cfg.wheel_radius
        self._right_wheel_actions[:] = right_speed.unsqueeze(-1)
        self._left_wheel_actions[:] = left_speed.unsqueeze(-1)

        fallen = self._pedestrian.data.root_pos_w[:, 2] < self.cfg.minimum_pedestrian_height
        if torch.any(fallen):
            reset_pedestrian_assets(
                self._env, torch.nonzero(fallen, as_tuple=False).squeeze(-1), replan=True
            )
        ended = self._navigation.pedestrian_progress >= self._navigation.pedestrian_path_lengths - 2
        if torch.any(ended):
            ended_ids = torch.nonzero(ended, as_tuple=False).squeeze(-1)
            self._navigation.reset_pedestrians(
                ended_ids, sample_spawn=False, current_positions=self._pedestrian.data.root_pos_w[:, :2]
            )
        trajectory = self._navigation.pedestrian_targets(sample_count=10, sample_interval=0.5)
        observation = build_mimickit_observation(
            root_position=self._pedestrian.data.root_pos_w,
            root_quaternion_wxyz=self._pedestrian.data.root_quat_w,
            root_linear_velocity=self._pedestrian.data.root_lin_vel_w,
            root_angular_velocity=self._pedestrian.data.root_ang_vel_w,
            joint_position=self._pedestrian.data.joint_pos[:, self._policy_joint_ids],
            joint_velocity=self._pedestrian.data.joint_vel[:, self._policy_joint_ids],
            key_body_position=self._pedestrian.data.body_pos_w[:, self._key_body_ids],
            trajectory_xy=trajectory,
        )
        with torch.inference_mode():
            self._pedestrian_actions[:] = self._animation_net(observation)
        self._navigation.advance_pedestrians(float(self._env.step_dt))

        if self._pedestrian_visuals is not None:
            self._pedestrian_visuals.update(
                self._pedestrian.data.body_pos_w,
                self._pedestrian.data.body_quat_w,
            )

        if self.cfg.follow_pedestrian_camera:
            target = self._pedestrian.data.root_pos_w[0].detach().cpu()
            target[2] += self.cfg.pedestrian_camera_target_height
            offset = torch.tensor(self.cfg.pedestrian_camera_offset)
            self._env.sim.set_camera_view(
                eye=(target + offset).tolist(),
                target=target.tolist(),
            )

        if self._trajectory_marker is not None:
            heights = torch.full((*trajectory.shape[:-1], 1), self.cfg.marker_height, device=self.device)
            self._trajectory_marker.visualize(translations=torch.cat((trajectory, heights), dim=-1).flatten(0, 1))

    def apply_actions(self) -> None:
        self._pedestrian.set_joint_position_target(
            self._pedestrian_actions, joint_ids=self._policy_joint_ids
        )
        self._asset.set_joint_velocity_target(
            self._right_wheel_actions, joint_ids=self._right_wheel_joint_ids
        )
        self._asset.set_joint_velocity_target(
            self._left_wheel_actions, joint_ids=self._left_wheel_joint_ids
        )


@configclass
class NavWithPedActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavWithPedAction
    pedestrian_asset_name: str = "ped"
    pedestrian_policy_path: str = str(DEFAULT_PEDESTRIAN_POLICY)
    active_robot_count: int = 1
    pedestrian_speed: float = 0.85
    pedestrian_speed_variation: float = 0.15
    fast_pedestrian_probability: float = 0.10
    fast_pedestrian_speed: float = 1.20
    fast_pedestrian_speed_variation: float = 0.15
    pedestrian_acceleration_max: float = 2.0
    pedestrian_turn_rate_max: float = 2.0
    path_spacing: float = 0.10
    robot_goal_min: float = 4.0
    robot_goal_max: float = 8.0
    pedestrian_goal_min: float = 6.0
    pedestrian_goal_max: float = 10.0
    map_scale: float = 0.25
    map_dilation: int = 7
    map_image_path: str = ""
    map_yaml_path: str = ""
    right_wheel_joint_names: list[str] = MISSING
    left_wheel_joint_names: list[str] = MISSING
    wheel_radius: float = 0.135
    wheel_base: float = 0.4132
    visualize_pedestrian_trajectory: bool = False
    marker_height: float = 0.05
    minimum_pedestrian_height: float = 0.4
    realistic_pedestrian_visuals: bool = False
    pedestrian_visual_asset: str = "data/usd/assets/people/tailornet_smpl_female.usd"
    pedestrian_visual_update_interval: int = 1
    hide_pedestrian_physics_visuals: bool = True
    follow_pedestrian_camera: bool = False
    pedestrian_camera_offset: tuple[float, float, float] = (3.2, -4.0, 1.8)
    pedestrian_camera_target_height: float = 0.8
