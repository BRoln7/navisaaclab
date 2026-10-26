"""CADRL training examples for selected IsaacNav scenes."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from socnav_sim.assets.jackal import JACKAL_CFG, WHEEL_BASE, WHEEL_RADIUS
from socnav_sim.assets.smpl_humanoid import MIMICKIT_SMPL_HUMANOID_CFG
from socnav_sim.config import load_task_config
from socnav_sim.tasks.socnav.nav_utils import action_base, event_base, reset_base
from socnav_sim.tasks.socnav.nav_utils.map_manager import get_navigation_scene_assets

import socnav_sim.tasks.socnav.cadrl.mdp as mdp


@configclass
class CadrlSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(
            visible=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )
    ped: ArticulationCfg = MIMICKIT_SMPL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Ped")
    robot: ArticulationCfg = JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    world = AssetBaseCfg(
        prim_path="/World/NavigationScene", collision_group=-1,
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=800.0,
        ),
    )


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    navcmd = action_base.NavWithPedActionCfg(
        asset_name="robot",
        right_wheel_joint_names=["front_right_wheel_joint", "rear_right_wheel_joint"],
        left_wheel_joint_names=["front_left_wheel_joint", "rear_left_wheel_joint"],
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_states_obs = ObsTerm(func=mdp.compute_joint_states_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_navigation = EventTerm(
        func=event_base.reset_navigation, mode="reset", params={"active_robot_count": 1}
    )


@configclass
class RewardsCfg:
    goal = RewTerm(func=mdp.compute_goal_reward, weight=30.0)
    collision = RewTerm(func=mdp.compute_collision_reward, weight=30.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=reset_base.time_runout, time_out=True)
    reach_goal = DoneTerm(func=reset_base.reach_goal, params={"goal_radius": 0.4})
    map_collision = DoneTerm(func=reset_base.robot_map_collision)


@configclass
class CurriculumCfg:
    pass


@configclass
class CadrlEnvCfg(ManagerBasedRLEnvCfg):
    """Shared CADRL task; concrete subclasses select an immutable scene."""

    scene_id: int = 27
    scene: CadrlSceneCfg = CadrlSceneCfg(num_envs=16, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        task = load_task_config()["env"]
        num_envs = int(task.get("num_envs", 16))
        active_robots = int(task.get("num_robots", num_envs))
        if not 0 < active_robots <= num_envs:
            raise ValueError(f"env.num_robots must be in [1, {num_envs}], got {active_robots}.")
        assets = get_navigation_scene_assets(self.scene_id)

        self.scene.num_envs = num_envs
        self.scene.world.spawn.usd_path = str(assets.usd)
        self.actions.navcmd.active_robot_count = active_robots
        self.actions.navcmd.pedestrian_speed = float(task.get("pedestrian_speed", 0.85))
        self.actions.navcmd.pedestrian_speed_variation = float(
            task.get("pedestrian_speed_variation", 0.15)
        )
        self.actions.navcmd.fast_pedestrian_probability = float(
            task.get("fast_pedestrian_probability", 0.10)
        )
        self.actions.navcmd.fast_pedestrian_speed = float(
            task.get("fast_pedestrian_speed", 1.20)
        )
        self.actions.navcmd.fast_pedestrian_speed_variation = float(
            task.get("fast_pedestrian_speed_variation", 0.15)
        )
        self.actions.navcmd.pedestrian_acceleration_max = float(
            task.get("pedestrian_acceleration_max", 2.0)
        )
        self.actions.navcmd.pedestrian_turn_rate_max = float(
            task.get("pedestrian_turn_rate_max", 2.0)
        )
        self.actions.navcmd.map_dilation = int(task.get("map_dilation", 7))
        self.actions.navcmd.pedestrian_policy_path = str(
            task.get("pedestrian_policy", self.actions.navcmd.pedestrian_policy_path)
        )
        self.actions.navcmd.realistic_pedestrian_visuals = bool(
            task.get("realistic_pedestrians", False)
        )
        self.actions.navcmd.pedestrian_visual_asset = str(
            task.get("pedestrian_visual_asset", self.actions.navcmd.pedestrian_visual_asset)
        )
        self.actions.navcmd.pedestrian_visual_update_interval = int(
            task.get("pedestrian_visual_update_interval", 1)
        )
        self.actions.navcmd.hide_pedestrian_physics_visuals = bool(
            task.get("hide_pedestrian_physics_visuals", True)
        )
        self.actions.navcmd.map_image_path = str(assets.map_image)
        self.actions.navcmd.map_yaml_path = str(assets.map_yaml)
        debug_vis = bool(task.get("debug_vis", False))
        self.actions.navcmd.visualize_pedestrian_trajectory = debug_vis
        self.events.reset_navigation.params.update(
            active_robot_count=active_robots,
            pedestrian_speed=self.actions.navcmd.pedestrian_speed,
            pedestrian_speed_variation=self.actions.navcmd.pedestrian_speed_variation,
            fast_pedestrian_probability=self.actions.navcmd.fast_pedestrian_probability,
            fast_pedestrian_speed=self.actions.navcmd.fast_pedestrian_speed,
            fast_pedestrian_speed_variation=self.actions.navcmd.fast_pedestrian_speed_variation,
            pedestrian_acceleration_max=self.actions.navcmd.pedestrian_acceleration_max,
            pedestrian_turn_rate_max=self.actions.navcmd.pedestrian_turn_rate_max,
            path_spacing=self.actions.navcmd.path_spacing,
            robot_goal_min=self.actions.navcmd.robot_goal_min,
            robot_goal_max=self.actions.navcmd.robot_goal_max,
            pedestrian_goal_min=self.actions.navcmd.pedestrian_goal_min,
            pedestrian_goal_max=self.actions.navcmd.pedestrian_goal_max,
            map_scale=self.actions.navcmd.map_scale,
            map_dilation=self.actions.navcmd.map_dilation,
            map_image_path=str(assets.map_image),
            map_yaml_path=str(assets.map_yaml),
            visualize_paths=debug_vis,
        )

        # Match the 30 Hz control frequency used to train the pedestrian actor.
        self.decimation = 4
        self.episode_length_s = 24.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.enable_stabilization = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


@configclass
class CadrlScene27EnvCfg(CadrlEnvCfg):
    scene_id: int = 27

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (42.0, -42.0, 48.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class CadrlScene12EnvCfg(CadrlEnvCfg):
    scene_id: int = 12

    def __post_init__(self):
        super().__post_init__()
        # The traversable area is roughly x=[-13, 8], y=[-9, 24].
        self.viewer.eye = (24.0, -20.0, 32.0)
        self.viewer.lookat = (-2.0, 8.0, 0.5)
        # Eight pedestrians and four robots need far fewer contact pairs than
        # Isaac Lab's large-scale default, leaving headroom for RTX recording.
        self.sim.physx.gpu_max_rigid_contact_count = 2**21


@configclass
class CadrlScene34EnvCfg(CadrlEnvCfg):
    scene_id: int = 34

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (38.0, -28.0, 48.0)
        self.viewer.lookat = (1.5, 13.5, 0.5)


@configclass
class CadrlScene35EnvCfg(CadrlEnvCfg):
    scene_id: int = 35

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (11.0, -11.0, 18.0)
        self.viewer.lookat = (2.0, 1.0, 0.5)
