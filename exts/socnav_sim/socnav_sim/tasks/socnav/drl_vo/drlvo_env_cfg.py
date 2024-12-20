# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from socnav_sim.assets.smpl_humanoid import SMPL_HUMANOID_CFG
from socnav_sim.assets.scout_mini import SCOUT_CFG
from socnav_sim.sensors.sensors.ray_caster import RTXRayCaster, RTXRayCasterCfg
from socnav_sim.sensors.sim.spawners.sensors.sensors_cfg import LidarCfg
import socnav_sim.tasks.socnav.drl_vo.mdp as mdp

##
# Scene definition
##

@configclass
class DrlvoSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with amp task."""

    # ground
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        collision_group=-1,
        terrain_type="plane",
        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.4, 0.6, 0.95)
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # robots
    ped: ArticulationCfg = SMPL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Ped") # type: ignore
    robot: ArticulationCfg = SCOUT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # sensor
    lidar = RTXRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/scout_mini/base_link/lidar",
        offset=RTXRayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=LidarCfg(lidar_type=LidarCfg.LidarType.VELODYNE_VLS128),
        debug_vis=True,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=5000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # navigate = mdp.JointPositionActionDrlvoCfg(
    #     asset_name="robot", joint_names=[".*"], scale=3.1416, offset=0.0,
    # )
    navigate = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, offset=0.0,
    )
    joint_pos = mdp.PacerActionCfg(
        asset_name="ped", joint_names=[".*"], scale=3.1416, offset=0.0,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        # peds_obs = ObsTerm(func=mdp.compute_peds_obs)
        # lidar_obs = ObsTerm(func=mdp.compute_lidar_obs)
        # subgoal_obs = ObsTerm(func=mdp.compute_subgoal_obs)
        peds_obs = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # reach_goal_reward = RewTerm(mdp.compute_location_reward,
    #                       weight=0.5)
    # collision_reward = RewTerm(mdp.compute_power_reward,
    #                        weight=0.5)
    # angular_reward = RewTerm(mdp.compute_direction_reward,
    #                        weight=0.0)
    # vo_reward = RewTerm(obs_computer.compute_amp_reward,
    #                      weight=1.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the trajectory ends
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    # reach_goal = DoneTerm(func=mdp.reach_goal, params={"minimum_height": 0.6})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class DrlvoEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: DrlvoSceneCfg = DrlvoSceneCfg(num_envs=8, env_spacing=6)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 1 / 120.0

        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.enable_stabilization = True
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0