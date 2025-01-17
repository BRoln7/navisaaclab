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
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from socnav_sim.assets.smpl_humanoid import SMPL_HUMANOID_CFG
from socnav_sim.assets.nova_carter import CARTER_CFG
from socnav_sim.tasks.socnav.nav_utils import action_base, event_base, reset_base

import socnav_sim.tasks.socnav.crowdnav.mdp as mdp
##
# Scene definition
##

@configclass
class CrowdnavSceneCfg(InteractiveSceneCfg):
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
    robot: ArticulationCfg = CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    # )

    # scene
    scene = AssetBaseCfg(
        prim_path="/World/scene",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, -0.05)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"data/usd/scenes/scene_crowdnav/scene_crowdnav.usd"),
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

    navcmd = action_base.NavWithPedActionCfg(
        asset_name="robot", 
        joint_names=["joint_wheel_right", "joint_wheel_left"], # not stand just to use this 2 joint
        scale=1.0, offset=0.0,
    )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        joint_states_obs = ObsTerm(func=mdp.compute_joint_states_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=event_base.reset_init_state_amp_motion,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reach_goal_reward = RewTerm(mdp.compute_goal_reward,
                          weight=1.0)
    collision_reward = RewTerm(mdp.compute_collision_reward,
                           weight=1.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    ############ for robots ##############
    # (1) Terminate if the time out
    time_out = DoneTerm(func=reset_base.time_runout, time_out=True)
    # (2) Terminate if the robot reach goal
    reach_goal = DoneTerm(func=reset_base.reach_goal, params={"goal_radius": 0.25})

    ############# for peds ###############
    ped_fall = DoneTerm(func=reset_base.ped_reset)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class CrowdnavEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: CrowdnavSceneCfg = CrowdnavSceneCfg(num_envs=8, env_spacing=3)
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