# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners import materials
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
import socnav_sim.tasks.amp.smpl_humanoid.mdp as mdp

##
# Scene definition
##

obs_computer = mdp.ObsComputer()

@configclass
class SmplHumanoidSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a smpl_humanoid robot."""

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
    robot: ArticulationCfg = SMPL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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

    joint_pos = mdp.JointPositionActionPacerCfg(
        asset_name="robot", joint_names=[".*"], scale=3.1416, offset=0.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        humanoid_obs = ObsTerm(func=obs_computer.compute_humanoid_obs)
        task_obs = ObsTerm(func=obs_computer.compute_task_obs)
        terrain_obs = ObsTerm(func=obs_computer.compute_height_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_motion,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    location_reward = RewTerm(mdp.compute_location_reward,
                              weight= 0.5)
    power_reward = RewTerm(mdp.compute_power_reward,
                           weight=0.5)
    amp_reward = RewTerm(obs_computer.compute_amp_reward,
                         weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the trajectory ends
    trajectory_end = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_state_below_minimum, params={"minimum_height": 0.6})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class SmplHumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: SmplHumanoidSceneCfg = SmplHumanoidSceneCfg(num_envs=8, env_spacing=2.5)
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
        self.episode_length_s = 32
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

        


