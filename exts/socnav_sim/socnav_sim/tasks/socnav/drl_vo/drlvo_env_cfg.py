"""DRL-VO observations on the shared Jackal / MimicKit pedestrian scenes."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from socnav_sim.tasks.socnav.cadrl.cadrl_env_cfg import CadrlEnvCfg, CadrlSceneCfg
from . import mdp
from .lidar import NavigationLidarCfg


@configclass
class DrlvoSceneCfg(CadrlSceneCfg):
    lidar = NavigationLidarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jackal/base_link/drlvo_lidar",
        update_period=1.0 / 30.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        peds_obs = ObsTerm(func=mdp.compute_peds_obs)
        lidar_obs = ObsTerm(func=mdp.compute_lidar_obs)
        subgoal_obs = ObsTerm(func=mdp.compute_subgoal_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DrlvoEnvCfg(CadrlEnvCfg):
    scene_id: int = 12
    scene: DrlvoSceneCfg = DrlvoSceneCfg(num_envs=16, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.active_robot_count = self.actions.navcmd.active_robot_count
        self.viewer.eye = (24.0, -20.0, 32.0) if self.scene_id == 12 else (11.0, -11.0, 18.0)
        self.viewer.lookat = (-2.0, 8.0, 0.5) if self.scene_id == 12 else (2.0, 1.0, 0.5)
        self.sim.physx.gpu_max_rigid_contact_count = 2**21


@configclass
class DrlvoScene12EnvCfg(DrlvoEnvCfg):
    scene_id: int = 12


@configclass
class DrlvoScene35EnvCfg(DrlvoEnvCfg):
    scene_id: int = 35
