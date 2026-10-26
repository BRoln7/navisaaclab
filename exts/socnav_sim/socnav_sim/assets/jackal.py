"""Jackal differential-drive robot configuration."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from socnav_sim.config import PROJECT_ROOT

WHEEL_RADIUS = 0.098
WHEEL_BASE = 0.37558

JACKAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PROJECT_ROOT / "data/usd/assets/robots/jackal.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            max_linear_velocity=100,
            max_angular_velocity=10,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=10_000_000.0,
        ),
    },
)
