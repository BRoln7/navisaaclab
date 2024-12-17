"""Configuration for a smpl_humanoid robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os

scout_wheel_radius = 0.098
scout_lenght = 0.262
scout_width = 0.37558

SCOUT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), 'usd', 'scout_mini.usd'),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            max_linear_velocity=1000,
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0., -0.,  0.)
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "front_left_wheel": 0.,
                "front_right_wheel": 0.,
                "rear_left_wheel": 0.,
                "rear_right_wheel": 0.,
                "inertial_joint": 0.,
            },
            damping={
                "front_left_wheel": 10000000.,
                "front_right_wheel": 10000000.,
                "rear_left_wheel": 10000000.,
                "rear_right_wheel": 10000000.,
                "inertial_joint": 10000000.,
            },
   ),}
)