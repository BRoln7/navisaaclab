"""Configuration for a smpl_humanoid robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os

wheel_radius = 0.098
lenght = 0.262
width = 0.37558

JACKAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), 'data', 'usd', 'robots', 'jackal.usda'),
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0., -0.,  0.2),
        # rot=(-0.7487, -0.0248,  0.0320, -0.6617),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "front_left_wheel_joint": 0.,
                "front_right_wheel_joint": 0.,
                "rear_left_wheel_joint": 0.,
                "rear_right_wheel_joint": 0.,
            },
            damping={
                "front_left_wheel_joint": 10000000.,
                "front_right_wheel_joint": 10000000.,
                "rear_left_wheel_joint": 10000000.,
                "rear_right_wheel_joint": 10000000.,
            },
   ),}
)