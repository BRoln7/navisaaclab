"""Configuration for a smpl_humanoid robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

turtlebot_wheel_radius = 0.0332
turtlebot_width = 0.16

TURTLEBOT_CFG3 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), 'data', 'usd', 'robots', 'turtlebot3.usda'),
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0., -0.,  0.),
        # rot=(-0.7487, -0.0248,  0.0320, -0.6617),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "wheel_left_joint": 0.,
                "wheel_right_joint": 0.,
            },
            damping={
                "wheel_left_joint": 10000000.,
                "wheel_right_joint": 10000000.,
            },
   ),}
)
