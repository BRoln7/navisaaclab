import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

limo_wheel_radius = 0.06
limo_lenght = 0.2
limo_width = 0.14

LIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), 'data', 'usd', 'robots', 'limo.usda'),
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
        pos=(1., 5.,  0.2)
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "front_left_wheel": 0.,
                "front_right_wheel": 0.,
                "rear_left_wheel": 0.,
                "rear_right_wheel": 0.,
            },
            damping={
                "front_left_wheel": 10000000.,
                "front_right_wheel": 10000000.,
                "rear_left_wheel": 10000000.,
                "rear_right_wheel": 10000000.,
            },
   ),}
)
