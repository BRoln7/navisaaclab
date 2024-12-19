"""Configuration for a smpl_humanoid robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os

SMPL_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), 'data', 'usd', 'robots', 'smpl_humanoid.usda'),
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
        pos=(-0.1443, -0.2548,  0.9556),
        rot=(-0.7487, -0.0248,  0.0320, -0.6617),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_Hip.*": 800,
                ".*_Knee.*": 800,
                ".*_Ankle.*": 800,
                "Torso.*": 1000,
                "Spine.*": 1000,
                "Chest.*": 1000,
                ".*_Toe.*": 500,
                "Neck.*": 500,
                "Head.*": 500,
                ".*_Thorax.*": 500,
                ".*_Shoulder.*": 500,
                ".*_Elbow.*": 500,
                ".*_Wrist.*": 300,
                ".*_Hand.*": 300,
            },
            damping={
                ".*_Hip.*": 80,
                ".*_Knee.*": 80,
                ".*_Ankle.*": 80,
                "Torso.*": 100,
                "Spine.*": 100,
                "Chest.*": 100,
                ".*_Toe.*": 50,
                "Neck.*": 50,
                "Head.*": 50,
                ".*_Thorax.*": 50,
                ".*_Shoulder.*": 50,
                ".*_Elbow.*": 50,
                ".*_Wrist.*": 30,
                ".*_Hand.*": 30,
            },
   ),}
)