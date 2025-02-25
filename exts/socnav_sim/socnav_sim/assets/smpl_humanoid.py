"""Configuration for a smpl_humanoid robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os

SMPL_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"data/usd/robots/new_human.usd",
        # usd_path=f"data/usd/robots/smpl_humanoid.usda",
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            max_linear_velocity=100,
            max_angular_velocity=10,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2, 2,  0.9556),
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

'''
body_mass

['Pelvis',
'L_Hip',
'R_Hip', 
'Torso', 
'L_Knee', 
'R_Knee', 
'Spine', 
'L_Ankle', 
'R_Ankle', 
'Chest', 
'L_Toe', 
'R_Toe', 
'Neck', 
'L_Thorax', 
'R_Thorax', 
'Head', 
L_Shoulder', 
'R_Shoulder', 
'L_Elbow', 
'R_Elbow', 
'L_Wrist', 
'R_Wrist', 'L_Hand', 'R_Hand']

smpl
[5.1249, 3.6615, 3.5949, 2.1559, 2.8735, 2.8748, 1.9085, 1.5240, 1.5976,
4.7059, 0.3793, 0.4081, 0.6445, 1.0855, 1.0598, 4.2519, 1.8960, 1.9828,
1.0497, 1.0857, 0.2981, 0.3164, 0.3978, 0.4076]

smpl-vis
[15.6621,  8.1512,  7.9639,  5.9568,  3.5758,  3.5963,  5.3418,  0.6771,
0.6887, 12.9372,  0.1587,  0.1637,  0.6445,  1.0750,  1.0531,  4.0104,
1.8793,  1.9705,  1.0224,  1.0528,  0.0879,  0.0879,  0.4723,  0.4729]

'''