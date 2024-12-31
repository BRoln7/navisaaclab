import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import os
import torch

wheel_radius = 0.135
width = 0.4132

CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), 'data', 'usd', 'robots', 'nova_carter.usda'),
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
        pos=(1., 5.,  0.),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "joint_wheel_right": 0.,
                "joint_wheel_left": 0.,
            },
            damping={
                "joint_wheel_right": 10000000.,
                "joint_wheel_left": 10000000.,
            },
   ),}
)

def two_wheel_differential(twist, wheel_r, width, _num_envs):
    velocity_tensor = torch.zeros(_num_envs, 7).to("cuda:0")
    
    velocity_tensor[:, 1] = (twist[:, 0] - (width/2) * twist[:, 1]) / wheel_r
    velocity_tensor[:, 2] = (twist[:, 0] + (width/2) * twist[:, 1]) / wheel_r

    return velocity_tensor