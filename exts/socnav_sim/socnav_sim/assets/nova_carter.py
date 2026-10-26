import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import torch

from socnav_sim.config import PROJECT_ROOT

wheel_radius = 0.135
width = 0.4132

CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PROJECT_ROOT / "data/usd/assets/robots/nova_carter.usda"),
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
    init_state=ArticulationCfg.InitialStateCfg(pos=(1.0, 5.0, 0.0)),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "joint_wheel_right": 0.0,
                "joint_wheel_left": 0.0,
            },
            damping={
                "joint_wheel_right": 10_000_000.0,
                "joint_wheel_left": 10_000_000.0,
            },
        ),
    },
)


def two_wheel_differential(twist: torch.Tensor, wheel_r: float, width: float) -> torch.Tensor:
    """Convert ``[linear_x, angular_z]`` commands to right/left wheel speeds."""
    wheel_speed = twist.new_empty((twist.shape[0], 2))
    wheel_speed[:, 0] = (twist[:, 0] - 0.5 * width * twist[:, 1]) / wheel_r
    wheel_speed[:, 1] = (twist[:, 0] + 0.5 * width * twist[:, 1]) / wheel_r
    return wheel_speed
