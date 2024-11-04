"""Configuration for a carter robot."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/social-navigation/USD-saver/nova_carter_sensors.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            articulation_enabled=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-15.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_wheel_left"],effort_limit=400.0,
                                                   velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "right_wheel_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_wheel_right"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "caster_base_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_caster_base"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "swing_left_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_swing_left"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "swing_right_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_swing_right"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "caster_left_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_caster_left"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),
        "caster_right_actuator": ImplicitActuatorCfg(joint_names_expr=["joint_caster_right"],effort_limit=400.0,
                                                    velocity_limit=100.0,stiffness=0.0,damping=10.0),                                                    
    },
)