# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from socnav_sim.assets.scout_mini import SCOUT_CFG, scout_wheel_radius, scout_lenght, scout_width  # isort:skip
from socnav_sim.sensors.sensors.ray_caster import RTXRayCaster, RTXRayCasterCfg
from socnav_sim.sensors.sim.spawners.sensors.sensors_cfg import LidarCfg

@configclass
class ScoutSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    scout: ArticulationCfg = SCOUT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    lidar = RTXRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/scout_mini/base_link/lidar",
        offset=RTXRayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=LidarCfg(lidar_type=LidarCfg.LidarType.VELODYNE_VLS128),
        debug_vis=True,
    )


def four_wheel_differential(twist, wheel_r, length, width):
    """
    twist: [num_envs, 2-dimension twist]
    wheel_r: radius of wheel
    length: distance of front and rear wheel
    width: distance of left and right wheel
    """
    velocity_tensor = torch.zeros(args_cli.num_envs, 4).to("cuda:0")
    velocity_tensor[:, 0] = twist[:, 0] - twist[:, 1] * length / 2 + twist[:, 1] * width / 2
    velocity_tensor[:, 1] = twist[:, 0] - twist[:, 1] * length / 2 - twist[:, 1] * width / 2
    velocity_tensor[:, 2] = twist[:, 0] + twist[:, 1] * length / 2 + twist[:, 1] * width / 2
    velocity_tensor[:, 3] = twist[:, 0] + twist[:, 1] * length / 2 - twist[:, 1] * width / 2
    velocity_tensor = velocity_tensor.clone() / wheel_r

    return velocity_tensor

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["scout"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")
        velocity_target = torch.tensor([[1.0, 1.0], 
                                        [1.0, -1.0],
                                        [1.0, 1.0], 
                                        [1.0, -1.0]], device='cuda:0')
        velocity_tensor = four_wheel_differential(velocity_target, scout_wheel_radius, scout_lenght, scout_width)
        velocity_tensor[..., [0, 2]] *= -1
        robot.set_joint_velocity_target(velocity_tensor)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([5, 0.0, 5.0], [0.0, 0.0, 2.0])
    scene_cfg = ScoutSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
