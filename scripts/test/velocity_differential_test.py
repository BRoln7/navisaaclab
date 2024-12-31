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
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
import socnav_sim.assets.scout_mini as scout  # isort:skip
import socnav_sim.assets.jackal as jackal
import socnav_sim.assets.nova_carter as carter  # isort:skip


@configclass
class ScoutSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        collision_group=-1,
        terrain_type="plane",
        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.4, 0.6, 0.95)
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # scout: ArticulationCfg = scout.SCOUT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    jackal: ArticulationCfg = jackal.JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    limo: ArticulationCfg = carter.CARTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")


def four_wheel_differential(twist, wheel_r, width):
    velocity_tensor = torch.zeros(args_cli.num_envs, 4).to("cuda:0")
    velocity_tensor[:, 0] = twist[:, 0] - twist[:, 1] * width / 2
    velocity_tensor[:, 2] = twist[:, 0] - twist[:, 1] * width / 2
    velocity_tensor[:, 1] = twist[:, 0] + twist[:, 1] * width / 2
    velocity_tensor[:, 3] = twist[:, 0] + twist[:, 1] * width / 2

    velocity_tensor = velocity_tensor.clone() / wheel_r

    return velocity_tensor

def two_wheel_differential(twist, wheel_r, width):
    velocity_tensor = torch.zeros(args_cli.num_envs, 7).to("cuda:0")
    
    velocity_tensor[:, 1] = (twist[:, 0] - (width/2) * twist[:, 1]) / wheel_r
    velocity_tensor[:, 2] = (twist[:, 0] + (width/2) * twist[:, 1]) / wheel_r

    return velocity_tensor

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # robot = scene["scout"]
    robot = scene["jackal"]
    robot2 = scene["limo"]    
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 3000 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            root_state = robot2.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot2.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot2.data.default_joint_pos.clone(), robot2.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot2.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting robot state...")
        velocity_target = torch.tensor([[1., 0.5], 
                                        [1., 0.5],
                                        [1., 0.5], 
                                        [1., 0.5]], device='cuda:0')
        
        # velocity_tensor1 = four_wheel_differential(velocity_target, scout.wheel_radius, scout.width)
        # velocity_tensor1[:, [1,3]] *= -1
        # robot.set_joint_velocity_target(velocity_tensor1)
        
        velocity_tensor1 = four_wheel_differential(velocity_target, jackal.wheel_radius, jackal.width)
        # velocity_tensor1[:, [1,3]] *= -1
        robot.set_joint_velocity_target(velocity_tensor1)

        velocity_tensor2 = two_wheel_differential(velocity_target, carter.wheel_radius, carter.width)
        robot2.set_joint_velocity_target(velocity_tensor2)

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
