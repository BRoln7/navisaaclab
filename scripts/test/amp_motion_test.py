# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to visualize a piece of amp-motion.
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="visualize a piece of amp-motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from socnav_sim.tasks.amp.amp_utils.utils.load_motion import MotionLoad
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import dof_gym_to_sim
from socnav_sim.tasks.amp.amp_utils.utils.obs_utils import rot_wxyz_to_xyzw
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from socnav_sim.assets.smpl_humanoid import SMPL_HUMANOID_CFG

import omni.isaac.lab.sim as sim_utils


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg = sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    amp_cfg = SMPL_HUMANOID_CFG.copy() # type: ignore
    amp_cfg.prim_path = "/World/Origin/Robot"
    amp_cfg = Articulation(cfg=amp_cfg)

    # return the scene information
    scene_entities = {"amp": amp_cfg}
    return scene_entities, origins


motion_load = MotionLoad(1)

def reset_init_state(asset:Articulation):
    # reset dof pose
    wxyz_to_xyzw = [3, 0, 1, 2]
    motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, \
      root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot = motion_load.viusual_from_start(1)
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_rot[:, wxyz_to_xyzw]], dim=-1))
    asset.write_root_velocity_to_sim(torch.cat([root_vel, root_ang_vel], dim=-1))
    asset.write_joint_state_to_sim(position=dof_pos[:,dof_gym_to_sim],\
                                    velocity=dof_vel[:,dof_gym_to_sim])
    
    return motion_ids, motion_times


def reset_amp_motion(motion_ids, motion_times):
    dt = 1 / 60

    motion_ids = torch.tile(motion_ids.unsqueeze(-1),[1, 500])
    motion_times = motion_times.unsqueeze(-1)

    time_steps = dt * (torch.arange(0, 500, device='cuda:0') + 1)
    motion_times = motion_times + time_steps

    motion_ids = motion_ids.view(-1)
    motion_times = motion_times.view(-1)

    motion_res = motion_load.get_smpl_state_from_motionlib_cache(motion_ids, motion_times)

    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, \
        key_pos, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
            motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["key_pos"], motion_res["motion_bodies"], \
                motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], \
                    motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
    
    return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["amp"]
    sim_dt = 1 / 120
    wxyz_to_xyzw = [3, 0, 1, 2]
    while simulation_app.is_running():
        motion_ids, motion_times = reset_init_state(robot)
        robot.reset()
        print("[INFO]: Resetting robot state...")
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel = reset_amp_motion(motion_ids, motion_times)
        for i in range(root_pos.size(0)):
            robot.write_root_pose_to_sim(torch.cat([root_pos[i], root_rot[i][wxyz_to_xyzw]], dim=-1)) # type: ignore
            robot.write_root_velocity_to_sim(torch.cat([root_vel[i], root_ang_vel[i]], dim=0)) # type: ignore
            robot.write_joint_state_to_sim(position=dof_pos[:, dof_gym_to_sim][i],\
                                velocity=dof_vel[:,dof_gym_to_sim][i]) # type: ignore
            sim.step()
            robot.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0]) # type: ignore
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins) # type: ignore


if __name__ == "__main__":
    main()
    simulation_app.close()