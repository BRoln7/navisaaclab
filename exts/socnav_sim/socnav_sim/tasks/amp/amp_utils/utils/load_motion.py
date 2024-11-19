import torch
import os
import numpy as np
from socnav_sim.tasks.amp.amp_utils.isaac_utils.flags import flags
from socnav_sim.tasks.amp.amp_utils.isaac_utils.motion_lib_smpl import MotionLib as MotionLibSMPL
from poselib.skeleton.skeleton3d import SkeletonTree
from socnav_sim.uhc.smpllib.smpl_parser import SMPL_Parser

class MotionLoad():
    def __init__(self, num_envs):
    
        self.key_body_ids = np.array([8, 7, 21, 20])
        motion_file = os.path.join(os.getcwd(), 'data', 'amass_isaac_run_upright_slim.pkl')
        self.motion_lib = MotionLibSMPL(
            motion_file=motion_file,
            key_body_ids=self.key_body_ids,
            device="cuda:0", masterfoot_conifg=None)

        self.humanoid_betas = torch.zeros(17, device='cuda:0')
        self.humanoid_limb_and_weights = torch.tensor([ 
            1.0268,  1.0334,  0.8325,  
            0.8418,  0.8470, 11.8577, 
            11.7424, 39.0816, 4.4896,  4.6094], device='cuda:0')

        HACK_MOTION_SYNC = False
        skeleton_trees_xml = os.path.join(os.getcwd(), 'data', 'cfg', 'smpl_skeletontree.xml')
        sk_tree = SkeletonTree.from_mjcf(skeleton_trees_xml)
        self.skeleton_trees = [sk_tree] * num_envs
        self.motion_lib.load_motions(skeleton_trees = self.skeleton_trees, 
                                gender_betas = self.humanoid_betas.cpu().repeat(num_envs), 
                                limb_weights = self.humanoid_limb_and_weights.cpu().repeat(num_envs), 
                                random_sample=not HACK_MOTION_SYNC)

        data_dir = "data/smpl"
        self.smpl_parser_n = SMPL_Parser(model_path=data_dir,
                                         gender="neutral").to('cuda:0')
        self.smpl_parser_m = SMPL_Parser(model_path=data_dir,
                                         gender="male").to('cuda:0')
        self.smpl_parser_f = SMPL_Parser(model_path=data_dir,
                                         gender="female").to('cuda:0')
        
        self.ref_motion_cache = {}


    def sample_ref_state(self, num_envs):
        curr_gender_betas = torch.zeros(num_envs, 17).to("cuda:0")
        motion_ids = self.motion_lib.sample_motions(num_envs)
        motion_times = self.motion_lib.sample_time(motion_ids)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, \
            dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel = self.get_fixed_smpl_state_from_motionlib(
            motion_ids, motion_times, curr_gender_betas)

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot

    def get_fixed_smpl_state_from_motionlib(self, motion_ids, motion_times, curr_gender_betas):
        motion_res = self.get_smpl_state_from_motionlib_cache(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, _, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["key_pos"], motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        with torch.no_grad():
            gender = curr_gender_betas[:, 0]
            betas = curr_gender_betas[:, 1:]
            B, _ = betas.shape

            genders_curr = gender == 2
            height_tolorance = 0.02
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_f.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)
                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = ((vertices_curr - offset[:, None])[..., -1].min(dim=-1).values - height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 1
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_m.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            genders_curr = gender == 0
            if genders_curr.sum() > 0:
                poses_curr = pose_aa[genders_curr]
                root_pos_curr = root_pos[genders_curr]
                betas_curr = betas[genders_curr]
                vertices_curr, joints_curr = self.smpl_parser_n.get_joints_verts(
                    poses_curr, betas_curr, root_pos_curr)

                offset = joints_curr[:, 0] - root_pos[genders_curr]
                diff_fix = (
                    (vertices_curr - offset[:, None])[..., -1].min(dim=-1).values -
                    height_tolorance)
                root_pos[genders_curr, ..., -1] -= diff_fix
                key_pos[genders_curr, ..., -1] -= diff_fix[:, None]
                rb_pos[genders_curr, ..., -1] -= diff_fix[:, None]

            return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel


    def get_smpl_state_from_motionlib_cache(self, motion_ids, motion_times):
        self.ref_motion_cache['motion_ids'] = motion_ids
        self.ref_motion_cache['motion_times'] = motion_times
        motion_res = self.motion_lib.get_motion_state_smpl(motion_ids, motion_times)
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache