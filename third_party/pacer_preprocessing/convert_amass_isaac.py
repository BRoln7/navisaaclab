# Adapted from nv-tlabs/pacer, commit baeb6241f03fc9d78370a64e3c2dcef9d830d97f.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# See README.md for source attribution, upstream notices and adaptation details.

from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from socnav_sim.uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names

amass_run_data = [
    '0-ACCAD_Female1Running_c3d_C25 -  side step right_poses',
    '0-ACCAD_Female1Running_c3d_C5 - walk to run_poses',
    '0-ACCAD_Female1Walking_c3d_B15 - walk turn around (same direction)_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B15 - Walk turn around_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses',
    '0-ACCAD_Male2Running_c3d_C17 - run change direction_poses',
    '0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses',
    '0-ACCAD_Male2Running_c3d_C24 - quick sidestep left_poses',
    '0-ACCAD_Male2Running_c3d_C3 - run_poses',
    '0-ACCAD_Male2Walking_c3d_B15 -  Walk turn around_poses',
    '0-ACCAD_Male2Walking_c3d_B17 -  Walk to hop to walk a_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk t2_poses',
    '0-ACCAD_Male2Walking_c3d_B18 -  Walk to leap to walk_poses',
    '0-BioMotionLab_NTroje_rub001_0017_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub020_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub027_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub076_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub077_0027_circle_walk_poses',
    '0-BioMotionLab_NTroje_rub104_0027_circle_walk_poses',
    '0-Eyes_Japan_Dataset_aita_walk-04-fast-aita_poses',
    '0-Eyes_Japan_Dataset_aita_walk-21-one leg-aita_poses',
    '0-Eyes_Japan_Dataset_frederic_walk-04-fast-frederic_poses',
    '0-Eyes_Japan_Dataset_hamada_walk-06-catwalk-hamada_poses',
    '0-Eyes_Japan_Dataset_kaiwa_walk-27-thinking-kaiwa_poses',
    '0-Eyes_Japan_Dataset_shiono_walk-09-handbag-shiono_poses',
    '0-HumanEva_S2_Jog_1_poses', '0-HumanEva_S2_Jog_3_poses',
    '0-KIT_10_WalkInClockwiseCircle10_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_10_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_12_WalkInClockwiseCircle09_poses',
    '0-KIT_12_WalkInClockwiseCircle11_poses',
    '0-KIT_12_WalkInCounterClockwiseCircle01_poses',
    '0-KIT_12_WalkingStraightForwards03_poses', '0-KIT_167_downstairs04_poses',
    '0-KIT_167_upstairs_downstairs01_poses',
    '0-KIT_167_walking_medium04_poses', '0-KIT_167_walking_run02_poses',
    '0-KIT_167_walking_run06_poses', '0-KIT_183_run04_poses',
    '0-KIT_183_upstairs10_poses', '0-KIT_183_walking_fast03_poses',
    '0-KIT_183_walking_fast05_poses', '0-KIT_183_walking_medium04_poses',
    '0-KIT_183_walking_run04_poses', '0-KIT_183_walking_run05_poses',
    '0-KIT_183_walking_run06_poses', '0-KIT_205_walking_medium04_poses',
    '0-KIT_205_walking_medium10_poses', '0-KIT_314_run02_poses',
    '0-KIT_314_run04_poses', '0-KIT_314_walking_fast06_poses',
    '0-KIT_314_walking_medium02_poses', '0-KIT_314_walking_medium07_poses',
    '0-KIT_314_walking_slow05_poses', '0-KIT_317_walking_medium09_poses',
    '0-KIT_348_walking_medium07_poses', '0-KIT_348_walking_run10_poses',
    '0-KIT_359_downstairs04_poses', '0-KIT_359_downstairs06_poses',
    '0-KIT_359_upstairs09_poses', '0-KIT_359_upstairs_downstairs03_poses',
    '0-KIT_359_walking_fast10_poses', '0-KIT_359_walking_run05_poses',
    '0-KIT_359_walking_slow02_poses', '0-KIT_359_walking_slow09_poses',
    '0-KIT_3_walk_6m_straight_line04_poses', '0-KIT_3_walking_medium07_poses',
    '0-KIT_3_walking_medium08_poses', '0-KIT_3_walking_run03_poses',
    '0-KIT_3_walking_slow08_poses', '0-KIT_424_run05_poses',
    '0-KIT_424_upstairs03_poses', '0-KIT_424_upstairs05_poses',
    '0-KIT_424_walking_fast04_poses', '0-KIT_425_walking_fast01_poses',
    '0-KIT_425_walking_fast04_poses', '0-KIT_425_walking_fast05_poses',
    '0-KIT_425_walking_medium08_poses',
    '0-KIT_4_WalkInClockwiseCircle02_poses',
    '0-KIT_4_WalkInClockwiseCircle05_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle02_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle07_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle08_poses',
    '0-KIT_513_downstairs06_poses', '0-KIT_513_upstairs07_poses',
    '0-KIT_675_walk_with_handrail_table_left003_poses',
    '0-KIT_6_WalkInClockwiseCircle04_1_poses',
    '0-KIT_6_WalkInClockwiseCircle05_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle01_1_poses',
    '0-KIT_6_WalkInCounterClockwiseCircle10_1_poses',
    '0-KIT_7_WalkInCounterClockwiseCircle09_poses',
    '0-KIT_7_WalkingStraightForwards04_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle03_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_8_WalkInCounterClockwiseCircle10_poses',
    '0-KIT_9_WalkInClockwiseCircle04_poses',
    '0-KIT_9_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_9_WalkingStraightForwards01_poses',
    '0-KIT_9_WalkingStraightForwards04_poses', '0-KIT_9_run01_poses',
    '0-KIT_9_run05_poses', '0-KIT_9_walking_medium02_poses',
    '0-KIT_9_walking_run02_poses', '0-KIT_9_walking_slow07_poses',
    '0-SFU_0005_0005_Jogging001_poses', '0-TotalCapture_s4_walking2_poses',
    '0-Transitions_mocap_mazen_c3d_crouchwalk_running_poses',
    "0-KIT_359_walking_slow10_poses",
    '0-KIT_513_downstairs07_poses', '0-KIT_9_walking_run07_poses',
    '0-KIT_183_downstairs01_poses', '0-KIT_167_downstairs05_poses', '0-TotalCapture_s3_walking1_poses',
    '0-KIT_675_walk_with_handrail_beam_right07_poses', '0-KIT_317_run02_poses', '0-KIT_348_walking_slow04_poses',
    '0-KIT_424_walking_fast03_poses', '0-KIT_11_WalkingStraightForwards03_poses', '0-KIT_3_walking_medium09_poses',
    '0-KIT_314_walking_medium06_poses', '0-SFU_0008_0008_Walking002_poses', '0-KIT_9_walking_run08_poses', '0-KIT_11_WalkingStraightForwards02_poses',
    '0-BioMotionLab_NTroje_rub021_0027_circle_walk_poses', '0-KIT_425_walking_medium09_poses', '0-KIT_348_run04_poses', '0-KIT_183_walking_fast10_poses',
    '0-KIT_424_walking_fast05_poses', '0-KIT_8_WalkInCounterClockwiseCircle01_poses', '0-KIT_317_run03_poses', '0-BioMotionLab_NTroje_rub047_0027_circle_walk_poses', '0-KIT_183_walking_fast04_poses', '0-KIT_183_walking_fast08_poses',
    '0-KIT_9_WalkInClockwiseCircle10_poses', '0-ACCAD_Male2Running_c3d_C15 - run turn right 45_poses', '0-KIT_11_WalkInCounterClockwiseCircle01_poses',
    '0-KIT_425_walking_slow02_poses', '0-KIT_167_walking_slow04_poses', '0-KIT_348_walking_fast04_poses', '0-KIT_183_run02_poses', '0-KIT_9_walking_slow05_poses', '0-KIT_317_walking_fast09_poses', '0-KIT_183_walking_fast07_poses', '0-KIT_7_WalkingStraightForwards08_poses',
    '0-KIT_3_downstairs05_poses', '0-BioMotionLab_NTroje_rub073_0027_circle_walk_poses', '0-KIT_10_WalkInCounterClockwiseCircle03_poses'
    , '0-BioMotionLab_NTroje_rub062_0027_circle_walk_poses', '0-KIT_8_WalkInCounterClockwiseCircle05_poses',
    '0-KIT_183_walking_slow04_poses', '0-KIT_424_walking_slow02_poses', '0-KIT_9_WalkInCounterClockwiseCircle08_poses',
    '0-KIT_3_walking_slow05_poses', '0-KIT_3_walking_slow06_poses', '0-KIT_348_walking_run10_poses', '0-KIT_513_upstairs_downstairs06_poses',
    '0-KIT_11_WalkInCounterClockwiseCircle07_poses', '0-KIT_424_downstairs01_poses', '0-KIT_6_WalkInClockwiseCircle01_1_poses',
    '0-ACCAD_Male2Walking_c3d_B23 -  side step right_poses', '0-KIT_348_walking_medium08_poses', '0-KIT_424_walking_run02_poses', '0-KIT_423_upstairs09_poses',
    '0-KIT_425_downstairs_07_poses', '0-KIT_359_walking_medium06_poses', '0-KIT_11_WalkInClockwiseCircle09_poses', '0-KIT_424_walking_slow01_poses',
    '0-Transitions_mocap_mazen_c3d_walksideways_turntwist180_poses', '0-KIT_9_WalkInClockwiseCircle08_poses', '0-KIT_317_walking_medium08_poses',
    '0-KIT_4_WalkInCounterClockwiseCircle03_poses', '0-KIT_425_walking_slow04_poses', '0-KIT_183_run01_poses',
    '0-KIT_9_WalkInCounterClockwiseCircle01_poses', '0-KIT_205_walking_run09_poses', '0-ACCAD_MartialArtsWalksTurns_c3d_E19 - dodge left_poses',
    '0-KIT_348_walking_fast07_poses', '0-KIT_4_WalkInCounterClockwiseCircle08_poses', '0-KIT_8_WalkInClockwiseCircle08_poses',
    '0-BioMotionLab_NTroje_rub042_0027_circle_walk_poses', '0-KIT_3_upstairs06_poses', '0-KIT_12_WalkInCounterClockwiseCircle08_poses', '0-KIT_424_run04_poses',
    '0-KIT_9_WalkingStraightForwards03_poses', '0-KIT_3_walking_medium05_poses', '0-ACCAD_Male1Walking_c3d_Walk B10 - Walk turn left 45_poses',
    '0-KIT_425_walking_fast02_poses', '0-KIT_11_WalkInCounterClockwiseCircle04_poses', '0-KIT_9_walking_slow10_poses',
    '0-KIT_359_walking_slow10_poses', '0-KIT_348_walking_fast08_poses',
    '0-KIT_425_walking_slow10_poses', '0-KIT_314_walking_run10_poses', '0-KIT_183_walking_run08_poses', '0-KIT_317_walking_medium06_poses',
    '0-KIT_9_run03_poses', '0-KIT_6_WalkInCounterClockwiseCircle06_1_poses', '0-KIT_8_WalkInCounterClockwiseCircle07_poses', '0-KIT_348_run01_poses',
    '0-KIT_348_walking_run05_poses', '0-KIT_8_WalkingStraightForwards04_poses', '0-KIT_314_walking_run03_poses',
    '0-KIT_167_walking_slow03_poses', '0-ACCAD_Male2Running_c3d_C11 - run turn left 90_poses', '0-KIT_317_walking_run03_poses', '0-KIT_3_upstairs03_poses',
    '0-ACCAD_Male1Running_c3d_Run C24 - quick side step left_poses', '0-KIT_3_walking_medium01_poses',
    '0-BioMotionLab_NTroje_rub098_0027_circle_walk_poses', '0-KIT_9_WalkInCounterClockwiseCircle02_poses', '0-KIT_424_walking_run09_poses',
    '0-KIT_167_walking_slow08_poses', '0-KIT_11_WalkingStraightForwards09_poses', '0-BioMotionLab_NTroje_rub041_0027_circle_walk_poses', '0-KIT_425_walking_03_poses',
    '0-KIT_314_walking_fast04_poses', '0-KIT_425_walking_medium01_poses', '0-KIT_167_upstairs01_poses', '0-KIT_167_walking_medium07_poses',
    '0-KIT_424_walking_slow06_poses', '0-TotalCapture_s1_walking1_poses', '0-KIT_424_walking_medium07_poses', '0-KIT_425_downstairs_02_poses',
    '0-BioMotionLab_NTroje_rub018_0027_circle_walk_poses', '0-KIT_205_walking_slow02_poses', '0-KIT_9_walking_slow08_poses',
    '0-KIT_359_walking_slow06_poses', '0-KIT_317_walking_medium04_poses', '0-KIT_205_walking_medium01_poses',
    '0-ACCAD_Male2Walking_c3d_B21 -  put down box to walk_poses', '0-BioMotionLab_NTroje_rub038_0027_circle_walk_poses',
    '0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses', '0-KIT_425_walking_medium06_poses',
    '0-Eyes_Japan_Dataset_hamada_walk-22-look around-hamada_poses', '0-Eyes_Japan_Dataset_frederic_walk-04-fast-frederic_poses',
    '0-HumanEva_S2_Walking_1_poses', '0-KIT_9_walking_medium05_poses', '0-BioMotionLab_NTroje_rub017_0027_circle_walk_poses', '0-KIT_359_walking_medium04_poses',
    '0-KIT_425_walking_medium07_poses', '0-KIT_12_WalkingStraightForwards09_1_poses', '0-HumanEva_S2_Jog_1_poses', '0-KIT_7_WalkingStraightForwards09_poses',
    '0-KIT_9_walking_slow03_poses', '0-BioMotionLab_NTroje_rub064_0027_circle_walk_poses', '0-BioMotionLab_NTroje_rub085_0027_circle_walk_poses',
    '0-KIT_424_run02_poses', '0-KIT_317_walking_slow03_poses', '0-KIT_317_walking_medium07_poses', '0-KIT_8_WalkingStraightForwards10_poses',
    '0-BioMotionLab_NTroje_rub084_0027_circle_walk_poses', '0-BioMotionLab_NTroje_rub092_0023_circle_walk_poses',
    '0-KIT_8_WalkInClockwiseCircle06_poses', '0-KIT_359_walking_fast06_poses', '0-KIT_359_walking_fast02_poses',
    '0-ACCAD_Female1Running_c3d_C4 - Run to walk1_poses', '0-KIT_4_WalkingStraightForward01_poses', '0-KIT_3_upstairs_downstairs01_poses',
    '0-KIT_425_walking_medium02_poses', '0-KIT_9_walking_run09_poses', '0-KIT_424_upstairs01_poses', '0-KIT_8_WalkInCounterClockwiseCircle02_poses',
    '0-KIT_348_walking_medium10_poses', '0-KIT_317_walking_fast08_poses', '0-BioMotionLab_NTroje_rub060_0028_circle_walk2_poses',
    '0-KIT_424_downstairs08_poses', '0-KIT_348_walking_medium02_poses', '0-ACCAD_Female1Walking_c3d_B2 - walk to stand_poses',
    '0-KIT_348_run03_poses', '0-KIT_8_WalkInCounterClockwiseCircle03_poses', '0-KIT_317_walking_run08_poses', '0-BioMotionLab_NTroje_rub054_0027_circle_walk_poses',
    '0-KIT_314_walking_medium03_poses', '0-KIT_424_downstairs02_poses', '0-KIT_348_walking_medium03_poses',
    '0-KIT_359_walking_medium05_poses', '0-Eyes_Japan_Dataset_shiono_walk-10-shoulder bag-shiono_poses', '0-KIT_3_walking_fast08_poses', '0-KIT_7_WalkingStraightForwards06_poses',
    '0-KIT_424_walking_run01_poses', '0-KIT_9_walking_run02_poses',
    '0-BioMotionLab_NTroje_rub029_0027_circle_walk_poses', '0-KIT_3_walking_fast10_poses', '0-KIT_359_walking_slow07_poses'
]

def run(in_file: str, out_file: str, skeleton_file: str):

    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    print(robot_cfg)

    # PACER uses beta=0 neutral for every clip. Reuse its fixed skeleton;
    # the paired MimicKit skeleton is applied in a separate native conversion.
    skeleton_tree = SkeletonTree.from_mjcf(str(skeleton_file))

    amass_data = joblib.load(in_file)

    double = False

    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']


    amass_full_motion_dict = {}
    for key_name in tqdm(amass_data.keys()):
        if not key_name in amass_run_data:
            continue

        smpl_data_entry = amass_data[key_name]
        B = smpl_data_entry['pose_aa'].shape[0]

        start, end = 0, 0

        pose_aa = smpl_data_entry['pose_aa'].copy()[start:]
        root_trans = smpl_data_entry['trans'].copy()[start:]
        B = pose_aa.shape[0]

        beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
        if len(beta.shape) == 2:
            beta = beta[0]

        gender = smpl_data_entry.get("gender", "neutral")
        fps = 30.0

        if isinstance(gender, np.ndarray):
            gender = gender.item()

        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
        elif gender == "male":
            gender_number = [1]
        elif gender == "female":
            gender_number = [2]
        else:
            raise Exception("Gender Not Supported!!")

        smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

        num = 1
        if double:
            num = 2
        for idx in range(num):
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

            gender_number, beta[:], gender = [0], 0, "neutral"
            print("using neutral model")



            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)

            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.numpy()

                ############################################################
                # key_name_dump = key_name + f"_{idx}"
                key_name_dump = key_name
                if idx == 1:
                    left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                    pose_quat_global = pose_quat_global[:, left_to_right_index]
                    pose_quat_global[..., 0] *= -1
                    pose_quat_global[..., 2] *= -1

                    root_trans_offset[..., 1] *= -1
                ############################################################

            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps
            amass_full_motion_dict[key_name_dump] = new_motion_out

    Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)
    joblib.dump(amass_full_motion_dict, out_file)
    return
