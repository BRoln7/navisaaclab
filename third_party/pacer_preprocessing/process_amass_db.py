# Adapted from nv-tlabs/pacer, commit baeb6241f03fc9d78370a64e3c2dcef9d830d97f.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# See README.md for source attribution, upstream notices and adaptation details.

import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace

# Initialized by the CPU-only runner before calling the original functions.
target_fr = 30
amass_occlusion = {}
flags = SimpleNamespace(debug=False)

def fix_height_smpl_vanilla(pose_aa, th_trans, th_betas, gender, seq_name):
    # no filtering, just fix height
    gender = gender.item() if isinstance(gender, np.ndarray) else gender
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")

    if gender == "neutral":
        smpl_parser = smpl_parser_n
    elif gender == "male":
        smpl_parser = smpl_parser_m
    elif gender == "female":
        smpl_parser = smpl_parser_f
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")

    batch_size = pose_aa.shape[0]
    verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1], th_betas.repeat((1, 1)), th_trans=th_trans[0:1])

    # vertices = verts[0].numpy()
    gp = torch.min(verts[:, :, 2])

    # if gp < 0:
    th_trans[:, 2] -= gp

    return th_trans

def process_qpos_list(qpos_list):
    amass_res = {}
    removed_k = []
    pbar = qpos_list
    for (k, v) in tqdm(pbar):
        # print("=" * 20)
        k = "0-" + k
        seq_name = k
        betas = v["betas"]
        gender = v["gender"]
        amass_fr = v["mocap_framerate"]
        skip = int(amass_fr / target_fr)
        amass_pose = v["poses"][::skip]
        amass_trans = v["trans"][::skip]

        bound = amass_pose.shape[0]
        if k in amass_occlusion:
            issue = amass_occlusion[k]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[k]:
                bound = amass_occlusion[k]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", k, bound)
                    continue
            else:
                print("issue irrecoverable", k, issue)
                continue

        seq_length = amass_pose.shape[0]
        if seq_length < 10:
            continue
        with torch.no_grad():
            amass_pose = amass_pose[:bound]
            batch_size = amass_pose.shape[0]
            # import ipdb; ipdb.set_trace()
            amass_pose = np.concatenate([amass_pose[:, :66], np.zeros((batch_size, 6))], axis=1) # We use SMPL and not SMPLH

            pose_aa = torch.tensor(amass_pose)  # After sampling the bound

            amass_trans = torch.tensor(amass_trans[:bound])  # After sampling the bound
            betas = torch.from_numpy(betas)


            amass_trans = fix_height_smpl_vanilla(
                pose_aa=pose_aa,
                th_betas=betas,
                th_trans=amass_trans,
                gender=gender,
                seq_name=k,
            )


            amass_res[seq_name] = {
                "pose_aa": pose_aa.numpy(),
                # "qpos": qpos,
                "trans": amass_trans.numpy(),
                "beta": betas.numpy(),
                "seq_name": seq_name,
                "gender": gender,
            }

        if flags.debug and len(amass_res) > 10:
            break
    print(removed_k)
    return amass_res
