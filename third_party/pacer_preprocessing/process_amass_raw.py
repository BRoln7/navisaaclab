# Adapted from nv-tlabs/pacer, commit baeb6241f03fc9d78370a64e3c2dcef9d830d97f.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# See README.md for source attribution, upstream notices and adaptation details.
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import glob

import os
import os.path as osp
import numpy as np
from tqdm import tqdm

def read_single_sequence(folder, seq_name, selected_keys=None):
    subjects = sorted(x for x in os.listdir(folder) if osp.isdir(osp.join(folder, x)))

    datas = {}

    for subject in tqdm(subjects):
        actions = [
            x for x in sorted(os.listdir(osp.join(folder, subject))) if x.endswith(".npz") and osp.isdir(osp.join(folder, subject))
        ]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            vid_name = f"{seq_name}_{subject}_{action[:-4]}"
            if selected_keys is not None and vid_name not in selected_keys:
                continue
            with np.load(fname, allow_pickle=False) as stream:
                data = dict(stream)
            # data['poses'] = pose = data['poses'][:, joints_to_use]

            # shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
            # theta = np.concatenate([pose,shape], axis=1)
            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = data
            # thetas.append(theta)

    return datas
