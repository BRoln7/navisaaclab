# PACER motion preprocessing subset

Source: [nv-tlabs/pacer](https://github.com/nv-tlabs/pacer), revision
`baeb6241f03fc9d78370a64e3c2dcef9d830d97f` (2023-11-06).

These are attributed adaptations of functions in the project's existing
PACER-derived scripts under `exts/socnav_sim/socnav_sim/uhc/data_process/`.
The important functions/conventions were cross-checked against the upstream
revision. They are **not** represented as byte-identical upstream snapshots.
`sources.json` records local source hashes and adapted module hashes.

Retained code:

- `process_amass_raw.read_single_sequence`: original NPZ reading/key names.
- `process_amass_db.fix_height_smpl_vanilla` and `process_qpos_list`: occlusion
  removal/truncation, first-frame mesh floor correction, 22-body SMPL pose plus
  zero hand rotations, SMPL beta/gender/trans fields.
- `convert_amass_isaac.amass_run_data` and `run`: original locomotion selection,
  SMPL-to-DFS ordering, beta=0 neutral, root offset, upright global/local rotations.

Adaptations:

1. Remove interactive CLI/debugger code and unused imports/helpers. The raw
   reader checks directories and can load the final selected pool early to avoid
   materializing every AMASS motion in memory. Its key/array conventions remain.
2. Initialize model/parser globals from the CPU runner; omit unused `pose_6d`
   output and its legacy TorchGeometry dependency. The height/filter functions
   otherwise retain the project's PACER code.
3. Reuse a supplied fixed PACER neutral skeleton in `run` rather than generate
   and overwrite an identical neutral XML inside every iteration. This removes
   an unnecessary MuJoCo/mesh-generation dependency. The runner validates its
   body order and pelvis offset against the installed neutral SMPL model.
4. Use the correctly installed `poselib.skeleton` package path and the project's
   Python 3.11-compatible SMPL parser. All processing tensors remain on CPU.
5. Exact real-time 30 Hz sampling is a separate adapter **before** PACER's
   processing function. `--time_mode pacer_legacy` bypasses this adapter and
   retains PACER's integer stride. It does not imply bitwise reproduction of
   upstream's entire training split, asset generation, or missing v3 annotations.

The new runner uses the whole selected pool across the downloaded eight sources;
it does not silently reproduce the upstream `vald`/`valid` split typo, select only
the old slim subset, or claim an independent train/test evaluation split.
Weights are uniformly one; there is no new walking/running reweighting.

The public `download_data.sh` supplies occlusion **v2**, while upstream
`process_amass_db.py` names **v3**. The runner defaults to the published v2 file,
records its hash, and accepts an explicit `--occlusion` file. No missing annotation
is silently replaced by an empty dictionary.

## Notices

PACER's top-level [LICENSE](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/LICENSE)
contains:

```text
SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-NC-SA-4.0
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
```

The additional MPG notice from the original raw-processing file is retained in
that file. Preserve applicable upstream notices when redistributing these
adaptations; the repository's own code license does not replace those notices.
SMPL model files, AMASS NPZs and downloaded annotations are kept outside this
source directory. The corresponding dataset/model terms remain their own.
