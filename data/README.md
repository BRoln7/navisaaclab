# Data required by NavIsaacLab

Follow the download and installation instructions in the root
[Data setup](../README.md#data-setup). This directory contains the files needed
by the AMP training, navigation, and RGB-D collection examples in that README.

Download the external data bundle from
[NavIsaacLab assets on Hugging Face](https://huggingface.co/datasets/hill77/NavIsaacLab_assets).
Extract it into the repository using the layout below. Source SMPL model PKLs
are downloaded separately; see [SMPL source models](#smpl-source-models-download-separately).

## Required files

| Workflow | Files to retain |
| --- | --- |
| AMP training | `mimickit/amass-pacer-v1/native/motions.yaml` and every motion clip referenced by that list |
| Official walk/run baseline (optional) | `mimickit/official_smpl/smpl_walk.pkl`, `smpl_run.pkl`, and `motions.yaml` |
| Navigation training and RGB-D collection | `checkpoints/policy.pth`, `usd/assets/robots/jackal.usda`, the selected scene USD, and its navigation-map PNG/YAML |
| Clothed pedestrians (optional) | `usd/assets/people/tailornet_smpl_female.usd` and its dependencies, enabled with `--realistic_pedestrians` |
| Trained CADRL playback (optional) | `checkpoints/cadrl_policy.zip` or another chosen CADRL `.zip` checkpoint |

Preserve the relative paths below when extracting the external data bundle:

```text
data/
├── cfg/                                  # tracked experiment configurations
├── checkpoints/
│   ├── policy.pth                        # required pedestrian controller
│   └── cadrl_policy.zip                  # optional navigation checkpoint
├── mimickit/
│   ├── amass-pacer-v1/native/
│   │   ├── motions.yaml
│   │   ├── manifest.json
│   │   └── clip_*.pkl                    # 248 final training clips
│   └── official_smpl/
│       ├── smpl_walk.pkl
│       ├── smpl_run.pkl
│       └── motions.yaml                 # equal walk/run sampling weights
└── usd_assets/assets/
    ├── people/tailornet_smpl_female.usd    # optional clothed appearance
    ├── robots/jackal.usda
    └── scenes/IsaacNav scenes2/
        └── Collected_scene_<id>/
            ├── scene_<id>.usd
            ├── scene_<id>_nav_map.png
            └── scene_<id>_nav_map.yaml
```

The bundle stores assets in `usd_assets/`. The current simulator accesses them
through `data/usd/assets/`, so create this alias once from the repository root:

```bash
ln -s usd_assets data/usd
```

The symlink is a local installation step; the uploaded asset directory remains
`usd_assets/`. The root README uses scene 12 and shows a scene 35 demo. Keep each selected
scene's complete asset directory, including the materials, textures, meshes,
and sub-assets referenced by its USD. The robot and pedestrian USDs also need
their referenced assets; the tree above lists entry files, not every dependency.

## Configurations and MimicKit assets

AMP uses these tuned configurations; retain their current values:

```text
data/cfg/mimickit_amp/agent.yaml
data/cfg/mimickit_amp/engine.yaml
data/cfg/mimickit_amp/env.yaml
data/cfg/mimickit_amp/env_tracking.yaml
```

Retain the `cadrl_scene<id>.yaml`, `drlvo_scene<id>.yaml`, and
`egonav_scene<id>.yaml` files for the
navigation scenes and collection examples you use. Other configurations should
be checked for imports or references before removal.

Keep `data/cfg/task.yaml` as the default used by shared configuration loading.
The current CADRL and DRL-VO examples share the same Jackal, MimicKit pedestrian
asset, exported pedestrian policy, and scene/map files. DRL-VO does not require
the legacy Carter robot or warehouse scene.

AMP training and the navigation pedestrian controller also require the official
MimicKit SMPL assets, installed outside this directory:

```text
third_party/MimicKit/data/assets/smpl/smpl.xml
third_party/MimicKit/data/assets/smpl/smpl.usd
```

Keep their companion assets and license files. Installation is documented in
the root [README](../README.md#installation).

## Source data and metadata

The runtime bundle retains only the final 248 native training clips (60,587
frames at 30 Hz) and the two official MimicKit walk/run PKLs. The native clips
use uniform per-clip weights of 1.0. The official motions are a separate
baseline; the default AMP task loads only the 248-clip native set. Preprocessing
does not run during ordinary training.
Run commands from the repository root: the motion lists contain paths relative
to that directory rather than private machine-specific absolute paths. Keep
`mimickit/amass-pacer-v1/native/manifest.json` and the official SMPL
`asset-manifest.json`: the loader uses them when present to validate skeleton
and timestep compatibility, and motion-review tools require the motion
manifest. Map metadata JSON files provide generation provenance.

Raw AMASS downloads, legacy motion caches and intermediate conversion PKLs are
excluded. Source SMPL models are kept locally as described below and are not
included in the public download.

### SMPL source models (download separately)

Register at the [official SMPL website](https://smpl.is.tue.mpg.de/) and use
its [Downloads page](https://smpl.is.tue.mpg.de/download.php) to obtain the
Python SMPL models. Place the female, male, and neutral PKLs at:

```text
data/smpl/SMPL_FEMALE.pkl
data/smpl/SMPL_MALE.pkl
data/smpl/SMPL_NEUTRAL.pkl
```

If the downloaded model filenames differ, rename them to the names above.
Use SMPL models, rather than SMPL-H or SMPL-X models. This local-only directory
is ignored by Git. **Exclude `data/smpl/` from the Hugging Face data archive
as well**; Git ignore rules do not exclude files from an archive or folder
upload.

Preprocessing defaults to this model directory; another location can be passed
with `process_amass.py --smpl_models`. Rebuilding the clothed mesh uses
`data/smpl/SMPL_FEMALE.pkl` by default:

```bash
./isaaclab.sh -p scripts/socnav/build_tailornet_smpl_visual.py \
  --smpl data/smpl/SMPL_FEMALE.pkl
```

These source models are needed for rebuilding assets, not for ordinary AMP
training with prepared motion clips or rendering the prepared clothed USD.

Keep `data/cfg/smpl_skeletontree.xml` for the retained preprocessing tools and
CPU regression tests. Optional legacy curation tools require their source
datasets to be rebuilt before use; those old caches are not shipped.

Scene 34 additionally uses `scene_34_cadrl.usda`; preserve it together with the
original scene and map if running that optional example.

Third-party motion data, body models, and visual assets retain their original
licenses. Packaging converted AMASS data does not grant redistribution rights;
verify the applicable permissions before publishing those files on Hugging Face.
