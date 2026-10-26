# TailorNet pedestrian visuals

CADRL pedestrians can use a clothed SMPL render mesh while retaining the
MimicKit SMPL articulation for physics and the trained PACER policy for motion.
The render layer does not change observations, actions, collisions, rewards or
checkpoints.

The current asset uses TailorNet's official `t-shirt_female_sample`: a shaped
SMPL body, one simulated T-shirt displacement field, dark trousers and shoes.
At runtime Isaac Sim applies standard SMPL linear blend skinning to the body and
garment from the 24 physical link transforms. This is intentionally lighter
than running TailorNet's pose-dependent wrinkle networks every frame.

## Use the prepared visual asset

When the prepared `tailornet_smpl_female.usd` is installed from the runtime
bundle, no source SMPL model or TailorNet network is needed for rendering.
Follow the root [data setup](../README.md#data-setup), then pass
`--realistic_pedestrians` to the navigation or recording command.

TailorNet's dataset license permits distribution for specified non-commercial
purposes. The generated body mesh also includes SMPL-derived content, whose
license must be respected separately. See the
[TailorNet dataset license](https://github.com/zycliao/TailorNet_dataset/blob/master/LICENSE.md)
and [SMPL model license](https://smpl.is.tue.mpg.de/modellicense.html).

## Rebuild from licensed source models

Rebuilding requires the TailorNet source sample and a separately obtained
`SMPL_FEMALE.pkl`. Register at the [official SMPL website](https://smpl.is.tue.mpg.de/),
download the Python model from its [Downloads page](https://smpl.is.tue.mpg.de/download.php),
and place it at `data/smpl/SMPL_FEMALE.pkl`. This is the builder's default path.
Keep source models local; exclude `data/smpl/` from Git and the Hugging Face
data upload. See the [source-model setup](../data/README.md#smpl-source-models-download-separately)
for all three model filenames.

From the Isaac Lab Python environment:

```bash
./isaaclab.sh -p scripts/socnav/download_tailornet_sample.py
./isaaclab.sh -p scripts/socnav/build_tailornet_smpl_visual.py \
  --smpl data/smpl/SMPL_FEMALE.pkl
```

The downloader verifies the official archive SHA-256 hashes. The builder writes
and reopens `data/usd/assets/people/tailornet_smpl_female.usd`, then validates
the 24-joint skeleton, 6,890 body vertices, 7,702 garment vertices and four LBS
influences per vertex.

The exporter deliberately reproduces TailorNet's CSC/COO Loop-subdivision
vertex order. Garment indices depend on this order; sorting the new edge
vertices selects unrelated body regions even though the resulting mesh remains
a valid USD.

## Use in CADRL scenes

Scene 12 and 35 configurations provide the visual asset with these keys:

```yaml
pedestrian_visual_asset: data/usd/assets/people/tailornet_smpl_female.usd
pedestrian_visual_update_interval: 1
hide_pedestrian_physics_visuals: true
```

The clothed mesh is disabled by default. Add `--realistic_pedestrians` to a
CADRL training, CADRL inference, overview recording or EgoNav recording command
to enable it. Omitting the flag keeps the lightweight physical humanoid visible.
This avoids high-resolution USD skinning and per-step animation authoring for
jobs that do not need photorealistic pedestrians; physics and policies remain
identical in both modes.

Record a close-up validation clip with:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene12.yaml" \
./isaaclab.sh -p scripts/policy_training/cadrl/train.py \
  --headless --video --video_length 150 \
  --realistic_pedestrians \
  --follow_pedestrian_camera --num_envs 1 --num_robots 1 \
  --max_iterations 6 --device cuda:0 --seed 0
```

`--camera_offset X Y Z` changes the follow-camera position. Normal multi-agent
recordings continue to use each scene's configured overview camera.

## Source references

- [TailorNet dataset and data format](https://github.com/zycliao/TailorNet_dataset)
- [TailorNet model implementation](https://github.com/chaitanya100100/TailorNet)
- [TailorNet paper](https://arxiv.org/abs/2003.04583)
