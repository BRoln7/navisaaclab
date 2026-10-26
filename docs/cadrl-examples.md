# CADRL navigation examples

The public navigation task has four concrete environments:

- `Isaac-Cadrl-Scene12`, configured by `data/cfg/cadrl_scene12.yaml`
- `Isaac-Cadrl-Scene27`, configured by `data/cfg/cadrl_scene27.yaml`
- `Isaac-Cadrl-Scene34`, configured by `data/cfg/cadrl_scene34.yaml`
- `Isaac-Cadrl-Scene35`, configured by `data/cfg/cadrl_scene35.yaml`

Each vectorized slot contributes one SMPL pedestrian and one Jackal. The Jackal
USD is self-contained and does not depend on legacy Nucleus material paths. Only
the first `num_robots` Jackal instances are active agents; the rest are parked
outside the map and excluded from PPO. All pedestrians remain active and follow
collision-aware A* paths computed from each scene's generated occupancy map.

Pedestrian speeds are sampled independently at every path reset. Scenes 12, 34
and 35 use the demo profile: 80% walk at `0.90-1.10 m/s` and 20% walk faster
at `1.25-1.45 m/s`. A* paths
are shortcut and corner-smoothed with collision checks before metric
resampling. The safe spatial path is then time-parameterized with the same
`2 m/s^2` acceleration and `2 rad/s` normal turn-rate limits used by the PACER
training trajectory; pedestrians slow before corners rather than receiving
unsafe positional noise. Planning uses a centered `7x7` obstacle-dilation kernel on the
`0.2 m` grid, providing approximately `0.6 m` clearance around furniture for
the non-reactive walking policy.

Pedestrians use `data/checkpoints/policy.pth`, exported from the PACER-style
MimicKit tracking checkpoint at iteration 2800. The navigation runtime uses the same official MimicKit SMPL USD,
255-dimensional observation layout, joint order, normalization and 30 Hz
control rate as training.

Train scene 12:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene12.yaml" ./isaaclab.sh -p \
  scripts/policy_training/cadrl/train.py --headless --device cuda:0 \
  --num_envs 32 --num_robots 8 --max_iterations 1000
```

Train scene 27:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene27.yaml" ./isaaclab.sh -p \
  scripts/policy_training/cadrl/train.py --headless --device cuda:0 \
  --num_envs 32 --num_robots 8 --max_iterations 1000
```

Train scene 34:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene34.yaml" ./isaaclab.sh -p \
  scripts/policy_training/cadrl/train.py --headless --device cuda:0 \
  --num_envs 32 --num_robots 8 --max_iterations 1000
```

Train scene 35:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene35.yaml" ./isaaclab.sh -p \
  scripts/policy_training/cadrl/train.py --headless --device cuda:0 \
  --num_envs 32 --num_robots 8 --max_iterations 1000
```

To record one 600-step video while training headlessly, add
`--video --video_length 600`. Add `--video_interval 10000` to record another
clip every 10,000 environment steps. Videos are written under the run's
`videos/train` directory. Add `--realistic_pedestrians` when a command should
render the TailorNet clothed SMPL overlay; without this flag, every task uses
the lightweight humanoid appearance. See [TailorNet pedestrian visuals](tailornet-pedestrians.md)
for asset setup and the close follow-camera command.

The YAML files already select the correct task ID, so an explicit `--task` is
normally unnecessary. Both examples enable `debug_vis` by default, drawing the
planned paths and ten moving target points for every pedestrian. Run inference
with:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/cadrl_scene27.yaml" ./isaaclab.sh -p \
  scripts/policy_training/cadrl/inference.py --device cuda:0 \
  --num_envs 16 --num_robots 4 --checkpoint /absolute/path/to/model.zip
```

Re-export a newer locomotion checkpoint reproducibly with:

```bash
python scripts/mimickit_amp/export_policy.py \
  logs/mimickit/smpl-amp/RUN/int_models/model_ITERATION.pt \
  --output data/checkpoints/policy.pth
```

Map loading and planning have no Isaac Sim or CUDA side effects. Mutable path
state belongs to each environment instance, so task discovery does not read
maps, allocate CUDA tensors or create visualization markers.
