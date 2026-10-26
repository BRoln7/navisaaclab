# EgoNav first-person data collection

`EgoNav` is a data-collection example built on the same maps, AMP pedestrians,
and global planning used by CADRL. Its recording script creates one external
RGB-D camera that follows the first Jackal robot. This matches the project's
existing video pipeline without modifying the replicated robot articulation.
The robot follows the collision-aware global path with a deterministic
look-ahead controller, so no learned robot checkpoint is required.

The default Scene 12 example contains one moving robot and eight independently
planned pedestrians:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/egonav_scene12.yaml" \
./isaaclab.sh -p scripts/data_collection/collect_egocentric_rgbd.py \
  --headless --device cuda:0 --num_envs 8 --steps 1800 \
  --realistic_pedestrians
```

The program prints the exact output directory. By default it is:

```text
logs/egonav/Isaac-EgoNav-Scene12/YYYYMMDD-HHMMSS/
├── robot_first_person.webm
├── calibration.json
├── frames.jsonl
├── rgb/000000.png
└── depth/000000.png
```

Depth PNGs use unsigned 16-bit millimetres (`depth_m = pixel * 0.001`); zero
means invalid or clipped. `frames.jsonl` records the synchronized action,
robot pose and velocity, camera eye/target, navigation goal, time, and episode-reset
flag for every frame. `calibration.json` contains the pinhole intrinsic matrix.

Use `--video_only` for a small visual demonstration without image folders. A
normal completion or `Ctrl+C` both close the VP9 encoder and leave a playable
WebM. A force kill (`kill -9`, power loss, or an OOM kill) cannot finalize the
container.

Scene 35 is available with the same interface:

```bash
SOCNAV_CONFIG="$PWD/data/cfg/egonav_scene35.yaml" \
./isaaclab.sh -p scripts/data_collection/collect_egocentric_rgbd.py \
  --headless --device cuda:0 --num_envs 8 --steps 1800 \
  --realistic_pedestrians
```
