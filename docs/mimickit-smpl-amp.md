# SMPL AMP with MimicKit

These integration notes describe the MimicKit-based locomotion pipeline.
The legacy handcrafted SB3 AMP implementation has been removed. For current
training commands and tuned presets, use the root [README](../README.md#amp-locomotion).

For a fresh download of the original AMASS data, see
[the source audit and download instructions](mimickit-amass-rebuild-20261020.md).
That downloader preserves raw FPS and does not change this feature's training data.

For policy-free reference inspection, see
[the complete CPU audit and four-view motion review workflow](mimickit-reference-review.md).

## What is reused

The dependency is [MimicKit](https://github.com/xbpeng/MimicKit), pinned to
`fbe1a36e878f036d6bf5521defb8eb3b8ab4170d` in
[`third_party/mimickit.lock.json`](../third_party/mimickit.lock.json).
The checkout is ignored by Git; `fetch.py` recreates it on another machine.
Upstream source is kept unmodified, under its Apache-2.0 license.

| Component | Implementation |
| --- | --- |
| Rollout, PPO, GAE and discriminator losses | Upstream `learning/amp_agent.py` and `ppo_agent.py` |
| Optimizer updates | Native logical batches/update counts; microbatch accumulation in `agent.py` and `optimizer.py` |
| Replay ring/shuffle semantics and bounded-memory staging | Upstream `ExperienceBuffer`, extended by `buffers.py`; chunked native demo/reward calls in `agent.py` |
| Actor, critic, discriminator, normalizers and checkpoint weights | Upstream MimicKit |
| SMPL skeleton parsing, forward kinematics, motion interpolation and velocities | Upstream `MJCFCharModel` and `MotionLib` |
| AMP observations, history and reference-state initialization | Upstream `AMPEnv` |
| Simulator, actuation and ground contact sensors | Upstream `IsaacLabEngine`, extended for the official SMPL USD and application startup |
| Trajectory observations/reward, power and trajectory termination | `socnav_sim/mimickit_amp/env.py` |
| Actor symmetry regularizer | `SymmetricAMPAgent(AMPAgent)._compute_actor_loss`, calling `super()` |

This feature uses **MimicKit's official `assets/smpl/smpl.usd` and its paired
`smpl.xml`**, downloaded from the data pack linked in the upstream README.
They are installed under `third_party/MimicKit/data/assets/smpl/`, together with
the supplied license and an SHA-256 manifest. The USD is the original supplied
file, not a conversion of this project's old humanoid. Runtime SMPL/chumpy model
loading is unnecessary.

The official skeleton differs from the legacy asset in toe offsets, collision
geometry, joint limits and actuator settings. The existing motion dictionary is
converted again using the official XML; legacy-model caches are not reused.
Startup checks the installed asset hashes and the motion cache's skeleton hash,
rejecting a mismatched USD/XML pair or motions prepared for another model.

Your input `pose_quat_global` is already upright and in the project's 24-body DFS
order. The converter reconstructs local rotations and saves native MimicKit
`[root_position, root_expmap, joint_expmap]` frames. It does not apply the raw
AMASS axis conversion again. Height correction is one constant translation per
clip, calculated from the XML foot collision boxes, preserving flight height.

The upstream discriminator uses BCE and its native logarithmic style reward.
We retain that implementation; old PACER discriminator values and reward scales
are not directly comparable. This is an integration of the pinned upstream
implementation, rather than a claim that all its choices equal the 2021 paper.

## Setup and CPU checks

From the repository root, in the existing Isaac Lab environment:

```bash
conda activate navisaaclab
python scripts/mimickit_amp/fetch.py
python -m pip install -r requirements-mimickit-amp.txt
python scripts/mimickit_amp/install_assets.py /path/to/MimicKit_Data.zip
python scripts/mimickit_amp/prepare_motions.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/test_mimickit_amp.py -q
```

The official SMPL assets are **already installed in this workspace**. On another
machine, download `MimicKit_Data.zip` using the asset link in the upstream README
and run `install_assets.py`; it extracts only `assets/smpl/` and leaves an
existing destination unchanged. The model files and their license stay outside
the Git repository, and the installer preserves the license provided in the ZIP.

The dependencies were already present in this workspace; no package installation
or environment replacement was performed during integration. Although upstream
eagerly imports its W&B logger, the runner selects **TensorBoard** and never
initializes a W&B run.

The default conversion produces:

- `data/mimickit/official_smpl/locomotion/motions.yaml`: 246 clips / 63,758 frames.
- `data/mimickit/official_smpl/standing/motions.yaml`: 1 clip / 1,000 frames.

The current default is `data/mimickit/amass-pacer-v1/native/motions.yaml`:
248 rebuilt PACER-subset clips, uniform weight 1 per clip. It uses the corrected
30 Hz time axis described in [the rebuild report](mimickit-amass-rebuild-20261020.md).
Build it with `python scripts/mimickit_amp/process_amass.py --time_mode exact`.
The default environment inherits the upstream SMPL **10 second** episodes and
`global_obs: true`. Tracking is disabled by default; `env_tracking.yaml` or
`--finetune_tracking` explicitly enables it.

The older `locomotion-curated-v1/motions.yaml` (55/30/10/5 group probabilities)
remains available for historical comparisons. It is not the current default.
See [the curation report](mimickit-curated-locomotion-20261020.md) for its rules.

The original 246-file `locomotion/motions.yaml` and prior
`locomotion/motions-run6.yaml` remain available for existing checkpoints and
controlled comparisons. The [earlier composition audit](mimickit-motion-composition-20261019.md)
describes that previous experiment.

No clips in the supplied files were too short for the default AMP history.
Output manifests record input/skeleton hashes, per-clip hashes, FPS, floor shifts
and conversion errors. Repeating conversion reuses an identical verified output;
a changed input requires a new output directory.

CPU validation of the real training file produced finite native discriminator
features of shape `(512, 2370)` and a maximum conversion FK coordinate error of
`2.24e-6 m`. CUDA was not initialized. The tests exercise actual upstream AMP
updates on a synthetic vector environment; they do not simulate locomotion.

An offline OpenUSD check of the supplied official pair found 24 bodies and 69
DOFs, identical body order, matching rest rotations and a maximum skeleton
translation difference of `2.98e-8 m`. This checks the downloaded file contents;
runtime PhysX validation is still required.

## Simulator checks before a long run

A 16-environment GPU smoke test completed two native training iterations and
saved readable TensorBoard metrics and checkpoints. The 12,288-environment
probe exposed inconsistent GPU numbering and a PhysX allocation failure on
the 2060. The runner now defaults to `cuda:0` with the caller's CUDA ordering;
a full first iteration at 12,288 environments still needs validation. The short test is not evidence
that the larger run fits in memory.

First check the standing reset, without loading a policy:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode check --reset_motion standing --num_envs 4 --steps 60 --headless
```

Then check reference resets from moving clips:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode check --num_envs 4 --steps 1 --headless
```

Every launch flushes the reset to physics without advancing environment time,
refreshes the actual simulator state and compares it with native forward
kinematics. A position error over 5 cm, rotation error over 0.15 rad, or non-finite
state aborts the launch. This checks joint order/representation instead of
silently assuming that the USD and XML match. Check output is under
`logs/mimickit/smpl-amp/<timestamp>/check.json`.

A sampled moving pose held at fixed joint targets can fall; the hold check is
a reset/PD diagnostic, not an untrained walking controller. Pose matching alone
does not prove correct dynamics or successful AMP learning.

Then do a small end-to-end training run:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --headless --num_envs 64 --max_iterations 10 --seed 0
```

`--amp_only` disables trajectory observations/rewards and trajectory failure
termination. Power is also disabled unless an explicit non-zero
`--power_coefficient` is supplied. **The symmetry regularizer remains enabled** in both
stages. Its PACER-compatible settings are coefficient 1 and internal MSE scale
50. The runner rejects a zero symmetry weight. The 20 goal input dimensions are
reserved and zeroed in this stage so a checkpoint can later transfer to the
trajectory task.

An isolated official SMPL walk plus energy-regularization experiment uses:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --power_coefficient 5e-5 \
  --motion_file third_party/MimicKit/data/motions/smpl/smpl_walk.pkl \
  --headless --device cuda:0 --num_envs 16384 --max_iterations 500 --seed 0 \
  --out_dir logs/mimickit/smpl-official-walk-power-5e-5
```

Its optimization reward is `AMP - 5e-5 * sum(abs(torque * joint_velocity))`.
The output manifest and TensorBoard diagnostics retain the coefficient and raw
power so the run can be compared against a zero-power baseline.

After the smoke test passes, start an initial locomotion experiment:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --headless --device cuda:0 --num_envs 12288 \
  --max_iterations 1000 --seed 0
```

No checkpoint is loaded by default. One iteration is 32 control steps per
environment. All environment counts share `agent.yaml`; change only `--num_envs`.
Replay retains the upstream capacity of 200,000 samples on CPU. A first rollout
larger than this is inserted in bounded chunks, overwriting the oldest samples
without expanding the ring.

Actor/critic/discriminator logical batch ratios remain upstream **4/2/2 samples
per environment**, giving **40/32/32 optimizer updates per iteration** for the
upstream 32 rollout steps and 5/2/2 epochs. At 16,384 environments the logical
batches are 65,536/32,768/32,768. `optimizer_microbatch_size` limits forward/backward
chunks to 16,384/8,192/4,096, with no parameter update between chunks.

Gradient accumulation weights each partial chunk by its sample count. PPO uses
the number of random actions, while symmetry uses all observations. Replay
indices are reserved once per logical discriminator batch. Gradient clipping,
distributed reduction, momentum, weight decay and the optimizer counter are
applied once per logical batch. Models do not use batch-dependent normalization
layers; floating-point summation can still differ from a single large batch.

This corrects the earlier capped-batch policy, which inadvertently multiplied
the number of optimizer updates at large environment counts. Saved legacy
`batch_size_scaling` configs are migrated to their original `per_env` ratios;
their old caps become microbatch limits, and `disc_buffer_size_min` restores the
intended replay capacity. Older observation conventions are retained when loading
checkpoints. Use a fresh run to establish the new baseline.

The runner prints motion path, episode length, observation convention, reward
weights, logical/microbatch sizes and update counts before starting the simulator.

The default device is `cuda:0`; `--device` can be omitted. The runner preserves
the caller's CUDA device ordering and visibility and prints the selected CUDA
model at startup. It does not set `CUDA_DEVICE_ORDER` or `CUDA_VISIBLE_DEVICES`.

### CPU staging and memory bounds

Demonstration features and historical replay are float32 arrays in ordinary CPU
memory. The active rollout and the simulator/model remain on the selected device.
Discriminator minibatches transfer only the selected demo/replay samples to the
GPU. Actor/critic updates sample only the fields their native loss consumes, so
they do not gather or transfer AMP histories they never use.

`data_batch_size: 4096` bounds three operations independently of `--num_envs`:
native demo motion sampling/FK/history construction, shuffled replay insertion,
and AMP reward normalization/inference. Demo samples are written directly into
the reusable CPU buffer. There is no full-rollout GPU demo tensor, shuffled
replay copy or normalized AMP observation temporary.

| Environments (32 steps) | Demo + replay CPU storage | Persistent GPU storage removed |
| --- | --- | --- |
| 8,192 | about 4.08 GiB | about 4.08 GiB |
| 12,288 | about 5.24 GiB | about 5.24 GiB |
| 16,384 | about 6.39 GiB | about 6.39 GiB |

These are array-size calculations for 2,370 float32 discriminator features,
not measured total process peaks. Chunking also reduces transient GPU usage.
CPU staging uses additional RAM and PCIe bandwidth. Demo distributions/sample
counts, normalizer update timing, loss formulas, ten-frame history and symmetry
remain intact. The replay capacity is now the upstream 200,000. Chunking/CPU storage can change
RNG consumption and floating-point reduction order, so runs need not be bitwise
identical to upstream.

CPU regression cases cover partial chunks, normalization/reward agreement,
replay filling/wraparound, full-batch versus accumulated gradients, momentum,
weight decay, clipping, PPO random-action masking, exact PACER trajectory draws,
task/power reward and the strict trajectory-failure boundary. The MimicKit CPU
suite passed 53 tests on 2026-10-21; rerun it after configuration or code changes:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_mimickit_amp.py
```

This update has not been GPU-tested. Do the 64-environment check above before a
large run. No claim is made that changing update cadence alone fixes walking.
The 64-environment run exercises the native small-batch path. After the CPU
equivalence checks pass, exercise accumulation for all three networks with:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --headless --device cuda:0 --num_envs 8192 \
  --max_iterations 2 --seed 0
```

Then a fresh 16,384-environment run uses the same configs:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --headless --device cuda:0 --num_envs 16384 \
  --max_iterations 5000 --seed 0
```

Here 5,000 iterations is an experiment budget, not an estimate of convergence.

Inspect naturalness, falls and episode length across several checkpoints. There
is no commanded direction or speed in the locomotion stage; arbitrary motion is
expected. The 1,000-iteration budget is an initial evaluation point, not a
convergence guarantee.

Once locomotion is satisfactory, explicitly transfer to the trajectory task:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --checkpoint "$AMP_CKPT" --finetune_tracking \
  --headless --num_envs 4096 --max_iterations 10000 --seed 0
```

`AMP_CKPT` may point to a custom pure-AMP run with its `params/` directory or to
an official MimicKit run that retains `env_config.yaml`, `agent_config.yaml` and
`engine_config.yaml`. `--finetune_tracking` reads the current tracking configs,
retains the learned character/discriminator
parameters and enables task, small power cost and trajectory termination. It
initializes only the previously unused goal input weights and normalization,
so the first goal observation does not abruptly change the loaded controller.
Symmetry stays enabled. Optimizers and replay start fresh.

For a custom 255-feature checkpoint, the transfer zeros its previously unused
20 goal columns. For an official 235-feature MimicKit checkpoint, it expands the
actor and critic first layers to 255 features, copies the 235 character columns
exactly and initializes the 20 new goal columns to zero. Thus arbitrary initial
paths leave the loaded locomotion actions and values unchanged before learning.

Without `--finetune_tracking`, loading a checkpoint retains its saved settings,
including whether tracking is enabled. To start the trajectory task from scratch,
omit both checkpoint flags and select
`--env_config data/cfg/mimickit_amp/env_tracking.yaml` (without `--amp_only`).
Tracking transfer also preserves a loaded checkpoint's global/local observation
convention; changing that convention without retraining changes input meaning.

Run a short GPU smoke test before allocating the large experiment:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --env_config data/cfg/mimickit_amp/env_tracking.yaml \
  --headless --device cuda:0 --num_envs 64 --max_iterations 10 --seed 0
```

Then start the fresh tracking run by changing only the scale and budget:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --env_config data/cfg/mimickit_amp/env_tracking.yaml \
  --headless --device cuda:0 --num_envs 8192 --max_iterations 5000 --seed 0
```

Loading an older checkpoint also retains its saved motion list. To explicitly
continue with the rebuilt uniform data, add
`--motion_file data/mimickit/amass-pacer-v1/native/motions.yaml`.
The override changes only the selected data; it does not enable tracking or
change the saved loss weights. It cannot be combined with `--reset_motion standing`.

## Configuration

The three independent configs are in `data/cfg/mimickit_amp/`. They inherit the
pinned upstream SMPL configs and override explicit fields. `SOCNAV_CONFIG` does
not select this feature.

Defaults:

- Simulation 120 Hz, policy and demo history spacing 30 Hz (`dt = 1/30 s`).
- Native 10-frame AMP history. Sample anchors leave enough past frames inside
  each clip; negative times are not clamped into repeated first frames.
- Actor observation: 235 native character features + 20 local timed path
  coordinates. Discriminator: 237 native features × 10 frames.
- 69 spherical joint action coordinates, using native absolute PD targets and
  action normalization. No legacy 57-DOF mask or symmetry index table is reused.
- Self collisions are enabled, matching the upstream character creation default.
- Native `global_obs: true`; episode length 10 s (300 policy steps).
- Pure AMP: task weight 0, style weight 1, no power cost or trajectory termination.
- Optional PACER tracking: 101 vertices over the 10-second episode (0.1-second
  generator segments), speed 0–3 m/s, acceleration bound 2 m/s², heading-change
  bound 2 rad/s and independent 0.02 sharp-turn probability per segment. The
  first heading is uniform over `[-pi, pi]`, as in PACER.
- Ten future XY targets are sampled every 0.5 s and expressed in the character's
  heading-local frame. Task reward is `exp(-2 * planar_error_m**2)`.
- Failure if planar error exceeds 4 m, plus native fall termination. There is no
  velocity-matching reward, recovery reward, backwards penalty or grace period.
- Effective combined reward **when tracking is enabled**:
  `0.5 * (task_reward - 0.000001 * sum(abs(torque * angular_velocity))) + 0.5 * native_amp_reward`.
  There is no hidden timestep multiplier. PACER's Isaac Gym coefficient is
  `0.0005`; the Isaac Lab backend instead exposes an implicit-PD torque estimate.
  On this SMPL rig the PACER number produced a mean power cost larger than the
  mean tracking reward, so initial tracking experiments use a near-neutral
  coefficient while retaining power diagnostics for later calibration.
- Symmetry coefficient 1 with PACER's `50 * mean(square(error))` scale.
  Reflection happens on raw observations before RMS normalization, using the
  current shuffled actor minibatch and gradients through both mean-action branches.
- Native actor/critic/discriminator network and optimizer defaults are retained.
  Replay is 200,000 samples, and critic/discriminator evaluation is chunked.

Symmetry reflects across the XZ plane, swaps left/right joints and key bodies,
and distinguishes polar vectors from angular/rotation vectors. It does not
modify the discriminator objective or augment demonstration motions.

The trajectory port and parameter values are pinned against PACER commit
`baeb6241f03fc9d78370a64e3c2dcef9d830d97f`. Its
[`traj_generator.py`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/util/traj_generator.py)
supplies the sampling/interpolation convention,
[`humanoid_traj.py`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_traj.py)
supplies observations, reward and failure distance,
[`humanoid_pedestrain_terrain.py`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py)
supplies the power formula/default coefficient, and the
[`amp_humanoid_smpl_sept_task.yaml`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml)
config supplies the 0.5/0.5 task/style mixture. The AMP learner, network layout,
SMPL asset and flat simulation environment remain MimicKit/Isaac Lab components;
this is parameter and task alignment rather than an identical PACER runtime.

## TensorBoard, playback and checkpoints

```bash
tensorboard --logdir logs/mimickit --port 6006
```

Logging writes to the run folder. TensorBoard creates a startup event immediately
and flushes metrics after every completed iteration. Stage progress (rollout,
demo preparation and optimization) is printed and appended to `progress.log`;
exceptions are saved to `failure.txt` before simulator shutdown. The native
evaluation/checkpoint interval remains 100 iterations, starting at iteration
zero. Task diagnostics include position error, raw power, power cost, net task
reward, trajectory failures, falls and timeouts. Startup also checks that every
generated path starts at its character and that future path observations are
nonzero.

To view a new MimicKit checkpoint in the simulator:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode play --checkpoint latest \
  --num_envs 4 --steps 600 --reset_motion standing
```

`--checkpoint latest` selects the most recently written top-level training
checkpoint under `logs/mimickit/smpl-amp`. An explicit absolute path remains
available for reproducible comparisons.

Playback loads saved configs from the checkpoint's `params/` directory, and
writes root and target XY positions, tracking reward, power, tracking error,
root height and termination flags to `playback.csv`, plus a summary JSON. Keep the checkpoint with its `params/`
directory. To save a headless four-camera preview instead:

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode play --checkpoint latest \
  --num_envs 4 --steps 600 --headless --video --device cuda:0
```

`--video` enables offscreen cameras and streams a labeled 2-by-2 grid to
`four-environments.webm` in the output directory (VP9, 1280-by-720, 30 fps).
Each camera follows one environment; a cyan USD curve shows its generated path,
and time and reset counters appear in its label.
Use `--out_dir` to choose the destination. The default reference resets use the
training motion set; add `--reset_motion standing` to assess standing starts.
Rendering does not advance physics and recording retains no full video in RAM.

Old SB3 `.zip` checkpoints cannot be loaded. Upstream `.pt` checkpoints contain
networks and observation/action/discriminator normalizers, **not** optimizer
momentum, replay or the training iteration. Training with `--checkpoint` is a
weight warm start, not an exact resume. Ctrl-C saves these weights to
`model_interrupted.pt`; a completed run additionally saves `model_final.pt`.
