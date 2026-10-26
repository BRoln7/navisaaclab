"""Run the pinned, unmodified MimicKit entry point from its repository root."""

from pathlib import Path
import os
import runpy
import shlex
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit


def _normalize_out_dir(argv):
    """Keep MimicKit output outside its checkout and reject an empty shell expansion."""
    key = "--out_dir"
    if key not in argv:
        return

    index = argv.index(key) + 1
    missing_value = index >= len(argv) or argv[index].startswith("--")
    empty_value = not missing_value and not argv[index].strip()
    if missing_value or empty_value:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output = ROOT / "logs/mimickit-official" / f"smpl-walk-{timestamp}"
        if missing_value:
            argv.insert(index, str(output))
        else:
            argv[index] = str(output)
        print(f"[MimicKit] Empty --out_dir; using {output}")
        return

    output = Path(argv[index]).expanduser()
    if not output.is_absolute():
        argv[index] = str((Path.cwd() / output).resolve())


def _arg_value(args, key):
    flag = f"--{key}"
    if flag not in args:
        return None
    index = args.index(flag) + 1
    if index >= len(args) or args[index].startswith("--"):
        raise ValueError(f"Missing value for {flag}")
    return args[index]


def _effective_arg_value(argv, key, checkout):
    value = _arg_value(argv, key)
    if value is not None:
        return value

    arg_file = _arg_value(argv, "arg_file")
    if arg_file is None:
        return None
    path = Path(arg_file).expanduser()
    if not path.is_absolute():
        path = checkout / path
    file_args = shlex.split(path.read_text(), comments=True)
    return _arg_value(file_args, key)


def _yaml_int(path, key):
    prefix = f"{key}:"
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return int(stripped[len(prefix):].split("#", 1)[0].strip())
    raise KeyError(f"Missing {key} in {path}")


def _normalize_max_iterations(argv, checkout):
    iterations_value = _arg_value(argv, "max_iterations")
    if iterations_value is None:
        return
    iterations = int(iterations_value)
    if iterations <= 0:
        raise ValueError("--max_iterations must be positive")

    num_envs = int(_effective_arg_value(argv, "num_envs", checkout) or 1)
    agent_config = _effective_arg_value(argv, "agent_config", checkout)
    if agent_config is None:
        raise ValueError("--max_iterations requires --agent_config or an arg file that defines it")
    config_path = Path(agent_config).expanduser()
    if not config_path.is_absolute():
        config_path = checkout / config_path
    steps_per_iter = _yaml_int(config_path, "steps_per_iter")
    max_samples = iterations * num_envs * steps_per_iter

    iterations_index = argv.index("--max_iterations")
    del argv[iterations_index:iterations_index + 2]
    if "--max_samples" in argv:
        samples_index = argv.index("--max_samples") + 1
        if samples_index >= len(argv) or argv[samples_index].startswith("--"):
            raise ValueError("Missing value for --max_samples")
        argv[samples_index] = str(max_samples)
    else:
        argv.extend(["--max_samples", str(max_samples)])
    print(
        f"[MimicKit] Training budget: {iterations} iterations x {num_envs} envs "
        f"x {steps_per_iter} steps = {max_samples} samples"
    )


def _set_arg_value(argv, key, value):
    flag = f"--{key}"
    if flag in argv:
        index = argv.index(flag) + 1
        if index >= len(argv) or argv[index].startswith("--"):
            raise ValueError(f"Missing value for {flag}")
        argv[index] = str(value)
    else:
        argv.extend([flag, str(value)])


def _prepare_video_output(argv):
    value = _arg_value(argv, "video_out")
    if value is None:
        return None
    mode = _arg_value(argv, "mode") or "train"
    if mode != "test":
        raise ValueError("--video_out is only supported with --mode test")

    output = Path(value).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if output.suffix == "":
        output = output.with_suffix(".webm")

    index = argv.index("--video_out")
    del argv[index:index + 2]
    _set_arg_value(argv, "video", "true")
    _set_arg_value(argv, "visualize", "false")
    return output


def _export_test_video(env, output):
    diagnostics = env.record_diagnostics()
    recording = diagnostics.get("sim_recording")
    if recording is None or recording.get_num_frames() == 0:
        raise RuntimeError("Video recording completed without any frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(output))
    print(
        f"[MimicKit] Video saved: {output} "
        f"({recording.get_num_frames()} frames at {recording.get_fps()} FPS)"
    )


def _install_test_video_export(output):
    import learning.base_agent as base_agent

    upstream_test_model = base_agent.BaseAgent.test_model

    def test_model_with_video(self, num_episodes):
        result = upstream_test_model(self, num_episodes)
        _export_test_video(self._env, output)
        return result

    base_agent.BaseAgent.test_model = test_model_with_video
    print(f"[MimicKit] Headless video export enabled: {output}")


def _needs_bounded_replay_insert(argv, checkout):
    mode = _effective_arg_value(argv, "mode", checkout) or "train"
    if mode != "train":
        return False

    num_envs = int(_effective_arg_value(argv, "num_envs", checkout) or 1)
    agent_config = _effective_arg_value(argv, "agent_config", checkout)
    if agent_config is None:
        return False
    config_path = Path(agent_config).expanduser()
    if not config_path.is_absolute():
        config_path = checkout / config_path

    steps_per_iter = _yaml_int(config_path, "steps_per_iter")
    disc_buffer_size = _yaml_int(config_path, "disc_buffer_size")
    rollout_samples = num_envs * steps_per_iter
    print(
        "[MimicKit] Replay preflight: "
        f"{num_envs} envs x {steps_per_iter} steps = {rollout_samples} samples; "
        f"disc_buffer_size = {disc_buffer_size}"
    )
    return rollout_samples > disc_buffer_size


def _install_bounded_replay_insert(chunk_size=4096):
    """Support arbitrary rollout sizes without growing replay GPU storage."""
    import torch
    import learning.amp_agent as amp_agent

    upstream_agent = amp_agent.AMPAgent

    class BoundedReplayAMPAgent(upstream_agent):
        @torch.no_grad()
        def _store_disc_replay_data(self):
            disc_obs = self._exp_buffer.get_data_flat("disc_obs")
            num_rollout_samples = disc_obs.shape[0]
            indices = torch.randperm(num_rollout_samples, device=self._device, dtype=torch.long)
            if self._disc_buffer.is_full():
                num_samples = min(num_rollout_samples, self._disc_replay_samples)
            else:
                num_samples = num_rollout_samples

            # Upstream ExperienceBuffer.push() supports at most one buffer-length
            # insertion. Multiple bounded pushes preserve its circular semantics.
            push_size = min(chunk_size, self._disc_buffer.get_capacity())
            for start in range(0, num_samples, push_size):
                end = min(start + push_size, num_samples)
                replay_disc_obs = disc_obs[indices[start:end]]
                self._disc_buffer.push({"disc_obs": replay_disc_obs.unsqueeze(1)})

    amp_agent.AMPAgent = BoundedReplayAMPAgent
    print(
        "[MimicKit] Enabled bounded replay insertion for this environment count: "
        f"bounded insert chunks <= {chunk_size}; replay capacity unchanged"
    )


def main():
    _normalize_out_dir(sys.argv)
    video_output = _prepare_video_output(sys.argv)
    checkout = ROOT / "third_party/MimicKit"
    _normalize_max_iterations(sys.argv, checkout)
    needs_bounded_insert = _needs_bounded_replay_insert(sys.argv, checkout)
    enable_mimickit()
    if needs_bounded_insert:
        _install_bounded_replay_insert()
    if video_output is not None:
        _install_test_video_export(video_output)
    motion = checkout / "data/motions/smpl/smpl_walk.pkl"
    if not motion.is_file():
        raise FileNotFoundError(
            f"Missing official MimicKit motion: {motion}\n"
            "Download MimicKit_Data.zip from the upstream README and run "
            "scripts/mimickit_amp/install_official_smpl_baseline.py first."
        )
    official = checkout / "mimickit/run.py"
    os.chdir(checkout)
    sys.argv[0] = str(official)
    runpy.run_path(str(official), run_name="__main__")


if __name__ == "__main__":
    main()
