"""Resolve independent MimicKit YAML configurations without importing the simulator."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import yaml


def latest_checkpoint(log_root):
    """Return the most recently written top-level training checkpoint."""
    root = Path(log_root)
    candidates = []
    for pattern in ("*/model.pt", "*/model_final.pt"):
        for checkpoint in root.glob(pattern):
            manifest = checkpoint.parent / "params" / "manifest.json"
            if manifest.is_file():
                candidates.append(checkpoint)
    if not candidates:
        raise FileNotFoundError(f"No MimicKit training checkpoint under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
from .bootstrap import PROJECT_ROOT


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return result


def load(path, seen=()):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic YAML base: {path}")
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    return merge(load(base, seen+(path,)), cfg) if base else cfg


def saved_config(checkpoint_params, name):
    """Resolve configs saved by this integration or native MimicKit."""
    folder = Path(checkpoint_params)
    for filename in (f"{name}.yaml", f"{name}_config.yaml"):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"No saved {name} config in {folder}")


def run_configurations(
    paths,
    checkpoint_params=None,
    amp_only=False,
    finetune_tracking=False,
    motion_file=None,
    power_coefficient=None,
):
    """Playback/restarts use saved settings; explicit task transfer uses current settings."""
    if amp_only and finetune_tracking:
        raise ValueError("amp_only and finetune_tracking are mutually exclusive")
    if finetune_tracking and checkpoint_params is None:
        raise ValueError("finetune_tracking requires a checkpoint with saved params")
    if power_coefficient is not None:
        power_coefficient = float(power_coefficient)
        if not math.isfinite(power_coefficient) or power_coefficient < 0:
            raise ValueError("power_coefficient must be finite and non-negative")
    selected = ([saved_config(checkpoint_params, name) for name in ("env", "agent", "engine")]
                if checkpoint_params is not None and not finetune_tracking else paths)
    env, agent, engine = (load(path) for path in selected)
    if finetune_tracking:
        previous = load(saved_config(checkpoint_params, "env"))
        # A global/local observation switch changes meaning without changing
        # tensor shape. Preserve the coordinate convention of loaded weights.
        env["global_obs"] = previous["global_obs"]
    if motion_file is not None:
        env["motion_file"] = str(motion_file)
    if amp_only:
        env["tracking"]["enabled"] = False
        env["tracking"]["power_coefficient"] = 0. if power_coefficient is None else power_coefficient
        # Reuse the environment-reward channel for an explicitly requested
        # energy regularizer.  With coefficient zero, native pure AMP remains
        # exactly task-free and receives only discriminator reward.
        agent["task_reward_weight"] = 1. if env["tracking"]["power_coefficient"] > 0 else 0.
        agent["disc_reward_weight"] = 1.
    elif finetune_tracking or env["tracking"]["enabled"]:
        # The default is now upstream pure AMP. Explicit tracking transfer or
        # env_tracking.yaml enables the additional task, using one shared agent.
        env["tracking"]["enabled"] = True
        if power_coefficient is not None:
            env["tracking"]["power_coefficient"] = power_coefficient
        agent["task_reward_weight"] = float(env["tracking"]["reward_weight"])
        agent["disc_reward_weight"] = float(env["tracking"]["style_reward_weight"])
    else:
        if power_coefficient is not None and power_coefficient > 0:
            raise ValueError("power_coefficient with tracking disabled requires amp_only")
        env["tracking"]["power_coefficient"] = 0.
        agent["task_reward_weight"] = 0.
        agent["disc_reward_weight"] = 1.
    if float(agent.get("symmetry_weight", 0.)) <= 0:
        raise ValueError("SMPL AMP requires a positive symmetry_weight in both locomotion and tracking stages")
    if finetune_tracking and (not env["tracking"]["enabled"] or agent["task_reward_weight"] <= 0):
        raise ValueError("Tracking fine-tuning requires an enabled trajectory task with positive task reward weight")
    return env, agent, engine


def resolve_agent_runtime(agent, num_envs):
    """Keep native logical batch ratios; bound memory through microbatches only.

    Migrate saved configs from the earlier capped-batch policy without changing
    their networks/observation settings. New runs inherit the upstream ratios.
    """
    if not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    agent = deepcopy(agent)
    rollout_samples = num_envs * int(agent["steps_per_iter"])
    if rollout_samples <= 0:
        raise ValueError("steps_per_iter must be positive")
    legacy_policy = agent.pop("batch_size_scaling", {})
    agent["disc_buffer_size"] = int(agent.pop("disc_buffer_size_min", agent["disc_buffer_size"]))
    if agent["disc_buffer_size"] <= 0:
        raise ValueError("disc_buffer_size must be positive")
    microbatches = agent.setdefault("optimizer_microbatch_size", {})
    for name in ("actor", "critic", "disc"):
        key = f"{name}_batch_size"
        policy = legacy_policy.get(name)
        ratio = float(policy["per_env"] if policy else agent[key])
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"{key} ratio must be positive and finite")
        samples = math.ceil(ratio * num_envs)
        if samples > rollout_samples:
            raise ValueError(f"{key} exceeds rollout capacity")
        if name == "disc" and samples > agent["disc_buffer_size"]:
            raise ValueError("Logical discriminator batch exceeds replay capacity; increase disc_buffer_size")
        if policy:
            microbatches.setdefault(name, int(policy["max_samples"]))
        agent[key] = ratio
    for name, size in microbatches.items():
        if name not in ("actor", "critic", "disc") or type(size) is not int or size <= 0:
            raise ValueError("optimizer_microbatch_size requires positive integer actor/critic/disc sizes")
    return agent


def resolve_paths(env, engine):
    for cfg, keys in ((env, ("char_file", "motion_file")), (engine, ("smpl_usd",))):
        for key in keys:
            path = Path(cfg[key]).expanduser()
            path = path if path.is_absolute() else PROJECT_ROOT / path
            if not path.is_file():
                hint = "prepare_motions.py" if key == "motion_file" else "install_assets.py with the official MimicKit data ZIP"
                raise FileNotFoundError(f"Missing {key}: {path}. Run scripts/mimickit_amp/{hint} first.")
            cfg[key] = str(path.resolve())
    if abs(1./engine["control_freq"] - 1/30) > 1e-8:
        raise ValueError("Reconvert motions and recheck AMP history before changing the 30 Hz control timestep")
    skeleton, usd = Path(env["char_file"]), Path(engine["smpl_usd"])
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    asset_manifest = skeleton.parent / "asset-manifest.json"
    if asset_manifest.is_file():
        hashes = json.loads(asset_manifest.read_text())["sha256"]
        if usd.parent != skeleton.parent or hashes.get(skeleton.name) != digest(skeleton) or hashes.get(usd.name) != digest(usd):
            raise ValueError("SMPL USD/XML do not match the installed official asset pair")
    motion_manifest = Path(env["motion_file"]).parent / "manifest.json"
    if motion_manifest.is_file():
        motion = json.loads(motion_manifest.read_text())
        if motion["skeleton_sha256"] != digest(skeleton):
            raise ValueError("Motion cache was converted for a different SMPL skeleton; reconvert into a new output directory")
        if motion["history_steps"] < env["num_disc_obs_steps"] or abs(motion["control_dt"] - 1./engine["control_freq"]) > 1e-8:
            raise ValueError("Motion cache does not match the configured AMP history interval")
    return env, engine
