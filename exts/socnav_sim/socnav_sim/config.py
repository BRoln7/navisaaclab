"""Shared task configuration, resolved independently of the working directory."""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_training_device():
    """Device selected by the launcher before task modules are imported."""
    return os.environ.get("SOCNAV_DEVICE", "cuda:0")


def load_task_config():
    path = Path(os.environ.get("SOCNAV_CONFIG", PROJECT_ROOT / "data/cfg/task.yaml"))
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not {"env", "training"}.issubset(config):
        raise ValueError(f"{path}: expected env and training configuration sections")
    for variable, key in (("SOCNAV_NUM_ENVS", "num_envs"), ("SOCNAV_TASK", "task")):
        if variable in os.environ:
            value = os.environ[variable]
            config["env"][key] = int(value) if key == "num_envs" else value
    if os.environ.get("SOCNAV_PLAY") == "1":
        config["training"]["train"] = False
        config["env"]["debug_vis"] = os.environ.get("SOCNAV_DEBUG_VIS", "1") == "1"
        if "SOCNAV_MOTION_FILE" in os.environ:
            config["training"]["data_file"] = os.environ["SOCNAV_MOTION_FILE"]
    return config
