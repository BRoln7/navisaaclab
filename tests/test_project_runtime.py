"""CPU checks for task discovery, launch paths and source-model compatibility."""

import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_python(code, cwd):
    env = os.environ.copy()
    env["CONDA_PREFIX"] = sys.prefix
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["bash", str(ROOT / "isaaclab.sh"), "-p", "-c", code],
        cwd=cwd, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_imports_poselib_from_another_working_directory(tmp_path):
    launch_python(
        "from socnav_sim.config import load_task_config; "
        "from poselib.core.rotation3d import quat_mul; "
        "from poselib.skeleton.skeleton3d import SkeletonTree",
        tmp_path,
    )


def test_task_registration_does_not_load_simulation_or_navigation_resources(tmp_path):
    launch_python('''
import importlib
from pathlib import Path
import sys
import gymnasium as gym
import yaml
import socnav_sim.tasks
for name in (
    'Isaac-Cadrl-Scene12', 'Isaac-Cadrl-Scene27', 'Isaac-Cadrl-Scene34',
    'Isaac-Cadrl-Scene35', 'Isaac-Drlvo', 'Isaac-Drlvo-Scene12',
    'Isaac-Drlvo-Scene35', 'Isaac-EgoNav-Scene12', 'Isaac-EgoNav-Scene35',
):
    spec = gym.spec(name)
    assert isinstance(spec.kwargs['env_cfg_entry_point'], str)
    module, filename = spec.kwargs['sb3_cfg_entry_point'].split(':')
    path = Path(importlib.import_module(module).__file__).parent / filename
    assert isinstance(yaml.safe_load(path.read_text()), dict)
assert 'Isaac-Pacer' not in gym.registry
assert not any(name.startswith(('isaacsim', 'isaaclab', 'omni.', 'socnav_sim.uhc',
                               'socnav_sim.tasks.socnav.nav_utils')) for name in sys.modules)
assert not any(name.endswith('_env_cfg') for name in sys.modules)
''', tmp_path)


def test_legacy_smpl_reader_restores_apis_on_success_and_failure(tmp_path):
    module = load_module("legacy_smpl", ROOT / "exts/socnav_sim/socnav_sim/uhc/smpllib/legacy_smpl.py")
    original_numpy = {name: vars(np).get(name) for name in
                      ("bool", "int", "float", "complex", "object", "str", "unicode")}
    original_argspec = vars(inspect).get("getargspec")
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "SMPL_NEUTRAL.pkl"
    path.write_bytes(pickle.dumps({"v_template": expected}))
    np.testing.assert_array_equal(module.load_smpl_data(tmp_path).v_template, expected)
    path.write_bytes(b"invalid pickle")
    with pytest.raises(pickle.UnpicklingError):
        module.load_smpl_data(tmp_path)
    assert {name: vars(np).get(name) for name in original_numpy} == original_numpy
    assert vars(inspect).get("getargspec") is original_argspec


def test_task_config_from_another_working_directory(tmp_path, monkeypatch):
    module = load_module("configuration", ROOT / "exts/socnav_sim/socnav_sim/config.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOCNAV_CONFIG", str(ROOT / "data/cfg/cadrl_scene12.yaml"))
    monkeypatch.setenv("SOCNAV_NUM_ENVS", "7")
    monkeypatch.delenv("SOCNAV_TASK", raising=False)
    cfg = module.load_task_config()
    assert cfg["env"]["task"] == "Isaac-Cadrl-Scene12"
    assert cfg["env"]["num_envs"] == 7


def test_playback_overrides_do_not_modify_training_config(monkeypatch):
    module = load_module("configuration_play", ROOT / "exts/socnav_sim/socnav_sim/config.py")
    path = ROOT / "data/cfg/cadrl_scene12.yaml"
    original = yaml.safe_load(path.read_text())
    monkeypatch.setenv("SOCNAV_CONFIG", str(path))
    for key in ("SOCNAV_TASK", "SOCNAV_NUM_ENVS", "SOCNAV_MOTION_FILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SOCNAV_PLAY", "1")
    monkeypatch.setenv("SOCNAV_DEBUG_VIS", "1")
    playback = module.load_task_config()
    assert playback["training"]["train"] is False
    assert playback["env"]["debug_vis"] is True
    monkeypatch.delenv("SOCNAV_PLAY")
    assert module.load_task_config() == original
