"""Four-camera grid composition keeps all synchronized views visible."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playback_video", ROOT / "exts/socnav_sim/socnav_sim/playback_video.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_four_view_layout_and_labels_do_not_mutate_camera_frames():
    frames = [np.full((40, 64, 3), value, dtype=np.uint8) for value in (30, 60, 90, 120)]
    result = module.four_view_frame(frames, [f"Env {i}" for i in range(4)])
    assert result.shape == (80, 128, 3)
    assert [int(result[y, x, 0]) for y, x in ((35, 10), (35, 75), (75, 10), (75, 75))] == [30, 60, 90, 120]
    assert [int(frame[0, 0, 0]) for frame in frames] == [30, 60, 90, 120]


def test_grid_rejects_missing_camera():
    with pytest.raises(ValueError, match="four"):
        module.four_view_frame([np.zeros((40, 64, 3), dtype=np.uint8)], ["Env 0"])
