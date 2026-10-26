"""CPU encode/decode validation of the browser-compatible playback writer."""

from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import imageio_ffmpeg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.video import FourEnvVideo, WebMGridWriter


def test_webm_grid_round_trip_and_repeated_close(tmp_path):
    output = tmp_path / "preview.webm"
    writer = WebMGridWriter(output, fps=30)
    for step in range(12):
        frames = [np.full((40, 64, 3), 30+40*i+step, dtype=np.uint8) for i in range(4)]
        writer.write(frames, [f"Env {i}" for i in range(4)])
    writer.close()
    writer.close()
    assert writer.frames == 12 and output.stat().st_size > 0
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    metadata = next(reader)
    assert metadata["size"] == (128, 80)
    assert metadata["fps"] == 30
    decoded = list(reader)
    assert len(decoded) == 12
    last = np.frombuffer(decoded[-1], dtype=np.uint8).reshape(80, 128, 3)
    # All four independent panels survive VP9 compression in the intended order.
    for i, (y, x) in enumerate(((35, 30), (35, 94), (75, 30), (75, 94))):
        assert abs(int(last[y, x].mean())-(30+40*i+11)) < 5


def test_camera_tracking_applies_environment_offsets_and_python_float_coordinates():
    # Gf.Vec3d accepts Python doubles, but not numpy.float32 coordinates.
    def vector(*values):
        assert all(type(value) is float for value in values)
        return np.array(values)
    class Matrix:
        def SetLookAt(self, eye, target, up):
            self.eye, self.target = eye, target
            return self
        def GetInverse(self):
            return self
    def tensor(array):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: array))
    roots = np.array([[i, 2*i, 1.] for i in range(4)], dtype=np.float32)
    offsets = np.array([[5*i, -5*i] for i in range(4)], dtype=np.float32)
    engine = SimpleNamespace(get_root_pos=lambda _: tensor(roots), _env_offsets=tensor(offsets))
    recorder = object.__new__(FourEnvVideo)
    recorder._env = SimpleNamespace(_engine=engine, _get_char_id=lambda: 0)
    recorder._Gf = SimpleNamespace(Vec3d=vector, Matrix4d=Matrix)
    poses = []
    recorder._cameras = [SimpleNamespace(Set=lambda value: poses.append(value)) for _ in range(4)]
    recorder._position_cameras()
    for index, pose in enumerate(poses):
        np.testing.assert_allclose(pose.target[:2], roots[index, :2]+offsets[index])
        np.testing.assert_allclose(pose.target[2], .85)
        np.testing.assert_allclose(pose.eye-pose.target, [3.2, -3.2, 2.])


def test_camera_cleanup_detaches_without_texture_list_and_is_repeatable():
    recorder = object.__new__(FourEnvVideo)
    calls = []
    recorder.writer = SimpleNamespace(close=lambda: None)
    # Replicator 1.12's detach(list_of_HydraTexture) fails; no-argument detach
    # correctly uses the annotator's internally stored render-product paths.
    recorder._annotators = [SimpleNamespace(detach=lambda: calls.append("detach")) for _ in range(4)]
    recorder._products = [SimpleNamespace(destroy=lambda: calls.append("destroy")) for _ in range(4)]
    recorder.close()
    recorder.close()
    assert calls == ["detach", "destroy"]*4


@pytest.mark.parametrize("failing_stage", ["detach", "destroy"])
def test_invalid_render_nodes_do_not_skip_remaining_cleanup(failing_stage):
    recorder = object.__new__(FourEnvVideo)
    calls = []
    recorder.writer = SimpleNamespace(close=lambda: None)
    def cleanup(stage):
        calls.append(stage)
        if stage == failing_stage:
            raise TypeError("Invalid NodeObj object")
    recorder._annotators = [SimpleNamespace(detach=lambda: cleanup("detach")) for _ in range(4)]
    recorder._products = [SimpleNamespace(destroy=lambda: cleanup("destroy")) for _ in range(4)]
    with pytest.warns(RuntimeWarning, match="Invalid NodeObj"):
        recorder.close()
    recorder.close()
    assert calls == ["detach", "destroy"]*4
