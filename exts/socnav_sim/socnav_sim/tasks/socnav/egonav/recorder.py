"""Synchronized RGB-D, robot-state, and WebM writer for EgoNav."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image


def _rotate_vector(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a three-vector without introducing a SciPy dependency."""
    w = quaternion_wxyz[0]
    xyz = quaternion_wxyz[1:]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + w * cross + np.cross(xyz, cross)


class EgoNavCamera:
    """External Replicator camera that follows env_0's robot."""

    def __init__(self, env, resolution: tuple[int, int] = (640, 360)):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        self._sim = env.sim
        self._robot = env.scene["robot"]
        self._Gf = Gf
        self.resolution = resolution
        self.focal_length = 10.5
        self.horizontal_aperture = 20.955
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, "/World/EgoNavCapture/RobotCamera")
        camera.CreateFocalLengthAttr(self.focal_length)
        camera.CreateHorizontalApertureAttr(self.horizontal_aperture)
        camera.CreateVerticalApertureAttr(
            self.horizontal_aperture * resolution[1] / resolution[0]
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 30.0))
        self._transform = UsdGeom.Xformable(camera).AddTransformOp()
        self._render_product = rep.create.render_product(str(camera.GetPath()), resolution)
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
        self._rgb.attach([self._render_product])
        self._depth.attach([self._render_product])
        self.position = np.zeros(3, dtype=np.float64)
        self.target = np.zeros(3, dtype=np.float64)
        self._position_camera()
        for _ in range(4):
            self._sim.render()

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        width, height = self.resolution
        focal_pixels = width * self.focal_length / self.horizontal_aperture
        return np.array(
            [[focal_pixels, 0.0, width / 2.0], [0.0, focal_pixels, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _position_camera(self) -> None:
        root = self._robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        quaternion = self._robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        self.position = root + _rotate_vector(quaternion, np.array([0.25, 0.0, 0.45]))
        forward = _rotate_vector(quaternion, np.array([1.0, 0.0, 0.0]))
        up = _rotate_vector(quaternion, np.array([0.0, 0.0, 1.0]))
        self.target = self.position + forward
        matrix = self._Gf.Matrix4d().SetLookAt(
            self._Gf.Vec3d(*map(float, self.position)),
            self._Gf.Vec3d(*map(float, self.target)),
            self._Gf.Vec3d(*map(float, up)),
        )
        self._transform.Set(matrix.GetInverse())

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        self._position_camera()
        self._sim.render()
        rgb = np.asarray(self._rgb.get_data())
        depth = np.asarray(self._depth.get_data())
        expected = (self.resolution[1], self.resolution[0])
        if rgb.ndim != 3 or rgb.shape[:2] != expected:
            raise RuntimeError(f"EgoNav RGB annotator returned {rgb.shape}, expected {expected}")
        if depth.shape[:2] != expected:
            raise RuntimeError(f"EgoNav depth annotator returned {depth.shape}, expected {expected}")
        return rgb[..., :3].copy(), depth.copy()

    def close(self) -> None:
        for annotator in (self._rgb, self._depth):
            try:
                annotator.detach()
            except Exception as exc:
                warnings.warn(f"EgoNav annotator cleanup: {exc}", RuntimeWarning, stacklevel=2)
        try:
            self._render_product.destroy()
        except Exception as exc:
            warnings.warn(f"EgoNav render-product cleanup: {exc}", RuntimeWarning, stacklevel=2)


def depth_metres_to_uint16(depth: np.ndarray) -> np.ndarray:
    """Encode metric depth as millimetres; zero denotes invalid or clipped pixels."""
    metric = np.asarray(depth, dtype=np.float32)
    if metric.ndim == 3 and metric.shape[-1] == 1:
        metric = metric[..., 0]
    if metric.ndim != 2:
        raise ValueError(f"Expected an HxW depth image, got {metric.shape}")
    valid = np.isfinite(metric) & (metric > 0.0)
    encoded = np.zeros(metric.shape, dtype=np.uint16)
    encoded[valid] = np.clip(np.rint(metric[valid] * 1000.0), 1, 65535).astype(np.uint16)
    return encoded


class EgoNavDatasetWriter:
    """Write a browser-playable preview and lossless synchronized sensor data."""

    def __init__(self, output_dir: str | Path, fps: float, *, save_frames: bool = True):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames
        if save_frames:
            self.rgb_dir.mkdir(exist_ok=True)
            self.depth_dir.mkdir(exist_ok=True)
        self.video_path = self.output_dir / "robot_first_person.webm"
        self.metadata_path = self.output_dir / "frames.jsonl"
        self._metadata = self.metadata_path.open("w", encoding="utf-8")
        self._writer = None
        self._fps = float(fps)
        self.frames = 0

    def write_calibration(self, intrinsic_matrix: np.ndarray, image_shape: tuple[int, int]) -> None:
        height, width = image_shape
        payload = {
            "camera_model": "pinhole",
            "width": int(width),
            "height": int(height),
            "intrinsic_matrix": np.asarray(intrinsic_matrix, dtype=float).tolist(),
            "rgb_encoding": "rgb8",
            "depth_encoding": "16UC1",
            "depth_scale_metres_per_unit": 0.001,
            "invalid_depth_value": 0,
            "frame_rate_hz": self._fps,
            "camera_pose_convention": "world frame; eye and target positions; +Z up",
        }
        (self.output_dir / "calibration.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _write_video(self, rgb: np.ndarray) -> None:
        if self._writer is None:
            import imageio_ffmpeg

            height, width = rgb.shape[:2]
            self._writer = imageio_ffmpeg.write_frames(
                str(self.video_path),
                (width, height),
                fps=self._fps,
                codec="libvpx-vp9",
                pix_fmt_out="yuv420p",
                macro_block_size=2,
                output_params=[
                    "-deadline", "realtime", "-cpu-used", "6", "-crf", "30",
                    "-b:v", "0", "-row-mt", "1", "-threads", "4",
                ],
            )
            self._writer.send(None)
        self._writer.send(np.ascontiguousarray(rgb))

    def write(self, rgb: np.ndarray, depth: np.ndarray, metadata: dict) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
        depth_mm = depth_metres_to_uint16(depth)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth_mm.shape != rgb.shape[:2]:
            raise ValueError(f"Mismatched RGB-D frame shapes: rgb={rgb.shape}, depth={depth_mm.shape}")
        stem = f"{self.frames:06d}"
        if self.save_frames:
            Image.fromarray(rgb, mode="RGB").save(self.rgb_dir / f"{stem}.png")
            Image.fromarray(depth_mm, mode="I;16").save(self.depth_dir / f"{stem}.png")
        self._write_video(rgb)
        record = {
            "frame": self.frames,
            "rgb": f"rgb/{stem}.png" if self.save_frames else None,
            "depth": f"depth/{stem}.png" if self.save_frames else None,
            **metadata,
        }
        self._metadata.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._metadata.flush()
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if not self._metadata.closed:
            self._metadata.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()
