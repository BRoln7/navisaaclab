"""Stream four synchronized tracking cameras to a browser-playable WebM file."""

from pathlib import Path
import warnings
import numpy as np
from socnav_sim.playback_video import four_view_frame


class WebMGridWriter:
    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self._writer = None

    def write(self, frames, labels):
        frame = four_view_frame(frames, labels)
        if self._writer is None:
            import imageio_ffmpeg
            self._writer = imageio_ffmpeg.write_frames(
                str(self.path), (frame.shape[1], frame.shape[0]), fps=self.fps,
                codec="libvpx-vp9", pix_fmt_out="yuv420p", macro_block_size=2,
                output_params=["-deadline", "realtime", "-cpu-used", "6", "-crf", "32",
                               "-b:v", "0", "-row-mt", "1", "-threads", "4"],
            )
            self._writer.send(None)
        self._writer.send(frame)
        self.frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class FourEnvVideo:
    """Offscreen USD cameras; no viewport or upstream in-memory video recorder."""

    def __init__(self, env, path, resolution=(640, 360)):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, UsdLux
        if env.get_num_envs() != 4:
            raise ValueError("Four-environment video requires --num_envs 4")
        self._env = env
        self._sim = env._engine.get_sim()
        self._Gf = Gf
        self._resolution = resolution
        self._cameras = []
        self._products = []
        self._annotators = []
        self.writer = WebMGridWriter(path, fps=round(1/env._engine.get_timestep()))
        stage = env._engine._stage
        if not env.get_visualize():
            # Native engine skips lights when visualize=False; add render-only light.
            light = UsdLux.DomeLight.Define(stage, "/World/MimicKitVideo/Light")
            light.CreateIntensityAttr(1800.)
            light.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
        for index in range(4):
            camera = UsdGeom.Camera.Define(stage, f"/World/MimicKitVideo/Camera{index}")
            camera.CreateFocalLengthAttr(24.)
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateVerticalApertureAttr(20.955*resolution[1]/resolution[0])
            camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 1000.))
            self._cameras.append(UsdGeom.Xformable(camera).AddTransformOp())
            product = rep.create.render_product(str(camera.GetPath()), resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([product])
            self._products.append(product)
            self._annotators.append(annotator)
        self._position_cameras()
        self._update_trajectory_markers()
        for _ in range(8):
            self._sim.render()

    def _position_cameras(self):
        engine = self._env._engine
        roots = engine.get_root_pos(self._env._get_char_id()).cpu().numpy().copy()
        roots[:, :2] += engine._env_offsets.cpu().numpy()
        for index, transform in enumerate(self._cameras):
            target = roots[index].copy()
            target[2] = .85
            eye = target + np.array([3.2, -3.2, 2.0])
            matrix = self._Gf.Matrix4d().SetLookAt(
                self._Gf.Vec3d(*map(float, eye)), self._Gf.Vec3d(*map(float, target)), self._Gf.Vec3d(0., 0., 1.))
            transform.Set(matrix.GetInverse())

    def capture(self, labels):
        self._position_cameras()
        self._update_trajectory_markers()
        # Two renders avoid reading a prior camera pose's texture; physics is frozen.
        self._sim.render()
        self._sim.render()
        frames = []
        for annotator in self._annotators:
            rgb = np.asarray(annotator.get_data())
            if rgb.ndim != 3 or rgb.shape[:2] != (self._resolution[1], self._resolution[0]):
                raise RuntimeError(f"Video camera returned an invalid RGB frame: {rgb.shape}")
            frames.append(rgb[..., :3].copy())
        self.writer.write(frames, labels)

    def _update_trajectory_markers(self):
        if self._env._tracking:
            self._env.update_trajectory_markers(self._env._env_ids[:4])

    def close(self):
        try:
            self.writer.close()
        finally:
            resources = list(zip(self._annotators, self._products))
            self._annotators.clear()
            self._products.clear()
            for annotator, product in resources:
                try:
                    # Replicator 1.12 converts a single HydraTexture to its
                    # path, but not a list of textures. Detach all of this
                    # annotator's products instead of passing such a list.
                    annotator.detach()
                except Exception as exc:
                    # Replicator can already have removed shared graph nodes.
                    # Teardown follows encoding; report it without discarding
                    # the completed playback or skipping the other cameras.
                    warnings.warn(f"Playback annotator cleanup: {exc}", RuntimeWarning, stacklevel=2)
                try:
                    product.destroy()
                except Exception as exc:
                    warnings.warn(f"Playback render-product cleanup: {exc}", RuntimeWarning, stacklevel=2)
