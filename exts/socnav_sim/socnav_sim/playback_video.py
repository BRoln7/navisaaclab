"""Stream four synchronized playback views to a labeled video."""

import numpy as np
from PIL import Image, ImageDraw


def four_view_frame(frames, labels):
    if len(frames) != 4 or len(labels) != 4:
        raise ValueError("four frames and labels are required")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("all camera frames must have the same shape")
    height, width, channels = frames[0].shape
    if channels != 3:
        raise ValueError("camera frames must be RGB")
    grid = np.concatenate((np.concatenate(frames[:2], axis=1),
                           np.concatenate(frames[2:], axis=1)), axis=0)
    image = Image.fromarray(grid.astype(np.uint8, copy=False))
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(labels):
        x, y = (index % 2) * width, (index // 2) * height
        draw.rectangle((x, y, x + width - 1, y + 25), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
    return np.asarray(image)


class GridVideoWriter:
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self._writer = None

    def write(self, frames, labels):
        frame = four_view_frame(frames, labels)
        if self._writer is None:
            import imageio_ffmpeg
            self._writer = imageio_ffmpeg.write_frames(
                str(self.path), (frame.shape[1], frame.shape[0]), fps=self.fps,
                codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
                output_params=["-movflags", "+faststart", "-profile:v", "baseline", "-crf", "22"],
            )
            self._writer.send(None)
        self._writer.send(frame)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
