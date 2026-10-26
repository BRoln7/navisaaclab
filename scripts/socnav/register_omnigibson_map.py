"""Register an OmniGibson traversability image to an Isaac occupancy map.

Collected IsaacNav scenes bundle cropped OmniGibson maps without coordinate
metadata.  Their white pixels encode traversable floor.  Isaac's occupancy
slice supplies the missing orientation, scale, translation, and ROS origin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traversability_png", type=Path)
    parser.add_argument("isaac_map_png", type=Path)
    parser.add_argument("isaac_map_yaml", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--name", default="nav_map")
    parser.add_argument(
        "--source_resolution",
        type=float,
        default=0.08,
        help="Resolution of the bundled OmniGibson traversability images in metres per pixel",
    )
    args = parser.parse_args()
    if args.source_resolution <= 0:
        parser.error("source_resolution must be positive")
    return args


def _place(mask: np.ndarray, shape: tuple[int, int], offset_x: int, offset_y: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    y0, x0 = max(0, offset_y), max(0, offset_x)
    y1, x1 = min(shape[0], offset_y + mask.shape[0]), min(shape[1], offset_x + mask.shape[1])
    if y0 >= y1 or x0 >= x1:
        return canvas
    canvas[y0:y1, x0:x1] = mask[y0 - offset_y : y1 - offset_y, x0 - offset_x : x1 - offset_x]
    return canvas


def main() -> None:
    args = _arguments()
    source = np.asarray(Image.open(args.traversability_png).convert("L")) == 255
    isaac_image = np.asarray(Image.open(args.isaac_map_png).convert("L"))
    # Both inputs use white for traversable/free space and black for blocked or
    # unknown space.  Register the two free-space footprints directly.
    target = isaac_image >= 128

    with args.isaac_map_yaml.open("r", encoding="utf-8") as stream:
        isaac_metadata = yaml.safe_load(stream)
    target_resolution = float(isaac_metadata["resolution"])
    scale = args.source_resolution / target_resolution

    best = None
    for rotations in range(4):
        rotated = np.rot90(source, rotations)
        for reflected in (False, True):
            oriented = np.fliplr(rotated) if reflected else rotated
            size = (
                max(1, round(oriented.shape[1] * scale)),
                max(1, round(oriented.shape[0] * scale)),
            )
            resized = np.asarray(
                Image.fromarray(oriented).resize(size, resample=Image.Resampling.NEAREST), dtype=bool
            )
            # Never let the optimizer improve its score by clipping most of a
            # candidate outside the Isaac canvas.  The source maps use a known
            # 0.08 m resolution, so a correctly oriented image fits fully in
            # the world-space bounds produced by the USD.
            if resized.shape[0] > target.shape[0] or resized.shape[1] > target.shape[1]:
                continue
            correlation = fftconvolve(
                target.astype(np.float32), resized[::-1, ::-1].astype(np.float32), mode="valid"
            )
            offset_y, offset_x = np.unravel_index(np.argmax(correlation), correlation.shape)
            offset_x, offset_y = int(offset_x), int(offset_y)
            registered = _place(resized, target.shape, offset_x, offset_y)
            intersection = int(np.count_nonzero(registered & target))
            union = int(np.count_nonzero(registered | target))
            iou = intersection / union if union else 0.0
            candidate = (iou, rotations, reflected, float(scale), offset_x, offset_y, registered)
            if best is None or iou > best[0]:
                best = candidate

    assert best is not None
    iou, rotations, reflected, scale, offset_x, offset_y, registered = best
    labels, component_count = ndimage.label(registered)
    component_sizes = np.bincount(labels.ravel())[1:]
    largest_component = int(component_sizes.max(initial=0))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.name}.png"
    yaml_path = output_dir / f"{args.name}.yaml"
    report_path = output_dir / f"{args.name}.json"
    Image.fromarray(np.where(registered, 255, 0).astype(np.uint8)).save(image_path)
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {image_path.name}",
                f"resolution: {float(isaac_metadata['resolution']):.9g}",
                f"origin: {isaac_metadata['origin']}",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "traversability_source": str(args.traversability_png.resolve()),
        "isaac_registration_map": str(args.isaac_map_png.resolve()),
        "resolution_m": float(isaac_metadata["resolution"]),
        "source_resolution_m": args.source_resolution,
        "origin": isaac_metadata["origin"],
        "dimensions_px": [int(registered.shape[1]), int(registered.shape[0])],
        "registration": {
            "iou": iou,
            "rotations_ccw_90": rotations,
            "reflected_left_right": reflected,
            "scale": scale,
            "offset_xy_px": [offset_x, offset_y],
        },
        "free_cells": int(np.count_nonzero(registered)),
        "free_fraction": float(registered.mean()),
        "connected_components": int(component_count),
        "largest_component_cells": largest_component,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[map-registration] IoU={iou:.3f}; rotation={rotations * 90}deg; reflected={reflected}; "
        f"scale={scale:.3f}; offset=({offset_x}, {offset_y}) px"
    )
    print(
        f"[map-registration] free={registered.mean():.1%}; components={component_count}; "
        f"largest={largest_component} cells"
    )
    print(f"[map-registration] image: {image_path}")
    print(f"[map-registration] yaml: {yaml_path}")
    print(f"[map-registration] report: {report_path}")


if __name__ == "__main__":
    main()
