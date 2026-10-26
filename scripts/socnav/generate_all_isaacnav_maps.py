"""Generate and validate navigation maps for every collected IsaacNav scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "data/usd/assets/scenes/IsaacNav scenes2"
GENERATOR = ROOT / "scripts/socnav/generate_occupancy_map.py"
REGISTER = ROOT / "scripts/socnav/register_omnigibson_map.py"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cell_size", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--registration_only",
        action="store_true",
        help="Reuse existing Isaac occupancy slices and only redo map registration",
    )
    parser.add_argument("--only", type=int, nargs="*", help="Optional scene numbers")
    return parser.parse_args()


def _run(command: list[str], log_path: Path, timeout: int = 900) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}\n{tail}")


def _validate(image_path: Path, yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    pixels = np.asarray(Image.open(image_path).convert("L")).astype(np.uint8)
    # Match map_manager.load_map(): it flips image Y and marks dark cells occupied.
    occupancy = (np.flipud(pixels) < float(metadata.get("occupied_thresh", 0.65)) * 255).astype(np.uint8)
    labels, count = ndimage.label(occupancy == 0)
    sizes = np.bincount(labels.ravel())
    if sizes.size <= 1:
        raise RuntimeError("Map contains no free connected component")
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    free_yx = np.argwhere(labels == largest_label)
    if free_yx.shape[0] < 2:
        raise RuntimeError("Largest free connected component is too small")
    a = free_yx[0]
    b = free_yx[np.argmax(np.sum((free_yx - a) ** 2, axis=1))]
    a = free_yx[np.argmax(np.sum((free_yx - b) ** 2, axis=1))]

    sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
    from socnav_sim.tasks.socnav.nav_utils.planning import astar_path

    path = astar_path(occupancy, a[::-1], b[::-1])
    if path is None or len(path) < 2:
        raise RuntimeError("A* could not connect two points in the largest free component")
    return {
        "dimensions_px": [int(pixels.shape[1]), int(pixels.shape[0])],
        "resolution_m": float(metadata["resolution"]),
        "origin": metadata["origin"],
        "free_fraction": float(np.mean(occupancy == 0)),
        "connected_components": int(count),
        "largest_component_cells": int(sizes[largest_label]),
        "astar_validation_path_cells": int(len(path)),
    }


def main() -> None:
    args = _args()
    selected = set(args.only or [])
    scene_usds = sorted(
        SCENES.glob("Collected_scene_*/scene_*.usd"),
        key=lambda path: int(path.stem.removeprefix("scene_")),
    )
    if selected:
        scene_usds = [path for path in scene_usds if int(path.stem.removeprefix("scene_")) in selected]
    if not scene_usds:
        raise RuntimeError("No matching IsaacNav scene USD files found")

    summary = {"completed": {}, "skipped": [], "failed": {}}
    started = time.monotonic()
    for index, usd_path in enumerate(scene_usds, start=1):
        scene_id = int(usd_path.stem.removeprefix("scene_"))
        folder = usd_path.parent
        stem = f"scene_{scene_id}"
        nav_image = folder / f"{stem}_nav_map.png"
        nav_yaml = folder / f"{stem}_nav_map.yaml"
        source_map = SCENES / "Omingibson occupancy maps" / f"scene_map_{scene_id}.png"
        print(f"[batch-map] [{index}/{len(scene_usds)}] scene {scene_id}", flush=True)
        if nav_image.is_file() and nav_yaml.is_file() and not args.force:
            try:
                summary["completed"][str(scene_id)] = _validate(nav_image, nav_yaml)
                summary["skipped"].append(scene_id)
                print("[batch-map] existing map validated; skipped generation", flush=True)
                continue
            except Exception:
                print("[batch-map] existing map failed validation; regenerating", flush=True)
        try:
            if not source_map.is_file():
                raise FileNotFoundError(f"Missing OmniGibson map: {source_map}")
            log_dir = folder / "map_generation_logs"
            slice_stem = f"{stem}_isaac_slice"
            slice_image = folder / f"{slice_stem}.png"
            slice_yaml = folder / f"{slice_stem}.yaml"
            if args.registration_only:
                if not slice_image.is_file() or not slice_yaml.is_file():
                    raise FileNotFoundError(f"Missing existing Isaac occupancy slice for scene {scene_id}")
            else:
                _run(
                    [
                        str(ROOT / "isaaclab.sh"),
                        "-p",
                        str(GENERATOR),
                        str(usd_path),
                        "--output_dir",
                        str(folder),
                        "--name",
                        slice_stem,
                        "--cell_size",
                        str(args.cell_size),
                        "--meters_per_unit",
                        "1.0",
                        "--device",
                        args.device,
                        "--headless",
                    ],
                    log_dir / "isaac_occupancy.log",
                )
            _run(
                [
                    sys.executable,
                    str(REGISTER),
                    str(source_map),
                    str(slice_image),
                    str(slice_yaml),
                    "--output_dir",
                    str(folder),
                    "--name",
                    f"{stem}_nav_map",
                ],
                log_dir / "registration.log",
            )
            validation = _validate(nav_image, nav_yaml)
            summary["completed"][str(scene_id)] = validation
            print(
                f"[batch-map] complete: {validation['dimensions_px'][0]}x{validation['dimensions_px'][1]}, "
                f"free={validation['free_fraction']:.1%}, A*={validation['astar_validation_path_cells']} cells",
                flush=True,
            )
        except Exception as exc:
            summary["failed"][str(scene_id)] = str(exc)
            print(f"[batch-map] FAILED: {exc}", flush=True)

    summary["elapsed_seconds"] = time.monotonic() - started
    summary_path = SCENES / "map_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[batch-map] finished: {len(summary['completed'])} valid, {len(summary['failed'])} failed; "
        f"summary={summary_path}",
        flush=True,
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
