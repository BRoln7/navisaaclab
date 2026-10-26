"""Generate a ROS-compatible 2-D occupancy map from a USD scene.

The script uses Isaac Sim's native ``isaacsim.asset.gen.omap`` extension, so
the resulting image is produced from the same PhysX collision geometry used by
the simulator.  The YAML origin follows the convention expected by
``socnav_sim.tasks.socnav.nav_utils.map_manager``: pixel ``(0, 0)`` after the
loader's vertical flip maps to ``origin[:2]`` in world coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="USD scene to rasterize")
    parser.add_argument("--output_dir", type=Path, help="Output folder; defaults beside the USD")
    parser.add_argument("--name", help="Output stem; defaults to the USD stem plus '_map'")
    parser.add_argument("--cell_size", type=float, default=0.05, help="Map resolution in metres per pixel")
    parser.add_argument(
        "--meters_per_unit",
        type=float,
        help="Override USD metersPerUnit (the IsaacNav/OmniGibson collection requires 1.0)",
    )
    parser.add_argument("--padding", type=float, default=0.25, help="Extra XY area around the scene in metres")
    parser.add_argument("--z_min", type=float, default=0.10, help="Lowest obstacle height relative to map origin")
    parser.add_argument("--z_max", type=float, default=1.50, help="Highest obstacle height relative to map origin")
    parser.add_argument(
        "--origin_xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Flood-fill origin in world metres; defaults to the scene XY centre",
    )
    parser.add_argument(
        "--origin_search_grid",
        type=int,
        default=5,
        help="Grid width used to find the largest free connected component when origin_xy is omitted",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
        help="Explicit world-space map bounds; defaults to the USD world bounds",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")
    args = parser.parse_args()
    if not math.isfinite(args.cell_size) or args.cell_size <= 0:
        parser.error("cell_size must be finite and positive")
    if args.meters_per_unit is not None and (
        not math.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0
    ):
        parser.error("meters_per_unit must be finite and positive")
    if not math.isfinite(args.padding) or args.padding < 0:
        parser.error("padding must be finite and non-negative")
    if not (math.isfinite(args.z_min) and math.isfinite(args.z_max) and args.z_min < args.z_max):
        parser.error("z_min and z_max must be finite and z_min < z_max")
    if args.origin_search_grid <= 0:
        parser.error("origin_search_grid must be positive")
    return args


def _wait_for_stage(app, context, max_updates: int = 20_000) -> None:
    for _ in range(max_updates):
        app.update()
        _, _, loading = context.get_stage_loading_status()
        if loading == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


def _world_xy_bounds(stage) -> tuple[float, float, float, float]:
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    # OmniGibson collection stages can have incomplete/stale extents hints on
    # their root Xform and attach CollisionAPI to non-boundable Xforms.  Actual
    # boundable geometry still has valid local extents and world transforms.
    combined = Gf.BBox3d()
    bounded_prims = 0
    # Include geometry inside native USD instances.  A regular Stage.Traverse
    # skips instance proxies, which would omit nearly the entire OG scene.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            continue
        bound = cache.ComputeWorldBound(prim)
        if not bound.GetRange().IsEmpty():
            combined = Gf.BBox3d.Combine(combined, Gf.BBox3d(bound.ComputeAlignedRange()))
            bounded_prims += 1
    if bounded_prims == 0:
        raise RuntimeError("USD stage contains no visible boundable geometry")
    extent = combined.ComputeAlignedRange()
    minimum, maximum = extent.GetMin(), extent.GetMax()
    values = (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1]))
    if not all(math.isfinite(value) for value in values) or values[0] >= values[2] or values[1] >= values[3]:
        raise RuntimeError(f"Invalid USD world bounds: {values}")
    return values


def _ensure_physics_scene(stage) -> None:
    from pxr import Sdf, UsdPhysics

    if not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))


def _write_outputs(
    image,
    output_dir: Path,
    stem: str,
    resolution: float,
    origin_xy: tuple[float, float],
    metadata: dict,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}.png"
    yaml_path = output_dir / f"{stem}.yaml"
    report_path = output_dir / f"{stem}.json"
    image.save(image_path)
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {image_path.name}",
                f"resolution: {resolution:.9g}",
                f"origin: [{origin_xy[0]:.9g}, {origin_xy[1]:.9g}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path, yaml_path, report_path


def main() -> None:
    args = _parse_args()
    usd_path = args.usd.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD scene does not exist: {usd_path}")
    output_dir = (args.output_dir or usd_path.parent).expanduser().resolve()
    stem = args.name or f"{usd_path.stem}_map"

    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import numpy as np
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from PIL import Image
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import UsdGeom, UsdPhysics

        enable_extension("isaacsim.asset.gen.omap")
        app.update()
        from isaacsim.asset.gen.omap.bindings import _omap
        from isaacsim.asset.gen.omap.utils import update_location

        print(f"[occupancy-map] opening USD: {usd_path}", flush=True)
        context = omni.usd.get_context()
        if not context.open_stage(str(usd_path)):
            raise RuntimeError(f"Isaac Sim could not open {usd_path}")
        _wait_for_stage(app, context)
        stage = context.get_stage()
        print("[occupancy-map] USD loaded; inspecting bounds and collision geometry", flush=True)
        _ensure_physics_scene(stage)
        app.update()

        authored_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        meters_per_unit = args.meters_per_unit or authored_meters_per_unit
        units_per_meter = 1.0 / meters_per_unit
        if args.bounds is None:
            min_x, min_y, max_x, max_y = _world_xy_bounds(stage)
            padding_units = args.padding * units_per_meter
            min_x -= padding_units
            min_y -= padding_units
            max_x += padding_units
            max_y += padding_units
        else:
            min_x, min_y, max_x, max_y = (value * units_per_meter for value in args.bounds)
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("Map bounds must have positive width and height")

        origin_z = 0.0
        cell_units = args.cell_size * units_per_meter
        z_min_units = args.z_min * units_per_meter
        z_max_units = args.z_max * units_per_meter

        collision_prims = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())
        if collision_prims == 0:
            raise RuntimeError(
                "The USD has no PhysX collision geometry. Add colliders first or extend this tool "
                "with a visual-mesh collision session layer."
            )

        occupancy = _omap.acquire_omap_interface()
        occupancy.set_cell_size(cell_units)
        timeline = omni.timeline.get_timeline_interface()
        print(
            f"[occupancy-map] bounds (m): "
            f"[{min_x * meters_per_unit:.3f}, {min_y * meters_per_unit:.3f}] to "
            f"[{max_x * meters_per_unit:.3f}, {max_y * meters_per_unit:.3f}]",
            flush=True,
        )
        print(
            f"[occupancy-map] generating {args.cell_size:g} m cells from {collision_prims} collision prims",
            flush=True,
        )
        if args.origin_xy is not None:
            candidates = [tuple(value * units_per_meter for value in args.origin_xy)]
        else:
            fractions = np.linspace(0.1, 0.9, args.origin_search_grid)
            centre = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
            candidates = [centre]
            candidates.extend(
                (min_x + fx * (max_x - min_x), min_y + fy * (max_y - min_y))
                for fy in fractions
                for fx in fractions
                if (fx, fy) != (0.5, 0.5)
            )

        timeline.play()
        for _ in range(3):
            app.update()
        best_raw = None
        best_origin = None
        best_free = -1
        best_is_enclosed = False
        width = height = 0
        for candidate_index, (candidate_x, candidate_y) in enumerate(candidates, start=1):
            update_location(
                occupancy,
                (candidate_x, candidate_y, origin_z),
                (min_x - candidate_x, min_y - candidate_y, z_min_units),
                (max_x - candidate_x, max_y - candidate_y, z_max_units),
            )
            occupancy.generate()
            app.update()
            dims = occupancy.get_dimensions()
            candidate_width, candidate_height = int(dims[0]), int(dims[1])
            candidate_raw = np.asarray(occupancy.get_buffer(), dtype=np.float32)
            if candidate_raw.size != candidate_width * candidate_height:
                continue
            candidate_free = int(np.count_nonzero(candidate_raw == 0.0))
            free_grid = candidate_raw.reshape(candidate_height, candidate_width) == 0.0
            touches_border = bool(
                free_grid[0].any()
                or free_grid[-1].any()
                or free_grid[:, 0].any()
                or free_grid[:, -1].any()
            )
            is_enclosed = not touches_border
            if (is_enclosed and not best_is_enclosed) or (
                is_enclosed == best_is_enclosed and candidate_free > best_free
            ):
                width, height = candidate_width, candidate_height
                best_free = candidate_free
                best_is_enclosed = is_enclosed
                best_raw = candidate_raw.copy()
                best_origin = (candidate_x, candidate_y)
        timeline.stop()
        if best_raw is None or width <= 0 or height <= 0 or best_origin is None:
            raise RuntimeError("Occupancy generator did not return a valid map for any flood origin")
        raw = best_raw
        origin_x, origin_y = best_origin
        print(
            f"[occupancy-map] selected flood origin "
            f"({origin_x * meters_per_unit:.3f}, {origin_y * meters_per_unit:.3f}) m "
            f"from {len(candidates)} candidates; free cells={best_free}, enclosed={best_is_enclosed}",
            flush=True,
        )

        # Treat unknown cells as blocked.  This is both safer for planning and
        # matches the black-background maps bundled with the IsaacNav scenes.
        pixels = np.zeros((height, width), dtype=np.uint8)
        pixels[raw.reshape(height, width) == 0.0] = 255
        # Isaac's raw buffer points from +X/-Y toward -X/+Y.  The UI's ROS
        # export rotates it by 180 degrees so the YAML origin is the world-space
        # lower-left corner used by our map loader.
        image = Image.fromarray(pixels).rotate(180, expand=True)
        map_origin_m = (
            (min_x + 0.5 * cell_units) * meters_per_unit,
            (min_y + 0.5 * cell_units) * meters_per_unit,
        )

        occupied = int(np.count_nonzero(raw == 1.0))
        free = int(np.count_nonzero(raw == 0.0))
        unknown = int(raw.size - occupied - free)
        metadata = {
            "usd": str(usd_path),
            "generator": "isaacsim.asset.gen.omap",
            "stage_meters_per_unit": meters_per_unit,
            "authored_stage_meters_per_unit": authored_meters_per_unit,
            "meters_per_unit_overridden": args.meters_per_unit is not None,
            "resolution_m": args.cell_size,
            "dimensions_px": [image.width, image.height],
            "origin_m": [map_origin_m[0], map_origin_m[1], 0.0],
            "requested_bounds_m": [
                min_x * meters_per_unit,
                min_y * meters_per_unit,
                max_x * meters_per_unit,
                max_y * meters_per_unit,
            ],
            "height_range_m": [args.z_min, args.z_max],
            "flood_origin_m": [origin_x * meters_per_unit, origin_y * meters_per_unit, origin_z],
            "collision_prims": collision_prims,
            "cells": {
                "total": int(raw.size),
                "occupied": occupied,
                "free": free,
                "unknown": unknown,
                "occupied_fraction": occupied / raw.size,
                "free_fraction": free / raw.size,
                "unknown_fraction": unknown / raw.size,
            },
        }
        image_path, yaml_path, report_path = _write_outputs(
            image, output_dir, stem, args.cell_size, map_origin_m, metadata
        )
        print(f"[occupancy-map] USD: {usd_path}")
        print(f"[occupancy-map] collision prims: {collision_prims}")
        print(
            f"[occupancy-map] {image.width}x{image.height} at {args.cell_size:g} m/px; "
            f"occupied={occupied/raw.size:.1%}, free={free/raw.size:.1%}, unknown={unknown/raw.size:.1%}"
        )
        print(f"[occupancy-map] image: {image_path}")
        print(f"[occupancy-map] yaml: {yaml_path}")
        print(f"[occupancy-map] report: {report_path}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
