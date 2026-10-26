"""Occupancy-map loading and coordinate conversions for navigation scenes.

This module deliberately has no Isaac Sim or CUDA side effects.  The map is
loaded only when an environment is created, which keeps task discovery cheap
and makes the planning code usable in CPU-only tests and tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage

from socnav_sim.config import PROJECT_ROOT


ISAACNAV_SCENES_DIR = PROJECT_ROOT / "data/usd/assets/scenes/IsaacNav scenes2"


@dataclass(frozen=True)
class NavigationSceneAssets:
    scene_id: int
    usd: Path
    map_image: Path
    map_yaml: Path


def get_navigation_scene_assets(scene_id: int) -> NavigationSceneAssets:
    """Return and validate the assets for an IsaacNav navigation example."""
    if scene_id not in (12, 27, 34, 35):
        raise ValueError(
            f"Only the open-source CADRL examples 12, 27, 34 and 35 are supported, got {scene_id}."
        )
    directory = ISAACNAV_SCENES_DIR / f"Collected_scene_{scene_id}"
    usd_name = f"scene_{scene_id}_cadrl.usda" if scene_id == 34 else f"scene_{scene_id}.usd"
    assets = NavigationSceneAssets(
        scene_id=scene_id,
        usd=directory / usd_name,
        map_image=directory / f"scene_{scene_id}_nav_map.png",
        map_yaml=directory / f"scene_{scene_id}_nav_map.yaml",
    )
    missing = [path for path in (assets.usd, assets.map_image, assets.map_yaml) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Scene {scene_id} is incomplete: {missing}")
    return assets


@dataclass(frozen=True)
class MapInfo:
    """A planning grid and its relation to world coordinates.

    Array coordinates are ``[row(y), column(x)]``.  Public map coordinates are
    ``[x, y]`` so their order matches world positions.
    """

    occupancy: np.ndarray
    raw_occupancy: np.ndarray
    resolution: float
    origin: np.ndarray

    @property
    def dilated_img(self) -> np.ndarray:
        """Compatibility alias used by old callers."""
        return self.occupancy

    @property
    def binary_img(self) -> np.ndarray:
        """Compatibility alias used by old callers."""
        return self.raw_occupancy

    @property
    def res(self) -> float:
        """Compatibility alias used by old callers."""
        return self.resolution

    @property
    def free_cells_xy(self) -> np.ndarray:
        cells_yx = np.argwhere(self.occupancy == 0)
        return cells_yx[:, ::-1].copy()

    def map_to_world_numpy(self, positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float32)
        return positions * self.resolution + self.origin[:2]

    def world_to_map_numpy(self, positions_xy: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xy, dtype=np.float32)
        return np.rint((positions - self.origin[:2]) / self.resolution).astype(np.int64)

    def is_free_xy(self, cell_xy: np.ndarray | tuple[int, int]) -> bool:
        x, y = np.asarray(cell_xy, dtype=np.int64).tolist()
        rows, cols = self.occupancy.shape
        return 0 <= x < cols and 0 <= y < rows and self.occupancy[y, x] == 0

    def sample_free_cells(self, count: int, rng: np.random.Generator) -> np.ndarray:
        free = self.free_cells_xy
        if free.shape[0] == 0:
            raise RuntimeError("The occupancy map contains no free cells.")
        indices = rng.choice(free.shape[0], size=count, replace=count > free.shape[0])
        return free[indices]

    def sample_goal(
        self,
        start_world: np.ndarray,
        min_distance: float,
        max_distance: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        free_world = self.map_to_world_numpy(self.free_cells_xy)
        distances = np.linalg.norm(free_world - np.asarray(start_world)[:2], axis=-1)
        valid = np.flatnonzero((distances >= min_distance) & (distances <= max_distance))
        if valid.size == 0:
            valid = np.flatnonzero(distances >= min_distance)
        if valid.size == 0:
            valid = np.arange(free_world.shape[0])
        return free_world[int(rng.choice(valid))]


def load_map(
    image_path: str | Path,
    yaml_path: str | Path,
    *,
    scale: float = 0.25,
    dilation_size: int = 4,
) -> MapInfo:
    """Load an occupancy map, optionally downsampling before planning."""
    if scale <= 0:
        raise ValueError(f"Map scale must be positive, got {scale}.")

    with Path(yaml_path).open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)

    image = Image.open(image_path).convert("L").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if scale != 1.0:
        size = tuple(max(1, round(dim * scale)) for dim in image.size)
        image = image.resize(size, resample=Image.Resampling.NEAREST)

    pixels = np.asarray(image)
    raw = (pixels < int(metadata.get("occupied_thresh", 0.65) * 255)).astype(np.uint8)
    if dilation_size > 1:
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        occupancy = ndimage.binary_dilation(raw, structure=structure).astype(np.uint8)
    else:
        occupancy = raw.copy()

    return MapInfo(
        occupancy=occupancy,
        raw_occupancy=raw,
        resolution=float(metadata["resolution"]) / scale,
        origin=np.asarray(metadata["origin"], dtype=np.float32),
    )


def transform_map_to_world(positions: torch.Tensor, map_info: MapInfo) -> torch.Tensor:
    origin = positions.new_tensor(map_info.origin[:2])
    return positions.to(dtype=torch.float32) * map_info.resolution + origin


def transform_world_to_map(positions: torch.Tensor, map_info: MapInfo) -> torch.Tensor:
    origin = positions.new_tensor(map_info.origin[:2])
    return torch.round((positions[..., :2] - origin) / map_info.resolution).to(torch.long)


def random_spawn(map_info: MapInfo, num_agents: int, *, device: torch.device | str = "cpu"):
    rng = np.random.default_rng()
    map_positions = torch.as_tensor(map_info.sample_free_cells(num_agents, rng), device=device)
    return map_positions.long(), transform_map_to_world(map_positions, map_info)


def publish_random_goal(
    start_points: torch.Tensor,
    map_info: MapInfo,
    min_length: float,
    max_length: float,
):
    rng = np.random.default_rng()
    starts = start_points.detach().cpu().numpy()
    goals_world = np.stack(
        [map_info.sample_goal(start, min_length, max_length, rng) for start in starts]
    )
    world = torch.as_tensor(goals_world, device=start_points.device, dtype=start_points.dtype)
    return transform_world_to_map(world, map_info), world
