"""Isaac Lab adapter for Isaac Sim 5.1's native RTX planar lidar."""

from __future__ import annotations

import re

import carb
import numpy as np
import torch
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils import configclass

from .features import resample_scan


class NavigationLidar(SensorBase):
    """Create RTX render products only for active robots, in numeric slot order."""

    def __init__(self, cfg):
        self._lidars = []
        self._data = None
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)
        super().__init__(cfg)

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        from isaacsim.sensors.rtx import LidarRtx

        parents = sorted(
            self._parent_prims,
            key=lambda prim: int(re.search(r"/env_(\d+)/", str(prim.GetPath())).group(1)),
        )
        if not 0 < self.cfg.active_robot_count <= len(parents):
            raise ValueError("RTX lidar active_robot_count exceeds the scene's robot slots")
        self._data = torch.full((len(parents), 720), 30.0, device=self.device)
        leaf = self.cfg.prim_path.rsplit("/", 1)[1]
        for index, parent in enumerate(parents[:self.cfg.active_robot_count]):
            sensor = LidarRtx(
                prim_path=f"{parent.GetPath()}/{leaf}", name=f"drlvo_lidar_{index}",
                translation=np.asarray(self.cfg.offset),
                config_file_name="Example_Rotary_2D",
            )
            # Pull directly from the annotator; no extra global frame callbacks.
            sensor.attach_annotator("IsaacComputeRTXLidarFlatScan")
            self._lidars.append(sensor)

    def _update_buffers_impl(self, env_ids):
        ids = range(len(self._lidars)) if isinstance(env_ids, slice) else env_ids
        for index in ids:
            index = int(index)
            if index >= len(self._lidars):
                continue
            output = self._lidars[index].get_annotators()["IsaacComputeRTXLidarFlatScan"].get_data()
            ranges = output.get("linearDepthData", ()) if isinstance(output, dict) else ()
            self._data[index] = resample_scan(torch.as_tensor(ranges, device=self.device, dtype=torch.float32))

    def close(self):
        for sensor in self._lidars:
            sensor.detach_all_annotators()
            sensor.detach_all_writers()
        self._lidars.clear()

    def _invalidate_initialize_callback(self, event):
        self.close()
        super()._invalidate_initialize_callback(event)


@configclass
class NavigationLidarCfg(SensorBaseCfg):
    class_type: type = NavigationLidar
    active_robot_count: int = 1
    offset: tuple[float, float, float] = (0.0, 0.0, 0.45)
