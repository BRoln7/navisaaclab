"""Render-only TailorNet skin driven by the MimicKit SMPL articulation."""

from __future__ import annotations

from pathlib import Path

import torch
from pxr import Gf, UsdGeom, UsdSkel

import omni.usd

from .pedestrian_policy import SMPL_BODY_NAMES


_PARENT_NAME = {
    "Pelvis": None,
    "L_Hip": "Pelvis", "L_Knee": "L_Hip", "L_Ankle": "L_Knee", "L_Toe": "L_Ankle",
    "R_Hip": "Pelvis", "R_Knee": "R_Hip", "R_Ankle": "R_Knee", "R_Toe": "R_Ankle",
    "Torso": "Pelvis", "Spine": "Torso", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "L_Thorax": "Chest", "L_Shoulder": "L_Thorax", "L_Elbow": "L_Shoulder",
    "L_Wrist": "L_Elbow", "L_Hand": "L_Wrist",
    "R_Thorax": "Chest", "R_Shoulder": "R_Thorax", "R_Elbow": "R_Shoulder",
    "R_Wrist": "R_Elbow", "R_Hand": "R_Wrist",
}

_SHIRT_COLORS = (
    (0.10, 0.28, 0.62),
    (0.58, 0.10, 0.12),
    (0.08, 0.42, 0.25),
    (0.55, 0.27, 0.05),
    (0.34, 0.12, 0.48),
    (0.05, 0.43, 0.52),
)


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    inverse = _quat_conjugate(quaternion)
    return _quat_multiply(_quat_multiply(inverse, pure), quaternion)[..., 1:]


class TailorNetPedestrianVisuals:
    """Spawn one UsdSkel overlay per environment and update only its 24 joints."""

    def __init__(
        self,
        *,
        asset_path: str | Path,
        num_envs: int,
        body_ids: list[int],
        update_interval: int = 1,
        hide_physics_visuals: bool = True,
    ) -> None:
        self._asset_path = Path(asset_path).expanduser().resolve()
        if not self._asset_path.is_file():
            raise FileNotFoundError(
                f"TailorNet visual USD does not exist: {self._asset_path}. "
                "Run scripts/socnav/build_tailornet_smpl_visual.py first."
            )
        if len(body_ids) != len(SMPL_BODY_NAMES):
            raise ValueError("TailorNet visual driver requires all 24 SMPL body ids")
        self._body_ids = body_ids
        self._update_interval = max(int(update_interval), 1)
        self._step = 0
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("Cannot create TailorNet visuals without an open USD stage")

        name_to_index = {name: index for index, name in enumerate(SMPL_BODY_NAMES)}
        self._parent_indices = [
            -1 if _PARENT_NAME[name] is None else name_to_index[_PARENT_NAME[name]]
            for name in SMPL_BODY_NAMES
        ]
        self._animations: list[UsdSkel.Animation] = []
        for env_id in range(num_envs):
            visual_path = f"/World/Visuals/SocNav/PedestrianSkins/env_{env_id}"
            prim = self._stage.DefinePrim(visual_path, "Xform")
            prim.GetReferences().AddReference(str(self._asset_path))
            self._animations.append(UsdSkel.Animation.Get(self._stage, f"{visual_path}/Animation"))
            self._set_color(visual_path, env_id)
            if hide_physics_visuals:
                self._hide_physics_visuals(env_id)

    def _set_color(self, root_path: str, env_id: int) -> None:
        color = _SHIRT_COLORS[env_id % len(_SHIRT_COLORS)]
        shader = self._stage.OverridePrim(f"{root_path}/Materials/Shirt/PreviewSurface")
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*color))

    def _hide_physics_visuals(self, env_id: int) -> None:
        for body_name in SMPL_BODY_NAMES:
            path = f"/World/envs/env_{env_id}/Ped/robot/{body_name}/visuals"
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()

    def update(self, body_position_w: torch.Tensor, body_quaternion_wxyz: torch.Tensor) -> None:
        self._step += 1
        if self._step % self._update_interval:
            return

        position = body_position_w[:, self._body_ids]
        quaternion = body_quaternion_wxyz[:, self._body_ids]
        local_position = torch.empty_like(position)
        local_quaternion = torch.empty_like(quaternion)
        local_position[:, 0] = position[:, 0]
        local_quaternion[:, 0] = quaternion[:, 0]
        for joint_index, parent_index in enumerate(self._parent_indices[1:], start=1):
            parent_quaternion = quaternion[:, parent_index]
            local_position[:, joint_index] = _quat_rotate_inverse(
                parent_quaternion, position[:, joint_index] - position[:, parent_index]
            )
            local_quaternion[:, joint_index] = _quat_multiply(
                _quat_conjugate(parent_quaternion), quaternion[:, joint_index]
            )

        local_position = local_position.detach().cpu()
        local_quaternion = local_quaternion.detach().cpu()
        for env_id, animation in enumerate(self._animations):
            translations = [Gf.Vec3f(*value.tolist()) for value in local_position[env_id]]
            rotations = [
                Gf.Quatf(float(value[0]), Gf.Vec3f(*value[1:].tolist()))
                for value in local_quaternion[env_id]
            ]
            animation.GetTranslationsAttr().Set(translations)
            animation.GetRotationsAttr().Set(rotations)

