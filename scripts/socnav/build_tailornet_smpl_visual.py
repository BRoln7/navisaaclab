#!/usr/bin/env python3
"""Build a render-only TailorNet character bound to the project's SMPL skeleton.

The generated USD deliberately contains no physics.  Navigation keeps using the
MimicKit articulation for dynamics and drives this UsdSkel asset as a visual
overlay.  The TailorNet source data and the SMPL body models are both
license-gated, so the generated file lives below ``data/`` and is not intended
for redistribution.
"""

from __future__ import annotations

import argparse
import inspect
import os
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data/usd/assets/people/source/tailornet"
DEFAULT_SMPL = PROJECT_ROOT / "data/smpl/SMPL_FEMALE.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/usd/assets/people/tailornet_smpl_female.usd"

SMPL_NAMES = (
    "Pelvis L_Hip R_Hip Torso L_Knee R_Knee Spine L_Ankle R_Ankle Chest "
    "L_Toe R_Toe Neck L_Thorax R_Thorax Head L_Shoulder R_Shoulder "
    "L_Elbow R_Elbow L_Wrist R_Wrist L_Hand R_Hand"
).split()

# MimicKit's depth-first body order.  The paths below also define the hierarchy
# consumed by UsdSkel.
BODY_NAMES = (
    "Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe "
    "Torso Spine Chest Neck Head L_Thorax L_Shoulder L_Elbow L_Wrist L_Hand "
    "R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand"
).split()

PARENTS = {
    "Pelvis": None,
    "L_Hip": "Pelvis", "L_Knee": "L_Hip", "L_Ankle": "L_Knee", "L_Toe": "L_Ankle",
    "R_Hip": "Pelvis", "R_Knee": "R_Hip", "R_Ankle": "R_Knee", "R_Toe": "R_Ankle",
    "Torso": "Pelvis", "Spine": "Torso", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "L_Thorax": "Chest", "L_Shoulder": "L_Thorax", "L_Elbow": "L_Shoulder",
    "L_Wrist": "L_Elbow", "L_Hand": "L_Wrist",
    "R_Thorax": "Chest", "R_Shoulder": "R_Thorax", "R_Elbow": "R_Shoulder",
    "R_Wrist": "R_Elbow", "R_Hand": "R_Wrist",
}


def _ensure_pxr() -> None:
    """Restart with Isaac Sim's USD libraries when running outside Kit."""
    try:
        import pxr  # noqa: F401
        return
    except ImportError:
        pass

    # ``CONDA_PREFIX`` may still point at a parent shell environment when the
    # script is launched through isaaclab.sh.  The running interpreter is the
    # authoritative environment.
    env_prefix = Path(sys.prefix)
    site_packages = env_prefix / "lib/python3.11/site-packages"
    candidates = sorted((site_packages / "isaacsim/extscache").glob("omni.usd.libs-*/"))
    if not candidates:
        raise RuntimeError("Isaac Sim USD libraries were not found. Run this script with ./isaaclab.sh -p.")

    usd_lib = candidates[-1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(usd_lib), env.get("PYTHONPATH", ""))))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(usd_lib / "bin"), str(env_prefix / "lib"), env.get("LD_LIBRARY_PATH", "")))
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _load_smpl(path: Path):
    # Official SMPL pickles contain legacy Chumpy objects.  These aliases let
    # them load under Python 3.11 / modern NumPy without changing the source.
    import numpy as np

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in (
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("unicode", str), ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)

    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def _as_array(value):
    import numpy as np

    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def _loop_subdivision_matrix(vertices, faces):
    """Return TailorNet's exact one-step Loop subdivision operator.

    TailorNet stores garment vertex ids in the high-resolution SMPL topology.
    Their meaning therefore depends on the *ordering* of the edge vertices,
    not only on the subdivision geometry.  Keep the CSC -> COO ordering used
    by TailorNet's ``smpl_lib/smpl_paths.py`` instead of sorting edge tuples.
    """
    import numpy as np
    import scipy.sparse as sp

    count = len(vertices)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    connectivity = sp.csc_matrix((count, count))
    for corner in range(3):
        source = faces[:, corner]
        target = faces[:, (corner + 1) % 3]
        edge_matrix = sp.csc_matrix(
            (np.ones(len(source)), (source, target)), shape=connectivity.shape
        )
        connectivity = connectivity + edge_matrix + edge_matrix.T

    opposites: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for i in range(3):
            a, b, opposite = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            edge = tuple(sorted((a, b)))
            opposites.setdefault(edge, []).append(opposite)

    for vertex in range(count):
        adjacent = np.nonzero(connectivity[:, vertex])[0]
        n = len(adjacent)
        weight = 0.0 if n < 3 else (3.0 / 16.0 if n == 3 else 3.0 / (8.0 * n))
        rows.append(vertex); cols.append(vertex); values.append(1.0 - weight * n)
        for other in adjacent:
            rows.append(vertex); cols.append(int(other)); values.append(weight)

    connectivity_coo = sp.coo_matrix(connectivity)
    edge_array = np.column_stack((connectivity_coo.row, connectivity_coo.col))
    edge_array = edge_array[edge_array[:, 0] < edge_array[:, 1]]
    edge_list = [tuple(map(int, edge)) for edge in edge_array]
    edge_vertex: dict[tuple[int, int], int] = {}
    for edge_index, edge in enumerate(edge_list):
        row = count + edge_index
        edge_vertex[edge] = row
        for endpoint in edge:
            rows.append(row); cols.append(endpoint); values.append(3.0 / 8.0)
        opposite = opposites[edge]
        for vertex in opposite:
            rows.append(row); cols.append(vertex); values.append(2.0 / (8.0 * len(opposite)))

    high_faces: list[list[int]] = []
    for a, b, c in faces.astype(int):
        ab = edge_vertex[tuple(sorted((a, b)))]
        bc = edge_vertex[tuple(sorted((b, c)))]
        ca = edge_vertex[tuple(sorted((c, a)))]
        high_faces.extend(([ab, b, bc], [bc, c, ca], [ca, a, ab], [ab, bc, ca]))

    matrix = sp.csr_matrix((values, (rows, cols)), shape=(count + len(edge_list), count))
    return matrix, np.asarray(high_faces, dtype=np.int32)


def _top_influences(weights, count: int = 4):
    import numpy as np

    indices = np.argpartition(-weights, kth=count - 1, axis=1)[:, :count]
    values = np.take_along_axis(weights, indices, axis=1)
    order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int32)
    values = np.take_along_axis(values, order, axis=1).astype(np.float32)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
    return indices, values


def _sim_coordinates(values):
    """Map SMPL (x right, y up, z forward) to simulator (x forward, y left, z up)."""
    return values[..., [2, 0, 1]]


def _joint_paths() -> list[str]:
    paths: dict[str, str] = {}
    for name in BODY_NAMES:
        parent = PARENTS[name]
        paths[name] = name if parent is None else f"{paths[parent]}/{name}"
    return [paths[name] for name in BODY_NAMES]


def _material(stage, path: str, color: tuple[float, float, float], roughness: float = 0.72):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_mesh(mesh, skeleton, indices, weights) -> None:
    from pxr import Gf, UsdGeom, UsdSkel

    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))
    joint_indices = binding.CreateJointIndicesPrimvar(False, indices.shape[1])
    joint_indices.Set(indices.reshape(-1).tolist())
    joint_indices.SetInterpolation(UsdGeom.Tokens.vertex)
    joint_weights = binding.CreateJointWeightsPrimvar(False, weights.shape[1])
    joint_weights.Set(weights.reshape(-1).tolist())
    joint_weights.SetInterpolation(UsdGeom.Tokens.vertex)


def _define_mesh(stage, path: str, vertices, faces, skeleton, indices, weights, material):
    from pxr import UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(vertices.astype("float32").tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.astype("int32").reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    _bind_mesh(mesh, skeleton, indices, weights)
    return mesh


def build(source: Path, smpl_path: Path, output: Path) -> None:
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel

    required = [
        source / "garment_class_info.pkl",
        source / "t-shirt_female/style_shape/beta000_gamma023.npy",
        source / "t-shirt_female/shape/beta_000.npy",
    ]
    missing = [str(path) for path in required + [smpl_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing TailorNet/SMPL inputs:\n  " + "\n  ".join(missing))

    model = _load_smpl(smpl_path)
    template = _as_array(model["v_template"]).astype(np.float64)
    faces = _as_array(model["f"]).astype(np.int32)
    shapedirs = _as_array(model["shapedirs"]).astype(np.float64)
    weights = _as_array(model["weights"]).astype(np.float64)
    beta = np.load(source / "t-shirt_female/shape/beta_000.npy").astype(np.float64)
    beta = beta[: shapedirs.shape[-1]]
    shaped = template + np.tensordot(shapedirs, beta, axes=([2], [0]))
    joint_regressor = model["J_regressor"]
    joints = np.asarray(joint_regressor.dot(shaped), dtype=np.float64)

    with (source / "garment_class_info.pkl").open("rb") as stream:
        garment_info = pickle.load(stream, encoding="latin1")["t-shirt"]
    garment_vertex_ids = np.asarray(garment_info["vert_indices"], dtype=np.int64)
    garment_faces = np.asarray(garment_info["f"], dtype=np.int32)
    garment_offset = np.load(
        source / "t-shirt_female/style_shape/beta000_gamma023.npy"
    ).astype(np.float64)

    subdivision, _ = _loop_subdivision_matrix(template, faces)
    high_shaped = subdivision.dot(shaped)
    high_weights = subdivision.dot(weights)
    high_weights /= np.maximum(high_weights.sum(axis=1, keepdims=True), 1e-8)
    garment_vertices = high_shaped[garment_vertex_ids] + garment_offset
    garment_weights = high_weights[garment_vertex_ids]

    # A t-shirt must remain on the upper body.  This catches accidental
    # changes to the high-resolution vertex ordering immediately: the same
    # ids applied to a differently ordered subdivision produce a full-body
    # cape while still forming a technically valid USD mesh.
    garment_vertical_span = np.ptp(garment_vertices[:, 1])
    body_vertical_span = np.ptp(high_shaped[:, 1])
    if garment_vertical_span > 0.75 * body_vertical_span:
        raise RuntimeError(
            "TailorNet garment ids do not match the high-resolution SMPL vertex order "
            f"(garment vertical span={garment_vertical_span:.3f} m, "
            f"body span={body_vertical_span:.3f} m)."
        )

    order = [SMPL_NAMES.index(name) for name in BODY_NAMES]
    weights = weights[:, order]
    garment_weights = garment_weights[:, order]
    joints = joints[order]

    pelvis = joints[0].copy()
    body_vertices = _sim_coordinates(shaped - pelvis)
    garment_vertices = _sim_coordinates(garment_vertices - pelvis)
    joints = _sim_coordinates(joints - pelvis)
    body_indices, body_weights = _top_influences(weights)
    garment_indices, garment_weights = _top_influences(garment_weights)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdSkel.Root.Define(stage, "/TailorNetSMPL")
    stage.SetDefaultPrim(root.GetPrim())
    skeleton = UsdSkel.Skeleton.Define(stage, "/TailorNetSMPL/Skeleton")
    joint_paths = _joint_paths()
    skeleton.CreateJointsAttr(joint_paths)

    rest = []
    bind = []
    name_to_index = {name: index for index, name in enumerate(BODY_NAMES)}
    for index, name in enumerate(BODY_NAMES):
        parent = PARENTS[name]
        translation = joints[index] if parent is None else joints[index] - joints[name_to_index[parent]]
        local = Gf.Matrix4d(1.0)
        local.SetTranslateOnly(Gf.Vec3d(*translation.tolist()))
        rest.append(local)
        global_transform = Gf.Matrix4d(1.0)
        global_transform.SetTranslateOnly(Gf.Vec3d(*joints[index].tolist()))
        bind.append(global_transform)
    skeleton.CreateRestTransformsAttr(rest)
    skeleton.CreateBindTransformsAttr(bind)

    animation = UsdSkel.Animation.Define(stage, "/TailorNetSMPL/Animation")
    animation.CreateJointsAttr(joint_paths)
    animation.CreateTranslationsAttr(
        [Gf.Vec3f(*matrix.ExtractTranslation()) for matrix in rest]
    )
    animation.CreateRotationsAttr([Gf.Quatf(1.0, Gf.Vec3f(0.0)) for _ in rest])
    animation.CreateScalesAttr([Gf.Vec3h(1.0) for _ in rest])
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )

    materials = "/TailorNetSMPL/Materials"
    skin = _material(stage, f"{materials}/Skin", (0.55, 0.30, 0.20), 0.78)
    shirt = _material(stage, f"{materials}/Shirt", (0.10, 0.28, 0.62), 0.66)
    body = _define_mesh(
        stage, "/TailorNetSMPL/Geometry/Body", body_vertices, faces,
        skeleton, body_indices, body_weights, skin,
    )
    _define_mesh(
        stage, "/TailorNetSMPL/Geometry/TShirt", garment_vertices, garment_faces,
        skeleton, garment_indices, garment_weights, shirt,
    )

    # Tight trousers and shoes are material regions on the SMPL body.  The
    # actual upper garment is the separate TailorNet cloth surface above.
    pants = _material(stage, f"{materials}/Pants", (0.055, 0.065, 0.085), 0.82)
    shoes = _material(stage, f"{materials}/Shoes", (0.025, 0.025, 0.025), 0.60)
    source_vertices = shaped - pelvis
    face_centers = source_vertices[faces].mean(axis=1)
    feet_height = min(joints[BODY_NAMES.index("L_Ankle"), 2], joints[BODY_NAMES.index("R_Ankle"), 2]) + 0.10
    # Source SMPL y is vertical.  Pants stop below the waist and shoes occupy
    # the lowest 10 cm above the ankle/foot region.
    sim_centers = _sim_coordinates(face_centers)
    shoes_faces = np.nonzero(sim_centers[:, 2] < feet_height)[0]
    pants_faces = np.nonzero(
        (sim_centers[:, 2] < 0.02) & (sim_centers[:, 2] >= feet_height)
    )[0]
    if len(pants_faces):
        subset = UsdGeom.Subset.CreateGeomSubset(
            body, "Pants", UsdGeom.Tokens.face, pants_faces.astype(np.int32).tolist(), "materialBind"
        )
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(pants)
    if len(shoes_faces):
        subset = UsdGeom.Subset.CreateGeomSubset(
            body, "Shoes", UsdGeom.Tokens.face, shoes_faces.astype(np.int32).tolist(), "materialBind"
        )
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(shoes)

    root.GetPrim().SetCustomDataByKey("socnav:bodyNames", ",".join(BODY_NAMES))
    root.GetPrim().SetCustomDataByKey("socnav:source", "TailorNet t-shirt_female sample beta000_gamma023")
    root.GetPrim().SetCustomDataByKey("socnav:license", "TailorNet non-commercial research license")
    stage.GetRootLayer().Save()
    print(f"[TailorNet] wrote {output}")
    print(f"[TailorNet] body vertices={len(body_vertices)}, garment vertices={len(garment_vertices)}")


def validate(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"USD could not be reopened: {path}")
    skeleton = UsdSkel.Skeleton.Get(stage, "/TailorNetSMPL/Skeleton")
    joints = skeleton.GetJointsAttr().Get()
    if len(joints) != 24:
        raise RuntimeError(f"Expected 24 SMPL joints, found {len(joints)}")
    for mesh_path, expected_points in (
        ("/TailorNetSMPL/Geometry/Body", 6890),
        ("/TailorNetSMPL/Geometry/TShirt", 7702),
    ):
        mesh = UsdGeom.Mesh.Get(stage, mesh_path)
        points = mesh.GetPointsAttr().Get()
        if len(points) != expected_points:
            raise RuntimeError(f"{mesh_path}: expected {expected_points} points, found {len(points)}")
        binding = UsdSkel.BindingAPI(mesh.GetPrim())
        if binding.GetJointIndicesPrimvar().GetElementSize() != 4:
            raise RuntimeError(f"{mesh_path}: expected four joint influences per vertex")
    print(f"[TailorNet] validation passed: 24 joints, 6890 body vertices, 7702 garment vertices")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--smpl", type=Path, default=DEFAULT_SMPL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _ensure_pxr()
    build(args.source.resolve(), args.smpl.resolve(), args.output.resolve())
    validate(args.output.resolve())


if __name__ == "__main__":
    main()
