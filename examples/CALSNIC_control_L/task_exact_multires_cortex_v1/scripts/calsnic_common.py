#!/usr/bin/env python3
"""Shared CALSNIC helpers for exact SDF labels and matched geometry evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import point_cloud_utils as pcu
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BULK_ROOT = Path("/mnt/bulk10tb").resolve()
DEFAULT_OUTPUT_ROOT = BULK_ROOT / "Deep3DComp" / "CALSNIC" / "control_L_exact_multires_v1"
DATASET_ROOT = Path(
    "/home/jakaria/CALSNIC/calsnic_pial_surface/mesh_dataset/pial_surface_L"
)
DEFAULT_APPROX_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "calsnic_control_L_approx.csv"
DEFAULT_EXACT_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifests" / "calsnic_control_L_exact.csv"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def require_bulk_path(
    value: str | Path,
    description: str = "persistent output",
    *,
    allow_non_bulk: bool = False,
) -> Path:
    path = resolve_path(value)
    if allow_non_bulk:
        return path
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must be below {BULK_ROOT}; refusing {path}") from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root itself.")
    return path


def manifest_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    source = resolve_path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_id", "subject_id", "split", "mesh_path", "sdf_npz_path"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if len({row["scan_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest scan IDs are not unique.")
    for row in rows:
        for key in ("mesh_path", "mesh_path_mm", "sdf_npz_path", "source_sdf_npz_path"):
            if row.get(key):
                row[key] = str(resolve_path(row[key]))
    order = {"train": 0, "val": 1, "test": 2}
    return sorted(rows, key=lambda row: (order.get(row["split"], 99), row["scan_id"]))


def atomic_write_csv(
    path: str | Path, rows: list[dict[str, Any]], *, allow_non_bulk: bool = False
) -> Path:
    if not rows:
        raise ValueError("Refusing to write an empty CSV.")
    output = require_bulk_path(path, allow_non_bulk=allow_non_bulk)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def atomic_write_json(
    path: str | Path, value: Any, *, allow_non_bulk: bool = False
) -> Path:
    output = require_bulk_path(path, allow_non_bulk=allow_non_bulk)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def atomic_write_npz(
    path: str | Path, arrays: dict[str, np.ndarray], *, allow_non_bulk: bool = False
) -> Path:
    output = require_bulk_path(path, "exact SDF archive", allow_non_bulk=allow_non_bulk)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_sha256(xyz: np.ndarray) -> str:
    values = np.ascontiguousarray(xyz)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(resolve_path(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"Invalid or empty mesh: {path}")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"Non-finite mesh vertices: {path}")
    return loaded


def row_center(row: dict[str, str]) -> np.ndarray:
    fields = ("mesh_center_x", "mesh_center_y", "mesh_center_z")
    if any(row.get(field, "") in {None, ""} for field in fields):
        raise ValueError(f"Manifest has no complete SDF mesh centre for {row['scan_id']}.")
    center = np.asarray([float(row[field]) for field in fields], dtype=np.float64)
    if not np.isfinite(center).all():
        raise ValueError(f"Non-finite SDF mesh centre for {row['scan_id']}.")
    return center


def load_sdf_space_mesh(row: dict[str, str], *, require_watertight: bool = True) -> trimesh.Trimesh:
    mesh = load_mesh(row["mesh_path"]).copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) - row_center(row)[None, :]
    report_bad = not mesh.is_watertight or np.any(mesh.area_faces <= 1.0e-14)
    if require_watertight and report_bad:
        raise ValueError(f"Mesh is non-watertight or degenerate: {row['mesh_path']}")
    mesh.fix_normals(multibody=True)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if require_watertight and (not mesh.is_watertight or not mesh.is_winding_consistent):
        raise ValueError(f"Cannot establish closed consistent mesh: {row['mesh_path']}")
    return mesh


def load_sdf_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(resolve_path(path), allow_pickle=False) as archive:
        if "pos" not in archive or "neg" not in archive:
            raise KeyError(f"SDF archive lacks pos/neg arrays: {path}")
        pos = np.asarray(archive["pos"])
        neg = np.asarray(archive["neg"])
    for name, array in (("pos", pos), ("neg", neg)):
        if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} array {array.shape}: {path}")
        if not len(array):
            raise ValueError(f"Empty {name} array: {path}")
    return pos, neg


def exact_signed_distance(
    mesh: trimesh.Trimesh, xyz: np.ndarray, *, chunk_size: int = 100_000
) -> np.ndarray:
    """Exact closest-triangle magnitude with a fast-winding inside/outside sign.

    ``point_cloud_utils.signed_distance_to_mesh`` reliably returns the closest
    face and its barycentric coordinates, but in the installed PCU version its
    returned signed *value* is not always the Euclidean distance to that point.
    We therefore use that value only for its fast-winding sign and explicitly
    calculate the Euclidean magnitude from the returned closest point.
    """
    points = np.asarray(xyz, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), int(chunk_size)):
        stop = min(start + int(chunk_size), len(points))
        query = points[start:stop]
        # PCU has a singleton-query edge case in this environment.  Duplicating
        # its sole point keeps the call on the normal batched code path.
        pcu_query = np.repeat(query, 2, axis=0) if len(query) == 1 else query
        signed_proxy, face_index, barycentric = pcu.signed_distance_to_mesh(
            pcu_query, vertices, faces
        )
        signed_proxy = np.asarray(signed_proxy, dtype=np.float64)[: len(query)]
        face_index = np.asarray(face_index, dtype=np.int64)[: len(query)]
        barycentric = np.asarray(barycentric, dtype=np.float64)[: len(query)]
        if barycentric.shape != (len(query), 3) or np.any(face_index < 0) or np.any(
            face_index >= len(faces)
        ):
            raise RuntimeError("PCU returned invalid closest-face coordinates.")
        closest = np.einsum("nij,ni->nj", vertices[faces[face_index]], barycentric)
        magnitude = np.linalg.norm(query - closest, axis=1)
        result[start:stop] = np.where(signed_proxy < 0.0, -magnitude, magnitude)
    if not np.isfinite(result).all():
        raise RuntimeError("Exact SDF backend returned non-finite values.")
    return result


def relabel_arrays(
    mesh: trimesh.Trimesh,
    source_pos: np.ndarray,
    source_neg: np.ndarray,
    *,
    chunk_size: int = 100_000,
    zero_epsilon: float = 1.0e-8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = np.concatenate((source_pos, source_neg), axis=0)
    xyz = np.ascontiguousarray(source[:, :3])
    exact = exact_signed_distance(mesh, xyz, chunk_size=chunk_size)
    exact[np.abs(exact) <= float(zero_epsilon)] = 0.0
    positive = exact >= 0.0
    source_index = np.arange(len(source), dtype=np.int32)
    labelled = np.concatenate(
        (xyz, exact[:, None].astype(xyz.dtype, copy=False)), axis=1
    )
    arrays = {
        "pos": np.ascontiguousarray(labelled[positive]),
        "neg": np.ascontiguousarray(labelled[~positive]),
        "pos_source_index": np.ascontiguousarray(source_index[positive]),
        "neg_source_index": np.ascontiguousarray(source_index[~positive]),
        "source_pos_count": np.asarray(len(source_pos), dtype=np.int64),
        "source_neg_count": np.asarray(len(source_neg), dtype=np.int64),
        "format_version": np.asarray(2, dtype=np.int64),
    }
    old_positive = np.arange(len(source)) < len(source_pos)
    metrics = {
        "query_count": int(len(source)),
        "source_pos_count": int(len(source_pos)),
        "source_neg_count": int(len(source_neg)),
        "exact_pos_count": int(positive.sum()),
        "exact_neg_count": int((~positive).sum()),
        "sign_partition_changes": int(np.sum(positive != old_positive)),
        "sign_partition_change_fraction": float(np.mean(positive != old_positive)),
        "source_coordinate_sha256": coordinate_sha256(xyz),
        "exact_abs_sdf_mean": float(np.abs(exact).mean()),
        "exact_abs_sdf_p99": float(np.quantile(np.abs(exact), 0.99)),
    }
    return arrays, metrics


def restore_source_order(
    pos: np.ndarray, neg: np.ndarray, pos_index: np.ndarray, neg_index: np.ndarray
) -> np.ndarray:
    indices = np.concatenate((pos_index, neg_index)).astype(np.int64, copy=False)
    count = len(indices)
    if not np.array_equal(np.sort(indices), np.arange(count)):
        raise ValueError("Exact archive source indices are not a complete permutation.")
    xyz = np.concatenate((pos[:, :3], neg[:, :3]), axis=0)
    restored = np.empty_like(xyz)
    restored[indices] = xyz
    return restored


def sdf_vertices_to_mm(vertices: np.ndarray, row: dict[str, str]) -> np.ndarray:
    scale = float(row["scaled_from_mm_scale"])
    translation = np.asarray(
        [
            float(row["scaled_from_mm_tx"]),
            float(row["scaled_from_mm_ty"]),
            float(row["scaled_from_mm_tz"]),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(translation).all():
        raise ValueError(f"Invalid scaled-to-mm transform for {row['scan_id']}.")
    scaled = np.asarray(vertices, dtype=np.float64) + row_center(row)[None, :]
    return (scaled - translation[None, :]) / scale


def mesh_sdf_to_mm(mesh: trimesh.Trimesh, row: dict[str, str]) -> trimesh.Trimesh:
    result = mesh.copy()
    result.vertices = sdf_vertices_to_mm(result.vertices, row)
    return result


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
