#!/usr/bin/env python3
"""Common, structure-agnostic utilities for exact-triangle SDF relabelling."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BULK_ROOT = Path("/mnt/bulk10tb").resolve()
DEFAULT_OUTPUT_ROOT = (
    BULK_ROOT
    / "Deep3DComp"
    / "synthseg_qc_v1"
    / "left_hippocampus"
    / "exact_sdf_relabel_v1"
)
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_inr_representations_v1"
    / "metadata"
    / "hippocampus_qc_sdf_manifest.csv"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def require_bulk_path(
    value: str | Path,
    description: str = "persistent output",
    *,
    allow_non_bulk: bool = False,
) -> Path:
    """Return an absolute path, rejecting persistent output outside the 10-TB mount."""
    path = resolve_path(value)
    if allow_non_bulk:
        return path
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(
            f"{description} must be below {BULK_ROOT}; refusing {path}"
        ) from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root itself.")
    return path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    source = resolve_path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_id", "subject_id", "split", "mesh_path", "sdf_npz_path"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if len({row["scan_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest scan_id values are not unique.")
    for row in rows:
        row["mesh_path"] = str(resolve_path(row["mesh_path"]))
        row["sdf_npz_path"] = str(resolve_path(row["sdf_npz_path"]))
        if row.get("source_sdf_npz_path"):
            row["source_sdf_npz_path"] = str(resolve_path(row["source_sdf_npz_path"]))
    return rows


def manifest_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def atomic_write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    allow_non_bulk: bool = False,
) -> Path:
    if not rows:
        raise ValueError("Refusing to write an empty manifest.")
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
    path: str | Path,
    value: Any,
    *,
    allow_non_bulk: bool = False,
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


def load_mesh_for_sdf(path: str | Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Load a closed mesh and orient a private in-memory copy consistently."""
    source = resolve_path(path)
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {source}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"Mesh is empty or invalid: {source}")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"Mesh has non-finite vertices: {source}")
    report = {
        "mesh_path": str(source),
        "vertices": int(len(loaded.vertices)),
        "faces": int(len(loaded.faces)),
        "watertight_before": bool(loaded.is_watertight),
        "winding_consistent_before": bool(loaded.is_winding_consistent),
        "signed_volume_before": float(loaded.volume),
        "degenerate_faces": int(np.sum(loaded.area_faces <= 1.0e-14)),
    }
    if not report["watertight_before"]:
        raise ValueError(
            f"Exact sign is undefined for a non-watertight mesh; refusing {source}"
        )
    if report["degenerate_faces"]:
        raise ValueError(f"Mesh contains degenerate triangles; refusing {source}")
    mesh = loaded.copy()
    mesh.fix_normals(multibody=True)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    report.update(
        {
            "watertight_after": bool(mesh.is_watertight),
            "winding_consistent_after": bool(mesh.is_winding_consistent),
            "signed_volume_after": float(mesh.volume),
            "orientation_changed_in_memory": bool(
                (not report["winding_consistent_before"])
                or report["signed_volume_before"] < 0.0
            ),
        }
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent or float(mesh.volume) <= 0.0:
        raise ValueError(f"Could not establish a closed outward mesh orientation: {source}")
    return mesh, report


def exact_signed_distance_outside_positive(
    mesh: trimesh.Trimesh,
    xyz: np.ndarray,
    *,
    chunk_size: int = 50_000,
    zero_epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Exact triangle magnitude with negative-inside/positive-outside convention."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("xyz must be a finite N x 3 array.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        # trimesh returns positive inside and negative outside; invert to match DeepSDF.
        result[start:stop] = -trimesh.proximity.signed_distance(mesh, points[start:stop])
    result[np.abs(result) <= zero_epsilon] = 0.0
    if not np.isfinite(result).all():
        raise RuntimeError("Exact SDF backend produced a non-finite value.")
    return result


def load_source_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source = resolve_path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "pos" not in archive or "neg" not in archive:
            raise KeyError(f"SDF archive must contain pos and neg arrays: {source}")
        pos = np.asarray(archive["pos"])
        neg = np.asarray(archive["neg"])
        extras = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"pos", "neg"}
        }
    for name, values in (("pos", pos), ("neg", neg)):
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite N x 4 array: {source}")
    if not len(pos) or not len(neg):
        raise ValueError(f"SDF archive has an empty sign partition: {source}")
    return pos, neg, extras


def coordinate_sha256(xyz: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(xyz))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def relabel_arrays(
    mesh: trimesh.Trimesh,
    source_pos: np.ndarray,
    source_neg: np.ndarray,
    *,
    chunk_size: int = 50_000,
    zero_epsilon: float = 1.0e-8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep all source xyz exactly, replace SDF labels, and rebuild sign partitions."""
    source = np.concatenate((source_pos, source_neg), axis=0)
    xyz = np.ascontiguousarray(source[:, :3])
    target = exact_signed_distance_outside_positive(
        mesh, xyz, chunk_size=chunk_size, zero_epsilon=zero_epsilon
    )
    positive = target >= 0.0
    source_index = np.arange(len(source), dtype=np.int32)
    labelled = np.concatenate((xyz, target[:, None].astype(xyz.dtype, copy=False)), axis=1)
    output = {
        "pos": np.ascontiguousarray(labelled[positive]),
        "neg": np.ascontiguousarray(labelled[~positive]),
        "pos_source_index": np.ascontiguousarray(source_index[positive]),
        "neg_source_index": np.ascontiguousarray(source_index[~positive]),
        "source_pos_count": np.asarray(len(source_pos), dtype=np.int64),
        "source_neg_count": np.asarray(len(source_neg), dtype=np.int64),
        "format_version": np.asarray(1, dtype=np.int64),
    }
    if not len(output["pos"]) or not len(output["neg"]):
        raise RuntimeError("Exact relabelling produced an empty sign partition.")
    old_positive = np.arange(len(source)) < len(source_pos)
    metrics = {
        "query_count": int(len(source)),
        "source_pos_count": int(len(source_pos)),
        "source_neg_count": int(len(source_neg)),
        "exact_pos_count": int(positive.sum()),
        "exact_neg_count": int((~positive).sum()),
        "sign_partition_changes": int(np.sum(positive != old_positive)),
        "source_coordinate_sha256": coordinate_sha256(xyz),
        "exact_abs_sdf_mean": float(np.abs(target).mean()),
        "exact_abs_sdf_median": float(np.median(np.abs(target))),
        "exact_abs_sdf_p99": float(np.quantile(np.abs(target), 0.99)),
        "exact_zero_count": int(np.sum(target == 0.0)),
    }
    return output, metrics


def atomic_write_npz(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    *,
    allow_non_bulk: bool = False,
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


def restore_source_coordinate_order(
    exact_pos: np.ndarray,
    exact_neg: np.ndarray,
    pos_source_index: np.ndarray,
    neg_source_index: np.ndarray,
) -> np.ndarray:
    count = len(exact_pos) + len(exact_neg)
    indices = np.concatenate((pos_source_index, neg_source_index)).astype(np.int64, copy=False)
    if len(indices) != count or not np.array_equal(np.sort(indices), np.arange(count)):
        raise ValueError("Exact archive source-index arrays are not a permutation of all queries.")
    xyz = np.concatenate((exact_pos[:, :3], exact_neg[:, :3]), axis=0)
    restored = np.empty_like(xyz)
    restored[indices] = xyz
    return restored


def split_subject_leakage(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    subject_splits: dict[str, set[str]] = {}
    for row in rows:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
    return {
        subject: sorted(splits)
        for subject, splits in subject_splits.items()
        if len(splits) > 1
    }
