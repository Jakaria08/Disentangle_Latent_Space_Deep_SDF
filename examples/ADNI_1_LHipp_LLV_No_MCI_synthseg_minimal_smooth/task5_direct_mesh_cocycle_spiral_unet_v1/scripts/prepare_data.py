#!/usr/bin/env python3
"""Prepare read-only source meshes for direct surface-cocycle training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import common as C
import data as D
import mesh_hierarchy as H


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _velocity_imports():
    scripts = C.TASK_ROOT.parent / "task4_velocity_reference_audit_v2" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from velocity_core import Candidate, fit_trajectory, generalized_rigid_alignment, mesh_volume

    return Candidate, fit_trajectory, generalized_rigid_alignment, mesh_volume


def load_vertices(split: str, scan_ids: np.ndarray, manifest: list[dict[str, str]]) -> np.ndarray:
    old_cache = C.AE_BULK_ROOT / "cache" / f"adni_{split}_V.npy"
    manifest_split = [row for row in manifest if row["split"] == split]
    manifest_ids = np.asarray([row["scan_id"] for row in manifest_split], dtype=str)
    if old_cache.is_file() and np.array_equal(manifest_ids, scan_ids.astype(str)):
        vertices = np.load(old_cache).astype(np.float32, copy=False)
    else:
        import openmesh as om

        by_scan = {row["scan_id"]: row for row in manifest}
        vertices = np.stack(
            [om.read_trimesh(by_scan[str(scan)]["mesh_path_mm"]).points().astype(np.float32) for scan in scan_ids]
        )
    if vertices.shape != (len(scan_ids), C.VERTEX_COUNT, 3):
        raise ValueError(f"{split} vertex cache has wrong shape: {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite vertices in {split}")
    return vertices


def fitted_velocity_reference(
    vertices: np.ndarray, ages: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Candidate, fit_trajectory, generalized_rigid_alignment, _mesh_volume = _velocity_imports()
    candidate = Candidate(
        "surface_linear_rigid",
        "Observed fitted longitudinal reference",
        "rigid",
        "surface",
        1,
    )
    velocity = np.zeros_like(vertices, dtype=np.float32)
    reliability = np.zeros(len(vertices), dtype=np.float32)
    for subject_index in range(len(offsets) - 1):
        first, last = int(offsets[subject_index]), int(offsets[subject_index + 1])
        subject_ages = ages[first:last].astype(np.float64)
        subject_vertices = vertices[first:last].astype(np.float64)
        if last - first < 2 or np.ptp(subject_ages) <= 0.0:
            continue
        aligned, _template = generalized_rigid_alignment(subject_vertices)
        trajectory = fit_trajectory(subject_ages, aligned, candidate)
        derivative = trajectory.derivative(subject_ages).astype(np.float32)
        velocity[first:last] = derivative
        visits = last - first
        span = float(np.ptp(subject_ages))
        visit_weight = 0.25 if visits == 2 else min(1.0, 0.5 + 0.2 * (visits - 3))
        span_weight = min(1.0, span / 2.0)
        reliability[first:last] = float(visit_weight * span_weight)
    return velocity, reliability


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_split(split: str, root: Path, manifest: list[dict[str, str]], overwrite: bool) -> dict:
    destination = D.split_cache_path(split, root)
    if destination.exists() and not overwrite:
        prepared = D.load_split(split, root)
        return {
            "split": split,
            "scans": len(prepared.scan_ids),
            "subjects": len(set(prepared.subject_ids.astype(str))),
            "path": str(destination),
            "reused": True,
        }
    source_path = C.source_sequence_path(split)
    with np.load(source_path, allow_pickle=False) as source:
        scan_ids = source["visit_scan_ids"].astype(str)
        subject_ids = source["visit_subject_ids"].astype(str)
        diagnoses = source["visit_diagnoses"].astype(str)
        labels = source["visit_label_ad"].astype(np.int64)
        orders = source["visit_orders"].astype(np.int64)
        ages = source["visit_age_years"].astype(np.float32)
        years = source["visit_time_years_from_baseline"].astype(np.float32)
        offsets = source["subject_visit_offsets"].astype(np.int64)
    vertices = load_vertices(split, scan_ids, manifest)
    velocity, reliability = fitted_velocity_reference(vertices, ages, offsets)
    faces = np.load(C.AE_BULK_ROOT / "cache" / "adni_faces.npy").astype(np.int64)
    _Candidate, _fit, _align, mesh_volume = _velocity_imports()
    volumes = np.asarray([mesh_volume(mesh, faces) for mesh in vertices], dtype=np.float32)
    rows = C.load_pairs(split, scan_ids, subject_ids, labels)
    atomic_npz(
        destination,
        vertices_mm=vertices,
        age_years=ages,
        years_from_baseline=years,
        label_ad=labels,
        volume_mm3=volumes,
        velocity_reference_mm_per_year=velocity,
        velocity_reference_weight=reliability,
        scan_ids=scan_ids,
        subject_ids=subject_ids,
        diagnoses=diagnoses,
        visit_orders=orders,
        subject_visit_offsets=offsets,
    )
    return {
        "split": split,
        "scans": len(scan_ids),
        "subjects": len(set(subject_ids)),
        "diagnoses": dict(Counter(diagnoses)),
        "pairs": len(rows),
        "path": str(destination),
        "reused": False,
    }


def training_statistics(train: D.PreparedSplit, faces: np.ndarray, rows: list[C.PairRow]) -> dict:
    vertices = train.vertices.numpy().astype(np.float64)
    template = vertices.mean(axis=0)
    coordinate_scale = float(np.sqrt(np.mean((vertices - template) ** 2)))
    age_mean = float(train.ages.mean())
    age_std = float(train.ages.std(unbiased=False))
    volume_log = np.log(np.maximum(train.volumes.numpy().astype(np.float64), 1.0e-8))
    endpoint_distances = []
    rates_by_diagnosis: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}}
    for row in rows:
        delta = vertices[row.target] - vertices[row.source]
        endpoint_distances.append(float(np.linalg.norm(delta, axis=1).mean()))
        rate = float((volume_log[row.target] - volume_log[row.source]) / max(row.delta_years, 1.0e-6))
        rates_by_diagnosis[row.diagnosis].setdefault(row.subject, []).append(rate)
    velocity = train.velocity_reference.numpy().astype(np.float64)
    weights = train.velocity_reference_weight.numpy() > 0
    velocity_rms = np.sqrt(np.mean(velocity[weights] ** 2, axis=(1, 2))) if weights.any() else np.asarray([1.0])
    velocity_scale = float(max(np.median(velocity_rms), 1.0e-3))
    targets = {
        diagnosis: float(np.mean([np.mean(values) for values in subjects.values()]))
        for diagnosis, subjects in rates_by_diagnosis.items()
    }
    return {
        "schema_version": 1,
        "source": "train split only",
        "template_vertices_mm": template.astype(np.float32),
        "faces": faces.astype(np.int64),
        "coordinate_scale_mm": max(coordinate_scale, 1.0e-3),
        "velocity_scale_mm_per_year": velocity_scale,
        "endpoint_scale_mm": max(float(np.median(endpoint_distances)), 1.0e-3),
        "velocity_reference_scale_mm_per_year": max(velocity_scale, 1.0e-3),
        "age_mean_years": age_mean,
        "age_std_years": max(age_std, 1.0e-3),
        "log_volume_mean": float(volume_log.mean()),
        "log_volume_std": max(float(volume_log.std()), 1.0e-3),
        "group_log_volume_rate_targets": targets,
        "ad_minus_cn_log_volume_rate_target": targets["AD"] - targets["CN"],
        "train_scans": len(train.scan_ids),
        "train_subjects": len(set(train.subject_ids.astype(str))),
        "train_pairs": len(rows),
    }


def verify_prepared(root: Path) -> dict:
    splits = {split: D.load_split(split, root) for split in C.SPLITS}
    D.verify_split_isolation(splits)
    statistics = D.load_statistics(root)
    hierarchy = H.load_hierarchy(root)
    if hierarchy.sizes != [2746, 1373, 344, 86]:
        raise ValueError("Hierarchy verification failed")
    return {
        "status": "passed",
        "split_scans": {name: len(value.scan_ids) for name, value in splits.items()},
        "split_subjects": {name: len(set(value.subject_ids.astype(str))) for name, value in splits.items()},
        "hierarchy_sizes": hierarchy.sizes,
        "coordinate_scale_mm": float(statistics["coordinate_scale_mm"]),
        "velocity_scale_mm_per_year": float(statistics["velocity_scale_mm_per_year"]),
    }


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    if args.verify_only:
        print(json.dumps(verify_prepared(root), indent=2))
        return 0
    manifest = C.read_manifest()
    summaries = [prepare_split(split, root, manifest, args.overwrite) for split in C.SPLITS]
    hierarchy_path = H.save_hierarchy(root, overwrite=args.overwrite)
    train = D.load_split("train", root)
    train_rows = C.load_pairs("train", train.scan_ids, train.subject_ids, train.labels.numpy())
    faces = np.load(C.AE_BULK_ROOT / "cache" / "adni_faces.npy").astype(np.int64)
    statistics = training_statistics(train, faces, train_rows)
    C.atomic_torch_save(D.statistics_path(root), statistics)
    contract = {
        "schema_version": 1,
        "task": "direct_mesh_cocycle_spiral_unet",
        "source_manifest": str(C.SOURCE_MANIFEST),
        "source_manifest_sha256": C.sha256(C.SOURCE_MANIFEST),
        "source_sequence_root": str(C.SOURCE_SEQUENCE_ROOT),
        "source_meshes_modified": False,
        "subject_level_splits": True,
        "strict_no_mci": True,
        "topology_hash": manifest[0]["correspondence_topology_hash"],
        "vertex_count": C.VERTEX_COUNT,
        "face_count": C.FACE_COUNT,
        "hierarchy": str(hierarchy_path),
        "splits": summaries,
    }
    C.atomic_json(root / "cache" / "dataset_contract.json", contract)
    result = verify_prepared(root)
    C.atomic_json(root / "cache" / "preparation_validation.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

