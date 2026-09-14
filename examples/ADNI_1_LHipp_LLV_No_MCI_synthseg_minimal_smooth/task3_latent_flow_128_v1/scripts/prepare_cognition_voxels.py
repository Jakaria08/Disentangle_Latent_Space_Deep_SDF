#!/usr/bin/env python3
"""Precompute solid, shared-grid hippocampal masks for the PCA128 CNN."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import brainode_cognition as B
import common as C


_WORKER_FACES: np.ndarray | None = None
_WORKER_GRID: B.VoxelGrid | None = None


def _initialize_worker(faces: np.ndarray, grid_mapping: dict[str, Any]) -> None:
    global _WORKER_FACES, _WORKER_GRID
    _WORKER_FACES = np.asarray(faces, dtype=np.int64)
    _WORKER_GRID = B.VoxelGrid.from_mapping(grid_mapping)


def _voxelize_worker(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, dict[str, float]]:
    index, vertices = item
    if _WORKER_FACES is None or _WORKER_GRID is None:
        raise RuntimeError("Voxel worker was not initialized")
    mask, quality = _WORKER_GRID.voxelize(vertices, _WORKER_FACES)
    return index, mask, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--padding-voxels", type=int, default=2)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--splits", nargs="+", choices=C.SPLITS, default=list(C.SPLITS))
    parser.add_argument("--max-scans", type=int, default=None, help="Diagnostic bound; incompatible with persistent writes.")
    parser.add_argument("--dry-run", action="store_true", help="Voxelize two training shapes and write nothing.")
    return parser.parse_args()


def voxelize_split(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: B.VoxelGrid,
    workers: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    count = len(vertices)
    masks = np.zeros((count, grid.resolution, grid.resolution, grid.resolution), dtype=np.uint8)
    quality: list[dict[str, float] | None] = [None] * count
    items = ((index, np.asarray(vertices[index], dtype=np.float32)) for index in range(count))
    if int(workers) == 1:
        _initialize_worker(faces, grid.mapping())
        iterator = map(_voxelize_worker, items)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_worker,
            initargs=(faces, grid.mapping()),
        )
        iterator = executor.map(_voxelize_worker, items, chunksize=8)
    try:
        for completed, (index, mask, current) in enumerate(iterator, start=1):
            masks[index] = mask
            quality[index] = current
            if completed % 250 == 0 or completed == count:
                print(f"  voxelized {completed}/{count}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return masks, [item for item in quality if item is not None]


def quality_summary(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        "scans": float(len(records)),
        "occupied_voxels_min": float(min(item["occupied_voxels"] for item in records)),
        "occupied_voxels_mean": float(np.mean([item["occupied_voxels"] for item in records])),
        "clipped_local_voxels_total": float(sum(item["clipped_local_voxels"] for item in records)),
        "voxel_to_mesh_volume_ratio_mean": float(np.mean([item["voxel_to_mesh_volume_ratio"] for item in records])),
        "voxel_to_mesh_volume_ratio_min": float(min(item["voxel_to_mesh_volume_ratio"] for item in records)),
        "voxel_to_mesh_volume_ratio_max": float(max(item["voxel_to_mesh_volume_ratio"] for item in records)),
    }


def main() -> int:
    args = parse_args()
    if args.max_scans is not None and not args.dry_run:
        raise ValueError("--max-scans is diagnostic only; partial persistent archives are prohibited")
    registry = C.load_registry()
    train_vertices = C.cached_vertices("train", registry)
    grid = B.VoxelGrid.from_training_vertices(train_vertices, args.resolution, args.padding_voxels)
    pca_root = C.resolve_path(registry["representations"]["pca128"]["pca_model_root"])
    faces_path = pca_root / "faces.npy"
    faces = np.load(faces_path, allow_pickle=False)
    if args.dry_run:
        count = min(int(args.max_scans or 2), len(train_vertices))
        started = time.time()
        masks, quality = voxelize_split(train_vertices[:count], faces, grid, min(args.workers, count))
        if masks.shape != (count, args.resolution, args.resolution, args.resolution) or not np.all(masks.sum((1, 2, 3)) > 0):
            raise RuntimeError("Voxel dry-run produced invalid masks")
        print("VOXEL DRY RUN PASSED — solid masks generated; no files written.")
        print(json.dumps({"grid": grid.mapping(), "quality": quality_summary(quality), "seconds": time.time() - started}, indent=2))
        return 0

    destinations = [B.voxel_archive_path(registry, split, args.resolution) for split in args.splits]
    root = destinations[0].parent
    manifest_path = root / "manifest.json"
    existing = [path for path in destinations + [manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing voxel artifacts: {existing}")
    split_records: dict[str, Any] = {}
    started = time.time()
    for split, destination in zip(args.splits, destinations):
        print(f"{split}: loading source metadata and raw physical-mm meshes", flush=True)
        archive = C.load_archive("pca128", split, registry)
        vertices = C.cached_vertices(split, registry)
        if len(vertices) != len(archive["visit_scan_ids"]):
            raise ValueError(f"Vertex/archive length mismatch for {split}")
        masks, quality = voxelize_split(vertices, faces, grid, args.workers)
        qualities = {key: np.asarray([item[key] for item in quality], dtype=np.float32) for key in quality[0]}
        arrays = {
            "masks_packed": B.pack_masks(masks),
            "visit_scan_ids": archive["visit_scan_ids"],
            "visit_subject_ids": archive["visit_subject_ids"],
            "visit_label_ad": archive["visit_label_ad"].astype(np.int8),
            "visit_diagnoses": archive["visit_diagnoses"],
            "resolution": np.asarray(grid.resolution, dtype=np.int32),
            "pitch_mm": np.asarray(grid.pitch_mm, dtype=np.float32),
            "minimum_center_mm": np.asarray(grid.minimum_center_mm, dtype=np.float32),
            "padding_voxels": np.asarray(grid.padding_voxels, dtype=np.int32),
            **qualities,
        }
        C.atomic_npz(destination, arrays)
        split_records[split] = {
            "path": str(destination),
            "sha256": C.sha256(destination),
            "subjects": int(len(np.unique(archive["visit_subject_ids"].astype(str)))),
            "scans": int(len(vertices)),
            "cn_scans": int(np.count_nonzero(archive["visit_label_ad"] == 0)),
            "ad_scans": int(np.count_nonzero(archive["visit_label_ad"] == 1)),
            "quality": quality_summary(quality),
        }
        print(f"WROTE {destination}", flush=True)
    manifest = {
        "schema_version": 1,
        "representation": "pca128",
        "anatomy": "left_hippocampus",
        "source": "raw correspondence mesh in final_ply_mm",
        "voxelization": "trimesh surface voxelization plus solid fill, projected to one train-bounded physical grid",
        "grid_bounds_fitted_from": "train raw meshes only",
        "grid": grid.mapping(),
        "faces_path": str(faces_path),
        "faces_sha256": C.sha256(faces_path),
        "split_records": split_records,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "scientific_contract": {
            "subject_split_unchanged": True,
            "test_labels_not_used_to_fit_grid": True,
            "mci_present": False,
            "source_meshes_modified": False,
        },
    }
    C.atomic_json(manifest_path, manifest)
    print(f"WROTE {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
