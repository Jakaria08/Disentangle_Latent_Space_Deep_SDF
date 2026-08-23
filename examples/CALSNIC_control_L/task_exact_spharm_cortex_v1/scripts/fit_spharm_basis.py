#!/usr/bin/env python3
"""Build the shared spherical embedding once, then fit/reconstruct SPHARM per scan.

Unlike PCA, the SPHARM basis is purely geometric (derived from mesh
connectivity, not from any subject's data), so there is no train/test leakage
concern: every split is fit and reconstructed the same way. This sweeps a
range of harmonic degrees and reports corresponded-vertex RMSE in the same
units and CSV schema as ``fit_matched_pca.py`` in the sibling multires task,
so the two curves overlay directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from spharm_common import (
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_SPHARM_DIR,
    load_sdf_space_mesh,
    read_manifest,
    require_bulk_path,
    sha256_file,
)
import spharm_embedding as se


SUMMARY_FIELDS = [
    "split",
    "degree",
    "requested_components",
    "effective_components",
    "count",
    "mean_corresponded_vertex_rmse_mm",
    "median_corresponded_vertex_rmse_mm",
    "p95_corresponded_vertex_rmse_mm",
]
PER_SCAN_FIELDS = [
    "scan_id",
    "split",
    "degree",
    "requested_components",
    "effective_components",
    "corresponded_vertex_rmse_sdf_units",
    "corresponded_vertex_rmse_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_SPHARM_DIR))
    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=(4, 8, 12, 16, 20, 24, 32),
        help="SPHARM degrees to sweep; total scalar coefficients per subject is 3*(degree+1)**2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-area-ratio", type=float, default=1000.0)
    return parser.parse_args()


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_or_build_embedding(output: Path, reference_faces: np.ndarray, vertex_count: int, max_area_ratio: float):
    angles_path = output / "sphere_angles.npy"
    xyz_path = output / "sphere_xyz.npy"
    faces_path = output / "faces.npy"
    anchors_path = output / "anchors.json"
    if angles_path.is_file() and xyz_path.is_file() and faces_path.is_file() and anchors_path.is_file():
        stored_faces = np.load(faces_path)
        if not np.array_equal(stored_faces, reference_faces):
            raise ValueError(f"Stored embedding faces at {faces_path} do not match the reference topology.")
        angles = np.load(angles_path)
        xyz = np.load(xyz_path)
        anchors = json.loads(anchors_path.read_text())
        print(f"Loaded existing spherical embedding from {output}.")
        return angles, xyz, anchors

    print("Building spherical embedding (one-time, topology-only)...")
    angles, xyz, anchors = se.build_spherical_embedding(reference_faces, vertex_count)
    signed = se.triangle_signed_areas(xyz, reference_faces)
    if np.any(signed <= 0):
        raise ValueError(f"Embedding has {int(np.sum(signed <= 0))} flipped/degenerate triangles.")
    areas = se.triangle_chord_areas(xyz, reference_faces)
    ratio = float(areas.max() / areas.min())
    if ratio > max_area_ratio:
        raise ValueError(f"Embedding area-distortion ratio {ratio:.1f} exceeds --max-area-ratio {max_area_ratio}.")
    output.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(angles_path, angles)
    atomic_save_npy(xyz_path, xyz)
    atomic_save_npy(faces_path, reference_faces)
    anchors_serializable = {
        "vertex_ids": np.asarray(anchors["vertex_ids"]).tolist(),
        "xyz": np.asarray(anchors["xyz"]).tolist(),
    }
    temporary = anchors_path.with_name(f".{anchors_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(anchors_serializable, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, anchors_path)
    print(f"Embedding valid: 0 flips, area-distortion ratio {ratio:.1f}. Saved to {output}.")
    return angles, xyz, anchors_serializable


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir, "SPHARM output")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        existing = {p.name for p in output.iterdir()}
        embedding_files = {"sphere_angles.npy", "sphere_xyz.npy", "faces.npy", "anchors.json"}
        if not embedding_files.issubset(existing):
            raise FileExistsError(f"Refusing non-empty SPHARM output {output}; pass --overwrite.")
    rows = read_manifest(args.manifest)
    if len(rows) < 1:
        raise ValueError("Manifest has no rows.")

    reference_faces: np.ndarray | None = None
    matrices: dict[str, np.ndarray] = {}
    scale_by_scan: dict[str, float] = {}
    print("Loading all manifest meshes (train + val + test; no leakage concern for a geometric basis)...")
    started = time.time()
    for number, row in enumerate(rows, start=1):
        mesh = load_sdf_space_mesh(row)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if reference_faces is None:
            reference_faces = faces
        elif not np.array_equal(faces, reference_faces):
            raise ValueError(f"{row['scan_id']}: face connectivity differs from the reference topology.")
        matrices[row["scan_id"]] = np.asarray(mesh.vertices, dtype=np.float64)
        scale_by_scan[row["scan_id"]] = float(row["scaled_from_mm_scale"])
        if number % 25 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] loaded, elapsed {time.time() - started:.1f}s", flush=True)

    assert reference_faces is not None
    vertex_count = int(reference_faces.max()) + 1
    angles, xyz, anchors = load_or_build_embedding(output, reference_faces, vertex_count, args.max_area_ratio)

    summaries: list[dict] = []
    per_scan: list[dict] = []
    for degree in sorted(set(args.degrees)):
        k_per_channel = (degree + 1) ** 2
        total_components = 3 * k_per_channel
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], degree)
        solver = se.SharedBasisSolver(basis)
        by_split: dict[str, list[float]] = {}
        for row in rows:
            vertices = matrices[row["scan_id"]]
            coefficients = solver.fit(vertices)
            reconstructed = solver.reconstruct(coefficients)
            sdf_rmse = float(
                np.sqrt(np.mean(np.sum((vertices - reconstructed) ** 2, axis=1)))
            )
            mm_rmse = sdf_rmse / scale_by_scan[row["scan_id"]]
            per_scan.append(
                {
                    "scan_id": row["scan_id"],
                    "split": row["split"],
                    "degree": degree,
                    "requested_components": total_components,
                    "effective_components": total_components,
                    "corresponded_vertex_rmse_sdf_units": sdf_rmse,
                    "corresponded_vertex_rmse_mm": mm_rmse,
                }
            )
            by_split.setdefault(row["split"], []).append(mm_rmse)
        for split, values in by_split.items():
            values_arr = np.asarray(values)
            summaries.append(
                {
                    "split": split,
                    "degree": degree,
                    "requested_components": total_components,
                    "effective_components": total_components,
                    "count": len(values_arr),
                    "mean_corresponded_vertex_rmse_mm": float(values_arr.mean()),
                    "median_corresponded_vertex_rmse_mm": float(np.median(values_arr)),
                    "p95_corresponded_vertex_rmse_mm": float(np.quantile(values_arr, 0.95)),
                }
            )
        print(f"degree={degree} (K={total_components} scalar coefficients) done.")

    atomic_write_csv(output / "reconstruction_summary.csv", summaries, SUMMARY_FIELDS)
    atomic_write_csv(output / "reconstruction_per_scan.csv", per_scan, PER_SCAN_FIELDS)
    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "scans": len(rows),
        "vertices": vertex_count,
        "faces": int(len(reference_faces)),
        "coordinate_space": "PreprocessMesh bbox-centred scaled OBJ coordinates",
        "comparison_role": "geometric (non-data-fitted) basis; every split fit independently, no leakage",
        "embedding_method": "anchored uniform-weight harmonic solve (12 icosahedral Dirichlet anchors)",
        "degrees_swept": sorted(set(args.degrees)),
        "files": {
            "sphere_angles": "sphere_angles.npy",
            "sphere_xyz": "sphere_xyz.npy",
            "faces": "faces.npy",
            "anchors": "anchors.json",
            "summary": "reconstruction_summary.csv",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"SPHARM fit complete: degrees={sorted(set(args.degrees))}, output={output}")


if __name__ == "__main__":
    main()
