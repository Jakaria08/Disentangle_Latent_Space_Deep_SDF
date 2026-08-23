#!/usr/bin/env python3
"""Fit train-only PCA to the same bbox-centred left meshes used by the SDF model."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import numpy as np

from calsnic_common import (
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    load_sdf_space_mesh,
    read_manifest,
    require_bulk_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / "pca" / "matched_left_train173"),
    )
    parser.add_argument("--requested-components", nargs="+", type=int, default=(128, 172, 256, 512))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def load_matrix(rows: list[dict[str, str]], faces_reference=None):
    arrays = []
    faces = faces_reference
    shape = None
    for number, row in enumerate(rows, start=1):
        mesh = load_sdf_space_mesh(row)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        current_faces = np.asarray(mesh.faces, dtype=np.int32)
        if shape is None:
            shape = vertices.shape
            if faces is None:
                faces = current_faces
        if vertices.shape != shape or not np.array_equal(current_faces, faces):
            raise ValueError(f"PCA topology/correspondence mismatch for {row['scan_id']}.")
        arrays.append(vertices.reshape(-1))
        if number % 25 == 0 or number == len(rows):
            print(f"Loaded PCA mesh {number}/{len(rows)}", flush=True)
    return np.stack(arrays).astype(np.float32), np.asarray(faces, dtype=np.int32), shape


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir, "PCA output")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty PCA output {output}; pass --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < 3:
        raise ValueError("PCA needs at least three training meshes.")
    matrix, faces, shape = load_matrix(train)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = matrix - mean[None, :]
    gram = np.asarray(centered @ centered.T, dtype=np.float64) / (len(train) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    tolerance = max(float(eigenvalues[0]) * 1.0e-8, 1.0e-12)
    rank = min(len(train) - 1, int(np.count_nonzero(eigenvalues > tolerance)))
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    singular = np.sqrt(eigenvalues * (len(train) - 1))
    components = (eigenvectors.T @ centered / singular[:, None]).astype(np.float32)
    scores_train = (centered @ components.T).astype(np.float32)

    assert shape is not None
    atomic_save(output / "mean_vertices.npy", mean.reshape(shape))
    atomic_save(output / "components.npy", components.reshape(rank, *shape))
    atomic_save(output / "faces.npy", faces)
    atomic_save(output / "explained_variance.npy", eigenvalues.astype(np.float32))
    atomic_save(output / "train_scores.npy", scores_train)

    summaries = []
    per_scan = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        values = matrix if split == "train" else load_matrix(split_rows, faces)[0]
        delta = values - mean[None, :]
        scores = scores_train if split == "train" else (delta @ components.T).astype(np.float32)
        for requested in args.requested_components:
            effective = min(int(requested), rank)
            reconstructed = mean[None, :] + scores[:, :effective] @ components[:effective]
            sdf_rmse = np.sqrt(
                np.mean(np.sum((values.reshape(len(values), -1, 3) - reconstructed.reshape(len(values), -1, 3)) ** 2, axis=2), axis=1)
            )
            mm_rmse = np.asarray(
                [error / float(row["scaled_from_mm_scale"]) for error, row in zip(sdf_rmse, split_rows)]
            )
            summaries.append(
                {
                    "split": split,
                    "requested_components": int(requested),
                    "effective_components": effective,
                    "count": len(split_rows),
                    "mean_corresponded_vertex_rmse_mm": float(mm_rmse.mean()),
                    "median_corresponded_vertex_rmse_mm": float(np.median(mm_rmse)),
                    "p95_corresponded_vertex_rmse_mm": float(np.quantile(mm_rmse, 0.95)),
                }
            )
            for row, error_sdf, error_mm in zip(split_rows, sdf_rmse, mm_rmse):
                per_scan.append(
                    {
                        "scan_id": row["scan_id"],
                        "split": split,
                        "requested_components": int(requested),
                        "effective_components": effective,
                        "corresponded_vertex_rmse_sdf_units": float(error_sdf),
                        "corresponded_vertex_rmse_mm": float(error_mm),
                    }
                )
    atomic_write_csv(output / "reconstruction_summary.csv", summaries)
    atomic_write_csv(output / "reconstruction_per_scan.csv", per_scan)
    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "training_meshes": len(train),
        "vertices": int(shape[0]),
        "faces": int(len(faces)),
        "rank": rank,
        "maximum_rank": len(train) - 1,
        "coordinate_space": "PreprocessMesh bbox-centred scaled OBJ coordinates",
        "comparison_role": "target-projected train-only oracle reconstruction baseline",
        "requested_components": list(args.requested_components),
        "files": {
            "mean": "mean_vertices.npy",
            "components": "components.npy",
            "faces": "faces.npy",
            "summary": "reconstruction_summary.csv",
        },
    }
    atomic_write_json(output / "metadata.json", metadata)
    print(f"Matched PCA complete: rank={rank}, output={output}")


if __name__ == "__main__":
    main()
