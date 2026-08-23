#!/usr/bin/env python3
"""SPHARM-then-PCA hybrid: a spectral pre-smoothing regularizer, tested against plain PCA.

Motivation (see the task README for the full writeup): plain PCA on raw vertices overfits badly at
its rank-172 ceiling (train RMSE ~0.00003mm vs val RMSE 3.86mm, from only 173 training scans against
122,886 raw dimensions). This script tests whether fitting a *moderate*-degree SPHARM basis first
(a spectral low-pass filter, applied identically to every scan — no data-fitting, no leakage) and
then running train-only PCA on the resulting coefficient vectors, instead of on raw vertices,
improves validation/test generalization. A near-lossless (very high degree) SPHARM step would just
be an expensive change of basis and is not expected to help; a genuinely truncating, moderate degree
is the version worth testing, which is what this script sweeps.

Unlike ``fit_spharm_basis.py``, the PCA step here *is* train-only and *does* need the same leakage
discipline ``fit_matched_pca.py`` uses: val/test scans are projected onto the train-fitted PCA basis,
never used to fit it. The SPHARM coefficient-fitting step underneath remains leakage-free (every scan
fit independently against the fixed geometric basis), exactly as in ``fit_spharm_basis.py``.
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
    "smoothing_degree",
    "pca_rank_requested",
    "pca_rank_effective",
    "count",
    "mean_corresponded_vertex_rmse_mm",
    "median_corresponded_vertex_rmse_mm",
    "p95_corresponded_vertex_rmse_mm",
]
PER_SCAN_FIELDS = [
    "scan_id",
    "split",
    "smoothing_degree",
    "pca_rank_requested",
    "pca_rank_effective",
    "corresponded_vertex_rmse_sdf_units",
    "corresponded_vertex_rmse_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument("--spharm-dir", default=str(DEFAULT_SPHARM_DIR), help="Existing embedding directory (angles/faces reused, not rebuilt).")
    parser.add_argument("--output-dir", default=str(DEFAULT_SPHARM_DIR.parent / "pca_hybrid"))
    parser.add_argument("--smoothing-degrees", nargs="+", type=int, default=(16, 24, 32, 48, 64))
    parser.add_argument("--pca-ranks", nargs="+", type=int, default=(16, 32, 64, 100, 128, 172))
    parser.add_argument("--overwrite", action="store_true")
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


def train_only_pca(train_matrix: np.ndarray) -> dict:
    """Mirrors task_exact_multires_cortex_v1/scripts/fit_matched_pca.py's eigh-based dual PCA,
    applied here to SPHARM coefficient vectors instead of raw vertex vectors."""
    n = train_matrix.shape[0]
    mean = train_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = train_matrix - mean[None, :]
    gram = np.asarray(centered @ centered.T, dtype=np.float64) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    tolerance = max(float(eigenvalues[0]) * 1.0e-8, 1.0e-12)
    rank = min(n - 1, int(np.count_nonzero(eigenvalues > tolerance)))
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    singular = np.sqrt(eigenvalues * (n - 1))
    components = (eigenvectors.T @ centered / singular[:, None]).astype(np.float32)
    scores_train = (centered @ components.T).astype(np.float32)
    return {"mean": mean, "components": components, "rank": rank, "scores_train": scores_train}


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir, "SPHARM+PCA hybrid output")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output {output}; pass --overwrite.")

    spharm_dir = Path(args.spharm_dir)
    angles = np.load(spharm_dir / "sphere_angles.npy")
    embedding_faces = np.load(spharm_dir / "faces.npy")
    vertex_count = angles.shape[0]

    rows = read_manifest(args.manifest)
    print("Loading all manifest meshes...")
    started = time.time()
    vertices_by_scan: dict[str, np.ndarray] = {}
    scale_by_scan: dict[str, float] = {}
    split_by_scan: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        mesh = load_sdf_space_mesh(row)
        if not np.array_equal(np.asarray(mesh.faces, dtype=np.int64), embedding_faces):
            raise ValueError(f"{row['scan_id']}: face connectivity differs from the embedding's reference topology.")
        vertices_by_scan[row["scan_id"]] = np.asarray(mesh.vertices, dtype=np.float64)
        scale_by_scan[row["scan_id"]] = float(row["scaled_from_mm_scale"])
        split_by_scan[row["scan_id"]] = row["split"]
        if number % 25 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] loaded, elapsed {time.time() - started:.1f}s", flush=True)

    train_scan_ids = [row["scan_id"] for row in rows if row["split"] == "train"]
    all_scan_ids = [row["scan_id"] for row in rows]

    summaries: list[dict] = []
    per_scan: list[dict] = []
    for degree in sorted(set(args.smoothing_degrees)):
        k_per_channel = (degree + 1) ** 2
        print(f"--- smoothing degree {degree} (K={k_per_channel} per channel) ---")
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], degree)
        solver = se.SharedBasisSolver(basis)

        coeffs_by_scan: dict[str, np.ndarray] = {}
        for scan_id in all_scan_ids:
            coefficients = solver.fit(vertices_by_scan[scan_id])  # (K, 3)
            coeffs_by_scan[scan_id] = coefficients.reshape(-1).astype(np.float32)  # flatten (3K,)

        train_matrix = np.stack([coeffs_by_scan[s] for s in train_scan_ids], axis=0)
        pca = train_only_pca(train_matrix)
        print(f"  train-only PCA on coefficients: achieved rank {pca['rank']} (requested up to {max(args.pca_ranks)})")

        train_index = {s: i for i, s in enumerate(train_scan_ids)}
        degree_dir = output / f"degree_{degree}"
        degree_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(degree_dir / "mean_coefficients.npy", pca["mean"])
        atomic_save_npy(degree_dir / "components.npy", pca["components"])
        (degree_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "smoothing_degree": degree,
                    "k_per_channel": k_per_channel,
                    "coefficient_dim": 3 * k_per_channel,
                    "achieved_rank": pca["rank"],
                    "train_scans": len(train_scan_ids),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        by_split_rank: dict[tuple[str, int], list[float]] = {}
        for scan_id in all_scan_ids:
            coeffs_flat = coeffs_by_scan[scan_id]
            if scan_id in train_index:
                scores = pca["scores_train"][train_index[scan_id]]
            else:
                scores = (coeffs_flat - pca["mean"]) @ pca["components"].T
            for requested in sorted(set(args.pca_ranks)):
                effective = min(int(requested), pca["rank"])
                reconstructed_flat = pca["mean"] + scores[:effective] @ pca["components"][:effective]
                reconstructed_coeffs = reconstructed_flat.reshape(k_per_channel, 3)
                reconstructed_vertices = solver.reconstruct(reconstructed_coeffs)
                target_vertices = vertices_by_scan[scan_id]
                sdf_rmse = float(np.sqrt(np.mean(np.sum((target_vertices - reconstructed_vertices) ** 2, axis=1))))
                mm_rmse = sdf_rmse / scale_by_scan[scan_id]
                split = split_by_scan[scan_id]
                per_scan.append(
                    {
                        "scan_id": scan_id,
                        "split": split,
                        "smoothing_degree": degree,
                        "pca_rank_requested": requested,
                        "pca_rank_effective": effective,
                        "corresponded_vertex_rmse_sdf_units": sdf_rmse,
                        "corresponded_vertex_rmse_mm": mm_rmse,
                    }
                )
                by_split_rank.setdefault((split, requested), []).append(mm_rmse)

        for (split, requested), values in by_split_rank.items():
            values_arr = np.asarray(values)
            effective = min(int(requested), pca["rank"])
            summaries.append(
                {
                    "split": split,
                    "smoothing_degree": degree,
                    "pca_rank_requested": requested,
                    "pca_rank_effective": effective,
                    "count": len(values_arr),
                    "mean_corresponded_vertex_rmse_mm": float(values_arr.mean()),
                    "median_corresponded_vertex_rmse_mm": float(np.median(values_arr)),
                    "p95_corresponded_vertex_rmse_mm": float(np.quantile(values_arr, 0.95)),
                }
            )

    atomic_write_csv(output / "reconstruction_summary.csv", summaries, SUMMARY_FIELDS)
    atomic_write_csv(output / "reconstruction_per_scan.csv", per_scan, PER_SCAN_FIELDS)
    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "embedding_dir": str(spharm_dir),
        "vertices": vertex_count,
        "smoothing_degrees_swept": sorted(set(args.smoothing_degrees)),
        "pca_ranks_swept": sorted(set(args.pca_ranks)),
        "method": "per-scan SPHARM coefficient fit (geometric, no leakage) -> train-only PCA on "
        "coefficients (leakage-guarded, val/test projected onto train basis) -> SPHARM reconstruct",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"SPHARM+PCA hybrid sweep complete: output={output}")


if __name__ == "__main__":
    main()
