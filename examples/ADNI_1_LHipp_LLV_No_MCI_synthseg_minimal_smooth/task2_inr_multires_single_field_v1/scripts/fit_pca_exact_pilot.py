#!/usr/bin/env python3
"""Refit PCA on ONLY the 401 exact-pilot training scans, for a fair-data comparison.

Every PCA number quoted against the exact-SDF INR arms (E1/E2/E3, multires_exact,
compact_exact) so far actually came from ``hippocampus_pca_cocycle_v4/pca/model``,
whose own ``pca_model_summary.json`` records ``fit_scans: 2037`` -- the full QC
cohort's training split.  The INR never saw more than the 401 scans in the
exact-SDF pilot manifest, because exact point-to-triangle SDF labels only exist
for that subset.  So every "PCA vs INR" comparison so far gave PCA a 5.1x data
advantage on top of its structural ones (vertex correspondence, linear-optimal
fit).  This script removes that confound: fit PCA from the same 401 scans, score
it with the exact same ``surface_metrics`` function used for the INR, on the
exact same 100 val / 100 test scan_ids.

Method: Gram-matrix PCA (n_train=401 << 8238 features), following
``task_spiral_ae_v1/scripts/fit_pca_baseline.py``.  Vertices are loaded with
trimesh(process=False) rather than that script's openmesh reader; process=False
disables any vertex merge/reorder, so file order -- and therefore the vertex
correspondence the whole pipeline depends on -- is preserved identically.  All
601 exact-pilot scans share one ``correspondence_topology_hash`` (verified
before fitting), so stacking their vertices is valid.

Writes: model/{mean,components_<k>,eigenvalues}.npy, metrics/per_scan_metrics.csv,
pca_401_summary.json.  Read-only with respect to every existing run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multires_common import load_manifest, require_bulk_path, write_csv, write_json
from periodic_evaluate_multires import load_mesh, surface_metrics

EXACT_PILOT_MANIFEST = (
    "/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/"
    "exact_sdf_relabel_v1/manifests/hippocampus_pilot_exact.csv"
)
DEFAULT_OUTPUT = (
    "/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/"
    "exact_sdf_relabel_v1/runs/pca_401_train"
)
# The INR arms this compares against all used 30000 sampled points and this
# component count; matching both keeps the comparison direct, not just close.
DEFAULT_COMPONENTS = (8, 16, 32, 64, 100, 128, 150)
DEFAULT_SURFACE_POINTS = 30000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=EXACT_PILOT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--components", nargs="+", type=int, default=list(DEFAULT_COMPONENTS))
    parser.add_argument("--surface-points", type=int, default=DEFAULT_SURFACE_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_split(rows: list[dict[str, str]], split: str) -> tuple[list[dict[str, str]], np.ndarray]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows for split={split!r} in {EXACT_PILOT_MANIFEST}")
    vertices = np.stack(
        [
            np.asarray(load_mesh(row["mesh_path_mm"]).vertices, dtype=np.float64)
            for row in selected
        ]
    )
    return selected, vertices


def fit_pca(train_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gram-matrix PCA, exact for n_samples << n_features. Returns (mean, components, eigenvalues)."""
    mean = train_flat.mean(axis=0)
    centered = train_flat - mean
    gram = np.asarray(centered @ centered.T, dtype=np.float64) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors.T @ centered
    norms = np.linalg.norm(components, axis=1, keepdims=True)
    components = components / np.maximum(norms, 1.0e-12)
    return mean, components, eigenvalues


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir)
    rows = load_manifest(args.manifest)

    topology = {row["correspondence_topology_hash"] for row in rows}
    if len(topology) != 1:
        raise ValueError(f"Exact-pilot scans do not share one topology: {topology}")
    faces = load_mesh(rows[0]["mesh_path_mm"]).faces

    print(f"[pca-401] loading vertices ({topology.pop()[:12]}...)", flush=True)
    started = time.time()
    train_rows, train_vertices = load_split(rows, "train")
    val_rows, val_vertices = load_split(rows, "val")
    test_rows, test_vertices = load_split(rows, "test")
    print(
        f"[pca-401] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"loaded in {time.time() - started:.1f}s",
        flush=True,
    )

    n_vertices = train_vertices.shape[1]
    train_flat = train_vertices.reshape(len(train_rows), -1)
    print(f"[pca-401] fitting on {len(train_rows)} train scans, {train_flat.shape[1]} features", flush=True)
    mean, components, eigenvalues = fit_pca(train_flat)
    total_variance = float(eigenvalues.sum())

    max_k = max(args.components)
    if max_k > components.shape[0]:
        raise ValueError(
            f"Requested {max_k} components but only {components.shape[0]} scans-1 available."
        )

    per_scan_rows = []
    summary_rows = []
    for k in sorted(args.components):
        basis = components[:k]
        cev = float(eigenvalues[:k].sum() / total_variance)
        for split, split_rows, split_vertices in (
            ("val", val_rows, val_vertices),
            ("test", test_rows, test_vertices),
        ):
            flat = split_vertices.reshape(len(split_rows), -1)
            coefficients = (flat - mean) @ basis.T
            reconstructed = (coefficients @ basis + mean).reshape(len(split_rows), n_vertices, 3)
            per_metric = []
            for row, recon_vertices in zip(split_rows, reconstructed):
                ground_truth = load_mesh(row["mesh_path_mm"])
                predicted = trimesh.Trimesh(vertices=recon_vertices, faces=faces, process=False)
                seed = args.seed
                metrics = surface_metrics(ground_truth, predicted, args.surface_points, seed)
                per_metric.append(metrics)
                per_scan_rows.append(
                    {
                        "components": k,
                        "split": split,
                        "scan_id": row["scan_id"],
                        "subject_id": row["subject_id"],
                        **metrics,
                    }
                )
            mean_assd = float(np.mean([m["assd_mm"] for m in per_metric]))
            mean_hd95 = float(np.mean([m["hd95_mm"] for m in per_metric]))
            summary_rows.append(
                {
                    "components": k,
                    "split": split,
                    "cumulative_explained_variance_train": cev,
                    "assd_mm_mean": mean_assd,
                    "hd95_mm_mean": mean_hd95,
                }
            )
            print(
                f"[pca-401] k={k:4d} {split:5s} assd_mm={mean_assd:.6f} "
                f"hd95_mm={mean_hd95:.6f} cev={cev:.6f}",
                flush=True,
            )

    model_dir = require_bulk_path(output / "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "mean.npy", mean.astype(np.float32))
    np.save(model_dir / "faces.npy", faces)
    np.save(model_dir / "eigenvalues.npy", eigenvalues.astype(np.float64))
    for k in sorted(args.components):
        np.save(model_dir / f"components_{k}.npy", components[:k].astype(np.float32))

    write_csv(output / "per_scan_metrics.csv", per_scan_rows)
    write_csv(output / "summary_by_k.csv", summary_rows)
    write_json(
        output / "pca_401_summary.json",
        {
            "manifest": args.manifest,
            "fit_scans": len(train_rows),
            "fit_subjects": len({row["subject_id"] for row in train_rows}),
            "features": int(train_flat.shape[1]),
            "components_scored": sorted(args.components),
            "surface_points": args.surface_points,
            "note": (
                "Fit on the 401-scan exact-SDF pilot train split only, NOT the "
                "2037-scan full-cohort PCA in hippocampus_pca_cocycle_v4. Comparable "
                "1:1 against the INR arms trained on the same manifest."
            ),
        },
    )
    print(f"[pca-401] wrote model + metrics to {output}")


if __name__ == "__main__":
    main()
