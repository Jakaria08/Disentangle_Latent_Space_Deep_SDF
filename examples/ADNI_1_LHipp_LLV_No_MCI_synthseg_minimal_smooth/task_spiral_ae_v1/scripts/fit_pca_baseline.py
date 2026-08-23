#!/usr/bin/env python3
"""Refit PCA on the train split and score it at 128 / 256 components.

The stored model in hippocampus_pca_cocycle_v4 caps at 150 components, so the 256-component
comparison this task needs does not exist yet. This script regenerates the basis and, as a
regression check on splits + metric, asserts that PCA-128 reproduces the published validation
number (0.033667) before writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spiral_common as sc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components", nargs="+", type=int, default=[32, 64, 128, 150, 256])
    p.add_argument("--output-dir", default=str(sc.OUTPUT_ROOT / "pca"))
    p.add_argument("--skip-reproduction-check", action="store_true")
    return p.parse_args()


def fit_pca(train_flat):
    """Gram-matrix PCA (n_samples << n_features). Returns (mean, components, eigenvalues)."""
    mean = train_flat.mean(axis=0)
    centered = train_flat - mean
    gram = np.asarray(centered @ centered.T, dtype=np.float64) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors.T @ centered
    norms = np.linalg.norm(components, axis=1, keepdims=True)
    components = components / np.maximum(norms, 1e-12)
    return mean, components, eigenvalues


def main():
    args = parse_args()
    out_dir = sc.makedirs(sc.require_bulk_path(args.output_dir, "PCA output"))
    rows = sc.read_manifest()
    faces = sc.load_faces(rows=rows)

    splits = {s: sc.load_split_vertices(s, rows=rows) for s in sc.SPLITS}
    flat = {s: v.reshape(len(v), -1).astype(np.float64) for s, v in splits.items()}
    n_vertices = splits["train"].shape[1]

    print(f"[pca] fitting on {len(flat['train'])} train scans, {flat['train'].shape[1]} features")
    mean, components, eigenvalues = fit_pca(flat["train"])
    total_variance = eigenvalues.sum()

    max_k = max(args.components)
    if max_k > components.shape[0]:
        raise ValueError(f"Requested {max_k} components but only {components.shape[0]} available")

    metric_rows = []
    for k in sorted(args.components):
        basis = components[:k]
        cev = float(eigenvalues[:k].sum() / total_variance)
        for split in sc.SPLITS:
            data = flat[split]
            coeffs = (data - mean) @ basis.T
            recon = coeffs @ basis + mean
            metrics = sc.reconstruction_metrics(
                recon.reshape(len(data), n_vertices, 3),
                data.reshape(len(data), n_vertices, 3),
                faces=faces,
            )
            metric_rows.append(
                {
                    "structure": "left_hippocampus",
                    "components": k,
                    "split": split,
                    "cumulative_explained_variance_train": cev,
                    **metrics,
                }
            )
            print(
                f"[pca] k={k:4d} {split:5s} rmse_mm={metrics['vertex_rmse_mm_mean']:.6f} "
                f"cev={cev:.6f}"
            )

    # Regression check: splits + metric must still reproduce the published table.
    if not args.skip_reproduction_check:
        ref = [r for r in metric_rows if r["components"] == 128 and r["split"] == "val"]
        if not ref:
            raise ValueError("PCA-128 must be included so the reproduction check can run.")
        got = ref[0]["vertex_rmse_mm_mean"]
        delta = abs(got - sc.PCA128_VAL_REFERENCE)
        if delta > sc.PCA_REPRODUCTION_TOL:
            raise SystemExit(
                f"PCA-128 val RMSE {got:.6f} does not reproduce published "
                f"{sc.PCA128_VAL_REFERENCE:.6f} (delta {delta:.2e}). "
                "Splits or metric convention have drifted; refusing to continue."
            )
        print(f"[pca] reproduction check OK: {got:.6f} vs published {sc.PCA128_VAL_REFERENCE:.6f}")

    model_dir = sc.makedirs(out_dir / "model")
    sc.atomic_save_npy(model_dir / "mean.npy", mean.astype(np.float32))
    for k in sorted(args.components):
        sc.atomic_save_npy(model_dir / f"components_{k}.npy", components[:k].astype(np.float32))
    sc.atomic_save_npy(model_dir / "eigenvalues.npy", eigenvalues.astype(np.float64))

    metrics_dir = sc.makedirs(out_dir / "metrics")
    sc.atomic_write_csv(metrics_dir / "pca_reconstruction_summary.csv", metric_rows)
    sc.atomic_write_json(
        out_dir / "pca_model_summary.json",
        {
            "fit_scans": int(len(flat["train"])),
            "features": int(flat["train"].shape[1]),
            "components_scored": sorted(args.components),
            "reproduction_check_val_128": float(
                [r for r in metric_rows if r["components"] == 128 and r["split"] == "val"][0][
                    "vertex_rmse_mm_mean"
                ]
            ),
            "published_val_128": sc.PCA128_VAL_REFERENCE,
        },
    )
    print(f"[pca] wrote model + metrics to {out_dir}")


if __name__ == "__main__":
    main()
