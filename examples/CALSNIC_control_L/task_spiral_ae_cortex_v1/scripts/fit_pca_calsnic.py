#!/usr/bin/env python3
"""Refit CALSNIC PCA in this pipeline and gate on reproducing the published numbers.

PCA is rank-limited here: 173 training meshes => at most 172 components. Requesting more
returns the 172-component result, so 3.844 mm (test) is the linear ceiling at ANY latent
budget -- which is why the K=256 autoencoder's honest comparator is PCA-172, not PCA-256.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calsnic_data as cd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components", nargs="+", type=int, default=[128, 172])
    p.add_argument("--skip-gate", action="store_true")
    return p.parse_args()


def fit_pca(train_flat):
    mean = train_flat.mean(axis=0)
    X = train_flat - mean
    gram = np.asarray(X @ X.T, dtype=np.float64) / (len(X) - 1)
    ev, evec = np.linalg.eigh(gram)
    order = np.argsort(ev)[::-1]
    ev, evec = np.clip(ev[order], 0, None), evec[:, order]
    comp = evec.T @ X
    comp /= np.maximum(np.linalg.norm(comp, axis=1, keepdims=True), 1e-12)
    return mean, comp, ev


def main():
    args = parse_args()
    out = cd.makedirs(cd.require_bulk_path(cd.OUTPUT_ROOT / "pca", "pca dir"))
    rows = cd.read_manifest()

    print("[gate] topology check across all 203 scans ...", flush=True)
    cd.topology_gate(rows)
    print(f"[gate] OK: one topology, {cd.EXPECTED_VERTICES} vertices", flush=True)

    V = {s: cd.load_split_vertices(s, rows=rows) for s in cd.SPLITS}
    F = {s: v.reshape(len(v), -1).astype(np.float64) for s, v in V.items()}
    S = {s: cd.scale_factors(rows, s) for s in cd.SPLITS}
    n_train = len(F["train"])
    max_rank = n_train - 1
    print(f"[pca] {n_train} train meshes, {F['train'].shape[1]} features, max rank {max_rank}")

    mean, comp, ev = fit_pca(F["train"])
    metric_rows, got = [], {}
    for k_req in sorted(args.components):
        k = min(k_req, comp.shape[0], max_rank)
        basis = comp[:k]
        for split in cd.SPLITS:
            recon = (F[split] - mean) @ basis.T @ basis + mean
            e = cd.corresponded_vertex_rmse_mm(recon, F[split], S[split])
            metric_rows.append({
                "split": split, "requested_components": k_req, "effective_components": k,
                "count": len(e), "mean_corresponded_vertex_rmse_mm": float(e.mean()),
                "median_corresponded_vertex_rmse_mm": float(np.median(e)),
                "p95_corresponded_vertex_rmse_mm": float(np.quantile(e, 0.95)),
            })
            got[(k, split)] = float(e.mean())
            print(f"[pca] k={k:4d} {split:5s} rmse_mm={e.mean():.6f}")

    if not args.skip_gate:
        bad = []
        for k, refs in cd.PCA_REFERENCE.items():
            for split, ref in refs.items():
                if (k, split) in got and abs(got[(k, split)] - ref) > cd.PCA_TOL:
                    bad.append(f"k={k} {split}: got {got[(k,split)]:.6f} published {ref:.6f}")
        if bad:
            raise SystemExit("PCA reproduction FAILED:\n  " + "\n  ".join(bad))
        print("[gate] PCA reproduction OK against published reconstruction_summary.csv")

    model = cd.makedirs(out / "model")
    cd.atomic_save_npy(model / "mean.npy", mean.astype(np.float32))
    for k_req in sorted(args.components):
        k = min(k_req, comp.shape[0], max_rank)
        cd.atomic_save_npy(model / f"components_{k}.npy", comp[:k].astype(np.float32))
    cd.atomic_write_csv(out / "reconstruction_summary.csv", metric_rows)
    cd.atomic_write_json(out / "pca_summary.json", {
        "train_meshes": n_train, "features": int(F["train"].shape[1]),
        "max_rank": max_rank, "components_scored": sorted(args.components),
        "linear_ceiling_test_mm": got.get((min(172, max_rank), "test")),
        "note": "PCA cannot exceed 172 components here; PCA-256 does not exist.",
    })
    print(f"[pca] wrote {out}")


if __name__ == "__main__":
    main()
