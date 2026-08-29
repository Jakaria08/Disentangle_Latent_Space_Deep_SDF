#!/usr/bin/env python3
"""Validate the all-flow cache before the notebook consumes it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    root = parse_args().cache_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Cache manifest is not complete")
    if not manifest.get("smoke_runs_excluded") or manifest.get("legacy_results_separate") is not True:
        raise ValueError("Run-separation contract failed")
    endpoint = pd.read_csv(root / "tables" / "current_endpoint_metrics.csv")
    velocity = pd.read_csv(root / "tables" / "current_velocity_per_scan.csv")
    inventory = pd.read_csv(root / "tables" / "checkpoint_inventory.csv")
    expected = {"pca", "spiral", "adaptive", "inr"}
    if set(endpoint.method) != expected or set(velocity.method) != expected or set(inventory.method) != expected:
        raise ValueError("Current method inventory mismatch")
    if len(endpoint) != 80 or endpoint.subject_id.astype(str).nunique() != 20:
        raise ValueError(f"Endpoint cohort mismatch: rows={len(endpoint)} subjects={endpoint.subject_id.nunique()}")
    if len(velocity) != 400 or velocity.subject_id.astype(str).nunique() != 20:
        raise ValueError(f"Velocity cohort mismatch: rows={len(velocity)} subjects={velocity.subject_id.nunique()}")
    if inventory.run_dir.astype(str).str.contains("/smoke_").any() or (inventory.test_evaluation_pairs < 100).any() or set(inventory.status) != {"full"}:
        raise ValueError("Smoke or incomplete checkpoint entered primary cache")
    numeric = endpoint.select_dtypes(include=[np.number]).to_numpy()
    velocity_numeric = velocity.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all() or not np.isfinite(velocity_numeric).all():
        raise ValueError("Non-finite current metrics")
    for filename in ("current_surface_maps.npz", "legacy_surface_maps.npz"):
        with np.load(root / "arrays" / filename, allow_pickle=False) as arrays:
            if any(not np.isfinite(arrays[key]).all() for key in arrays.files):
                raise ValueError(f"Non-finite array in {filename}")
            if arrays["template_vertices"].ndim != 2 or arrays["faces"].shape[1] != 3:
                raise ValueError(f"Mesh contract failed in {filename}")
    forbidden = ("128", "150", "256")
    if any(any(token in str(label) for token in forbidden) for label in endpoint.method_label.unique()):
        raise ValueError("Ordinary current method label exposes latent dimension")
    print("ANALYSIS CACHE VALIDATION PASSED")
    print(json.dumps({
        "current_endpoint_rows": len(endpoint), "current_velocity_rows": len(velocity),
        "subjects": endpoint.subject_id.astype(str).nunique(), "methods": sorted(endpoint.method_label.unique()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
