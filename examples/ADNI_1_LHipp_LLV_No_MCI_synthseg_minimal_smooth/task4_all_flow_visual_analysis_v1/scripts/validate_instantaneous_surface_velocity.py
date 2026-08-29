#!/usr/bin/env python3
"""Validate the velocity-only supplemental cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = TASK / "derived_cache" / "instantaneous_surface_velocity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def validate_stage(root: Path, stage: str, expected_rows: int, expected_scans: int, expected_methods: set[str]) -> dict[str, object]:
    table_path = root / "tables" / f"{stage}_instantaneous_surface_velocity.csv"
    array_path = root / "arrays" / f"{stage}_instantaneous_surface_maps.npz"
    manifest_path = root / f"manifest_{stage}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete {stage} manifest")
    frame = pd.read_csv(table_path)
    if len(frame) != expected_rows or frame.scan_id.astype(str).nunique() != expected_scans:
        raise ValueError(f"{stage} cohort mismatch: rows={len(frame)}, scans={frame.scan_id.nunique()}")
    if set(frame.method_label.astype(str)) != expected_methods:
        raise ValueError(f"{stage} method mismatch: {sorted(frame.method_label.unique())}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"{stage} contains NaN/Inf")
    required_columns = {
        "observed_gt_normal_rms_mm_per_year",
        "model_normal_rms_mm_per_year",
        "normal_mae_mm_per_year",
        "normal_pearson",
        "normal_sign_agreement",
        "conditional_ad_minus_cn_normal_rms_mm_per_year",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{stage} missing columns: {sorted(required_columns - set(frame.columns))}")
    with np.load(array_path, allow_pickle=False) as arrays:
        vertices = arrays["template_vertices"]
        faces = arrays["faces"]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{stage} invalid template topology")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{stage} non-finite template")
        for key in arrays.files:
            if key in {"template_vertices", "faces"}:
                continue
            value = arrays[key]
            if value.shape != (len(vertices),) or not np.isfinite(value).all():
                raise ValueError(f"{stage} invalid map {key}: {value.shape}")
    return {
        "stage": stage,
        "rows": len(frame),
        "subjects": int(frame.subject_id.astype(str).nunique()),
        "scans": int(frame.scan_id.astype(str).nunique()),
        "methods": sorted(expected_methods),
    }


def main() -> int:
    root = parse_args().cache_dir.expanduser().resolve()
    reports = [
        validate_stage(root, "current", 400, 100, {"PCA", "Spiral", "Adaptive", "INR"}),
        validate_stage(root, "legacy", 856, 214, {"PCA Cocycle", "INR Cocycle", "Latent ODE", "BrainODE"}),
    ]
    print(json.dumps({"status": "passed", "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
