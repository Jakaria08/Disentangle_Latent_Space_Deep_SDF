#!/usr/bin/env python3
"""Finalize a cache whose current-model phase completed before legacy maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import build_analysis_cache as B


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.cache_dir.expanduser().resolve()
    manifest = B.read_json(root / "manifest.json")
    if manifest.get("status") != "building":
        raise ValueError("Expected a building-state partial cache")
    required = [
        root / "tables" / "current_endpoint_metrics.csv",
        root / "tables" / "current_velocity_per_scan.csv",
        root / "arrays" / "current_surface_maps.npz",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Partial cache did not finish its current-model phase")
    registry = B.read_json(B.REGISTRY_PATH)
    legacy = B.copy_legacy(registry, root / "tables")
    B.legacy_surface_maps(
        legacy,
        root / "arrays" / "legacy_surface_maps.npz",
        B.resolve(registry["legacy"]["pca_group_map_root"]),
    )
    endpoint = pd.read_csv(root / "tables" / "current_endpoint_metrics.csv")
    velocity = pd.read_csv(root / "tables" / "current_velocity_per_scan.csv")
    complete = {
        "status": "complete", "schema_version": 1, "recovered_from_completed_current_phase": True,
        "seed": 42, "surface_resolution": 256, "surface_samples": 5000,
        "voxel_pitch_mm": 0.5, "bootstrap_samples": 2000, "velocity_eps_years": 0.05,
        "shared_subjects": int(endpoint.subject_id.astype(str).nunique()),
        "shared_scans": int(velocity.scan_id.astype(str).nunique()),
        "current_endpoint_rows": len(endpoint), "current_velocity_rows": len(velocity),
        "legacy_endpoint_rows": len(legacy), "primary_methods": list(B.CURRENT_ORDER),
        "legacy_results_separate": True, "smoke_runs_excluded": True,
        "source_meshes_modified": False,
        "surface_distance_definition": "deterministic bidirectional sampled-surface nearest-neighbour distance via cKDTree",
        "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"),
    }
    B.atomic_json(root / "manifest.json", complete)
    print(f"FINALIZED {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
