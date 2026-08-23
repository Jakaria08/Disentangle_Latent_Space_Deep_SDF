#!/usr/bin/env python3
"""Validate the CALSNIC exact-SDF, centred-mesh, split, AABB, and audit contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from calsnic_common import exact_signed_distance, load_sdf_arrays, load_sdf_space_mesh, read_manifest
from delegate_multires import GENERIC_SCRIPTS

sys.path.insert(0, str(GENERIC_SCRIPTS))
from multires_common import load_config, require_bulk_path, validate_manifest_contract, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scans", type=int, default=8)
    parser.add_argument("--points-per-scan", type=int, default=2048)
    parser.add_argument("--output-report", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_manifest(config["manifest"])
    contract = validate_manifest_contract(rows, config["network_specs"]["grid_aabb"])
    expected = {"train": 173, "val": 15, "test": 15}
    if contract["split_scan_counts"] != expected:
        raise ValueError(f"Expected split counts {expected}, found {contract['split_scan_counts']}")
    audit_path = Path(config["required_exact_audit"])
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit.get("training_allowed") or audit.get("exact_manifest_sha256") != sha256(Path(config["manifest"])):
        raise RuntimeError("The full exact-SDF audit is absent, failed, or stale.")

    selected_indices = np.linspace(0, len(rows) - 1, min(args.scans, len(rows)), dtype=int)
    per_scan = []
    errors = []
    sign_violations = 0
    outside_near = 0
    total_near = 0
    aabb = np.asarray(config["network_specs"]["grid_aabb"], dtype=np.float64)
    for index in selected_indices:
        row = rows[int(index)]
        pos, neg = load_sdf_arrays(row["sdf_npz_path"])
        sign_violations += int(np.sum(pos[:, 3] < 0.0) + np.sum(neg[:, 3] >= 0.0))
        combined = np.concatenate((pos, neg), axis=0)
        near = np.abs(combined[:, 3]) <= float(config["sampling"]["near_band"])
        outside = np.logical_or(combined[:, :3] < aabb[0], combined[:, :3] > aabb[1]).any(axis=1)
        outside_near += int(np.logical_and(near, outside).sum())
        total_near += int(near.sum())
        rng = np.random.default_rng(int(config["seed"]) + int(index))
        chosen = combined[rng.integers(0, len(combined), size=min(args.points_per_scan, len(combined)))]
        exact = exact_signed_distance(load_sdf_space_mesh(row), chosen[:, :3])
        error = np.abs(exact - chosen[:, 3])
        errors.append(error)
        per_scan.append(
            {
                "scan_id": row["scan_id"],
                "absolute_signed_distance_mae": float(error.mean()),
                "absolute_signed_distance_max": float(error.max()),
                "near_fraction": float(near.mean()),
            }
        )
    all_errors = np.concatenate(errors)
    tolerance = float(config["sdf_supervision"]["exact_distance_mae_tolerance"])
    if sign_violations:
        raise RuntimeError(f"Found {sign_violations} pos/neg sign violations.")
    if float(all_errors.mean()) > tolerance:
        raise RuntimeError(f"Exact-label MAE {all_errors.mean():.6g} exceeds {tolerance:.6g}.")
    grid_parameters = sum(int(value) ** 3 for value in config["network_specs"]["grid_resolutions"]) * int(config["network_specs"]["grid_channels_per_level"])
    report = {
        "passed": True,
        "config": str(Path(args.config).resolve()),
        "manifest": str(Path(config["manifest"]).resolve()),
        "contract": contract,
        "exact_audit": str(audit_path),
        "sampled_scan_reports": per_scan,
        "sampled_exact_signed_distance_mae": float(all_errors.mean()),
        "sampled_exact_signed_distance_max": float(all_errors.max()),
        "fraction_near_samples_outside_feature_roi": outside_near / max(total_near, 1),
        "grid_feature_parameters": grid_parameters,
        "grid_feature_fp32_mib": grid_parameters * 4 / 1024**2,
        "test_policy": "test split locked during model and checkpoint selection",
    }
    output = require_bulk_path(
        args.output_report
        or Path(config["output_dir"]).parent / "implementation_validation" / f"{config['name']}_input_validation.json"
    )
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
