#!/usr/bin/env python3
"""Validate the SDF/mesh contract before multiresolution training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from multires_common import (
    level_weights_for_epoch,
    load_config,
    load_manifest,
    load_sdf_arrays,
    require_bulk_path,
    stable_seed,
    validate_manifest_contract,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scans", type=int, default=8)
    parser.add_argument("--points-per-scan", type=int, default=2048)
    parser.add_argument(
        "--require-exact-triangle-sdf",
        action="store_true",
        help="Fail when the sampled magnitudes do not meet the configured exact-distance tolerance.",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional report path below /mnt/bulk10tb (defaults under the run family).",
    )
    return parser.parse_args()


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = require_bulk_path(config["output_dir"])
    rows = load_manifest(config["manifest"])
    contract = validate_manifest_contract(rows, config["network_specs"]["grid_aabb"])
    split_subjects = {
        split: {row["subject_id"] for row in rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    if any(
        split_subjects[left].intersection(split_subjects[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Subject leakage detected.")

    selected_indices = np.linspace(0, len(rows) - 1, min(args.scans, len(rows)), dtype=int)
    aabb = np.asarray(config["network_specs"]["grid_aabb"], dtype=np.float32)
    distance_errors = []
    outside_all = 0
    outside_near = 0
    total_all = 0
    total_near = 0
    sign_violations = 0
    scan_reports = []
    for index in selected_indices:
        row = rows[int(index)]
        pos, neg = load_sdf_arrays(row["sdf_npz_path"])
        sign_violations += int(np.sum(pos[:, 3] < 0.0) + np.sum(neg[:, 3] > 0.0))
        combined = np.concatenate((pos, neg), axis=0)
        outside = np.logical_or(combined[:, :3] < aabb[0], combined[:, :3] > aabb[1]).any(axis=1)
        near = np.abs(combined[:, 3]) <= 0.1
        outside_all += int(outside.sum())
        outside_near += int(np.logical_and(outside, near).sum())
        total_all += len(combined)
        total_near += int(near.sum())

        rng = np.random.default_rng(stable_seed(row["scan_id"], int(config["seed"]) + 11001))
        chosen = combined[rng.integers(0, len(combined), size=args.points_per_scan)]
        mesh = load_mesh(row["mesh_path"])
        exact = trimesh.proximity.ProximityQuery(mesh).on_surface(chosen[:, :3])[1]
        error = np.abs(np.abs(chosen[:, 3]) - exact)
        distance_errors.append(error)
        scan_reports.append(
            {
                "scan_id": row["scan_id"],
                "absolute_distance_mae": float(error.mean()),
                "absolute_distance_p99": float(np.quantile(error, 0.99)),
            }
        )

    errors = np.concatenate(distance_errors)
    if sign_violations:
        raise RuntimeError(f"SDF sign partition has {sign_violations} violations.")
    if not np.isfinite(errors).all():
        raise RuntimeError("Non-finite exact-distance validation error.")
    schedule_epochs = sorted(
        {1, int(config["total_epochs"]), *[int(item["start_epoch"]) for item in config["level_schedule"]], *[int(item["end_epoch"]) for item in config["level_schedule"]]}
    )
    schedule = {str(epoch): level_weights_for_epoch(epoch, config) for epoch in schedule_epochs}
    grid_parameters = sum(
        int(resolution) ** 3
        for resolution in config["network_specs"]["grid_resolutions"]
    ) * int(config["network_specs"]["grid_channels_per_level"])
    supervision = config.get("sdf_supervision", {})
    exact_tolerance = float(supervision.get("exact_distance_mae_tolerance", 0.002))
    exact_distance_verified = bool(float(errors.mean()) <= exact_tolerance)
    eikonal_enabled = bool(config.get("eikonal", {}).get("enabled", False))
    warnings = []
    if not exact_distance_verified:
        warnings.append(
            "Stored magnitudes are not exact point-to-triangle SDF values at the configured "
            "tolerance. This is expected for this repository's PreprocessMesh sampler, which "
            "uses nearest rendered surface points and a close-range point-plane approximation."
        )
    if eikonal_enabled and not exact_distance_verified:
        warnings.append(
            "Eikonal is enabled with approximate DeepSDF targets; treat this configuration as "
            "an ablation and select it only from validation mesh/gradient metrics."
        )
    report = {
        "passed": bool(not args.require_exact_triangle_sdf or exact_distance_verified),
        "training_input_contract_passed": True,
        "exact_triangle_sdf_contract_passed": exact_distance_verified,
        "requested_exact_triangle_sdf": bool(args.require_exact_triangle_sdf),
        "persistent_output_dir": str(output),
        "contract": contract,
        "sdf_supervision": supervision,
        "sdf_exact_distance_scans": scan_reports,
        "sdf_exact_distance_mae": float(errors.mean()),
        "sdf_exact_distance_median": float(np.median(errors)),
        "sdf_exact_distance_p99": float(np.quantile(errors, 0.99)),
        "sdf_exact_distance_mae_tolerance": exact_tolerance,
        "sdf_sign_violations": sign_violations,
        "fraction_all_samples_outside_feature_roi": outside_all / total_all,
        "fraction_near_samples_outside_feature_roi": outside_near / max(1, total_near),
        "grid_feature_parameters": grid_parameters,
        "grid_feature_fp32_megabytes": grid_parameters * 4 / 1024**2,
        "level_schedule": schedule,
        "eikonal_enabled": eikonal_enabled,
        "recommended_primary_run": (
            "no_eikonal_until_ablation_demonstrates_validation_benefit"
            if not exact_distance_verified
            else "eikonal_is_supported_by_exact_distance_audit"
        ),
        "warnings": warnings,
        "interpretation": (
            "The data are valid signed near/off-surface supervision. Exact triangle-distance "
            "agreement is a separate, stricter contract relevant to an unqualified unit-Eikonal claim."
        ),
    }
    report_path = require_bulk_path(
        args.output_report
        or Path(config["_output_dir"]).parent
        / "implementation_validation"
        / "input_data_validation_report.json"
    )
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_exact_triangle_sdf and not exact_distance_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
