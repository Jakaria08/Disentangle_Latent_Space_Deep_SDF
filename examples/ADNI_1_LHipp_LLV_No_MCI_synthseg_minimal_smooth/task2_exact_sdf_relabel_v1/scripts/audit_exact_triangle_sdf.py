#!/usr/bin/env python3
"""Independently audit exact-SDF archives before any model is trained."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import (
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    exact_signed_distance_outside_positive,
    load_mesh_for_sdf,
    load_source_samples,
    read_manifest,
    require_bulk_path,
    restore_source_coordinate_order,
    split_subject_leakage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approx-manifest",
        default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "hippocampus_pilot_approx.csv"),
    )
    parser.add_argument(
        "--exact-manifest",
        default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "hippocampus_pilot_exact.csv"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--points-per-scan", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def audit_row(
    approximate: dict[str, str],
    exact: dict[str, str],
    args: argparse.Namespace,
    ordinal: int,
) -> dict[str, Any]:
    source_pos, source_neg, _extras = load_source_samples(
        approximate.get("source_sdf_npz_path") or approximate["sdf_npz_path"]
    )
    with np.load(exact["sdf_npz_path"], allow_pickle=False) as archive:
        required = {"pos", "neg", "pos_source_index", "neg_source_index", "format_version"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing exact archive fields {sorted(missing)}")
        pos = np.asarray(archive["pos"])
        neg = np.asarray(archive["neg"])
        pos_index = np.asarray(archive["pos_source_index"])
        neg_index = np.asarray(archive["neg_source_index"])
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != 4 or neg.shape[1] != 4:
        raise ValueError("Exact pos/neg arrays must be N x 4.")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Exact archive contains non-finite values.")
    if np.any(pos[:, 3] < 0.0) or np.any(neg[:, 3] >= 0.0):
        raise ValueError("Exact archive violates negative-inside pos/neg partition.")

    source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
    restored_xyz = restore_source_coordinate_order(pos, neg, pos_index, neg_index)
    coordinates_identical = bool(
        restored_xyz.dtype == source_xyz.dtype and np.array_equal(restored_xyz, source_xyz)
    )
    if not coordinates_identical:
        raise ValueError("Exact archive did not preserve all source coordinates bit-for-bit.")

    combined = np.concatenate((pos, neg), axis=0)
    combined_index = np.concatenate((pos_index, neg_index)).astype(np.int64, copy=False)
    rng = np.random.default_rng(int(args.seed) + ordinal * 7919)
    count = min(int(args.points_per_scan), len(combined))
    selected = rng.choice(len(combined), size=count, replace=False)
    mesh, mesh_report = load_mesh_for_sdf(exact["mesh_path"])
    recalculated = exact_signed_distance_outside_positive(
        mesh,
        combined[selected, :3],
        chunk_size=int(args.chunk_size),
    )
    stored = combined[selected, 3].astype(np.float64)
    error = np.abs(recalculated - stored)
    maximum_error = float(error.max(initial=0.0))
    if maximum_error > float(args.absolute_tolerance):
        raise ValueError(
            f"Recomputed exact SDF maximum error {maximum_error:.3g} exceeds "
            f"tolerance {args.absolute_tolerance:.3g}."
        )
    source_order_of_selected = combined_index[selected]
    if not np.array_equal(combined[selected, :3], source_xyz[source_order_of_selected]):
        raise ValueError("Source-index coordinate mapping is internally inconsistent.")
    return {
        "scan_id": exact["scan_id"],
        "split": exact["split"],
        "query_count": int(len(combined)),
        "coordinates_bitwise_identical": coordinates_identical,
        "audited_point_count": int(count),
        "sdf_absolute_error_mean": float(error.mean()),
        "sdf_absolute_error_p99": float(np.quantile(error, 0.99)),
        "sdf_absolute_error_max": maximum_error,
        "mesh_watertight": mesh_report["watertight_after"],
        "mesh_winding_consistent": mesh_report["winding_consistent_after"],
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.points_per_scan < 1 or args.absolute_tolerance <= 0.0:
        raise ValueError("workers/points-per-scan must be positive and tolerance must exceed zero.")
    root = require_bulk_path(args.output_root, allow_non_bulk=args.allow_non_bulk_output)
    approximate_rows = read_manifest(args.approx_manifest)
    exact_rows = read_manifest(args.exact_manifest)
    approximate_by_id = {row["scan_id"]: row for row in approximate_rows}
    exact_by_id = {row["scan_id"]: row for row in exact_rows}
    if set(approximate_by_id) != set(exact_by_id):
        raise ValueError("Approximate and exact manifests do not contain identical scan IDs.")
    for scan_id in approximate_by_id:
        for field in ("subject_id", "split", "mesh_path"):
            if approximate_by_id[scan_id][field] != exact_by_id[scan_id][field]:
                raise ValueError(f"Paired manifest mismatch for {scan_id}: {field}")
    leakage = split_subject_leakage(exact_rows)
    if leakage:
        raise ValueError(f"Exact manifest has subject leakage: {dict(list(leakage.items())[:5])}")

    results = []
    failures = []
    pairs = [
        (approximate_by_id[row["scan_id"]], row, ordinal)
        for ordinal, row in enumerate(exact_rows)
    ]
    if args.workers == 1:
        for approximate, exact, ordinal in pairs:
            try:
                results.append(audit_row(approximate, exact, args, ordinal))
                print(f"[{ordinal + 1}/{len(pairs)}] {exact['scan_id']} audited", flush=True)
            except Exception as error:
                failures.append(
                    {"scan_id": exact["scan_id"], "error_type": type(error).__name__, "error": str(error)}
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(audit_row, approximate, exact, args, ordinal): exact["scan_id"]
                for approximate, exact, ordinal in pairs
            }
            for future in as_completed(futures):
                scan_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    failures.append(
                        {"scan_id": scan_id, "error_type": type(error).__name__, "error": str(error)}
                    )
    results.sort(key=lambda row: row["scan_id"])
    if results:
        atomic_write_csv(
            root / "audits" / "exact_sdf_audit_per_scan.csv",
            results,
            allow_non_bulk=args.allow_non_bulk_output,
        )
    split_counts = {
        split: sum(row["split"] == split for row in exact_rows)
        for split in ("train", "val", "test")
    }
    summary = {
        "passed": not failures and len(results) == len(exact_rows),
        "approximate_manifest": str(Path(args.approx_manifest).resolve()),
        "exact_manifest": str(Path(args.exact_manifest).resolve()),
        "scan_count": len(exact_rows),
        "subject_count": len({row["subject_id"] for row in exact_rows}),
        "split_scan_counts": split_counts,
        "subject_split_disjoint": True,
        "source_coordinates_bitwise_preserved": bool(
            results and all(row["coordinates_bitwise_identical"] for row in results)
        ),
        "exact_recomputation_points_per_scan": int(args.points_per_scan),
        "exact_recomputation_absolute_tolerance": float(args.absolute_tolerance),
        "maximum_observed_absolute_error": (
            max(row["sdf_absolute_error_max"] for row in results) if results else None
        ),
        "failure_count": len(failures),
        "failures": failures,
    }
    atomic_write_json(
        root / "audits" / "exact_sdf_audit_summary.json",
        summary,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
