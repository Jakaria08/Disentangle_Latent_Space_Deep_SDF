#!/usr/bin/env python3
"""Preserve CALSNIC query XYZ and replace approximate labels by triangle SDF values."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from calsnic_common import (
    DEFAULT_APPROX_MANIFEST,
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    load_sdf_arrays,
    load_sdf_space_mesh,
    read_manifest,
    relabel_arrays,
    require_bulk_path,
    restore_source_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approx-manifest", default=str(DEFAULT_APPROX_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--scan-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def verify_existing(path: Path, source_pos: np.ndarray, source_neg: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"pos", "neg", "pos_source_index", "neg_source_index", "format_version"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Incomplete existing exact archive {path}: {sorted(missing)}")
        pos, neg = np.asarray(archive["pos"]), np.asarray(archive["neg"])
        format_version = int(np.asarray(archive["format_version"]).item())
        if format_version != 2:
            raise ValueError(
                f"Existing archive has label format {format_version}; expected exact-label format 2. "
                "Rebuild it with --overwrite."
            )
        restored = restore_source_order(
            pos, neg, np.asarray(archive["pos_source_index"]), np.asarray(archive["neg_source_index"])
        )
    source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
    if restored.dtype != source_xyz.dtype or not np.array_equal(restored, source_xyz):
        raise ValueError(f"Existing archive does not preserve source XYZ: {path}")
    return {"query_count": int(len(pos) + len(neg)), "status": "resumed_verified_existing"}


def process(row: dict[str, str], root: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    scan_id = row["scan_id"]
    source_path = Path(row["sdf_npz_path"])
    output = require_bulk_path(root / "sdf_exact" / f"{scan_id}.npz", allow_non_bulk=args.allow_non_bulk_output)
    source_pos, source_neg = load_sdf_arrays(source_path)
    if output.exists():
        if args.resume:
            metrics = verify_existing(output, source_pos, source_neg)
        elif args.overwrite:
            mesh = load_sdf_space_mesh(row)
            arrays, metrics = relabel_arrays(
                mesh, source_pos, source_neg, chunk_size=args.chunk_size, zero_epsilon=args.zero_epsilon
            )
            atomic_write_npz(output, arrays, allow_non_bulk=args.allow_non_bulk_output)
            metrics["status"] = "overwritten"
        else:
            raise FileExistsError(f"Refusing to overwrite {output}; use --resume or --overwrite.")
    else:
        mesh = load_sdf_space_mesh(row)
        arrays, metrics = relabel_arrays(
            mesh, source_pos, source_neg, chunk_size=args.chunk_size, zero_epsilon=args.zero_epsilon
        )
        atomic_write_npz(output, arrays, allow_non_bulk=args.allow_non_bulk_output)
        metrics["status"] = "written"
    exact_row = dict(row)
    exact_row["source_sdf_npz_path"] = str(source_path.resolve())
    exact_row["sdf_npz_path"] = str(output)
    exact_row["sdf_label_kind"] = "exact_triangle_magnitude_fast_winding_sign"
    exact_row["exact_sdf_backend"] = (
        "PCU closest-face barycentric Euclidean magnitude + fast-winding sign"
    )
    return exact_row, {
        "scan_id": scan_id,
        "subject_id": row["subject_id"],
        "split": row["split"],
        "source_sdf_npz_path": str(source_path.resolve()),
        "exact_sdf_npz_path": str(output),
        "mesh_watertight": True,
        "mesh_transform": "subtract_manifest_AABB_midpoint",
        **metrics,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive.")
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    root = require_bulk_path(args.output_root, allow_non_bulk=args.allow_non_bulk_output)
    rows = read_manifest(args.approx_manifest)
    requested = set(args.scan_id)
    if requested:
        missing = requested.difference(row["scan_id"] for row in rows)
        if missing:
            raise KeyError(f"Unknown scan IDs: {sorted(missing)}")
        rows = [row for row in rows if row["scan_id"] in requested]

    successes = []
    failures = []
    if args.workers == 1:
        work = [(index, row, None) for index, row in enumerate(rows)]
        for index, row, _unused in work:
            try:
                successes.append(process(row, root, args))
                print(f"[{index + 1}/{len(rows)}] {row['scan_id']} OK", flush=True)
            except Exception as error:
                failures.append({"scan_id": row["scan_id"], "error": repr(error), "traceback": traceback.format_exc()})
                print(f"[{index + 1}/{len(rows)}] {row['scan_id']} FAILED: {error}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, row, root, args): (index, row) for index, row in enumerate(rows)}
            for future in as_completed(futures):
                index, row = futures[future]
                try:
                    successes.append(future.result())
                    print(f"[{index + 1}/{len(rows)}] {row['scan_id']} OK", flush=True)
                except Exception as error:
                    failures.append({"scan_id": row["scan_id"], "error": repr(error), "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))})
                    print(f"[{index + 1}/{len(rows)}] {row['scan_id']} FAILED: {error}", flush=True)

    successes.sort(key=lambda item: item[0]["scan_id"])
    reports = [item[1] for item in successes]
    suffix = "partial" if requested else "full"
    if reports:
        atomic_write_csv(root / "audits" / f"relabel_per_scan_{suffix}.csv", reports, allow_non_bulk=args.allow_non_bulk_output)
    atomic_write_json(
        root / "audits" / f"relabel_failures_{suffix}.json",
        failures,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    if failures:
        raise RuntimeError(f"Exact relabelling failed for {len(failures)} scans.")
    if not requested:
        exact_rows = [item[0] for item in successes]
        exact_manifest = root / "manifests" / DEFAULT_EXACT_MANIFEST.name
        atomic_write_csv(exact_manifest, exact_rows, allow_non_bulk=args.allow_non_bulk_output)
        print(json.dumps({"exact_manifest": str(exact_manifest), "scans": len(exact_rows)}, indent=2))
    else:
        print("Partial relabelling completed; the full training manifest was intentionally not written.")


if __name__ == "__main__":
    main()
