#!/usr/bin/env python3
"""Replace existing SDF labels with exact point-to-triangle signed distances."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import (
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    load_mesh_for_sdf,
    load_source_samples,
    read_manifest,
    relabel_arrays,
    require_bulk_path,
    restore_source_coordinate_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approx-manifest",
        default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "hippocampus_pilot_approx.csv"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--sign-check-points", type=int, default=4096)
    parser.add_argument("--sign-check-margin", type=float, default=1.0e-5)
    parser.add_argument("--maximum-sign-disagreement-fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--scan-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe path component in manifest: {value!r}")
    return value


def read_existing_archive(path: Path, source_pos: np.ndarray, source_neg: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"pos", "neg", "pos_source_index", "neg_source_index", "format_version"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Existing exact archive is incomplete ({sorted(missing)}): {path}")
        pos = np.asarray(archive["pos"])
        neg = np.asarray(archive["neg"])
        restored = restore_source_coordinate_order(
            pos,
            neg,
            np.asarray(archive["pos_source_index"]),
            np.asarray(archive["neg_source_index"]),
        )
    source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
    if restored.dtype != source_xyz.dtype or not np.array_equal(restored, source_xyz):
        raise ValueError(f"Existing exact archive does not preserve source xyz exactly: {path}")
    if np.any(pos[:, 3] < 0.0) or np.any(neg[:, 3] >= 0.0):
        raise ValueError(f"Existing exact archive violates its pos/neg partition: {path}")
    return {
        "query_count": int(len(pos) + len(neg)),
        "exact_pos_count": int(len(pos)),
        "exact_neg_count": int(len(neg)),
        "status": "resumed_verified_existing",
    }


def sign_cross_check(
    mesh,
    arrays: dict[str, np.ndarray],
    *,
    count: int,
    margin: float,
    seed: int,
) -> dict[str, Any]:
    pos = arrays["pos"]
    neg = arrays["neg"]
    combined = np.concatenate((pos, neg), axis=0)
    eligible = np.flatnonzero(np.abs(combined[:, 3]) > margin)
    if not len(eligible) or count <= 0:
        return {"sign_check_count": 0, "sign_disagreements": 0, "sign_disagreement_fraction": 0.0}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(count, len(eligible)), replace=False)
    ray_inside = np.asarray(mesh.contains(combined[chosen, :3]), dtype=bool)
    exact_inside = combined[chosen, 3] < 0.0
    disagreements = int(np.sum(ray_inside != exact_inside))
    return {
        "sign_check_count": int(len(chosen)),
        "sign_disagreements": disagreements,
        "sign_disagreement_fraction": disagreements / len(chosen),
    }


def process_row(
    row: dict[str, str],
    root: Path,
    args: argparse.Namespace,
    ordinal: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    structure = safe_component(row.get("structure", "structure"))
    scan_id = safe_component(row["scan_id"])
    output = root / "sdf_exact" / structure / f"{scan_id}.npz"
    source_path = Path(row.get("source_sdf_npz_path") or row["sdf_npz_path"]).resolve()
    source_pos, source_neg, source_extras = load_source_samples(source_path)
    if output.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(f"Exact archive already exists; pass --resume: {output}")
        metrics = read_existing_archive(output, source_pos, source_neg)
        mesh_report = {"mesh_path": row["mesh_path"], "validation": "deferred_to_full_audit"}
    else:
        mesh, mesh_report = load_mesh_for_sdf(row["mesh_path"])
        arrays, metrics = relabel_arrays(
            mesh,
            source_pos,
            source_neg,
            chunk_size=int(args.chunk_size),
            zero_epsilon=float(args.zero_epsilon),
        )
        cross_check = sign_cross_check(
            mesh,
            arrays,
            count=int(args.sign_check_points),
            margin=float(args.sign_check_margin),
            seed=int(args.seed) + ordinal,
        )
        metrics.update(cross_check)
        if cross_check["sign_disagreement_fraction"] > float(
            args.maximum_sign_disagreement_fraction
        ):
            raise RuntimeError(
                "Exact normal/ray sign and independent all-ray sign disagree for "
                f"{cross_check['sign_disagreement_fraction']:.3%} of checked points."
            )
        # Preserve non-conflicting source metadata without allowing it to replace the
        # explicit provenance and coordinate-mapping fields of this format.
        for key, value in source_extras.items():
            if key not in arrays:
                arrays[f"source_extra_{key}"] = value
        arrays.update(
            {
                "backend": np.asarray("trimesh_exact_triangle_hybrid_sign"),
                "sign_convention": np.asarray("negative_inside_positive_outside"),
                "source_sdf_npz_path": np.asarray(str(source_path)),
                "mesh_path": np.asarray(str(Path(row["mesh_path"]).resolve())),
                "zero_epsilon": np.asarray(float(args.zero_epsilon), dtype=np.float64),
            }
        )
        atomic_write_npz(
            output,
            arrays,
            allow_non_bulk=args.allow_non_bulk_output,
        )
        metrics["status"] = "written"

    exact_row = dict(row)
    exact_row["source_sdf_npz_path"] = str(source_path)
    exact_row["sdf_npz_path"] = str(output)
    exact_row["sdf_label_kind"] = "exact_triangle_signed_distance"
    exact_row["exact_sdf_backend"] = "trimesh_exact_triangle_hybrid_sign"
    report = {
        "scan_id": scan_id,
        "subject_id": row["subject_id"],
        "split": row["split"],
        "source_sdf_npz_path": str(source_path),
        "exact_sdf_npz_path": str(output),
        **metrics,
        **{f"mesh_{key}": value for key, value in mesh_report.items()},
    }
    return exact_row, report


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive.")
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both.")
    root = require_bulk_path(args.output_root, allow_non_bulk=args.allow_non_bulk_output)
    approximate_stem = Path(args.approx_manifest).stem
    cohort_stem = (
        approximate_stem[: -len("_approx")]
        if approximate_stem.endswith("_approx")
        else approximate_stem
    )
    safe_component(cohort_stem)
    rows = read_manifest(args.approx_manifest)
    requested = set(args.scan_id)
    if requested:
        known = {row["scan_id"] for row in rows}
        missing = sorted(requested - known)
        if missing:
            raise KeyError(f"Requested scan IDs are absent from the manifest: {missing[:5]}")
        work_rows = [row for row in rows if row["scan_id"] in requested]
    else:
        work_rows = rows

    successes: list[tuple[dict[str, str], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    if args.workers == 1:
        for ordinal, row in enumerate(work_rows):
            try:
                successes.append(process_row(row, root, args, ordinal))
                print(f"[{ordinal + 1}/{len(work_rows)}] {row['scan_id']} OK", flush=True)
            except Exception as error:  # retain a machine-readable failure report
                failures.append(
                    {
                        "scan_id": row["scan_id"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[{ordinal + 1}/{len(work_rows)}] {row['scan_id']} FAILED: {error}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, row, root, args, ordinal): (ordinal, row)
                for ordinal, row in enumerate(work_rows)
            }
            for future in as_completed(futures):
                ordinal, row = futures[future]
                try:
                    successes.append(future.result())
                    print(f"[{ordinal + 1}/{len(work_rows)}] {row['scan_id']} OK", flush=True)
                except Exception as error:
                    failures.append(
                        {
                            "scan_id": row["scan_id"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": "".join(
                                traceback.format_exception(type(error), error, error.__traceback__)
                            ),
                        }
                    )
                    print(f"[{ordinal + 1}/{len(work_rows)}] {row['scan_id']} FAILED: {error}", flush=True)

    successes.sort(key=lambda item: item[1]["scan_id"])
    reports = [item[1] for item in successes]
    audit_dir = root / "audits"
    suffix = "partial" if requested else "full"
    if reports:
        atomic_write_csv(
            audit_dir / f"relabel_per_scan_{suffix}.csv",
            reports,
            allow_non_bulk=args.allow_non_bulk_output,
        )
    summary = {
        "approximate_manifest": str(Path(args.approx_manifest).resolve()),
        "backend": "trimesh exact closest triangle; oriented-normal sign with ray fallback",
        "sign_convention": "negative_inside_positive_outside",
        "coordinate_policy": "source xyz retained bit-for-bit; source row indices saved",
        "requested_scan_count": len(work_rows),
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
        "partial_run": bool(requested),
        "exact_manifest_written": False,
    }
    if not failures and len(successes) == len(rows) and not requested:
        exact_rows_by_id = {row[0]["scan_id"]: row[0] for row in successes}
        exact_rows = [exact_rows_by_id[row["scan_id"]] for row in rows]
        exact_manifest = root / "manifests" / f"{cohort_stem}_exact.csv"
        if exact_manifest.exists() and not args.overwrite and not args.resume:
            raise FileExistsError(exact_manifest)
        atomic_write_csv(
            exact_manifest,
            exact_rows,
            allow_non_bulk=args.allow_non_bulk_output,
        )
        summary["exact_manifest_written"] = True
        summary["exact_manifest"] = str(exact_manifest)
    summary_path = audit_dir / f"relabel_summary_{suffix}.json"
    atomic_write_json(summary_path, summary, allow_non_bulk=args.allow_non_bulk_output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    if not summary["exact_manifest_written"]:
        print("Partial relabelling completed; the training manifest was intentionally not written.")


if __name__ == "__main__":
    main()
