#!/usr/bin/env python3
"""Audit CALSNIC exact labels, preserved coordinates, transforms, and mesh topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from calsnic_common import (
    DEFAULT_APPROX_MANIFEST,
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    coordinate_sha256,
    exact_signed_distance,
    load_sdf_arrays,
    load_sdf_space_mesh,
    read_manifest,
    require_bulk_path,
    restore_source_order,
    sha256_file,
    split_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approx-manifest", default=str(DEFAULT_APPROX_MANIFEST))
    parser.add_argument("--exact-manifest", default=str(DEFAULT_EXACT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--points-per-scan", type=int, default=2048)
    parser.add_argument("--independent-points-per-scan", type=int, default=64)
    parser.add_argument("--max-independent-magnitude-error", type=float, default=2.0e-5)
    parser.add_argument("--max-recomputed-label-error", type=float, default=2.0e-6)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--scans", type=int, default=None, help="Audit a deterministic subset only.")
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def independent_ray_signed_distance(
    mesh: trimesh.Trimesh, xyz: np.ndarray
) -> np.ndarray:
    """Return Trimesh distance magnitude with a global ray-containment sign.

    ``trimesh.proximity.signed_distance`` normally assigns sign from the
    closest face normal when the closest-point projection lies on that face.
    For highly folded cortical surfaces, that local sign can disagree with
    both global containment and the mesh winding number.  The audit needs an
    independent *global* inside/outside test, so use Trimesh's closest-triangle
    magnitude and ray containment as two explicit operations.

    The CALSNIC/DeepSDF convention is negative inside and positive outside.
    """
    points = np.asarray(xyz, dtype=np.float64)
    _closest, magnitude, _triangle_id = trimesh.proximity.closest_point(mesh, points)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    inside = np.asarray(mesh.contains(points), dtype=bool)
    if magnitude.shape != (len(points),) or inside.shape != (len(points),):
        raise RuntimeError("Trimesh independent audit returned invalid shapes.")
    if not np.isfinite(magnitude).all():
        raise RuntimeError("Trimesh independent audit returned non-finite distances.")
    return np.where(inside, -magnitude, magnitude)


def main() -> None:
    args = parse_args()
    root = require_bulk_path(args.output_root, allow_non_bulk=args.allow_non_bulk_output)
    approximate = read_manifest(args.approx_manifest)
    exact = read_manifest(args.exact_manifest)
    approx_by_id = {row["scan_id"]: row for row in approximate}
    exact_by_id = {row["scan_id"]: row for row in exact}
    if set(approx_by_id) != set(exact_by_id):
        raise ValueError("Approximate and exact manifests contain different scan IDs.")
    if split_counts(approximate) != split_counts(exact):
        raise ValueError("Approximate and exact manifests have different split counts.")
    ids = sorted(exact_by_id)
    if args.scans is not None and args.scans < len(ids):
        indices = np.linspace(0, len(ids) - 1, max(1, args.scans), dtype=int)
        ids = [ids[int(index)] for index in indices]

    reports = []
    for number, scan_id in enumerate(ids, start=1):
        source_row = approx_by_id[scan_id]
        row = exact_by_id[scan_id]
        source_pos, source_neg = load_sdf_arrays(source_row["sdf_npz_path"])
        with np.load(row["sdf_npz_path"], allow_pickle=False) as archive:
            required = {"pos", "neg", "pos_source_index", "neg_source_index", "format_version"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{scan_id}: exact archive missing {sorted(missing)}")
            pos = np.asarray(archive["pos"])
            neg = np.asarray(archive["neg"])
            restored = restore_source_order(
                pos,
                neg,
                np.asarray(archive["pos_source_index"]),
                np.asarray(archive["neg_source_index"]),
            )
        source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
        if restored.dtype != source_xyz.dtype or not np.array_equal(restored, source_xyz):
            raise ValueError(f"{scan_id}: exact relabelling changed source XYZ values.")
        if np.any(pos[:, 3] < 0.0) or np.any(neg[:, 3] >= 0.0):
            raise ValueError(f"{scan_id}: exact pos/neg sign partition is invalid.")
        combined = np.concatenate((pos, neg), axis=0)
        rng = np.random.default_rng(args.seed + number)
        chosen_index = rng.integers(0, len(combined), size=min(args.points_per_scan, len(combined)))
        chosen = combined[chosen_index]
        mesh = load_sdf_space_mesh(row)
        recomputed = exact_signed_distance(mesh, chosen[:, :3])
        recomputed_error = np.abs(recomputed - chosen[:, 3])
        if float(recomputed_error.max()) > args.max_recomputed_label_error:
            raise ValueError(
                f"{scan_id}: recomputed exact-label max error {recomputed_error.max():.6g} exceeds tolerance."
            )

        independent_count = min(args.independent_points_per_scan, len(chosen))
        independent = chosen[:independent_count]
        trimesh_signed = independent_ray_signed_distance(mesh, independent[:, :3])
        magnitude_error = np.abs(np.abs(trimesh_signed) - np.abs(independent[:, 3]))
        sign_mask = np.abs(independent[:, 3]) > 1.0e-5
        sign_disagreements = int(
            np.sum((trimesh_signed[sign_mask] >= 0.0) != (independent[sign_mask, 3] >= 0.0))
        )
        if len(magnitude_error) and float(magnitude_error.max()) > args.max_independent_magnitude_error:
            raise ValueError(
                f"{scan_id}: independent triangle-distance max error {magnitude_error.max():.6g} exceeds tolerance."
            )
        if sign_disagreements:
            raise ValueError(f"{scan_id}: independent sign audit found {sign_disagreements} disagreements.")
        reports.append(
            {
                "scan_id": scan_id,
                "split": row["split"],
                "query_count": int(len(combined)),
                "coordinate_sha256": coordinate_sha256(source_xyz),
                "recomputed_points": int(len(chosen)),
                "recomputed_label_mae": float(recomputed_error.mean()),
                "recomputed_label_max": float(recomputed_error.max()),
                "independent_points": int(independent_count),
                "independent_magnitude_mae": float(magnitude_error.mean()),
                "independent_magnitude_max": float(magnitude_error.max()),
                "independent_sign_disagreements": sign_disagreements,
                "mesh_vertices": int(len(mesh.vertices)),
                "mesh_faces": int(len(mesh.faces)),
                "mesh_watertight": bool(mesh.is_watertight),
                "mesh_winding_consistent": bool(mesh.is_winding_consistent),
            }
        )
        print(f"[{number}/{len(ids)}] {scan_id} audit passed", flush=True)

    full_audit = len(ids) == len(exact_by_id)
    report = {
        "passed": True,
        "full_cohort_audit": full_audit,
        "audited_scans": len(ids),
        "manifest_scans": len(exact_by_id),
        "split_counts": split_counts(exact),
        "approx_manifest": str(Path(args.approx_manifest).resolve()),
        "exact_manifest": str(Path(args.exact_manifest).resolve()),
        "approx_manifest_sha256": sha256_file(args.approx_manifest),
        "exact_manifest_sha256": sha256_file(args.exact_manifest),
        "coordinate_contract": "bitwise-identical source XYZ after source-index restoration",
        "distance_backend": "PCU closest-face barycentric Euclidean magnitude with fast-winding sign",
        "independent_backend": (
            "Trimesh closest-triangle magnitude + global ray-containment sign"
        ),
        "training_allowed": full_audit,
    }
    suffix = "full" if full_audit else "partial"
    atomic_write_csv(
        root / "audits" / f"exact_sdf_audit_per_scan_{suffix}.csv",
        reports,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    atomic_write_json(
        root / "audits" / f"exact_sdf_audit_{suffix}.json",
        report,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    if full_audit:
        atomic_write_json(
            root / "audits" / "exact_sdf_audit.json",
            report,
            allow_non_bulk=args.allow_non_bulk_output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
