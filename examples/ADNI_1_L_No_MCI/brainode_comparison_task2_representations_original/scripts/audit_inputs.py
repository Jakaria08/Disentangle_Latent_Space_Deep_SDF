#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

from task2_common import (
    TASK_DIR,
    load_config,
    load_json,
    load_manifest,
    load_obj_arrays,
    resolve_repo_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Task 1 meshes and SDF samples before representation fitting."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
        help="Task 2 pipeline configuration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check SDF structure for 10 scans per split instead of all scans.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = resolve_repo_path(config["manifest"])
    rows = load_manifest(manifest_path)
    output_dir = TASK_DIR / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    split_counts = Counter(row["split"] for row in rows)
    expected = config["expected_counts"]
    if len(rows) != int(expected["total"]):
        errors.append(f"Expected {expected['total']} rows, found {len(rows)}.")
    for split in ("train", "val", "test"):
        if split_counts[split] != int(expected[split]):
            errors.append(
                f"Expected {expected[split]} {split} rows, found {split_counts[split]}."
            )

    scan_ids = [row["scan_id"] for row in rows]
    filenames = [row["filename"] for row in rows]
    if len(scan_ids) != len(set(scan_ids)):
        errors.append("Duplicate scan_id values exist.")
    if len(filenames) != len(set(filenames)):
        errors.append("Duplicate filenames exist.")

    task1_report_path = resolve_repo_path(config["task1_validation_report"])
    if not task1_report_path.is_file():
        errors.append(f"Task 1 validation report is missing: {task1_report_path}")
        task1_report = None
    else:
        task1_report = load_json(task1_report_path)
        if task1_report.get("errors"):
            errors.append("Task 1 validation report contains errors.")

    expected_vertices = int(config["mesh"]["vertex_count"])
    expected_faces = int(config["mesh"]["face_count"])
    coordinate_min = float(config["mesh"]["coordinate_min"])
    coordinate_max = float(config["mesh"]["coordinate_max"])
    reference_faces = None
    reference_face_hash = None
    mesh_stats = {
        "vertex_count_distribution": Counter(),
        "face_count_distribution": Counter(),
        "watertight": 0,
        "winding_consistent": 0,
        "finite": 0,
        "topology_match": 0,
        "inside_configured_domain": 0,
        "volume_min": float("inf"),
        "volume_max": float("-inf"),
        "global_coordinate_min": [float("inf")] * 3,
        "global_coordinate_max": [float("-inf")] * 3,
    }

    for row in rows:
        mesh_path = Path(row["mesh_path"])
        sdf_path = Path(row["sdf_npz_path"])
        if not mesh_path.is_file():
            errors.append(f"Missing mesh: {mesh_path}")
            continue
        if not sdf_path.is_file():
            errors.append(f"Missing SDF: {sdf_path}")

        vertices, faces = load_obj_arrays(mesh_path)
        mesh_stats["vertex_count_distribution"][len(vertices)] += 1
        mesh_stats["face_count_distribution"][len(faces)] += 1
        if len(vertices) != expected_vertices or len(faces) != expected_faces:
            errors.append(
                f"Unexpected topology size for {row['scan_id']}: "
                f"{len(vertices)} vertices, {len(faces)} faces."
            )
        if np.isfinite(vertices).all():
            mesh_stats["finite"] += 1
        else:
            errors.append(f"Non-finite mesh vertices: {row['scan_id']}")

        if reference_faces is None:
            reference_faces = faces.copy()
            reference_face_hash = sha256_file(mesh_path)
        if np.array_equal(faces, reference_faces):
            mesh_stats["topology_match"] += 1
        else:
            errors.append(f"Face ordering mismatch: {row['scan_id']}")

        current_min = vertices.min(axis=0)
        current_max = vertices.max(axis=0)
        mesh_stats["global_coordinate_min"] = np.minimum(
            mesh_stats["global_coordinate_min"], current_min
        ).tolist()
        mesh_stats["global_coordinate_max"] = np.maximum(
            mesh_stats["global_coordinate_max"], current_max
        ).tolist()
        if current_min.min() >= coordinate_min and current_max.max() <= coordinate_max:
            mesh_stats["inside_configured_domain"] += 1
        else:
            errors.append(f"Mesh leaves [{coordinate_min}, {coordinate_max}]: {row['scan_id']}")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_stats["watertight"] += int(mesh.is_watertight)
        mesh_stats["winding_consistent"] += int(mesh.is_winding_consistent)
        volume = abs(float(mesh.volume))
        mesh_stats["volume_min"] = min(mesh_stats["volume_min"], volume)
        mesh_stats["volume_max"] = max(mesh_stats["volume_max"], volume)

    if reference_faces is not None:
        np.save(output_dir / "reference_faces.npy", reference_faces)

    sdf_rows = rows
    if args.quick:
        sdf_rows = []
        for split in ("train", "val", "test"):
            sdf_rows.extend([row for row in rows if row["split"] == split][:10])

    sdf_stats = {
        "mode": "quick" if args.quick else "full",
        "scans_checked": 0,
        "finite_scans": 0,
        "sign_correct_scans": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "positive_sdf_min": float("inf"),
        "positive_sdf_max": float("-inf"),
        "negative_sdf_min": float("inf"),
        "negative_sdf_max": float("-inf"),
    }
    sign_tolerance = 1e-6
    for row in sdf_rows:
        path = Path(row["sdf_npz_path"])
        if not path.is_file():
            continue
        try:
            with np.load(path) as archive:
                if "pos" not in archive or "neg" not in archive:
                    errors.append(f"SDF lacks pos/neg arrays: {row['scan_id']}")
                    continue
                pos = np.asarray(archive["pos"])
                neg = np.asarray(archive["neg"])
        except Exception as exc:
            errors.append(f"Could not read SDF {row['scan_id']}: {exc}")
            continue
        sdf_stats["scans_checked"] += 1
        if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != 4 or neg.shape[1] != 4:
            errors.append(f"Unexpected SDF shape: {row['scan_id']}")
            continue
        finite = bool(np.isfinite(pos).all() and np.isfinite(neg).all())
        sdf_stats["finite_scans"] += int(finite)
        if not finite:
            errors.append(f"Non-finite SDF values: {row['scan_id']}")
            continue
        sign_correct = bool(
            np.min(pos[:, 3]) >= -sign_tolerance
            and np.max(neg[:, 3]) <= sign_tolerance
        )
        sdf_stats["sign_correct_scans"] += int(sign_correct)
        if not sign_correct:
            errors.append(f"SDF sign mismatch: {row['scan_id']}")
        sdf_stats["positive_rows"] += int(len(pos))
        sdf_stats["negative_rows"] += int(len(neg))
        sdf_stats["positive_sdf_min"] = min(
            sdf_stats["positive_sdf_min"], float(np.min(pos[:, 3]))
        )
        sdf_stats["positive_sdf_max"] = max(
            sdf_stats["positive_sdf_max"], float(np.max(pos[:, 3]))
        )
        sdf_stats["negative_sdf_min"] = min(
            sdf_stats["negative_sdf_min"], float(np.min(neg[:, 3]))
        )
        sdf_stats["negative_sdf_max"] = max(
            sdf_stats["negative_sdf_max"], float(np.max(neg[:, 3]))
        )

    mesh_stats["vertex_count_distribution"] = dict(
        sorted(mesh_stats["vertex_count_distribution"].items())
    )
    mesh_stats["face_count_distribution"] = dict(
        sorted(mesh_stats["face_count_distribution"].items())
    )
    report = {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "reference_mesh_file_sha256": reference_face_hash,
        "mesh": mesh_stats,
        "sdf": sdf_stats,
        "task1_validation_status": (
            task1_report.get("status") if isinstance(task1_report, dict) else None
        ),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_dir / "input_audit.json", report)
    write_json(output_dir / "mesh_topology_report.json", mesh_stats)
    write_json(output_dir / "sdf_report.json", sdf_stats)

    print(json.dumps({"status": report["status"], "errors": len(errors)}, indent=2))
    print(f"Wrote audit reports to {output_dir}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
