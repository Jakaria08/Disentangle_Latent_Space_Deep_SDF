#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from task2_common import (
    TASK_DIR,
    load_config,
    load_manifest,
    stable_seed,
    summarize_values,
    write_json,
)


METRICS = (
    "chamfer_l2_squared",
    "assd",
    "hd95",
    "volume_absolute_error",
    "volume_relative_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PCA and INR reconstruction meshes with common metrics."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
        help="Task 2 pipeline configuration.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method names; defaults to all configured methods.",
    )
    parser.add_argument(
        "--splits", default="train,val,test", help="Comma-separated splits."
    )
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def sample_surface(
    mesh: trimesh.Trimesh, count: int, seed: int
) -> np.ndarray:
    points, _faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def surface_metrics(
    ground_truth_points: np.ndarray,
    predicted_points: np.ndarray,
) -> dict[str, float]:
    predicted_tree = cKDTree(predicted_points)
    ground_truth_tree = cKDTree(ground_truth_points)
    gt_to_pred = predicted_tree.query(ground_truth_points, k=1, workers=-1)[0]
    pred_to_gt = ground_truth_tree.query(predicted_points, k=1, workers=-1)[0]
    return {
        "chamfer_l2_squared": float(
            np.mean(gt_to_pred**2) + np.mean(pred_to_gt**2)
        ),
        "assd": float(0.5 * (np.mean(gt_to_pred) + np.mean(pred_to_gt))),
        "hd95": float(
            max(np.quantile(gt_to_pred, 0.95), np.quantile(pred_to_gt, 0.95))
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rows = load_manifest(config["manifest"])
    evaluation_config = config["evaluation"]
    configured_methods = evaluation_config["methods"]
    methods = (
        [value.strip() for value in args.methods.split(",") if value.strip()]
        if args.methods
        else list(configured_methods)
    )
    unknown = set(methods).difference(configured_methods)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    selected_rows = [row for row in rows if row["split"] in splits]
    if args.limit > 0:
        selected_rows = selected_rows[: args.limit]
    point_count = int(args.surface_points or evaluation_config["surface_points"])
    base_seed = int(evaluation_config["seed"])
    ground_truth_dir = TASK_DIR / "evaluation" / "ground_truth_points"
    metric_dir = TASK_DIR / "evaluation" / "metrics"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    failures = []
    for scan_index, row in enumerate(selected_rows, start=1):
        gt_mesh = as_mesh(Path(row["mesh_path"]))
        gt_points_path = ground_truth_dir / f"{row['scan_id']}.npz"
        if gt_points_path.is_file():
            with np.load(gt_points_path) as archive:
                gt_points = np.asarray(archive["points"], dtype=np.float32)
            if len(gt_points) != point_count:
                gt_points = sample_surface(
                    gt_mesh,
                    point_count,
                    stable_seed(row["scan_id"], base_seed),
                )
                np.savez_compressed(gt_points_path, points=gt_points)
        else:
            gt_points = sample_surface(
                gt_mesh,
                point_count,
                stable_seed(row["scan_id"], base_seed),
            )
            np.savez_compressed(gt_points_path, points=gt_points)
        gt_volume = abs(float(gt_mesh.volume))

        for method in methods:
            mesh_path = TASK_DIR / configured_methods[method] / f"{row['scan_id']}.ply"
            if not mesh_path.is_file():
                failures.append(
                    {
                        "method": method,
                        "scan_id": row["scan_id"],
                        "error": "missing reconstructed mesh",
                    }
                )
                continue
            try:
                predicted_mesh = as_mesh(mesh_path)
                predicted_points = sample_surface(
                    predicted_mesh,
                    point_count,
                    stable_seed(f"{method}:{row['scan_id']}", base_seed),
                )
                metrics = surface_metrics(gt_points, predicted_points)
                predicted_volume = abs(float(predicted_mesh.volume))
                absolute_volume_error = abs(predicted_volume - gt_volume)
                relative_volume_error = absolute_volume_error / max(gt_volume, 1e-12)
                result_rows.append(
                    {
                        "method": method,
                        "scan_id": row["scan_id"],
                        "subject_id": row["subject_id"],
                        "split": row["split"],
                        "diagnosis": row["diagnosis"],
                        "label_ad": row["label_ad"],
                        "visit_order": row["visit_order"],
                        **metrics,
                        "ground_truth_volume": gt_volume,
                        "predicted_volume": predicted_volume,
                        "volume_absolute_error": absolute_volume_error,
                        "volume_relative_error": relative_volume_error,
                        "predicted_watertight": bool(predicted_mesh.is_watertight),
                        "predicted_winding_consistent": bool(
                            predicted_mesh.is_winding_consistent
                        ),
                        "predicted_mesh_path": str(mesh_path),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "method": method,
                        "scan_id": row["scan_id"],
                        "error": str(exc),
                    }
                )
        print(f"[{scan_index}/{len(selected_rows)}] evaluated {row['scan_id']}")

    if not result_rows:
        raise RuntimeError("No reconstruction metrics were produced.")
    write_csv(metric_dir / "reconstruction_per_scan.csv", result_rows)

    summary_rows = []
    summary_json = {}
    group_specs = [
        ("overall", lambda row: True),
        *[(f"split:{split}", lambda row, value=split: row["split"] == value) for split in sorted(splits)],
        *[
            (
                f"diagnosis:{diagnosis}",
                lambda row, value=diagnosis: row["diagnosis"] == value,
            )
            for diagnosis in ("CN", "AD")
        ],
    ]
    for method in methods:
        method_rows = [row for row in result_rows if row["method"] == method]
        summary_json[method] = {}
        for group_name, predicate in group_specs:
            group_rows = [row for row in method_rows if predicate(row)]
            if not group_rows:
                continue
            summary_json[method][group_name] = {}
            for metric in METRICS:
                stats = summarize_values(float(row[metric]) for row in group_rows)
                summary_json[method][group_name][metric] = stats
                summary_rows.append(
                    {
                        "method": method,
                        "group": group_name,
                        "metric": metric,
                        **stats,
                    }
                )
            volumes_true = np.asarray(
                [float(row["ground_truth_volume"]) for row in group_rows]
            )
            volumes_pred = np.asarray(
                [float(row["predicted_volume"]) for row in group_rows]
            )
            correlation = (
                float(np.corrcoef(volumes_true, volumes_pred)[0, 1])
                if len(group_rows) > 1
                else None
            )
            summary_json[method][group_name]["volume_correlation"] = correlation

    write_csv(metric_dir / "reconstruction_summary.csv", summary_rows)
    write_json(metric_dir / "reconstruction_summary.json", summary_json)
    write_json(
        metric_dir / "evaluation_run.json",
        {
            "methods": methods,
            "splits": sorted(splits),
            "surface_points": point_count,
            "selected_scan_count": len(selected_rows),
            "metric_row_count": len(result_rows),
            "failure_count": len(failures),
            "failures": failures,
            "metric_definitions": {
                "chamfer_l2_squared": "mean(gt_to_pred_distance^2) + mean(pred_to_gt_distance^2)",
                "assd": "0.5 * (mean(gt_to_pred_distance) + mean(pred_to_gt_distance))",
                "hd95": "max(q95(gt_to_pred_distance), q95(pred_to_gt_distance))",
                "alignment": "none",
            },
        },
    )
    print(
        json.dumps(
            {
                "metric_rows": len(result_rows),
                "failures": len(failures),
                "output": str(metric_dir),
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
