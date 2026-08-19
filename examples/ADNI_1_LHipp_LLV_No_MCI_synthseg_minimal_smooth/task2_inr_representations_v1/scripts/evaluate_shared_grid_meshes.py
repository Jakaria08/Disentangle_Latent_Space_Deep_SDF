#!/usr/bin/env python3
"""Evaluate reconstructed meshes against manifest ground truth surfaces."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from shared_grid_common import (
    load_checkpoint_training_scan_ids,
    load_config,
    load_manifest,
    stable_seed,
    write_json,
)


METRICS = ("chamfer_l1", "chamfer_l2_squared", "assd", "hd95", "volume_relative_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--meshes-dir", required=True, help="Directory containing split/scan_id.ply.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val", "test"))
    parser.add_argument("--surface-points", type=int, default=30000)
    parser.add_argument("--max-scans", type=int, default=None)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Empty or invalid mesh: {path}")
    return mesh


def sample(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def mesh_metrics(ground_truth: trimesh.Trimesh, predicted: trimesh.Trimesh, count: int, seed: int) -> dict:
    gt = sample(ground_truth, count, seed)
    pred = sample(predicted, count, seed + 1)
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    gt_to_pred = pred_tree.query(gt, k=1, workers=-1)[0]
    pred_to_gt = gt_tree.query(pred, k=1, workers=-1)[0]
    chamfer_l1 = float(gt_to_pred.mean() + pred_to_gt.mean())
    gt_volume = abs(float(ground_truth.volume))
    pred_volume = abs(float(predicted.volume))
    return {
        "chamfer_l1": chamfer_l1,
        "chamfer_l2_squared": float(np.square(gt_to_pred).mean() + np.square(pred_to_gt).mean()),
        "assd": 0.5 * chamfer_l1,
        "hd95": float(max(np.quantile(gt_to_pred, 0.95), np.quantile(pred_to_gt, 0.95))),
        "ground_truth_volume": gt_volume,
        "predicted_volume": pred_volume,
        "volume_relative_error": abs(pred_volume - gt_volume) / max(gt_volume, 1.0e-12),
        "predicted_watertight": bool(predicted.is_watertight),
        "predicted_winding_consistent": bool(predicted.is_winding_consistent),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    result = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"split:{row['split']}"].append(row)
        provenance = "seen" if row["source_pretrain_subject_seen"] else "unseen"
        groups[f"source_pretrain_subject:{provenance}"].append(row)
    for name, group in groups.items():
        result[name] = {
            metric: {
                "count": len(group),
                "mean": float(np.mean([float(row[metric]) for row in group])),
                "median": float(np.median([float(row[metric]) for row in group])),
                "std": float(np.std([float(row[metric]) for row in group])),
            }
            for metric in METRICS
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    meshes_dir = Path(args.meshes_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else meshes_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(config["manifest"])
    rows = [row for row in manifest_rows if row["split"] in args.splits]
    source_training_ids = load_checkpoint_training_scan_ids(
        config["global_checkpoint"], str(config.get("global_checkpoint_scan_id_suffix", ""))
    )
    source_seen_subjects = {
        row["subject_id"] for row in manifest_rows if row["scan_id"] in source_training_ids
    }
    if args.max_scans is not None:
        rows = rows[: args.max_scans]
    results = []
    failures = []
    for number, row in enumerate(rows, start=1):
        predicted_path = meshes_dir / row["split"] / f"{row['scan_id']}.ply"
        try:
            metrics = mesh_metrics(
                load_mesh(Path(row["mesh_path"])),
                load_mesh(predicted_path),
                args.surface_points,
                stable_seed(row["scan_id"], int(config["seed"])),
            )
            results.append(
                {
                    "scan_id": row["scan_id"],
                    "subject_id": row["subject_id"],
                    "split": row["split"],
                    "diagnosis": row.get("diagnosis", ""),
                    "source_pretrain_scan_seen": row["scan_id"] in source_training_ids,
                    "source_pretrain_subject_seen": row["subject_id"] in source_seen_subjects,
                    **metrics,
                    "predicted_mesh_path": str(predicted_path),
                }
            )
        except Exception as error:
            failures.append({"scan_id": row["scan_id"], "split": row["split"], "error": str(error)})
        print(f"[{number:04d}/{len(rows):04d}] {row['scan_id']}", flush=True)
    if not results:
        raise RuntimeError(f"No meshes were evaluated. Failures: {failures[:3]}")
    write_csv(output_dir / "per_scan_metrics.csv", results)
    write_json(
        output_dir / "summary.json",
        {
            "surface_points": args.surface_points,
            "evaluated_count": len(results),
            "failure_count": len(failures),
            "failures": failures,
            "metrics": summarize(results),
            "definitions": {
                "chamfer_l1": "mean(gt_to_pred) + mean(pred_to_gt)",
                "chamfer_l2_squared": "mean(gt_to_pred^2) + mean(pred_to_gt^2)",
                "assd": "0.5 * chamfer_l1",
                "hd95": "max(q95(gt_to_pred), q95(pred_to_gt))",
                "alignment": "none; meshes use the common normalized coordinates",
            },
        },
    )
    print(f"Evaluated {len(results)} meshes; failures={len(failures)}; output={output_dir}")


if __name__ == "__main__":
    main()
