#!/usr/bin/env python3
"""Paired subject-bootstrap comparison of completed Spiral and Adaptive evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiral-run", required=True)
    parser.add_argument("--adaptive-run", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def subject_means(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if metric not in row or row[metric] in (None, ""):
            continue
        grouped.setdefault(row["subject"], []).append(float(row[metric]))
    return {subject: float(np.mean(values)) for subject, values in grouped.items()}


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    samples: int,
    seed: int,
    higher_is_better: bool,
) -> dict:
    subjects = sorted(set(left) & set(right))
    if not subjects:
        raise ValueError("No common subjects between evaluations")
    difference = np.asarray([right[subject] - left[subject] for subject in subjects], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(subjects), size=(int(samples), len(subjects)))
    draws = difference[indices].mean(axis=1)
    return {
        "subjects": len(subjects),
        "adaptive_minus_spiral_mean": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "better_direction": "higher" if higher_is_better else "lower",
        "probability_adaptive_better": float(np.mean(draws > 0.0 if higher_is_better else draws < 0.0)),
    }


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    spiral_path = root / "runs" / args.spiral_run / "evaluation" / args.split / "per_pair_metrics.csv"
    adaptive_path = root / "runs" / args.adaptive_run / "evaluation" / args.split / "per_pair_metrics.csv"
    spiral = read_rows(spiral_path)
    adaptive = read_rows(adaptive_path)
    keys_left = {(row["source_scan_id"], row["target_scan_id"]) for row in spiral}
    keys_right = {(row["source_scan_id"], row["target_scan_id"]) for row in adaptive}
    if keys_left != keys_right:
        raise ValueError("Spiral and Adaptive evaluations do not contain identical pairs")
    metric_directions = {
        "mean_vertex_error_mm": False,
        "vertex_rmse_mm": False,
        "volume_relative_error": False,
        "volume_rate_abs_error_per_year": False,
        "assd_mm": False,
        "hd95_mm": False,
        "chamfer_l2_squared_mm2": False,
        "flipped_face_fraction": False,
        "normal_signed_cosine": True,
        "mesh_dice": True,
    }
    metrics = [
        "mean_vertex_error_mm",
        "vertex_rmse_mm",
        "volume_relative_error",
        "volume_rate_abs_error_per_year",
    ]
    for optional in (
        "assd_mm",
        "hd95_mm",
        "chamfer_l2_squared_mm2",
        "flipped_face_fraction",
        "normal_signed_cosine",
        "mesh_dice",
    ):
        if any(row.get(optional, "") != "" for row in spiral) and any(
            row.get(optional, "") != "" for row in adaptive
        ):
            metrics.append(optional)
    comparison = {
        metric: paired_bootstrap(
            subject_means(spiral, metric),
            subject_means(adaptive, metric),
            args.bootstrap_samples,
            args.seed + index,
            metric_directions[metric],
        )
        for index, metric in enumerate(metrics)
    }
    output = root / "comparisons" / f"{args.spiral_run}_vs_{args.adaptive_run}_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "split": args.split,
        "spiral_run": args.spiral_run,
        "adaptive_run": args.adaptive_run,
        "pairs": len(spiral),
        "subject_is_independent_unit": True,
        "comparison": comparison,
    }
    C.atomic_json(output / "paired_bootstrap.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
