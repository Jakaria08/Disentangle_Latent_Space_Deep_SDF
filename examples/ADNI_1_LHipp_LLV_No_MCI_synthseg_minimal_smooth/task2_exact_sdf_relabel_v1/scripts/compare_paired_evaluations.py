#!/usr/bin/env python3
"""Compare exact-label and approximate-label INR metrics on identical scans."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import DEFAULT_OUTPUT_ROOT, atomic_write_csv, atomic_write_json, require_bulk_path


ERROR_METRICS = (
    "assd_mm",
    "hd95_mm",
    "chamfer_l1_mm",
    "chamfer_l2_squared_mm2",
    "volume_absolute_error_mm3",
    "volume_relative_error",
)
HIGHER_IS_BETTER = ("fscore_0_5mm", "fscore_1mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("compact", "multires"), required=True)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--approx-metrics", default=None)
    parser.add_argument("--exact-metrics", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def default_metrics_path(family: str, labels: str, epoch: int) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / f"{family}_{labels}"
        / "periodic_evaluation"
        / f"epoch_{epoch:04d}"
        / "per_scan_metrics.csv"
    )


def read_inr_rows(path: str | Path, split: str) -> dict[str, dict[str, str]]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == split and row.get("method", "inr") == "inr"
        ]
    result = {row["scan_id"]: row for row in rows}
    if not result:
        raise ValueError(f"No INR rows for split={split}: {source}")
    if len(result) != len(rows):
        raise ValueError(f"Duplicate INR scan IDs in {source}")
    return result


def subject_cluster_bootstrap(
    differences: np.ndarray,
    subjects: np.ndarray,
    replicates: int,
    seed: int,
) -> list[float]:
    unique = np.unique(subjects)
    by_subject = {subject: differences[subjects == subject] for subject in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([by_subject[subject] for subject in sampled])
        means[index] = values.mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    approximate_path = Path(
        args.approx_metrics or default_metrics_path(args.family, "approx", args.epoch)
    ).resolve()
    exact_path = Path(
        args.exact_metrics or default_metrics_path(args.family, "exact", args.epoch)
    ).resolve()
    approximate = read_inr_rows(approximate_path, args.split)
    exact = read_inr_rows(exact_path, args.split)
    if set(approximate) != set(exact):
        only_approx = sorted(set(approximate) - set(exact))
        only_exact = sorted(set(exact) - set(approximate))
        raise ValueError(
            "Paired evaluation requires identical scans; "
            f"only approximate={only_approx[:3]}, only exact={only_exact[:3]}"
        )

    per_scan: list[dict[str, Any]] = []
    for scan_id in sorted(approximate):
        approx = approximate[scan_id]
        current = exact[scan_id]
        if approx["subject_id"] != current["subject_id"]:
            raise ValueError(f"Subject mismatch for {scan_id}")
        row: dict[str, Any] = {
            "scan_id": scan_id,
            "subject_id": current["subject_id"],
            "split": args.split,
        }
        for metric in (*ERROR_METRICS, *HIGHER_IS_BETTER):
            if metric not in approx or metric not in current:
                continue
            approximate_value = float(approx[metric])
            exact_value = float(current[metric])
            row[f"approx_{metric}"] = approximate_value
            row[f"exact_{metric}"] = exact_value
            row[f"exact_minus_approx_{metric}"] = exact_value - approximate_value
        per_scan.append(row)

    summary: dict[str, Any] = {
        "family": args.family,
        "epoch": int(args.epoch),
        "split": args.split,
        "approximate_metrics": str(approximate_path),
        "exact_metrics": str(exact_path),
        "scan_count": len(per_scan),
        "subject_count": len({row["subject_id"] for row in per_scan}),
        "comparison": "exact minus approximate on identical scans",
        "metrics": {},
    }
    subjects = np.asarray([row["subject_id"] for row in per_scan])
    for metric in (*ERROR_METRICS, *HIGHER_IS_BETTER):
        key = f"exact_minus_approx_{metric}"
        if key not in per_scan[0]:
            continue
        difference = np.asarray([row[key] for row in per_scan], dtype=np.float64)
        higher = metric in HIGHER_IS_BETTER
        improved = difference > 0.0 if higher else difference < 0.0
        interval = subject_cluster_bootstrap(
            difference,
            subjects,
            int(args.bootstrap_replicates),
            int(args.seed) + len(summary["metrics"]) * 1009,
        )
        summary["metrics"][metric] = {
            "mean_exact_minus_approx": float(difference.mean()),
            "median_exact_minus_approx": float(np.median(difference)),
            "subject_cluster_bootstrap_mean_95ci": interval,
            "fraction_scans_improved": float(improved.mean()),
            "direction_for_improvement": "positive" if higher else "negative",
            "ci_excludes_zero_in_improvement_direction": bool(
                interval[0] > 0.0 if higher else interval[1] < 0.0
            ),
        }

    output = require_bulk_path(
        args.output_dir
        or DEFAULT_OUTPUT_ROOT
        / "evaluations"
        / "paired_exact_vs_approx"
        / args.family
        / f"epoch_{args.epoch:04d}"
        / args.split,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    atomic_write_csv(
        output / "per_scan_paired_differences.csv",
        per_scan,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    atomic_write_json(
        output / "summary.json",
        summary,
        allow_non_bulk=args.allow_non_bulk_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
