#!/usr/bin/env python3
"""Paired subject-cluster comparison of sampling-ablation mesh evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ablation_common import atomic_write_json, require_bulk_path  # noqa: E402


LOWER_IS_BETTER = (
    "assd_mm",
    "hd95_mm",
    "chamfer_l1_mm",
    "chamfer_l2_squared_mm2",
    "volume_absolute_error_mm3",
    "volume_relative_error",
)
HIGHER_IS_BETTER = (
    "fscore_0_1mm",
    "fscore_0_25mm",
    "fscore_0_5mm",
    "fscore_1mm",
    "normal_absolute_cosine",
)
TARGET_ONE = (
    "curvature_ratio_to_ground_truth",
    "surface_area_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Control per_scan_metrics.csv on bulk disk.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Repeat NAME=/mnt/.../per_scan_metrics.csv for each candidate.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Candidate must have the form NAME=/mnt/.../per_scan_metrics.csv")
    name, raw_path = value.split("=", 1)
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError(f"Unsafe candidate name {name!r}.")
    path = require_bulk_path(raw_path, f"{name} candidate metrics")
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def read_inr(path: Path, split: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("method", "inr") == "inr" and row["split"] == split
        ]
    result = {row["scan_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate scan IDs in {path}.")
    if not result:
        raise ValueError(f"No {split} INR rows in {path}.")
    return result


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    repeats: int,
    seed: int,
) -> list[float]:
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    positions = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    for index in range(repeats):
        sampled = unique[rng.integers(0, len(unique), size=len(unique))]
        draw = np.concatenate([positions[cluster] for cluster in sampled])
        means[index] = values[draw].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def signed_rank_pvalue(values: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if not np.any(values != 0.0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def metric_values(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    scan_ids: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray([float(baseline[scan_id][metric]) for scan_id in scan_ids])
    cand = np.asarray([float(candidate[scan_id][metric]) for scan_id in scan_ids])
    if metric in TARGET_ONE:
        base = np.abs(base - 1.0)
        cand = np.abs(cand - 1.0)
    return base, cand, cand - base


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_test:
        raise PermissionError("Test comparison is locked; add --confirm-test only for the final model.")
    if args.bootstrap_replicates < 1000:
        raise ValueError("Use at least 1000 subject-cluster bootstrap replicates.")
    baseline_path = require_bulk_path(args.baseline, "control metrics")
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    output = require_bulk_path(args.output_dir)
    baseline = read_inr(baseline_path, args.split)
    candidates = [parse_candidate(value) for value in args.candidate]
    metric_rows: list[dict[str, Any]] = []
    per_scan_rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for candidate_index, (name, path) in enumerate(candidates):
        candidate = read_inr(path, args.split)
        if set(candidate) != set(baseline):
            missing = sorted(set(baseline).difference(candidate))
            extra = sorted(set(candidate).difference(baseline))
            raise ValueError(f"Paired scan mismatch for {name}: missing={missing[:3]} extra={extra[:3]}.")
        scan_ids = sorted(baseline)
        cluster_ids = np.asarray([baseline[scan_id]["subject_id"] for scan_id in scan_ids])
        available = set(next(iter(baseline.values()))).intersection(next(iter(candidate.values())))
        metrics = [
            metric for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER, *TARGET_ONE)
            if metric in available
        ]
        summaries: dict[str, dict[str, Any]] = {}
        for metric_index, metric in enumerate(metrics):
            base, cand, difference = metric_values(baseline, candidate, scan_ids, metric)
            interval = cluster_bootstrap(
                difference,
                cluster_ids,
                args.bootstrap_replicates,
                args.seed + 1009 * candidate_index + metric_index,
            )
            direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
            if direction == "higher":
                wins = int(np.sum(difference > 0.0))
            else:
                wins = int(np.sum(difference < 0.0))
            row = {
                "candidate": name,
                "split": args.split,
                "metric": metric,
                "preferred_direction": direction,
                "count": len(difference),
                "subject_count": len(np.unique(cluster_ids)),
                "baseline_mean": float(base.mean()),
                "candidate_mean": float(cand.mean()),
                "candidate_minus_control_mean": float(difference.mean()),
                "candidate_minus_control_median": float(np.median(difference)),
                "bootstrap_mean_ci95_lower": interval[0],
                "bootstrap_mean_ci95_upper": interval[1],
                "candidate_wins": wins,
                "control_wins": int(np.sum(difference != 0.0)) - wins,
                "ties": int(np.sum(difference == 0.0)),
                "wilcoxon_two_sided_p": signed_rank_pvalue(difference),
            }
            metric_rows.append(row)
            summaries[metric] = row
            for scan_id, base_value, candidate_value, delta in zip(scan_ids, base, cand, difference):
                per_scan_rows.append(
                    {
                        "candidate": name,
                        "split": args.split,
                        "scan_id": scan_id,
                        "subject_id": baseline[scan_id]["subject_id"],
                        "metric": metric,
                        "control": float(base_value),
                        "candidate_value": float(candidate_value),
                        "candidate_minus_control": float(delta),
                    }
                )
        base_watertight = sum(baseline[scan_id].get("predicted_watertight", "False") == "True" for scan_id in scan_ids)
        cand_watertight = sum(candidate[scan_id].get("predicted_watertight", "False") == "True" for scan_id in scan_ids)
        base_multi = sum(int(float(baseline[scan_id].get("predicted_connected_components", 1))) > 1 for scan_id in scan_ids)
        cand_multi = sum(int(float(candidate[scan_id].get("predicted_connected_components", 1))) > 1 for scan_id in scan_ids)
        assd = summaries["assd_mm"]
        hd95 = summaries["hd95_mm"]
        decisions[name] = {
            "assd_mean_improves": assd["candidate_minus_control_mean"] < 0.0,
            "assd_bootstrap_ci_excludes_zero_in_improvement_direction": assd["bootstrap_mean_ci95_upper"] < 0.0,
            "hd95_noninferior_margin_mm": 0.005,
            "hd95_noninferior": hd95["candidate_minus_control_mean"] <= 0.005,
            "control_watertight_count": base_watertight,
            "candidate_watertight_count": cand_watertight,
            "control_multicomponent_count": base_multi,
            "candidate_multicomponent_count": cand_multi,
            "topology_not_worse": cand_watertight >= base_watertight and cand_multi <= base_multi,
        }
        decisions[name]["passes_primary_rule"] = bool(
            decisions[name]["assd_mean_improves"]
            and decisions[name]["assd_bootstrap_ci_excludes_zero_in_improvement_direction"]
            and decisions[name]["hd95_noninferior"]
            and decisions[name]["topology_not_worse"]
        )
    report = {
        "baseline": str(baseline_path),
        "split": args.split,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "subject cluster",
        "difference_definition": "candidate minus continued-training control",
        "test_locked_during_selection": args.split != "test",
        "decisions": decisions,
        "metrics": metric_rows,
    }
    atomic_write_json(output / "comparison_summary.json", report)
    write_csv(output / "paired_metric_summary.csv", metric_rows)
    write_csv(output / "paired_per_scan.csv", per_scan_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
