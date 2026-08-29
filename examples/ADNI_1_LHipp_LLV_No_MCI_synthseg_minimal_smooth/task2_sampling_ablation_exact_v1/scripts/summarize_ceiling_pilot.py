#!/usr/bin/env python3
"""Summarize paired ASSD/HD95 changes in the 20-scan evaluation-ceiling pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ablation_common import atomic_write_json, require_bulk_path  # noqa: E402
from compare_sampling_runs import cluster_bootstrap  # noqa: E402


COMBINATIONS = ((256, 500), (384, 500), (512, 500), (256, 1000), (256, 2000))
METRICS = ("assd_mm", "hd95_mm", "chamfer_l1_mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("method", "inr") == "inr"]
    result = {row["scan_id"]: row for row in rows}
    if len(result) != len(rows) or not result:
        raise ValueError(f"Invalid or empty INR rows in {path}.")
    return result


def main() -> None:
    args = parse_args()
    root = require_bulk_path(args.root, "ceiling pilot root")
    runs = {}
    for resolution, steps in COMBINATIONS:
        path = root / f"resolution_{resolution}_steps_{steps}" / "per_scan_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        runs[(resolution, steps)] = read_rows(path)
    baseline = runs[(256, 500)]
    scan_ids = sorted(baseline)
    clusters = np.asarray([baseline[scan_id]["subject_id"] for scan_id in scan_ids])
    report_rows = []
    for combination_index, ((resolution, steps), rows) in enumerate(runs.items()):
        if set(rows) != set(baseline):
            raise ValueError(f"Paired scan mismatch for resolution={resolution}, steps={steps}.")
        for metric_index, metric in enumerate(METRICS):
            base = np.asarray([float(baseline[scan_id][metric]) for scan_id in scan_ids])
            values = np.asarray([float(rows[scan_id][metric]) for scan_id in scan_ids])
            delta = values - base
            report_rows.append(
                {
                    "resolution": resolution,
                    "latent_steps": steps,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "mean_delta_vs_resolution256_steps500": float(delta.mean()),
                    "bootstrap_delta_mean_ci95": cluster_bootstrap(
                        delta,
                        clusters,
                        args.bootstrap_replicates,
                        args.seed + 101 * combination_index + metric_index,
                    ),
                }
            )
    report = {
        "root": str(root),
        "baseline": {"resolution": 256, "latent_steps": 500},
        "difference_definition": "configuration minus resolution-256/500-step baseline",
        "bootstrap_unit": "subject cluster",
        "rows": report_rows,
    }
    atomic_write_json(root / "ceiling_comparison.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
