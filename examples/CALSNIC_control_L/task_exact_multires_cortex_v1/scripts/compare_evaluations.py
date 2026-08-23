#!/usr/bin/env python3
"""Compare paired CALSNIC validation/test metrics from multiple evaluation directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from calsnic_common import atomic_write_csv, atomic_write_json, require_bulk_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, help="LABEL=/bulk/evaluation_directory")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_metrics(value: str, split: str):
    label, path = value.split("=", 1)
    with (Path(path) / "per_scan_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split and row["method"] == "inr"]
    return label, {row["scan_id"]: row for row in rows}


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir)
    evaluations = [read_metrics(value, args.split) for value in args.evaluation]
    reference_label, reference = evaluations[0]
    metrics = ("assd_mm", "hd95_mm", "high_curvature_gt_to_prediction_mm", "fscore_1mm")
    rows = []
    for label, values in evaluations:
        common = sorted(set(reference).intersection(values))
        for metric in metrics:
            delta = np.asarray([float(values[scan][metric]) - float(reference[scan][metric]) for scan in common])
            rows.append(
                {
                    "split": args.split,
                    "reference": reference_label,
                    "candidate": label,
                    "metric": metric,
                    "paired_count": len(common),
                    "mean_candidate_minus_reference": float(delta.mean()),
                    "median_candidate_minus_reference": float(np.median(delta)),
                }
            )
    atomic_write_csv(output / "paired_model_comparison.csv", rows)
    atomic_write_json(output / "summary.json", {"split": args.split, "reference": reference_label, "evaluations": [label for label, _ in evaluations], "rows": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
