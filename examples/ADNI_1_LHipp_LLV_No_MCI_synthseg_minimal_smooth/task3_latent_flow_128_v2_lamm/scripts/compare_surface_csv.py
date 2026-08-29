#!/usr/bin/env python3
"""Paired subject bootstrap for any two rows in a surface-evaluation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _bootstrap import activate

activate()
import common as C


LOWER_IS_BETTER = (
    "prediction_coordinate_rmse_mm",
    "prediction_mean_vertex_euclidean_mm",
    "prediction_assd_mm",
    "prediction_hd95_mm",
    "prediction_chamfer_l1_mm",
    "prediction_chamfer_l2_squared_mm2",
    "prediction_flipped_face_fraction_vs_ground_truth",
    "prediction_volume_absolute_error_mm3",
    "prediction_volume_relative_error",
    "prediction_log_volume_rate_absolute_error_per_year",
)
DERIVED_ERRORS: dict[str, Callable[[dict[str, str]], float]] = {
    "prediction_normal_absolute_cosine_error": lambda row: 1.0 - float(row["prediction_normal_absolute_cosine"]),
    "prediction_surface_area_ratio_absolute_error": lambda row: abs(float(row["prediction_surface_area_ratio"]) - 1.0),
    "prediction_curvature_ratio_absolute_error": lambda row: abs(float(row["prediction_curvature_ratio_to_ground_truth"]) - 1.0),
    "prediction_connected_components_absolute_error": lambda row: abs(float(row["prediction_predicted_connected_components"]) - 1.0),
    "prediction_not_watertight": lambda row: float(str(row["prediction_predicted_watertight"]).lower() not in {"true", "1"}),
    "prediction_winding_inconsistent": lambda row: float(str(row["prediction_predicted_winding_consistent"]).lower() not in {"true", "1"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def subject_rows(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    selected = [row for row in rows if row["representation"] == name]
    output = {row["subject_id"]: row for row in selected}
    if len(output) != len(selected):
        raise ValueError(f"Duplicate subject rows for {name}")
    if not output:
        raise ValueError(f"No rows for representation {name!r}")
    return output


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    csv_path = args.csv.expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = subject_rows(rows, args.baseline)
    candidate = subject_rows(rows, args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError(
            "Surface rows are not subject matched: "
            f"baseline_only={len(set(baseline)-set(candidate))}, "
            f"candidate_only={len(set(candidate)-set(baseline))}"
        )
    subjects = sorted(baseline)
    extractors: dict[str, Callable[[dict[str, str]], float]] = {
        metric: (lambda row, key=metric: float(row[key])) for metric in LOWER_IS_BETTER
    }
    extractors.update(DERIVED_ERRORS)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(subjects), size=(args.bootstrap_samples, len(subjects)))
    metrics: dict[str, Any] = {}
    for metric, extract in extractors.items():
        baseline_values = np.asarray([extract(baseline[subject]) for subject in subjects])
        candidate_values = np.asarray([extract(candidate[subject]) for subject in subjects])
        differences = candidate_values - baseline_values
        estimates = differences[indices].mean(axis=1)
        metrics[metric] = {
            "baseline_mean": float(baseline_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "candidate_minus_baseline_mean": float(differences.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
            "fraction_subjects_candidate_better": float(np.mean(differences < 0.0)),
            "negative_favors_candidate": True,
        }
    output = {
        "schema_version": 1,
        "source_csv": str(csv_path),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "subjects": len(subjects),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "subject",
        "metrics": metrics,
    }
    C.assert_finite_mapping(output)
    destination = C.require_bulk_path(args.output, "surface paired-comparison output")
    if destination.suffix.lower() != ".json":
        raise ValueError("--output must end in .json")
    if destination.exists():
        raise FileExistsError(destination)
    C.atomic_json(destination, output)
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
