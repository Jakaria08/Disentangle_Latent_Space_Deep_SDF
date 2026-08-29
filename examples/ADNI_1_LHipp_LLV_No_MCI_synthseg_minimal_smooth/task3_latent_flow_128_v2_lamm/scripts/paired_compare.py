#!/usr/bin/env python3
"""Subject-paired comparison of two unified C4 evaluation reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import activate

activate()
import common as C


METRICS = (
    "latent",
    "coordinate",
    "euclidean",
    "end_to_end_coordinate_rmse",
    "end_to_end_coordinate_mae",
    "end_to_end_euclidean",
    "volume_relative",
    "rate",
)
NOCHANGE_COMPARATORS = {
    "latent": "nochange_latent",
    "coordinate": "nochange_coordinate",
    "euclidean": "nochange_euclidean",
    "end_to_end_coordinate_mae": "nochange_end_to_end_coordinate_mae",
    "end_to_end_euclidean": "nochange_end_to_end_euclidean",
    "volume_relative": "nochange_volume_relative",
    "rate": "nochange_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_report(run: Path, evaluation_name: str, split: str) -> dict[str, Any]:
    C.validate_run_name(evaluation_name)
    path = run.expanduser().resolve() / evaluation_name / split / "summary.json"
    report = C.read_json(path)
    if report.get("split") != split:
        raise ValueError(f"Split mismatch in {path}")
    if report.get("method") != "direct_c4":
        raise ValueError(f"Expected direct_c4 in {path}")
    if bool(report.get("test_loaded_during_training", False)):
        raise ValueError(f"Training/test leakage recorded in {path}")
    return report


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    missing = {"subject", "source_index", "target_index", "pair_type"}.difference(row)
    if missing:
        raise KeyError(
            f"Evaluation row lacks {sorted(missing)}. Re-run evaluate.py with the current core."
        )
    return (
        str(row["subject"]),
        int(row["source_index"]),
        int(row["target_index"]),
        str(row["pair_type"]),
    )


def indexed_rows(report: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = report["pair_metrics"]["all_forward"].get("row_metrics", [])
    output = {pair_key(row): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("Duplicate pair keys in evaluation row_metrics")
    return output


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    baseline = load_report(args.baseline_run, args.evaluation_name, args.split)
    candidate = load_report(args.candidate_run, args.evaluation_name, args.split)
    baseline_rows = indexed_rows(baseline)
    candidate_rows = indexed_rows(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError(
            "Evaluations are not pair-matched: "
            f"baseline_only={len(set(baseline_rows) - set(candidate_rows))}, "
            f"candidate_only={len(set(candidate_rows) - set(baseline_rows))}"
        )
    keys = sorted(baseline_rows)
    by_subject: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for key in keys:
        by_subject[str(key[0])].append(key)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(subjects), size=(args.bootstrap_samples, len(subjects))
    )
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        subject_differences = np.asarray([
            np.mean([
                float(candidate_rows[key][metric]) - float(baseline_rows[key][metric])
                for key in by_subject[subject]
            ])
            for subject in subjects
        ], dtype=np.float64)
        estimates = subject_differences[bootstrap_indices].mean(axis=1)
        metrics[metric] = {
            "baseline_mean": float(np.mean([float(baseline_rows[key][metric]) for key in keys])),
            "candidate_mean": float(np.mean([float(candidate_rows[key][metric]) for key in keys])),
            "candidate_minus_baseline_subject_mean": float(subject_differences.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
            "fraction_subjects_candidate_better": float(np.mean(subject_differences < 0.0)),
            "negative_favors_candidate": True,
        }
    nochange_improvement: dict[str, Any] = {}
    for metric, nochange_metric in NOCHANGE_COMPARATORS.items():
        baseline_subject = np.asarray([
            np.mean([
                1.0 - float(baseline_rows[key][metric])
                / max(float(baseline_rows[key][nochange_metric]), 1.0e-12)
                for key in by_subject[subject]
            ])
            for subject in subjects
        ])
        candidate_subject = np.asarray([
            np.mean([
                1.0 - float(candidate_rows[key][metric])
                / max(float(candidate_rows[key][nochange_metric]), 1.0e-12)
                for key in by_subject[subject]
            ])
            for subject in subjects
        ])
        differences = candidate_subject - baseline_subject
        estimates = differences[bootstrap_indices].mean(axis=1)
        nochange_improvement[metric] = {
            "baseline_fractional_improvement_mean": float(baseline_subject.mean()),
            "candidate_fractional_improvement_mean": float(candidate_subject.mean()),
            "candidate_minus_baseline_improvement_mean": float(differences.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
            "fraction_subjects_candidate_improvement_larger": float(np.mean(differences > 0.0)),
            "positive_favors_candidate": True,
        }
    primary = "end_to_end_coordinate_rmse"
    baseline_floor = float(baseline["representation_floor"]["coordinate_rmse_mm_mean"])
    candidate_floor = float(candidate["representation_floor"]["coordinate_rmse_mm_mean"])
    output = {
        "schema_version": 1,
        "split": args.split,
        "baseline": baseline["representation"],
        "candidate": candidate["representation"],
        "baseline_run": str(args.baseline_run.expanduser().resolve()),
        "candidate_run": str(args.candidate_run.expanduser().resolve()),
        "subjects": len(subjects),
        "pairs": len(keys),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "subject",
        "metric_interpretation": {
            "primary": "end_to_end_coordinate_rmse compares predictions to raw target meshes",
            "transport_only": "coordinate/euclidean compare predictions to each representation's decoded target",
            "latent": "standardized latent MSE is diagnostic only because the two latent bases differ",
        },
        "metrics": metrics,
        "fractional_improvement_over_nochange": nochange_improvement,
        "floor_adjusted_primary": {
            "metric": primary,
            "baseline_representation_floor_rmse_mm": baseline_floor,
            "candidate_representation_floor_rmse_mm": candidate_floor,
            "baseline_error_over_floor": metrics[primary]["baseline_mean"] / max(baseline_floor, 1.0e-12),
            "candidate_error_over_floor": metrics[primary]["candidate_mean"] / max(candidate_floor, 1.0e-12),
            "baseline_excess_over_floor_mm": metrics[primary]["baseline_mean"] - baseline_floor,
            "candidate_excess_over_floor_mm": metrics[primary]["candidate_mean"] - candidate_floor,
        },
    }
    C.assert_finite_mapping(output)
    destination = C.require_bulk_path(args.output, "paired comparison output")
    if destination.suffix.lower() != ".json":
        raise ValueError("--output must end in .json")
    if destination.exists():
        raise FileExistsError(destination)
    C.atomic_json(destination, output)
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
