#!/usr/bin/env python3
"""Compare volume-coboundary V2 with matched ODE, BrainODE, and direct-C4 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REPRESENTATIONS = ("pca128", "spiralnet128", "adaptive128")
METHODS = {
    "plain_ode": "plain_ode/{representation}_plain_ode_s42",
    "brainode": "brainode/{representation}_brainode_s42",
    "direct_c4": "direct_c4/{representation}_direct_c4_s42",
    "volume_exact_coboundary_v2": (
        "volume_exact_coboundary_c4_v2/{representation}_volume_exact_coboundary_v2_s42"
    ),
}
TARGET_METHOD = "volume_exact_coboundary_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training"),
    )
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def load_reports(args: argparse.Namespace) -> dict[tuple[str, str, str], dict[str, Any]]:
    reports = {}
    for representation in REPRESENTATIONS:
        for method, template in METHODS.items():
            run = args.training_root / representation / template.format(representation=representation)
            for split in ("val", "test"):
                path = run / args.evaluation_name / split / "summary.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("split") != split or report.get("representation") != representation:
                    raise ValueError(f"Report identity mismatch: {path}")
                reports[(representation, method, split)] = report
    return reports


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Invalid metric ratio")
    return float(numerator / denominator)


def summary_row(representation: str, method: str, split: str, report: dict[str, Any]) -> dict[str, Any]:
    overall = report["pair_metrics"]["all_forward"]["groups"]["overall"]
    first_last = report["pair_metrics"]["first_last_forward"]["groups"]
    defects = report["consistency_defects"]
    row = {
        "representation": representation,
        "method": method,
        "split": split,
        "all_coordinate_mae_mm": overall["coordinate_mean"],
        "all_coordinate_ratio_to_nochange": safe_ratio(
            overall["coordinate_mean"], overall["nochange_coordinate_mean"]
        ),
        "all_euclidean_mm": overall["euclidean_mean"],
        "all_euclidean_ratio_to_nochange": safe_ratio(
            overall["euclidean_mean"], overall["nochange_euclidean_mean"]
        ),
        "all_end_to_end_rmse_mm": overall["end_to_end_coordinate_rmse_mean"],
        "all_volume_relative_error": overall["volume_relative_mean"],
        "all_volume_ratio_to_nochange": safe_ratio(
            overall["volume_relative_mean"], overall["nochange_volume_relative_mean"]
        ),
        "all_log_volume_rate_error_per_year": overall["rate_mean"],
        "semigroup_defect": defects["relative_semigroup_defect_mean"],
        "inverse_defect": defects["relative_inverse_defect_mean"],
        "structural_audit_passed": bool(report.get("structural_audit_passed", False)),
    }
    for diagnosis in ("CN", "AD"):
        group = first_last[diagnosis]
        key = diagnosis.lower()
        row[f"first_last_{key}_coordinate_ratio"] = safe_ratio(
            group["coordinate_mean"], group["nochange_coordinate_mean"]
        )
        row[f"first_last_{key}_euclidean_ratio"] = safe_ratio(
            group["euclidean_mean"], group["nochange_euclidean_mean"]
        )
        row[f"first_last_{key}_volume_ratio"] = safe_ratio(
            group["volume_relative_mean"], group["nochange_volume_relative_mean"]
        )
        row[f"first_last_{key}_predicted_signed_rate"] = group["predicted_signed_rate_mean"]
        row[f"first_last_{key}_observed_signed_rate"] = group["observed_signed_rate_mean"]
    row["macro_first_last_shape_ratio"] = 0.25 * sum(
        row[f"first_last_{diagnosis}_{metric}_ratio"]
        for diagnosis in ("cn", "ad") for metric in ("coordinate", "euclidean")
    )
    row["predicted_ad_minus_cn_rate_gap"] = (
        row["first_last_ad_predicted_signed_rate"] - row["first_last_cn_predicted_signed_rate"]
    )
    row["observed_ad_minus_cn_rate_gap"] = (
        row["first_last_ad_observed_signed_rate"] - row["first_last_cn_observed_signed_rate"]
    )
    return row


def subject_means(report: dict[str, Any], metric: str) -> dict[str, float]:
    rows = report["pair_metrics"]["all_forward"]["row_metrics"]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject"]), []).append(float(row[metric]))
    return {subject: float(np.mean(values)) for subject, values in grouped.items()}


def paired_bootstrap(
    target: dict[str, Any], baseline: dict[str, Any], metric: str, samples: int, seed: int,
) -> dict[str, Any]:
    left, right = subject_means(target, metric), subject_means(baseline, metric)
    subjects = sorted(set(left) & set(right))
    if set(left) != set(right) or not subjects:
        raise ValueError(f"Subject mismatch for paired {metric} comparison")
    differences = np.asarray([left[subject] - right[subject] for subject in subjects], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draws[index] = np.mean(generator.choice(differences, size=len(differences), replace=True))
    return {
        "metric": metric,
        "difference": "volume_exact_coboundary_v2_minus_baseline; negative favors V2",
        "subjects": len(subjects),
        "mean_difference": float(np.mean(differences)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_v2_better": float(np.mean(draws < 0.0)),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = (
        "representation", "method", "split", "macro_first_last_shape_ratio",
        "all_coordinate_ratio_to_nochange", "all_end_to_end_rmse_mm",
        "all_volume_ratio_to_nochange", "predicted_ad_minus_cn_rate_gap",
        "observed_ad_minus_cn_rate_gap", "semigroup_defect", "inverse_defect",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column]) for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *body)) + "\n"


def main() -> int:
    args = parse_args()
    reports = load_reports(args)
    rows = [
        summary_row(representation, method, split, reports[(representation, method, split)])
        for representation in REPRESENTATIONS for split in ("val", "test") for method in METHODS
    ]
    paired = []
    for representation_index, representation in enumerate(REPRESENTATIONS):
        for split_index, split in enumerate(("val", "test")):
            target = reports[(representation, TARGET_METHOD, split)]
            for baseline_index, baseline_name in enumerate(("plain_ode", "brainode", "direct_c4")):
                baseline = reports[(representation, baseline_name, split)]
                for metric_index, metric in enumerate(("coordinate", "euclidean", "volume_relative")):
                    paired.append({
                        "representation": representation, "split": split, "baseline": baseline_name,
                        **paired_bootstrap(
                            target, baseline, metric, args.bootstrap_samples,
                            10_000 * representation_index + 1_000 * split_index
                            + 100 * baseline_index + metric_index + 42,
                        ),
                    })
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"summary_rows": rows, "paired_subject_bootstrap": paired}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.md").write_text(markdown_table(rows), encoding="utf-8")
    print(f"WROTE comparison files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
