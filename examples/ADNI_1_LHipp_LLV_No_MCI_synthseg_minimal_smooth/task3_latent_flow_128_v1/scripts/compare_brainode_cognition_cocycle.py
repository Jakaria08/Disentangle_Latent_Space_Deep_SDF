#!/usr/bin/env python3
"""Compare PCA128 BrainODE cognition modes against the PCA128 direct cocycle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainode-run", type=Path, required=True)
    parser.add_argument("--cocycle-run", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--brainode-evaluation-name", default="brainode_cognition_evaluation")
    parser.add_argument("--cocycle-evaluation-name", default="brainode_cognition_comparator")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pair_report(report: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "pca_cocycle":
        return report["pair_metrics"]
    return report["transports"][mode]["pair_metrics"]


def summary_rows(brain: dict[str, Any], cocycle: dict[str, Any], split: str) -> list[dict[str, Any]]:
    modes = {
        "brainode_fixed_label": pair_report(brain, "fixed_observed_label"),
        "brainode_voxel_feedback": pair_report(brain, "voxel_cognition_feedback"),
        "pca_cocycle": pair_report(cocycle, "pca_cocycle"),
    }
    output = []
    for method, report in modes.items():
        for stratum, values in sorted(report.items()):
            for diagnosis, metrics in sorted(values["groups"].items()):
                if int(metrics.get("rows", 0)) == 0:
                    continue
                output.append({
                    "representation": "pca128",
                    "method": method,
                    "split": split,
                    "stratum": stratum,
                    "diagnosis": diagnosis,
                    **metrics,
                })
    # No-change is identical for all three methods and is therefore emitted once.
    nochange = modes["pca_cocycle"]["all_forward"]["groups"]
    for diagnosis, metrics in sorted(nochange.items()):
        output.append({
            "representation": "pca128",
            "method": "no_change",
            "split": split,
            "stratum": "all_forward",
            "diagnosis": diagnosis,
            "rows": metrics["rows"],
            "coordinate_mean": metrics["nochange_coordinate_mean"],
            "euclidean_mean": metrics["nochange_euclidean_mean"],
            "end_to_end_coordinate_mae_mean": metrics["nochange_end_to_end_coordinate_mae_mean"],
            "end_to_end_euclidean_mean": metrics["nochange_end_to_end_euclidean_mean"],
            "volume_relative_mean": metrics["nochange_volume_relative_mean"],
            "rate_mean": metrics["nochange_rate_mean"],
        })
    return output


def indexed_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(row["subject"]), int(row["source_index"]), int(row["target_index"])): row
        for row in rows
    }


def paired_bootstrap(
    candidate_rows: list[dict[str, Any]],
    cocycle_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate = indexed_rows(candidate_rows)
    cocycle = indexed_rows(cocycle_rows)
    keys = sorted(set(candidate) & set(cocycle))
    if set(candidate) != set(cocycle):
        raise ValueError("BrainODE/cocycle evaluated pair sets differ")
    subjects = sorted({key[0] for key in keys})
    metrics = ("coordinate", "euclidean", "end_to_end_coordinate_rmse", "volume_relative", "rate")
    subject_differences: dict[str, dict[str, float]] = {}
    for subject in subjects:
        current = [key for key in keys if key[0] == subject]
        subject_differences[subject] = {
            metric: float(np.mean([candidate[key][metric] - cocycle[key][metric] for key in current]))
            for metric in metrics
        }
    generator = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(int(samples)):
        chosen = generator.choice(subjects, size=len(subjects), replace=True)
        for metric in metrics:
            draws[metric].append(float(np.mean([subject_differences[str(subject)][metric] for subject in chosen])))
    return {
        "difference": "candidate minus PCA cocycle; negative favors candidate",
        "unit": "subject",
        "subjects": len(subjects),
        "pairs": len(keys),
        "samples": int(samples),
        "metrics": {
            metric: {
                "mean_difference": float(np.mean([subject_differences[subject][metric] for subject in subjects])),
                "ci95_low": float(np.quantile(draws[metric], 0.025)),
                "ci95_high": float(np.quantile(draws[metric], 0.975)),
                "subject_win_fraction": float(np.mean([subject_differences[subject][metric] < 0.0 for subject in subjects])),
            }
            for metric in metrics
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata = ["representation", "method", "split", "stratum", "diagnosis"]
    fields = metadata + sorted({key for row in rows for key in row}.difference(metadata))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    brain_path = args.brainode_run.expanduser().resolve() / args.brainode_evaluation_name / args.split / "summary.json"
    cocycle_path = args.cocycle_run.expanduser().resolve() / args.cocycle_evaluation_name / args.split / "summary.json"
    brain, cocycle = C.read_json(brain_path), C.read_json(cocycle_path)
    if brain["split"] != args.split or cocycle["split"] != args.split:
        raise ValueError("Evaluation split mismatch")
    if brain["representation"] != "pca128" or cocycle["representation"] != "pca128":
        raise ValueError("This comparison requires PCA128 on both sides")
    if bool(brain.get("test_loaded_during_training", True)) or bool(cocycle.get("test_loaded_during_training", True)):
        raise ValueError("Training/test leakage flag found")
    rows = summary_rows(brain, cocycle, args.split)
    brain_fixed_rows = brain["transports"]["fixed_observed_label"]["pair_metrics"]["all_forward"]["row_metrics"]
    brain_feedback_rows = brain["transports"]["voxel_cognition_feedback"]["pair_metrics"]["all_forward"]["row_metrics"]
    cocycle_rows = cocycle["pair_metrics"]["all_forward"]["row_metrics"]
    paired = {
        "brainode_fixed_label_vs_pca_cocycle": paired_bootstrap(
            brain_fixed_rows, cocycle_rows, args.bootstrap_samples, 421
        ),
        "brainode_voxel_feedback_vs_pca_cocycle": paired_bootstrap(
            brain_feedback_rows, cocycle_rows, args.bootstrap_samples, 422
        ),
    }
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must end in .csv")
    json_path = output.with_suffix(".json")
    if output.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {json_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    C.atomic_json(json_path, {
        "split": args.split,
        "brainode_report": str(brain_path),
        "cocycle_report": str(cocycle_path),
        "summary_rows": rows,
        "paired_subject_bootstrap": paired,
        "cognition_estimator_metrics": brain["cognition_estimator"],
        "condition_injectivity": brain["condition_injectivity"],
        "paper_mapping": brain["paper_mapping"],
        "claim_boundary": brain["cohort_contract"],
    })
    print(f"WROTE {output} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
