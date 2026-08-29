#!/usr/bin/env python3
"""Assemble INR-256 and previous 128-D direct-C4 results without hiding scope differences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as C


PREVIOUS_ROOT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inr-run", type=Path, required=True)
    parser.add_argument("--inr-evaluation-name", default="evaluation")
    parser.add_argument("--previous-evaluation-name", default="evaluation")
    parser.add_argument("--output-name", default="comparison_previous_128d")
    return parser.parse_args()


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def old_row(representation: str, evaluation_name: str):
    run_name = f"{representation}_direct_c4_s42"
    run = PREVIOUS_ROOT / representation / "direct_c4" / run_name
    status = load(run / "training_status.json")
    evaluation = run / evaluation_name / "test" / "summary.json"
    row = {
        "representation": representation,
        "latent_dim": 128,
        "method": "direct_c4",
        "cohort": "full 128-D task test split (277 scans, 61 subjects)",
        "best_epoch": status["best_epoch"],
        "validation_selection_score": status["best_validation_selection_score"],
        "test_metric_kind": "fixed-topology decoded vertex coordinate/euclidean ratio",
        "test_macro_first_last_ratio": None,
        "test_macro_first_last_surface_assd_ratio": None,
        "test_macro_first_last_volume_ratio": None,
        "semigroup_defect_mean": None,
        "inverse_defect_mean": None,
        "representation_floor_metric": None,
        "representation_floor_value": None,
        "test_evaluation": str(evaluation) if evaluation.is_file() else None,
    }
    if evaluation.is_file():
        report = load(evaluation)
        group = report["pair_metrics"]["first_last_forward"]["groups"]
        ratios = []
        for diagnosis in ("CN", "AD"):
            values = group[diagnosis]
            ratios.append(0.5 * (values["coordinate_mean"] / max(values["nochange_coordinate_mean"], 1.0e-8) + values["euclidean_mean"] / max(values["nochange_euclidean_mean"], 1.0e-8)))
        row["test_macro_first_last_ratio"] = float(np.mean(ratios))
        row["test_macro_first_last_volume_ratio"] = float(np.mean([
            group[diagnosis]["volume_relative_mean"] / max(group[diagnosis]["nochange_volume_relative_mean"], 1.0e-8)
            for diagnosis in ("CN", "AD")
        ]))
        row["semigroup_defect_mean"] = report["consistency_defects"]["relative_semigroup_defect_mean"]
        row["inverse_defect_mean"] = report["consistency_defects"]["relative_inverse_defect_mean"]
        row["representation_floor_metric"] = "coordinate_rmse_mm"
        row["representation_floor_value"] = report["representation_floor"]["coordinate_rmse_mm_mean"]
    return row


def main():
    args = parse_args()
    inr_run = args.inr_run.expanduser().resolve()
    inr_status = load(inr_run / "training_status.json")
    inr_evaluation = inr_run / args.inr_evaluation_name / "test" / "summary.json"
    if not inr_evaluation.is_file():
        raise FileNotFoundError(inr_evaluation)
    report = load(inr_evaluation)
    first = report["proxy_pair_metrics"]["first_last_forward"]["groups"]
    inr_ratio = float(np.mean([first[diagnosis]["sdf_mean"] / max(first[diagnosis]["nochange_sdf_mean"], 1.0e-8) for diagnosis in ("CN", "AD")]))
    surface = report["first_last_surface_metrics"]["groups"]
    surface_ratio = float(np.mean([
        surface[diagnosis]["predicted_assd_mm_mean"] / max(surface[diagnosis]["nochange_assd_mm_mean"], 1.0e-8)
        for diagnosis in ("CN", "AD")
    ]))
    volume_ratio = float(np.mean([
        surface[diagnosis]["predicted_volume_relative_error_mean"] / max(surface[diagnosis]["nochange_volume_relative_error_mean"], 1.0e-8)
        for diagnosis in ("CN", "AD")
    ]))
    rows = [{
        "representation": "inr256",
        "latent_dim": 256,
        "method": "direct_c4",
        "cohort": "exact-SDF INR subset test split (100 scans, 20 subjects)",
        "best_epoch": inr_status["best_epoch"],
        "validation_selection_score": inr_status["best_validation_selection_score"],
        "test_metric_kind": "decoded exact-SDF MAE ratio",
        "test_macro_first_last_ratio": inr_ratio,
        "test_macro_first_last_surface_assd_ratio": surface_ratio,
        "test_macro_first_last_volume_ratio": volume_ratio,
        "semigroup_defect_mean": report["consistency_defects"]["relative_semigroup_defect_mean"],
        "inverse_defect_mean": report["consistency_defects"]["relative_inverse_defect_mean"],
        "representation_floor_metric": "exact_point_to_triangle_assd_mm",
        "representation_floor_value": surface["overall"]["representation_floor_assd_mm_mean"],
        "test_evaluation": str(inr_evaluation),
    }]
    rows += [old_row(name, args.previous_evaluation_name) for name in ("pca128", "spiralnet128", "adaptive128")]
    comparison = {
        "rows": rows,
        "interpretation": [
            "Ratios below 1 improve over carrying the source shape forward unchanged.",
            "INR SDF and 128-D coordinate ratios use different decoder-native proxy metrics and different test cohort sizes; compare direction and relative improvement, not raw metric magnitude.",
            "The INR surface ASSD ratio uses exact sampled-point-to-triangle MC-256 metrics. The older evaluator did not report surface ASSD, so that column is intentionally blank for the 128-D rows.",
            "All 100 INR test scans are contained in the older 277-scan test cohort, but the original published evaluator currently reports its complete split.",
        ],
    }
    C.validate_run_name(args.output_name)
    destination = inr_run / args.output_name
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "comparison.json", comparison)
    C.write_csv(destination / "comparison.csv", rows)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
