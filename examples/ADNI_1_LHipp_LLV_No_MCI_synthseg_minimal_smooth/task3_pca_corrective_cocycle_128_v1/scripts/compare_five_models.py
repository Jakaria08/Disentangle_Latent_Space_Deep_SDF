#!/usr/bin/env python3
"""Build full-cohort and matched-cohort PCA/Spiral/Adaptive/corrective/INR tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import common as C


FIXED_MODELS = ("pca128", "spiralnet128", "adaptive128", "pca_corrective128")
CORRECTIVE_PCA_BRANCH = "pca_corrective128_same_flow_pca_decode"
LATENT_DIMS = {**{name: 128 for name in FIXED_MODELS}, CORRECTIVE_PCA_BRANCH: 128, "inr256": 256}
DISPLAY_NAMES = {
    "pca128": "PCA-128",
    "spiralnet128": "SpiralNet++-128",
    "adaptive128": "Adaptive-Spiral-128",
    "pca_corrective128": "PCA-corrective-Spiral-128",
    CORRECTIVE_PCA_BRANCH: "Corrective flow + PCA-only decode",
    "inr256": "INR-256",
}
EXACT_ERROR_METRICS = (
    "prediction_assd_mm",
    "prediction_hd95_mm",
    "prediction_chamfer_l2_squared_mm2",
    "prediction_volume_absolute_error_mm3",
    "prediction_volume_relative_error",
    "prediction_log_volume_rate_absolute_error_per_year",
)
MEAN_METRICS = (
    "prediction_coordinate_rmse_mm",
    "prediction_mean_vertex_euclidean_mm",
    "prediction_assd_mm",
    "prediction_hd95_mm",
    "prediction_chamfer_l1_mm",
    "prediction_chamfer_l2_squared_mm2",
    "prediction_normal_signed_cosine",
    "prediction_flipped_face_fraction_vs_ground_truth",
    "prediction_volume_absolute_error_mm3",
    "prediction_volume_relative_error",
    "prediction_surface_area_ratio",
    "prediction_curvature_ratio_to_ground_truth",
    "prediction_predicted_connected_components",
    "prediction_predicted_euler_number",
    "nochange_coordinate_rmse_mm",
    "nochange_mean_vertex_euclidean_mm",
    "nochange_assd_mm",
    "nochange_hd95_mm",
    "nochange_chamfer_l1_mm",
    "nochange_chamfer_l2_squared_mm2",
    "nochange_volume_absolute_error_mm3",
    "nochange_volume_relative_error",
    "floor_coordinate_rmse_mm",
    "floor_mean_vertex_euclidean_mm",
    "floor_assd_mm",
    "floor_hd95_mm",
    "floor_chamfer_l2_squared_mm2",
    "floor_volume_relative_error",
    "transport_coordinate_rmse_mm",
    "transport_mean_vertex_euclidean_mm",
    "source_true_volume_mm3",
    "target_true_volume_mm3",
    "prediction_volume_mm3",
    "observed_signed_log_volume_rate_per_year",
    "prediction_signed_log_volume_rate_raw_anchor_per_year",
    "prediction_log_volume_rate_absolute_error_per_year",
    "nochange_log_volume_rate_absolute_error_per_year",
    "observed_volume_change_mm3_per_year",
    "prediction_volume_change_raw_anchor_mm3_per_year",
    "observed_annualized_percent_change",
    "prediction_annualized_percent_change_raw_anchor",
    "prediction_atrophy_direction_agreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-surface-dir", type=Path, required=True)
    parser.add_argument("--inr-run", type=Path, required=True)
    parser.add_argument("--inr-exact-evaluation-name", default="evaluation_mc256_exact")
    parser.add_argument("--inr-volume-analysis-name", default="volume_trend_analysis_mc256")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    output = float(value)
    return output if np.isfinite(output) else None


def mean_available(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := finite_float(row, key)) is not None]
    return float(np.mean(values)) if values else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator / max(denominator, 1.0e-12))


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int((int(digest[:8], 16) + base) % (2**31 - 1))


def summarize_rows(
    rows: list[dict[str, Any]], cohort: str, expected_models: Iterable[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in expected_models:
        model_rows = [row for row in rows if row["representation"] == model]
        if not model_rows:
            raise ValueError(f"No surface rows for {model}")
        for diagnosis in ("CN", "AD", "overall"):
            group = model_rows if diagnosis == "overall" else [
                row for row in model_rows if row["diagnosis"] == diagnosis
            ]
            if not group:
                continue
            record: dict[str, Any] = {
                "representation": model,
                "display_name": DISPLAY_NAMES[model],
                "latent_dim": LATENT_DIMS[model],
                "cohort": cohort,
                "diagnosis": diagnosis,
                "subjects": len(group),
            }
            for metric in MEAN_METRICS:
                record[f"{metric}_mean"] = mean_available(group, metric)
            for prefix in ("prediction", "nochange", "floor"):
                for flag in ("predicted_watertight", "predicted_winding_consistent"):
                    key = f"{prefix}_{flag}"
                    values = [str(row[key]).lower() == "true" for row in group if row.get(key) not in (None, "")]
                    record[f"{key}_fraction"] = float(np.mean(values)) if values else None
            for metric in ("assd_mm", "hd95_mm", "chamfer_l2_squared_mm2", "volume_relative_error"):
                record[f"prediction_{metric}_ratio_to_nochange"] = ratio(
                    record.get(f"prediction_{metric}_mean"),
                    record.get(f"nochange_{metric}_mean"),
                )
            record["prediction_volume_absolute_error_mm3_ratio_to_nochange"] = ratio(
                record.get("prediction_volume_absolute_error_mm3_mean"),
                record.get("nochange_volume_absolute_error_mm3_mean"),
            )
            record["prediction_rate_mae_ratio_to_nochange"] = ratio(
                record.get("prediction_log_volume_rate_absolute_error_per_year_mean"),
                record.get("nochange_log_volume_rate_absolute_error_per_year_mean"),
            )
            output.append(record)
    return output


def inr_rows(exact: dict[str, Any], volume_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_rows = exact["first_last_surface_metrics"]["row_metrics"]
    exact_by_subject = {str(row["subject"]): row for row in exact_rows}
    volume_by_subject = {str(row["subject_id"]): row for row in volume_rows}
    if set(exact_by_subject) != set(volume_by_subject):
        raise ValueError("INR exact-surface and volume subject sets differ")
    output = []
    for subject in sorted(exact_by_subject):
        surface, volume = exact_by_subject[subject], volume_by_subject[subject]
        if str(surface["diagnosis"]) != str(volume["diagnosis"]):
            raise ValueError(f"INR diagnosis disagreement for {subject}")
        source_volume = float(volume["source_true_volume_mm3"])
        target_volume = float(volume["target_true_volume_mm3"])
        predicted_volume = float(volume["predicted_target_volume_mm3_mc"])
        output.append({
            "representation": "inr256",
            "source_representation": "inr256",
            "split": "test",
            "subject_id": subject,
            "diagnosis": str(volume["diagnosis"]),
            "source_scan_id": str(volume["source_scan_id"]),
            "target_scan_id": str(volume["target_scan_id"]),
            "followup_years": float(volume["followup_years"]),
            "prediction_assd_mm": float(surface["predicted_assd_mm"]),
            "prediction_hd95_mm": float(surface["predicted_hd95_mm"]),
            "prediction_chamfer_l2_squared_mm2": float(surface["predicted_chamfer_l2_squared_mm2"]),
            "prediction_volume_relative_error": float(volume["predicted_target_volume_relative_error"]),
            "nochange_assd_mm": float(surface["nochange_assd_mm"]),
            "nochange_hd95_mm": float(surface["nochange_hd95_mm"]),
            "nochange_chamfer_l2_squared_mm2": float(surface["nochange_chamfer_l2_squared_mm2"]),
            "nochange_volume_relative_error": float(volume["nochange_target_volume_relative_error"]),
            "floor_assd_mm": float(surface["representation_floor_assd_mm"]),
            "floor_hd95_mm": float(surface["representation_floor_hd95_mm"]),
            "floor_chamfer_l2_squared_mm2": float(surface["representation_floor_chamfer_l2_squared_mm2"]),
            "floor_volume_relative_error": float(surface["representation_floor_volume_relative_error"]),
            "source_true_volume_mm3": source_volume,
            "target_true_volume_mm3": target_volume,
            "prediction_volume_mm3": predicted_volume,
            "prediction_volume_absolute_error_mm3": abs(predicted_volume - target_volume),
            "nochange_volume_absolute_error_mm3": abs(source_volume - target_volume),
            "observed_signed_log_volume_rate_per_year": float(volume["observed_signed_log_rate_per_year"]),
            "prediction_signed_log_volume_rate_raw_anchor_per_year": float(volume["predicted_signed_log_rate_per_year"]),
            "prediction_log_volume_rate_absolute_error_per_year": float(volume["absolute_rate_error_per_year"]),
            "nochange_log_volume_rate_absolute_error_per_year": float(volume["nochange_absolute_rate_error_per_year"]),
            "observed_volume_change_mm3_per_year": (target_volume - source_volume) / float(volume["followup_years"]),
            "prediction_volume_change_raw_anchor_mm3_per_year": (predicted_volume - source_volume) / float(volume["followup_years"]),
            "observed_annualized_percent_change": float(volume["observed_annualized_percent_change"]),
            "prediction_annualized_percent_change_raw_anchor": float(volume["predicted_annualized_percent_change"]),
            "prediction_atrophy_direction_agreement": float(volume["atrophy_direction_agreement"]),
        })
    return output


def trend_gaps(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {(row["representation"], row["diagnosis"]): row for row in summary}
    output = []
    for model in sorted({row["representation"] for row in summary}):
        cn, ad = grouped[(model, "CN")], grouped[(model, "AD")]
        observed = float(ad["observed_signed_log_volume_rate_per_year_mean"] - cn["observed_signed_log_volume_rate_per_year_mean"])
        predicted = float(ad["prediction_signed_log_volume_rate_raw_anchor_per_year_mean"] - cn["prediction_signed_log_volume_rate_raw_anchor_per_year_mean"])
        output.append({
            "representation": model,
            "display_name": DISPLAY_NAMES[model],
            "cohort": cn["cohort"],
            "cn_subjects": cn["subjects"],
            "ad_subjects": ad["subjects"],
            "cn_observed_percent_per_year": cn["observed_annualized_percent_change_mean"],
            "cn_predicted_percent_per_year": cn["prediction_annualized_percent_change_raw_anchor_mean"],
            "ad_observed_percent_per_year": ad["observed_annualized_percent_change_mean"],
            "ad_predicted_percent_per_year": ad["prediction_annualized_percent_change_raw_anchor_mean"],
            "observed_ad_minus_cn_log_rate_gap": observed,
            "predicted_ad_minus_cn_log_rate_gap": predicted,
            "gap_absolute_error": abs(predicted - observed),
            "gap_recovery_fraction": predicted / observed if abs(observed) > 1.0e-12 else None,
            "ad_more_atrophy_than_cn_predicted": bool(predicted < 0.0),
        })
    return output


def bootstrap_against_pca(
    rows: list[dict[str, Any]], samples: int, seed: int, models: Iterable[str]
) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model[str(row["representation"])][str(row["subject_id"])] = row
    output = []
    for model in models:
        if model == "pca128":
            continue
        subjects = sorted(set(by_model["pca128"]).intersection(by_model[model]))
        if not subjects:
            raise ValueError(f"No subjects shared by PCA and {model}")
        for metric in EXACT_ERROR_METRICS:
            pairs = [
                (finite_float(by_model[model][subject], metric), finite_float(by_model["pca128"][subject], metric))
                for subject in subjects
            ]
            valid = [(current, baseline) for current, baseline in pairs if current is not None and baseline is not None]
            if not valid:
                continue
            differences = np.asarray([current - baseline for current, baseline in valid], dtype=np.float64)
            rng = np.random.default_rng(stable_seed(model + metric, seed))
            indices = rng.integers(0, len(differences), size=(samples, len(differences)))
            estimates = differences[indices].mean(axis=1)
            output.append({
                "representation": model,
                "display_name": DISPLAY_NAMES[model],
                "baseline": "pca128",
                "metric": metric,
                "subjects": len(differences),
                "current_minus_pca_mean": float(differences.mean()),
                "ci95_low": float(np.quantile(estimates, 0.025)),
                "ci95_high": float(np.quantile(estimates, 0.975)),
                "fraction_subjects_better": float(np.mean(differences < 0.0)),
                "negative_favors_current": True,
            })
    return output


def decoder_ablation_bootstrap(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    """Compare corrected versus PCA-only decoding of the exact same transported latent."""
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_model[str(row["representation"])][str(row["subject_id"])] = row
    current_name = "pca_corrective128"
    baseline_name = CORRECTIVE_PCA_BRANCH
    subjects = sorted(set(by_model[current_name]).intersection(by_model[baseline_name]))
    if not subjects:
        raise ValueError("Corrective decoder ablation has no paired subjects")
    output = []
    for metric in (*EXACT_ERROR_METRICS, "floor_assd_mm", "floor_volume_relative_error"):
        differences = np.asarray([
            float(by_model[current_name][subject][metric])
            - float(by_model[baseline_name][subject][metric])
            for subject in subjects
        ], dtype=np.float64)
        rng = np.random.default_rng(stable_seed("decoder_ablation" + metric, seed))
        indices = rng.integers(0, len(differences), size=(samples, len(differences)))
        estimates = differences[indices].mean(axis=1)
        output.append({
            "representation": current_name,
            "display_name": DISPLAY_NAMES[current_name],
            "baseline": baseline_name,
            "baseline_display_name": DISPLAY_NAMES[baseline_name],
            "latent_transport": "identical transported corrective-PCA latent",
            "metric": metric,
            "subjects": len(subjects),
            "corrective_minus_pca_decode_mean": float(differences.mean()),
            "ci95_low": float(np.quantile(estimates, 0.025)),
            "ci95_high": float(np.quantile(estimates, 0.975)),
            "fraction_subjects_corrective_better": float(np.mean(differences < 0.0)),
            "negative_favors_corrective": True,
        })
    return output


def standard_row(
    model: str, report: dict[str, Any], run: Path,
    surface_groups: dict[str, dict[str, Any]], cohort: str
) -> dict[str, Any]:
    status = C.read_json(run / "training_status.json")
    overall = report["pair_metrics"]["first_last_forward"]["groups"]["overall"]
    all_pairs = report["pair_metrics"]["all_forward"]["groups"]["overall"]
    defects = report["consistency_defects"]
    floor = report["representation_floor"]
    surface = surface_groups["overall"]
    macro_primary = float(np.mean([
        0.5 * (
            report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["coordinate_mean"]
            / report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["nochange_coordinate_mean"]
            + report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["euclidean_mean"]
            / report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["nochange_euclidean_mean"]
        )
        for diagnosis in ("CN", "AD")
    ]))
    macro_volume = float(np.mean([
        report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["volume_relative_mean"]
        / report["pair_metrics"]["first_last_forward"]["groups"][diagnosis]["nochange_volume_relative_mean"]
        for diagnosis in ("CN", "AD")
    ]))
    macro_assd = float(np.mean([
        surface_groups[diagnosis]["prediction_assd_mm_mean"]
        / surface_groups[diagnosis]["nochange_assd_mm_mean"]
        for diagnosis in ("CN", "AD")
    ]))
    return {
        "representation": model,
        "display_name": DISPLAY_NAMES[model],
        "latent_dim": LATENT_DIMS[model],
        "cohort": cohort,
        "best_epoch": status["best_epoch"],
        "validation_selection_score": status["best_validation_selection_score"],
        "method": "direct_c4",
        "test_metric_kind": "fixed-topology decoded vertex coordinate/euclidean ratio",
        "test_macro_first_last_ratio": macro_primary,
        "test_macro_first_last_surface_assd_ratio": macro_assd,
        "test_macro_first_last_volume_ratio": macro_volume,
        "primary_decoder_metric": "fixed-topology decoded vertex Euclidean distance",
        "first_last_primary_error": overall["euclidean_mean"],
        "first_last_primary_nochange": overall["nochange_euclidean_mean"],
        "first_last_primary_ratio_to_nochange": ratio(overall["euclidean_mean"], overall["nochange_euclidean_mean"]),
        "first_last_end_to_end_rmse_mm": overall["end_to_end_coordinate_rmse_mean"],
        "all_pair_transport_coordinate_mae_mm": all_pairs["coordinate_mean"],
        "first_last_volume_relative_error": overall["volume_relative_mean"],
        "first_last_volume_ratio_to_nochange": ratio(overall["volume_relative_mean"], overall["nochange_volume_relative_mean"]),
        "first_last_rate_mae_ratio_to_nochange": ratio(overall["rate_mean"], overall["nochange_rate_mean"]),
        "surface_assd_mm": surface["prediction_assd_mm_mean"],
        "surface_assd_ratio_to_nochange": surface["prediction_assd_mm_ratio_to_nochange"],
        "surface_hd95_mm": surface["prediction_hd95_mm_mean"],
        "surface_volume_absolute_error_mm3": surface["prediction_volume_absolute_error_mm3_mean"],
        "surface_volume_relative_error": surface["prediction_volume_relative_error_mean"],
        "surface_rate_mae_ratio_to_nochange": surface["prediction_rate_mae_ratio_to_nochange"],
        "representation_floor_metric": "coordinate_rmse_mm",
        "representation_floor_value": floor["coordinate_rmse_mm_mean"],
        "surface_representation_floor_assd_mm": surface["floor_assd_mm_mean"],
        "semigroup_defect_mean": defects["relative_semigroup_defect_mean"],
        "inverse_defect_mean": defects["relative_inverse_defect_mean"],
        "relative_semigroup_defect_mean": defects["relative_semigroup_defect_mean"],
        "relative_inverse_defect_mean": defects["relative_inverse_defect_mean"],
        "evaluation": str(run / "evaluation" / "test" / "summary.json"),
    }


def inr_standard_row(
    exact: dict[str, Any], volume_report: dict[str, Any], run: Path,
    exact_path: Path, surface_groups: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    status = C.read_json(run / "training_status.json")
    proxy = exact["proxy_pair_metrics"]["first_last_forward"]["groups"]["overall"]
    defects = exact["consistency_defects"]
    proxy_groups = exact["proxy_pair_metrics"]["first_last_forward"]["groups"]
    surface = surface_groups["overall"]
    macro_primary = float(np.mean([
        proxy_groups[diagnosis]["sdf_mean"] / proxy_groups[diagnosis]["nochange_sdf_mean"]
        for diagnosis in ("CN", "AD")
    ]))
    macro_assd = float(np.mean([
        surface_groups[diagnosis]["prediction_assd_mm_mean"]
        / surface_groups[diagnosis]["nochange_assd_mm_mean"]
        for diagnosis in ("CN", "AD")
    ]))
    exact_surface_groups = exact["first_last_surface_metrics"]["groups"]
    macro_volume = float(np.mean([
        exact_surface_groups[diagnosis]["predicted_volume_relative_error_mean"]
        / exact_surface_groups[diagnosis]["nochange_volume_relative_error_mean"]
        for diagnosis in ("CN", "AD")
    ]))
    return {
        "representation": "inr256",
        "display_name": DISPLAY_NAMES["inr256"],
        "latent_dim": 256,
        "cohort": "matched INR subset: 20 subjects / 100 scans",
        "best_epoch": status["best_epoch"],
        "validation_selection_score": status["best_validation_selection_score"],
        "method": "direct_c4",
        "test_metric_kind": "decoded exact-SDF MAE ratio",
        "test_macro_first_last_ratio": macro_primary,
        "test_macro_first_last_surface_assd_ratio": macro_assd,
        "test_macro_first_last_volume_ratio": macro_volume,
        "primary_decoder_metric": "decoded exact-SDF MAE",
        "first_last_primary_error": proxy["sdf_mean"],
        "first_last_primary_nochange": proxy["nochange_sdf_mean"],
        "first_last_primary_ratio_to_nochange": ratio(proxy["sdf_mean"], proxy["nochange_sdf_mean"]),
        "first_last_end_to_end_rmse_mm": None,
        "all_pair_transport_coordinate_mae_mm": None,
        "first_last_volume_relative_error": surface["prediction_volume_relative_error_mean"],
        "first_last_volume_ratio_to_nochange": surface["prediction_volume_relative_error_ratio_to_nochange"],
        "first_last_rate_mae_ratio_to_nochange": volume_report["groups"]["overall"]["rate_mae_ratio_to_nochange"],
        "surface_assd_mm": surface["prediction_assd_mm_mean"],
        "surface_assd_ratio_to_nochange": surface["prediction_assd_mm_ratio_to_nochange"],
        "surface_hd95_mm": surface["prediction_hd95_mm_mean"],
        "surface_volume_absolute_error_mm3": surface["prediction_volume_absolute_error_mm3_mean"],
        "surface_volume_relative_error": surface["prediction_volume_relative_error_mean"],
        "surface_rate_mae_ratio_to_nochange": surface["prediction_rate_mae_ratio_to_nochange"],
        "representation_floor_metric": "exact point-to-triangle ASSD (mm)",
        "representation_floor_value": surface["floor_assd_mm_mean"],
        "surface_representation_floor_assd_mm": surface["floor_assd_mm_mean"],
        "semigroup_defect_mean": defects["relative_semigroup_defect_mean"],
        "inverse_defect_mean": defects["relative_inverse_defect_mean"],
        "relative_semigroup_defect_mean": defects["relative_semigroup_defect_mean"],
        "relative_inverse_defect_mean": defects["relative_inverse_defect_mean"],
        "evaluation": str(exact_path),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    surface_dir = args.fixed_surface_dir.expanduser().resolve()
    surface_report = C.read_json(surface_dir / "summary.json")
    if surface_report["split"] != "test" or not surface_report.get("test_was_explicitly_authorized", False):
        raise ValueError("The fixed-surface input must be an explicitly authorized test report")
    if int(surface_report["surface_points_per_direction"]) != 30000:
        raise ValueError("Five-model comparison requires 30,000 surface samples per direction")
    fixed_rows = read_csv(surface_dir / "per_subject.csv")
    present = {str(row["representation"]) for row in fixed_rows}
    required = set(FIXED_MODELS) | {CORRECTIVE_PCA_BRANCH}
    if not required.issubset(present):
        raise ValueError(f"Missing fixed-surface predictions: {sorted(required - present)}")
    for model in required:
        subjects = {str(row["subject_id"]) for row in fixed_rows if row["representation"] == model}
        if len(subjects) != 61:
            raise ValueError(f"Expected 61 full-cohort subjects for {model}; found {len(subjects)}")

    inr_run = args.inr_run.expanduser().resolve()
    exact_path = inr_run / args.inr_exact_evaluation_name / "test" / "summary.json"
    volume_path = inr_run / args.inr_volume_analysis_name / "summary.json"
    exact = C.read_json(exact_path)
    volume_report = C.read_json(volume_path)
    if exact.get("split") != "test" or bool(exact.get("test_loaded_during_training", False)):
        raise ValueError("Invalid or leakage-marked INR exact test report")
    if volume_report.get("split") != "test":
        raise ValueError("INR volume report is not the test split")
    if int(exact["first_last_surface_metrics"]["surface_samples"]) != 30000:
        raise ValueError("INR exact report must use 30,000 surface samples")
    converted_inr = inr_rows(
        exact, read_csv(inr_run / args.inr_volume_analysis_name / "per_subject.csv")
    )
    if len(converted_inr) != 20:
        raise ValueError(f"Expected 20 INR test subjects; found {len(converted_inr)}")
    inr_subjects = {str(row["subject_id"]) for row in converted_inr}
    matched_fixed = [
        row for row in fixed_rows
        if row["representation"] in FIXED_MODELS and str(row["subject_id"]) in inr_subjects
    ]
    for model in FIXED_MODELS:
        found = {str(row["subject_id"]) for row in matched_fixed if row["representation"] == model}
        if found != inr_subjects:
            raise ValueError(f"Matched test cohort differs for {model}: missing={sorted(inr_subjects-found)}")

    full_models = (*FIXED_MODELS, CORRECTIVE_PCA_BRANCH)
    full_summary = summarize_rows(
        fixed_rows, "full fixed-topology test: 61 subjects / 277 scans", full_models
    )
    matched_summary = summarize_rows(
        [*matched_fixed, *converted_inr], "matched test: 20 subjects", (*FIXED_MODELS, "inr256")
    )
    matched_surface_rows = [*matched_fixed, *converted_inr]
    bootstrap = bootstrap_against_pca(
        matched_surface_rows, args.bootstrap_samples, args.seed, (*FIXED_MODELS, "inr256")
    )
    decoder_ablation = decoder_ablation_bootstrap(
        fixed_rows, args.bootstrap_samples, args.seed
    )
    full_groups = {
        model: {
            diagnosis: next(
                row for row in full_summary
                if row["representation"] == model and row["diagnosis"] == diagnosis
            )
            for diagnosis in ("CN", "AD", "overall")
        }
        for model in FIXED_MODELS
    }
    matched_inr_groups = {
        diagnosis: next(
            row for row in matched_summary
            if row["representation"] == "inr256" and row["diagnosis"] == diagnosis
        )
        for diagnosis in ("CN", "AD", "overall")
    }

    runs = {name: Path(path) for name, path in surface_report["runs"].items()}
    if set(FIXED_MODELS) - set(runs):
        raise ValueError(f"Surface report run manifest is incomplete: {sorted(runs)}")
    standard = []
    for model in FIXED_MODELS:
        report = C.read_json(runs[model] / "evaluation" / "test" / "summary.json")
        if report["representation"] != model or report["method"] != "direct_c4":
            raise ValueError(f"Unexpected standard report contract for {model}")
        if bool(report.get("test_loaded_during_training", False)):
            raise ValueError(f"Training/test leakage recorded for {model}")
        standard.append(standard_row(
            model, report, runs[model], full_groups[model],
            "full fixed-topology test: 61 subjects / 277 scans",
        ))
    standard.append(inr_standard_row(
        exact, volume_report, inr_run, exact_path, matched_inr_groups
    ))

    destination = C.require_bulk_path(args.output_dir, "five-model comparison output")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {
        "full_fixed_surface": destination / "full_fixed_surface_summary.csv",
        "matched_five_model": destination / "matched_five_model_summary.csv",
        "matched_trends": destination / "matched_volume_trends.csv",
        "prior_metrics": destination / "prior_metrics_plus_exact_surface.csv",
        "paired_bootstrap": destination / "paired_bootstrap_vs_pca.csv",
        "decoder_ablation": destination / "corrective_vs_same_flow_pca_decode.csv",
        "matched_per_subject": destination / "matched_per_subject.csv",
    }
    write_csv(outputs["full_fixed_surface"], full_summary)
    write_csv(outputs["matched_five_model"], matched_summary)
    write_csv(outputs["matched_trends"], trend_gaps(matched_summary))
    write_csv(outputs["prior_metrics"], standard)
    write_csv(outputs["paired_bootstrap"], bootstrap)
    write_csv(outputs["decoder_ablation"], decoder_ablation)
    write_csv(outputs["matched_per_subject"], matched_surface_rows)
    C.atomic_json(destination / "summary.json", {
        "schema_version": 1,
        "comparison_contract": {
            "full_fixed_cohort": "61 subjects; PCA, SpiralNet++, Adaptive-Spiral, PCA-corrective and its PCA-only decoder ablation",
            "matched_cross_representation_cohort": "the INR model's immutable 20-subject test subset",
            "surface_metric": "30,000 deterministic samples per direction with exact point-to-triangle proximity",
            "volume_metric": "absolute triangle-mesh or MC256 mesh volume in mm3",
            "trend_metric": "first-to-last signed log-volume rate/year using the observed source mesh as anchor",
            "important_caveat": "INR uses an MC256 implicit surface; the other models use the common fixed topology. Matched subjects and metric definitions improve fairness, but discretization differs.",
        },
        "fixed_surface_report": str(surface_dir / "summary.json"),
        "inr_exact_report": str(exact_path),
        "inr_volume_report": str(volume_path),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "full_fixed_summary": full_summary,
        "matched_five_model_summary": matched_summary,
        "matched_volume_trends": trend_gaps(matched_summary),
        "prior_metrics_plus_exact_surface": standard,
        "paired_bootstrap_vs_pca": bootstrap,
        "corrective_vs_same_flow_pca_decode": decoder_ablation,
    })
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
