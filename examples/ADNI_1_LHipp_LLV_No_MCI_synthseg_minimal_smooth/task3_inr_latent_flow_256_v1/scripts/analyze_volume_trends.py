#!/usr/bin/env python3
"""Exact MC volume-trend analysis for the validation-selected INR cocycle flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common as C
from evaluate import load_transport
from inr_geometry import build_geometry


DEFAULT_PREVIOUS_ROOT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training"
)
PREVIOUS_RUNS = {
    "PCA-128": ("pca128", "pca128_direct_c4_s42"),
    "SpiralNet++-128": ("spiralnet128", "spiralnet128_direct_c4_s42"),
    "Adaptive-Spiral-128": ("adaptive128", "adaptive128_direct_c4_s42"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--surface-resolution", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-evaluation-name", default="evaluation_mc256_exact")
    parser.add_argument("--output-name", default="volume_trend_analysis_mc256")
    parser.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    return parser.parse_args()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def bootstrap_group(rows: list[dict[str, Any]], samples: int, rng: np.random.Generator) -> dict[str, Any]:
    observed = np.asarray([row["observed_signed_log_rate_per_year"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_signed_log_rate_per_year"] for row in rows], dtype=np.float64)
    errors = np.abs(predicted - observed)
    directions = np.asarray([row["atrophy_direction_agreement"] for row in rows], dtype=np.float64)
    indices = rng.integers(0, len(rows), size=(samples, len(rows)))
    metrics = {
        "observed_signed_log_rate_per_year": observed[indices].mean(axis=1),
        "predicted_signed_log_rate_per_year": predicted[indices].mean(axis=1),
        "rate_mae_per_year": errors[indices].mean(axis=1),
        "atrophy_direction_agreement": directions[indices].mean(axis=1),
    }
    return {name: {"mean": float(values.mean()), **percentile_interval(values)} for name, values in metrics.items()}


def summarize_group(rows: list[dict[str, Any]], samples: int, rng: np.random.Generator) -> dict[str, Any]:
    predicted_rate = mean(rows, "predicted_signed_log_rate_per_year")
    observed_rate = mean(rows, "observed_signed_log_rate_per_year")
    rate_mae = mean(rows, "absolute_rate_error_per_year")
    nochange_rate_mae = mean(rows, "nochange_absolute_rate_error_per_year")
    predicted_volume_error = mean(rows, "predicted_target_volume_relative_error")
    nochange_volume_error = mean(rows, "nochange_target_volume_relative_error")
    return {
        "subjects": len(rows),
        "mean_followup_years": mean(rows, "followup_years"),
        "predicted_signed_log_volume_rate_per_year": predicted_rate,
        "observed_signed_log_volume_rate_per_year": observed_rate,
        "predicted_annualized_percent_change": float(100.0 * np.expm1(predicted_rate)),
        "observed_annualized_percent_change": float(100.0 * np.expm1(observed_rate)),
        "rate_mae_per_year": rate_mae,
        "nochange_rate_mae_per_year": nochange_rate_mae,
        "rate_mae_ratio_to_nochange": rate_mae / max(nochange_rate_mae, 1.0e-12),
        "rate_mae_reduction_percent": 100.0 * (1.0 - rate_mae / max(nochange_rate_mae, 1.0e-12)),
        "atrophy_direction_agreement": mean(rows, "atrophy_direction_agreement"),
        "predicted_target_volume_relative_error": predicted_volume_error,
        "nochange_target_volume_relative_error": nochange_volume_error,
        "volume_error_ratio_to_nochange": predicted_volume_error / max(nochange_volume_error, 1.0e-12),
        "bootstrap": bootstrap_group(rows, samples, rng),
    }


def bootstrap_gap(
    ad_rows: list[dict[str, Any]], cn_rows: list[dict[str, Any]], samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    ad_pred = np.asarray([row["predicted_signed_log_rate_per_year"] for row in ad_rows])
    cn_pred = np.asarray([row["predicted_signed_log_rate_per_year"] for row in cn_rows])
    ad_obs = np.asarray([row["observed_signed_log_rate_per_year"] for row in ad_rows])
    cn_obs = np.asarray([row["observed_signed_log_rate_per_year"] for row in cn_rows])
    ad_indices = rng.integers(0, len(ad_rows), size=(samples, len(ad_rows)))
    cn_indices = rng.integers(0, len(cn_rows), size=(samples, len(cn_rows)))
    predicted = ad_pred[ad_indices].mean(axis=1) - cn_pred[cn_indices].mean(axis=1)
    observed = ad_obs[ad_indices].mean(axis=1) - cn_obs[cn_indices].mean(axis=1)
    return {
        "predicted_gap": {"mean": float(predicted.mean()), **percentile_interval(predicted)},
        "observed_gap": {"mean": float(observed.mean()), **percentile_interval(observed)},
        "gap_error": {"mean": float((predicted - observed).mean()), **percentile_interval(predicted - observed)},
        "probability_predicted_ad_atrophy_exceeds_cn": float(np.mean(predicted < 0.0)),
    }


def enrich_existing_metrics(exact: dict[str, Any]) -> dict[str, Any]:
    proxy_groups = exact["proxy_pair_metrics"]["first_last_forward"]["groups"]
    surface_groups = exact["first_last_surface_metrics"]["groups"]
    output = {}
    for diagnosis in ("CN", "AD", "overall"):
        proxy, surface = proxy_groups[diagnosis], surface_groups[diagnosis]
        output[diagnosis] = {
            "sdf_mae": proxy["sdf_mean"],
            "nochange_sdf_mae": proxy["nochange_sdf_mean"],
            "sdf_ratio_to_nochange": proxy["sdf_mean"] / proxy["nochange_sdf_mean"],
            "latent_mse": proxy["latent_mean"],
            "nochange_latent_mse": proxy["nochange_latent_mean"],
            "latent_ratio_to_nochange": proxy["latent_mean"] / proxy["nochange_latent_mean"],
            "assd_mm": surface["predicted_assd_mm_mean"],
            "nochange_assd_mm": surface["nochange_assd_mm_mean"],
            "assd_ratio_to_nochange": surface["predicted_assd_mm_mean"] / surface["nochange_assd_mm_mean"],
            "hd95_mm": surface["predicted_hd95_mm_mean"],
            "nochange_hd95_mm": surface["nochange_hd95_mm_mean"],
            "chamfer_l2_squared_mm2": surface["predicted_chamfer_l2_squared_mm2_mean"],
            "nochange_chamfer_l2_squared_mm2": surface["nochange_chamfer_l2_squared_mm2_mean"],
            "representation_floor_assd_mm": surface["representation_floor_assd_mm_mean"],
        }
    return output


def comparison_row(
    name: str,
    cohort: str,
    groups: dict[str, Any],
    defects: dict[str, Any],
    geometry_metric: str,
    geometry_ratio: float,
    volume_ratio: float,
) -> dict[str, Any]:
    cn, ad, overall = groups["CN"], groups["AD"], groups["overall"]
    predicted_gap = float(ad["predicted_signed_log_volume_rate_per_year"] - cn["predicted_signed_log_volume_rate_per_year"])
    observed_gap = float(ad["observed_signed_log_volume_rate_per_year"] - cn["observed_signed_log_volume_rate_per_year"])
    return {
        "representation": name,
        "test_cohort": cohort,
        "cn_subjects": int(cn["subjects"]),
        "ad_subjects": int(ad["subjects"]),
        "cn_predicted_percent_per_year": float(100.0 * np.expm1(cn["predicted_signed_log_volume_rate_per_year"])),
        "cn_observed_percent_per_year": float(100.0 * np.expm1(cn["observed_signed_log_volume_rate_per_year"])),
        "ad_predicted_percent_per_year": float(100.0 * np.expm1(ad["predicted_signed_log_volume_rate_per_year"])),
        "ad_observed_percent_per_year": float(100.0 * np.expm1(ad["observed_signed_log_volume_rate_per_year"])),
        "predicted_ad_minus_cn_log_rate_gap": predicted_gap,
        "observed_ad_minus_cn_log_rate_gap": observed_gap,
        "gap_absolute_error": abs(predicted_gap - observed_gap),
        "gap_recovery_fraction": predicted_gap / observed_gap if abs(observed_gap) > 1.0e-12 else None,
        "ad_more_atrophy_than_cn_predicted": bool(predicted_gap < 0.0),
        "overall_rate_mae_ratio_to_nochange": float(overall["rate_mae_ratio_to_nochange"]),
        "overall_volume_error_ratio_to_nochange": float(volume_ratio),
        "geometry_metric": geometry_metric,
        "overall_geometry_error_ratio_to_nochange": float(geometry_ratio),
        "relative_semigroup_defect_mean": float(defects["relative_semigroup_defect_mean"]),
        "relative_inverse_defect_mean": float(defects["relative_inverse_defect_mean"]),
    }


def previous_comparisons(previous_root: Path) -> list[dict[str, Any]]:
    rows = []
    for name, (representation, run_name) in PREVIOUS_RUNS.items():
        path = previous_root / representation / "direct_c4" / run_name / "evaluation" / "test" / "summary.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        source_groups = report["pair_metrics"]["first_last_forward"]["groups"]
        groups = {}
        for diagnosis in ("CN", "AD", "overall"):
            source = source_groups[diagnosis]
            groups[diagnosis] = {
                "subjects": int(source["rows"]),
                "predicted_signed_log_volume_rate_per_year": float(source["predicted_signed_rate_mean"]),
                "observed_signed_log_volume_rate_per_year": float(source["observed_signed_rate_mean"]),
                "rate_mae_ratio_to_nochange": float(source["rate_mean"] / source["nochange_rate_mean"]),
            }
        overall = source_groups["overall"]
        rows.append(comparison_row(
            name,
            "legacy test: 61 subjects / 277 scans",
            groups,
            report["consistency_defects"],
            "fixed-topology mean Euclidean vertex distance",
            overall["euclidean_mean"] / overall["nochange_euclidean_mean"],
            overall["volume_relative_mean"] / overall["nochange_volume_relative_mean"],
        ))
    return rows


@torch.no_grad()
def main() -> int:
    args = parse_args()
    C.validate_run_name(args.output_name)
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    if args.surface_resolution < 32:
        raise ValueError("surface-resolution must be at least 32")
    C.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run_dir = args.run_dir.expanduser().resolve()
    destination = run_dir / args.output_name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    resolved = C.read_json(run_dir / "resolved_config.json")
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError("Checkpoint contract says test was loaded during training")
    registry = C.load_registry()
    train_archive = C.load_archive("train", registry)
    test_archive = C.load_archive("test", registry)
    device = C.choose_device(args.device)
    geometry = build_geometry(train_archive, device, registry)
    transport = load_transport(resolved["config"], checkpoint, device)
    values = C.values_on_device(test_archive, device)
    pairs = C.first_last_pairs(test_archive)

    per_subject = []
    for number, row in enumerate(pairs, start=1):
        source = values["z"][row.source:row.source + 1]
        source_age = values["age"][row.source:row.source + 1]
        target_age = values["age"][row.target:row.target + 1]
        label = values["label"][row.source:row.source + 1]
        prediction = transport.transport(source, source_age, target_age, label)
        predicted_mesh = geometry.mesh(prediction, args.surface_resolution)
        predicted_volume = abs(float(predicted_mesh.volume))
        source_volume = abs(float(test_archive["visit_volume_mm3"][row.source]))
        target_volume = abs(float(test_archive["visit_volume_mm3"][row.target]))
        years = float(test_archive["visit_time_years_from_baseline"][row.target] - test_archive["visit_time_years_from_baseline"][row.source])
        if min(source_volume, target_volume, predicted_volume, years) <= 0.0:
            raise ValueError(f"Non-positive volume or follow-up for {row.subject}")
        observed_rate = float(np.log(target_volume / source_volume) / years)
        predicted_rate = float(np.log(predicted_volume / source_volume) / years)
        record = {
            "subject_id": row.subject,
            "diagnosis": row.diagnosis,
            "source_scan_id": str(test_archive["visit_scan_ids"][row.source]),
            "target_scan_id": str(test_archive["visit_scan_ids"][row.target]),
            "followup_years": years,
            "source_true_volume_mm3": source_volume,
            "target_true_volume_mm3": target_volume,
            "predicted_target_volume_mm3_mc": predicted_volume,
            "observed_signed_log_rate_per_year": observed_rate,
            "predicted_signed_log_rate_per_year": predicted_rate,
            "observed_annualized_percent_change": float(100.0 * np.expm1(observed_rate)),
            "predicted_annualized_percent_change": float(100.0 * np.expm1(predicted_rate)),
            "observed_total_percent_change": float(100.0 * (target_volume / source_volume - 1.0)),
            "predicted_total_percent_change": float(100.0 * (predicted_volume / source_volume - 1.0)),
            "absolute_rate_error_per_year": abs(predicted_rate - observed_rate),
            "nochange_absolute_rate_error_per_year": abs(observed_rate),
            "atrophy_direction_agreement": int((predicted_rate < 0.0) == (observed_rate < 0.0)),
            "predicted_target_volume_relative_error": abs(predicted_volume - target_volume) / target_volume,
            "nochange_target_volume_relative_error": abs(source_volume - target_volume) / target_volume,
        }
        per_subject.append(record)
        print(f"volume {number:03d}/{len(pairs):03d} {row.subject} {row.diagnosis}", flush=True)

    groups = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = per_subject if diagnosis == "overall" else [row for row in per_subject if row["diagnosis"] == diagnosis]
        groups[diagnosis] = summarize_group(current, args.bootstrap_samples, rng)
    predicted_gap = groups["AD"]["predicted_signed_log_volume_rate_per_year"] - groups["CN"]["predicted_signed_log_volume_rate_per_year"]
    observed_gap = groups["AD"]["observed_signed_log_volume_rate_per_year"] - groups["CN"]["observed_signed_log_volume_rate_per_year"]
    gap = {
        "definition": "AD mean signed log-volume rate minus CN mean; more negative means greater AD atrophy",
        "predicted_ad_minus_cn_log_rate_per_year": predicted_gap,
        "observed_ad_minus_cn_log_rate_per_year": observed_gap,
        "absolute_gap_error": abs(predicted_gap - observed_gap),
        "gap_recovery_fraction": predicted_gap / observed_gap if abs(observed_gap) > 1.0e-12 else None,
        "predicted_order_correct": bool(predicted_gap < 0.0),
        "bootstrap": bootstrap_gap(
            [row for row in per_subject if row["diagnosis"] == "AD"],
            [row for row in per_subject if row["diagnosis"] == "CN"],
            args.bootstrap_samples,
            rng,
        ),
    }

    exact_path = run_dir / args.exact_evaluation_name / "test" / "summary.json"
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    other_metrics = enrich_existing_metrics(exact)
    inr_comparison = comparison_row(
        "INR-256 cocycle",
        "INR test subset: 20 subjects / 100 scans",
        groups,
        exact["consistency_defects"],
        "MC256 sampled surface ASSD",
        other_metrics["overall"]["assd_ratio_to_nochange"],
        groups["overall"]["volume_error_ratio_to_nochange"],
    )
    comparisons = [inr_comparison, *previous_comparisons(args.previous_root.expanduser().resolve())]
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": "test",
        "surface_resolution": int(args.surface_resolution),
        "volume_definition": "absolute watertight marching-cubes mesh volume in mm^3",
        "trend_definition": "signed log(V_target/V_source)/follow-up-years; negative is atrophy",
        "source_volume_basis": "observed source mesh volume from the immutable test archive",
        "bootstrap_samples": int(args.bootstrap_samples),
        "groups": groups,
        "ad_vs_cn_gap": gap,
        "other_inr_metrics": other_metrics,
        "consistency_defects": exact["consistency_defects"],
        "previous_comparison_caveat": (
            "PCA/Spiral/Adaptive use the legacy 61-subject/277-scan test cohort and fixed-topology geometry; "
            "INR uses its available 20-subject/100-scan test subset and MC256 SDF surfaces. Volume trends are "
            "defined identically, but geometry-error magnitudes are not cross-representation comparable."
        ),
        "comparison": comparisons,
    }
    C.assert_finite_mapping(report)
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "summary.json", report)
    C.write_csv(destination / "per_subject.csv", per_subject)
    C.write_csv(destination / "comparison.csv", comparisons)
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
