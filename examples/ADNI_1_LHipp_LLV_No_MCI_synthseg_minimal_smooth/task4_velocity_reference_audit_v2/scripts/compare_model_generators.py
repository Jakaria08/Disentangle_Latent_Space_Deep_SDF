#!/usr/bin/env python3
"""Compare trained flow generators with the fitted-trajectory velocity reference.

Current models are compared on their shared test cohort and shared registered
topology.  Legacy ODE/BrainODE results remain a separate cohort and retain their
interval-derived comparator; they are never pooled with the current models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
DEFAULT_CONFIG = TASK / "configs" / "velocity_reference_audit.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from velocity_core import subject_group_mean, vertex_geometry, weighted_field_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force deliberately")


def current_surface_comparison(config: dict[str, Any], cache: Path) -> pd.DataFrame:
    reference_path = cache / "arrays" / "test_velocity_reference.npz"
    source_root = resolve(config["existing_surface_velocity_cache"])
    model_path = source_root / "arrays" / "current_instantaneous_surface_maps.npz"
    model_table = pd.read_csv(source_root / "tables" / "current_instantaneous_surface_velocity.csv", dtype={"scan_id": str, "subject_id": str})
    with np.load(reference_path, allow_pickle=False) as loaded:
        reference = {key: loaded[key] for key in loaded.files}
    with np.load(model_path, allow_pickle=False) as loaded:
        model = {key: loaded[key] for key in loaded.files}
    shared_scans = set(model_table.scan_id.astype(str))
    mask = np.isin(reference["scan_ids"].astype(str), list(shared_scans))
    if int(mask.sum()) != model_table.scan_id.nunique():
        raise ValueError("Current model/reference shared scan cohort mismatch")
    shared_subjects = reference["subject_ids"][mask].astype(str)
    shared_diagnoses = reference["diagnoses"][mask].astype(str)
    shared_fields = reference["fitted_normal_velocity"][mask]
    _, area, _ = vertex_geometry(reference["template_vertices"], reference["faces"])
    rows = []
    labels = {"pca": "PCA", "spiral": "Spiral", "adaptive": "Adaptive", "inr": "INR"}
    for method, label in labels.items():
        for diagnosis in ("CN", "AD"):
            ref_field = subject_group_mean(shared_fields, shared_subjects, shared_diagnoses, diagnosis)
            model_field = model[f"{method}_model_{diagnosis.lower()}_group"]
            row = {
                "cohort": "current_shared_test",
                "method": method,
                "method_label": label,
                "diagnosis": diagnosis,
                "comparator": "Observed — fitted trajectory",
                "subjects": int(np.unique(shared_subjects[shared_diagnoses == diagnosis]).size),
                "scans": int(np.sum(shared_diagnoses == diagnosis)),
            }
            row.update(weighted_field_metrics(model_field, ref_field, area))
            rows.append(row)
        ref_gap = subject_group_mean(shared_fields, shared_subjects, shared_diagnoses, "AD") - subject_group_mean(shared_fields, shared_subjects, shared_diagnoses, "CN")
        model_gap = model[f"{method}_model_ad_group"] - model[f"{method}_model_cn_group"]
        row = {
            "cohort": "current_shared_test",
            "method": method,
            "method_label": label,
            "diagnosis": "AD minus CN",
            "comparator": "Observed — fitted trajectory group difference",
            "subjects": int(np.unique(shared_subjects).size),
            "scans": int(mask.sum()),
        }
        row.update(weighted_field_metrics(model_gap, ref_gap, area))
        rows.append(row)
    return pd.DataFrame(rows)


def current_comparator_sensitivity(config: dict[str, Any], cache: Path, fitted: pd.DataFrame) -> pd.DataFrame:
    """Put the previous interval comparator beside the corrected comparator."""
    source_root = resolve(config["existing_surface_velocity_cache"])
    with np.load(source_root / "arrays" / "current_instantaneous_surface_maps.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    _, area, _ = vertex_geometry(arrays["template_vertices"], arrays["faces"])
    labels = {"pca": "PCA", "spiral": "Spiral", "adaptive": "Adaptive", "inr": "INR"}
    rows = []
    for method, label in labels.items():
        for diagnosis in ("CN", "AD"):
            row = {
                "cohort": "current_shared_test",
                "method": method,
                "method_label": label,
                "diagnosis": diagnosis,
                "comparator": "Previous: Observed — adjacent interval",
            }
            row.update(weighted_field_metrics(
                arrays[f"{method}_model_{diagnosis.lower()}_group"],
                arrays[f"observed_gt_{diagnosis.lower()}"], area,
            ))
            rows.append(row)
        row = {
            "cohort": "current_shared_test",
            "method": method,
            "method_label": label,
            "diagnosis": "AD minus CN",
            "comparator": "Previous: Observed — adjacent interval group difference",
        }
        row.update(weighted_field_metrics(arrays[f"{method}_model_group_gap"], arrays["observed_gt_group_gap"], area))
        rows.append(row)
    previous = pd.DataFrame(rows)
    corrected = fitted.drop(columns=["subjects", "scans"]).copy()
    corrected["comparator"] = corrected.comparator.str.replace(
        "Observed — fitted trajectory", "Corrected: Observed — fitted trajectory", regex=False
    )
    return pd.concat([previous, corrected], ignore_index=True)


def legacy_surface_comparison(config: dict[str, Any]) -> pd.DataFrame:
    root = resolve(config["existing_surface_velocity_cache"])
    with np.load(root / "arrays" / "legacy_instantaneous_surface_maps.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    table = pd.read_csv(root / "tables" / "legacy_instantaneous_surface_velocity.csv")
    _, area, _ = vertex_geometry(arrays["template_vertices"], arrays["faces"])
    labels = {"pca_cocycle": "PCA Cocycle", "inr_cocycle": "INR Cocycle", "latent_ode": "Latent ODE", "brainode": "BrainODE"}
    rows = []
    for method, label in labels.items():
        for diagnosis in ("CN", "AD"):
            model_field = arrays[f"{method}_model_{diagnosis.lower()}_group"]
            observed = arrays[f"observed_gt_{diagnosis.lower()}"]
            subset = table[(table.method == method) & (table.diagnosis == diagnosis)]
            row = {
                "cohort": "legacy_reference_test",
                "method": method,
                "method_label": label,
                "diagnosis": diagnosis,
                "comparator": "Observed — adjacent interval (legacy cohort; not instantaneous GT)",
                "subjects": int(subset.subject_id.nunique()),
                "scans": int(subset.scan_id.nunique()),
            }
            row.update(weighted_field_metrics(model_field, observed, area))
            rows.append(row)
    return pd.DataFrame(rows)


def merged_latent_comparison(config: dict[str, Any], cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_csv(cache / "tables" / "latent_velocity_reference_per_scan.csv", dtype={"scan_id": str, "subject_id": str})
    model = pd.read_csv(resolve(config["existing_task4_cache"]) / "tables" / "current_velocity_per_scan.csv", dtype={"scan_id": str, "subject_id": str})
    merged = model.merge(
        reference,
        on=["method", "method_label", "subject_id", "scan_id", "diagnosis", "age_years"],
        how="inner",
        validate="one_to_one",
        suffixes=("_old", "_reference"),
    )
    if len(merged) != len(model):
        raise ValueError(f"Latent merge lost rows: model={len(model)}, merged={len(merged)}")
    merged["model_to_fitted_speed_ratio"] = merged.model_rms_per_coordinate_per_year / merged.fitted_rms_per_coordinate_per_year.clip(lower=1.0e-12)
    merged["adjacent_to_fitted_speed_ratio"] = merged.adjacent_rms_per_coordinate_per_year / merged.fitted_rms_per_coordinate_per_year.clip(lower=1.0e-12)
    rows = []
    for (method, label, diagnosis), frame in merged.groupby(["method", "method_label", "diagnosis"], sort=False):
        rows.append({
            "method": method,
            "method_label": label,
            "diagnosis": diagnosis,
            "subjects": int(frame.subject_id.nunique()),
            "scans": len(frame),
            "mean_model_speed": float(frame.model_rms_per_coordinate_per_year.mean()),
            "mean_fitted_reference_speed": float(frame.fitted_rms_per_coordinate_per_year.mean()),
            "mean_adjacent_interval_speed": float(frame.adjacent_rms_per_coordinate_per_year.mean()),
            "mean_model_to_fitted_speed_ratio": float(frame.model_to_fitted_speed_ratio.mean()),
            "mean_adjacent_to_fitted_speed_ratio": float(frame.adjacent_to_fitted_speed_ratio.mean()),
            "model_fitted_speed_pearson": float(frame.model_rms_per_coordinate_per_year.corr(frame.fitted_rms_per_coordinate_per_year)),
            "unit_note": "train-standardized latent units/year; magnitude comparisons are within representation only",
        })
    return merged, pd.DataFrame(rows)


def merged_surface_scalar_comparison(config: dict[str, Any], cache: Path) -> pd.DataFrame:
    reference = pd.read_csv(cache / "tables" / "velocity_reference_per_scan.csv", dtype={"scan_id": str, "subject_id": str})
    reference = reference[reference.split.eq("test")]
    model = pd.read_csv(resolve(config["existing_surface_velocity_cache"]) / "tables" / "current_instantaneous_surface_velocity.csv", dtype={"scan_id": str, "subject_id": str})
    columns = ["subject_id", "scan_id", "diagnosis", "age_years", "fit_normal_mean_mm_per_year", "fit_normal_abs_mean_mm_per_year", "fit_normal_rms_mm_per_year", "fit_surface_integral_log_volume_rate_percent_per_year"]
    merged = model.merge(reference[columns], on=["subject_id", "scan_id", "diagnosis", "age_years"], how="inner", validate="many_to_one")
    if len(merged) != len(model):
        raise ValueError(f"Surface scalar merge lost rows: model={len(model)}, merged={len(merged)}")
    merged["model_to_fitted_normal_speed_ratio"] = merged.model_normal_rms_mm_per_year / merged.fit_normal_rms_mm_per_year.clip(lower=1.0e-12)
    return merged


def main() -> int:
    args = parse_args()
    config = read_json(args.config.expanduser().resolve())
    cache = resolve(args.cache_root if args.cache_root is not None else config["output_root"])
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(f"Build the velocity reference first: {cache}")
    targets = [
        cache / "tables" / "current_model_surface_group_comparison.csv",
        cache / "tables" / "current_model_comparator_sensitivity.csv",
        cache / "tables" / "legacy_model_surface_group_comparison.csv",
        cache / "tables" / "latent_model_vs_fitted_reference_per_scan.csv",
        cache / "tables" / "latent_model_vs_fitted_reference_summary.csv",
        cache / "tables" / "current_model_surface_scalar_per_scan.csv",
        cache / "model_comparison_manifest.json",
    ]
    for target in targets:
        check_output(target, args.force)
    current = current_surface_comparison(config, cache)
    sensitivity = current_comparator_sensitivity(config, cache, current)
    legacy = legacy_surface_comparison(config)
    latent, latent_summary = merged_latent_comparison(config, cache)
    surface_scalar = merged_surface_scalar_comparison(config, cache)
    for frame, target in zip((current, sensitivity, legacy, latent, latent_summary, surface_scalar), targets[:-1]):
        frame.to_csv(target, index=False)
        print(f"saved {target} ({len(frame)} rows)")
    explicit = current.copy()
    explicit["latent_dimension"] = explicit.method.map({"pca": 128, "spiral": 128, "adaptive": 128, "inr": 256})
    explicit["comparison_scope"] = "dimension is shown only in the dedicated representation-size table"
    explicit.to_csv(cache / "tables" / "dedicated_128_256_surface_comparison.csv", index=False)
    targets[-1].write_text(json.dumps({
        "status": "complete",
        "current_comparator": "Observed — fitted trajectory",
        "current_scope": "same test subjects available to all four current models; subject-then-group aggregation",
        "legacy_comparator": "Observed — adjacent interval; not directly measured instantaneous motion",
        "legacy_scope": "separate legacy cohort; no numeric pooling with current models",
        "latent_unit_contract": "train-standardized latent units/year; compare magnitude only within each representation",
        "ordinary_labels_hide_latent_dimension": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {targets[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
