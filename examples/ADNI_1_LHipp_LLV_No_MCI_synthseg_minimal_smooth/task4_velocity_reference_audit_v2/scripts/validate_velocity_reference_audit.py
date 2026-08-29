#!/usr/bin/env python3
"""Validate scientific, schema, and presentation contracts for the audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
DEFAULT_CONFIG = TASK / "configs" / "velocity_reference_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{name}: missing columns {missing}")
    require(np.isfinite(frame[columns].to_numpy(dtype=float)).all(), f"{name}: non-finite required values")


def main() -> int:
    args = parse_args()
    config = read_json(args.config.expanduser().resolve())
    root = resolve(args.cache_root if args.cache_root is not None else config["output_root"])
    manifest = read_json(root / "manifest.json")
    comparison_manifest = read_json(root / "model_comparison_manifest.json")
    selected = read_json(root / "selected_estimator.json")
    require(manifest["status"] == "complete", "reference manifest is not complete")
    require(comparison_manifest["status"] == "complete", "model comparison manifest is not complete")
    require(args.allow_smoke or manifest["mode"] == "full", "smoke cache is not accepted as a final result")
    require(manifest["selection_split"] == "val" and manifest["evaluation_split"] == "test", "split-selection contract failed")
    require(selected["selected_on"] == "validation subjects only" and selected["test_was_not_used_for_selection"] is True, "test leakage contract failed")
    require(manifest["interpretation"]["instantaneous_ground_truth_exists"] is False, "instantaneous truth disclaimer missing")
    require(manifest["source_meshes_modified"] is False, "source meshes must remain read-only")
    require(config["scientific_contract"]["rigid_alignment_has_scale"] is False, "rigid alignment must not scale")
    require(config["scientific_contract"]["rigid_alignment_allows_reflection"] is False, "rigid alignment must not reflect")

    tables = root / "tables"
    reference = pd.read_csv(tables / "velocity_reference_per_scan.csv", dtype={"scan_id": str, "subject_id": str})
    intervals = pd.read_csv(tables / "adjacent_interval_diagnostics.csv", dtype={"source_scan_id": str, "target_scan_id": str})
    stability = pd.read_csv(tables / "leave_one_visit_out_derivative_stability.csv")
    val_folds = pd.read_csv(tables / "val_heldout_visit_predictions.csv")
    test_folds = pd.read_csv(tables / "test_heldout_visit_predictions.csv")
    current = pd.read_csv(tables / "current_model_surface_group_comparison.csv")
    sensitivity = pd.read_csv(tables / "current_model_comparator_sensitivity.csv")
    legacy = pd.read_csv(tables / "legacy_model_surface_group_comparison.csv")
    latent = pd.read_csv(tables / "latent_model_vs_fitted_reference_per_scan.csv")
    volume = pd.read_csv(tables / "diagnosis_volume_slope_bootstrap.csv")
    require(set(reference.split) == {"val", "test"}, "reference split labels are wrong")
    require(reference.scan_id.is_unique, "reference scans must be unique")
    require(set(reference.reference_label) == {"Observed — fitted trajectory"}, "reference display label changed")
    require(set(val_folds.split) == {"val"} and set(test_folds.split) == {"test"}, "held-out prediction split contamination")
    require((intervals.registered_gap_years > 0).all(), "non-positive visit interval")
    require((intervals.rigid_removed_to_registered_speed_ratio >= 0).all(), "invalid rigid speed ratio")
    require(((current.normal_sign_agreement >= 0) & (current.normal_sign_agreement <= 1)).all(), "invalid sign agreement")
    require(((current.hotspot_dice >= 0) & (current.hotspot_dice <= 1)).all(), "invalid hotspot Dice")
    require(set(current.method) == {"pca", "spiral", "adaptive", "inr"} and len(current) == 12, "current method coverage failed")
    require(set(legacy.method) == {"pca_cocycle", "inr_cocycle", "latent_ode", "brainode"} and len(legacy) == 8, "legacy method coverage failed")
    require(current.comparator.str.contains("fitted trajectory", regex=False).all(), "current model comparator is not fitted trajectory")
    require(len(sensitivity) == 24, "old-versus-corrected comparator coverage failed")
    require(sensitivity.comparator.str.startswith("Previous:").sum() == 12, "previous comparator rows missing")
    require(sensitivity.comparator.str.startswith("Corrected:").sum() == 12, "corrected comparator rows missing")
    require(legacy.comparator.str.contains("not instantaneous GT", regex=False).all(), "legacy limitation label missing")
    require(len(latent) == 400 and latent.scan_id.nunique() == 100, "shared latent comparison cohort changed")
    require(set(volume.contrast) == {"CN", "AD", "AD minus CN"}, "volume contrast coverage failed")
    require_finite(reference, ["fit_vector_rms_mm_per_year", "adjacent_vector_rms_mm_per_year", "fit_normal_rms_mm_per_year", "observed_volume_mm3"], "reference")
    require_finite(intervals, ["registered_vector_rms_mm_per_year", "rigid_removed_vector_rms_mm_per_year", "registered_gap_years"], "intervals")
    require_finite(stability, ["derivative_cosine", "relative_derivative_error", "speed_ratio"], "stability")
    require_finite(current, ["normal_rmse_mm_per_year", "normal_pearson", "hotspot_dice"], "current comparison")

    for split in ("val", "test"):
        with np.load(root / "arrays" / f"{split}_velocity_reference.npz", allow_pickle=False) as arrays:
            required = {"faces", "template_vertices", "fitted_normal_velocity", "adjacent_normal_velocity", "endpoint_normal_velocity", "scan_ids", "subject_ids", "diagnoses"}
            require(required.issubset(arrays.files), f"{split} array keys missing")
            n_scans = manifest["counts"][split]["scans"]
            require(arrays["fitted_normal_velocity"].shape == (n_scans, 2746), f"{split} fitted field shape changed")
            require(arrays["faces"].shape == (5488, 3), f"{split} topology changed")
            for key in ("template_vertices", "fitted_normal_velocity", "adjacent_normal_velocity", "endpoint_normal_velocity"):
                require(np.isfinite(arrays[key]).all(), f"{split}: non-finite {key}")

    notebook = TASK / "notebooks" / "velocity_reference_audit.ipynb"
    require(notebook.is_file(), "analysis notebook is missing")
    notebook_text = notebook.read_text(encoding="utf-8").lower()
    require("there is no directly measured instantaneous ground truth" in notebook_text, "notebook truth explanation missing")
    for forbidden in ("proxy", "sparse"):
        require(forbidden not in notebook_text, f"forbidden presentation word appears in notebook: {forbidden}")
    report = {
        "status": "pass",
        "cache_mode": manifest["mode"],
        "selected_estimator": manifest["selected_estimator"],
        "subjects": {split: manifest["counts"][split]["subjects"] for split in ("val", "test")},
        "scans": {split: manifest["counts"][split]["scans"] for split in ("val", "test")},
        "current_methods": sorted(current.method_label.unique().tolist()),
        "legacy_methods": sorted(legacy.method_label.unique().tolist()),
        "checks": "scientific labels, split isolation, finite values, topology, cohort coverage, and notebook explanations",
    }
    report_path = TASK / "reports" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
