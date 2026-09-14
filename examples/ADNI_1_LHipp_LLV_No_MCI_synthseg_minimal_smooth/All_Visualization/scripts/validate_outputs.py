#!/usr/bin/env python3
"""Validate generated tables, figures, HTML, and executed notebook outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--notebook", type=Path, default=EXPERIMENT_DIR / "all_methods_longitudinal_analysis.ipynb")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    required_tables = {
        "prediction_metrics_strict_test.csv": 12,
        "representation_reconstruction_floor.csv": 5,
        "inr_128_vs_256_reconstruction.csv": 4,
        "volume_rate_ad_cn.csv": 10,
        "instantaneous_velocity_per_scan.csv": 100,
        "instantaneous_velocity_5year.csv": 20,
        "instantaneous_velocity_annual.csv": 20,
        "instantaneous_velocity_summary.csv": 12,
        "validation_best_method_ranking.csv": 5,
        "cocycle_consistency.csv": 5,
        "ood_best_vs_brainode_trajectory.csv": 6,
        "brainode_selected_mesh_metrics.csv": 4,
        "matched_velocity_summary.csv": 20,
        "regional_surface_velocity.csv": 4000,
        "regional_surface_velocity_summary.csv": 30,
        "interval_integrated_velocity.csv": 500,
        "tangent_consistency.csv": 500,
        "paired_condition_progression.csv": 1000,
        "paired_condition_velocity_ci.csv": 50,
        "paired_condition_shape_ci.csv": 150,
        "observed_shape_progression.csv": 100,
    }
    checks = []
    for name, minimum in required_tables.items():
        path = root / "tables" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        infinite = int(np.isinf(numeric.to_numpy(float)).sum()) if not numeric.empty else 0
        if len(frame) < minimum or infinite:
            raise ValueError(f"Invalid table {name}: rows={len(frame)}, infinite={infinite}")
        checks.append({"artifact": name, "rows": len(frame), "infinite_values": infinite})

    velocity = pd.read_csv(root / "tables" / "instantaneous_velocity_per_scan.csv")
    required_methods = {
        "latent_pca", "latent_spiral", "latent_adaptive", "latent_inr", "lamm_n3",
        "mesh_spiral", "mesh_adaptive", "lamm_global_256", "lamm_global_384", "lamm_regional_tokens",
        "pca_plain_ode", "pca_brainode",
    }
    missing = sorted(required_methods - set(velocity.method.astype(str)))
    if missing:
        raise ValueError(f"Velocity methods missing: {missing}")
    if not set(velocity.diagnosis.astype(str)) >= {"CN", "AD"}:
        raise ValueError("Velocity table lacks CN or AD")
    pca_velocity_scans = set(velocity.loc[velocity.method.eq("latent_pca"), "scan_id"].astype(str))
    for method in ("pca_plain_ode", "pca_brainode"):
        scans = set(velocity.loc[velocity.method.eq(method), "scan_id"].astype(str))
        if scans != pca_velocity_scans:
            raise ValueError(f"{method} velocity scans do not match latent PCA")

    matched_velocity = pd.read_csv(root / "tables" / "matched_velocity_summary.csv")
    expected_matched = {
        "latent_pca", "latent_spiral", "latent_adaptive", "lamm_n3",
        "mesh_spiral", "mesh_adaptive", "pca_plain_ode", "pca_brainode",
    }
    present_matched = set(matched_velocity.method.astype(str))
    if present_matched != expected_matched:
        raise ValueError(
            f"Matched velocity table has the wrong method set: {sorted(present_matched)}"
        )
    if set(matched_velocity.diagnosis.astype(str)) != {"CN", "AD", "overall"}:
        raise ValueError("Matched velocity summary lacks CN, AD, or overall rows")

    regional = pd.read_csv(root / "tables" / "regional_surface_velocity.csv")
    detailed_methods = {
        "mesh_spiral", "mesh_adaptive", "latent_spiral", "latent_adaptive",
        "latent_pca", "lamm_n3", "pca_plain_ode", "pca_brainode"
    }
    if set(regional.method.astype(str)) != detailed_methods:
        raise ValueError("Regional analysis does not contain every detailed method")
    if set(regional.region.astype(str)) != {
        "Whole surface", "Head-side third", "Middle third", "Tail-side third"
    }:
        raise ValueError("Regional analysis has an unexpected long-axis partition")

    progression = json.loads((root / "surface_progression_manifest.json").read_text(encoding="utf-8"))
    if progression.get("models_retrained") is not False or progression.get("split") != "validation":
        raise ValueError("Surface progression manifest violates the frozen validation-only contract")

    prediction = pd.read_csv(root / "tables" / "prediction_metrics_strict_test.csv")
    overall = prediction[prediction.diagnosis.eq("overall")]
    if overall.method.nunique() < 12 or overall.subjects.min() < 20:
        raise ValueError("Strict prediction comparison is incomplete")
    ode = overall[overall.method.isin(["pca_plain_ode", "pca_brainode"])]
    if set(ode.method) != {"pca_plain_ode", "pca_brainode"}:
        raise ValueError("Matched PCA ODE endpoint rows are missing")
    required_ode_metrics = ["mean_vertex_error_mm", "assd_mm", "hd95_mm", "volume_relative_error"]
    if not np.isfinite(ode[required_ode_metrics].to_numpy(float)).all():
        raise ValueError("Matched PCA ODE endpoint metrics contain unavailable values")
    ranking = pd.read_csv(root / "tables" / "validation_best_method_ranking.csv")
    if int(ranking.selected_for_ood.astype(bool).sum()) != 1:
        raise ValueError("Validation selection must choose exactly one method")

    required_figures = [
        "prediction_metrics_strict_test.png", "prediction_vs_nochange.png",
        "representation_reconstruction_floor.png", "inr_128_vs_256_reconstruction.png",
        "volume_rate_ad_cn.png", "velocity_matched_cohort_summary.png",
        "velocity_regional_inward.png", "velocity_interval_integrated.png",
        "velocity_tangent_consistency.png", "velocity_paired_condition_trajectory.png",
        "surface_inward_displacement_progression.png",
        "surface_radial_narrowing_progression.png",
        "surface_nonuniform_area_progression.png",
        "velocity_ad_minus_cn_surface_map.png",
        "ood_best_vs_brainode_volume.png",
        "prediction_metrics_strict_test_without_direct_mesh.png",
        "prediction_vs_nochange_without_direct_mesh.png",
        "volume_rate_ad_cn_without_direct_mesh.png",
        "velocity_matched_cohort_summary_without_direct_mesh.png",
        "velocity_regional_inward_without_direct_mesh.png",
        "velocity_interval_integrated_without_direct_mesh.png",
        "velocity_tangent_consistency_without_direct_mesh.png",
        "velocity_paired_condition_trajectory_without_direct_mesh.png",
        "surface_inward_displacement_progression_without_direct_mesh.png",
        "surface_radial_narrowing_progression_without_direct_mesh.png",
        "surface_nonuniform_area_progression_without_direct_mesh.png",
        "velocity_ad_minus_cn_surface_map_without_direct_mesh.png",
        "ood_brainode_only_volume_without_direct_mesh.png",
    ]
    for name in required_figures:
        path = root / "figures" / name
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"Missing or suspiciously small figure: {path}")

    html = EXPERIMENT_DIR / "html" / "best_cocycle_vs_brainode_ad_age105.html"
    if not html.is_file() or html.stat().st_size < 50_000:
        raise ValueError(f"Interactive HTML missing or too small: {html}")
    content = html.read_text(encoding="utf-8", errors="ignore")
    for token in ("BrainODE", "age 105", "Observed"):
        if token not in content:
            raise ValueError(f"Interactive HTML is missing label: {token}")

    notebook = json.loads(args.notebook.expanduser().resolve().read_text(encoding="utf-8"))
    errors = []
    executed = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            executed += int(cell.get("execution_count") is not None)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append({"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")})
    if errors or executed == 0:
        raise ValueError(f"Notebook validation failed: executed_code_cells={executed}, errors={errors}")

    report = {
        "schema_version": 1, "status": "validated", "table_checks": checks,
        "velocity_methods": sorted(required_methods), "overall_prediction_methods": int(overall.method.nunique()),
        "selected_method": ranking.loc[ranking.selected_for_ood.astype(bool), "method"].iloc[0],
        "figures": len(required_figures), "html": str(html), "executed_code_cells": executed,
        "surface_progression_methods": progression["methods"],
    }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
