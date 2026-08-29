#!/usr/bin/env python3
"""Validate the all-method age-stratified velocity cache without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task4_all_flow_visual_analysis_v1/age_velocity_all_methods_v1"
)
FULL_METHODS = {
    "mesh_spiral",
    "mesh_adaptive",
    "latent_pca",
    "latent_spiral",
    "latent_adaptive",
    "lamm_n3",
}
ALL_METHODS = FULL_METHODS | {"latent_inr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def finite_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"scan_id": str, "subject_id": str})
    if frame.empty:
        raise ValueError(f"Empty table: {path}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    # CI columns are intentionally NaN when an age cell contains one subject.
    non_ci = [column for column in frame.select_dtypes(include=[np.number]).columns if "_ci95_" not in column]
    if not np.isfinite(frame[non_ci].to_numpy()).all():
        raise ValueError(f"Non-finite primary values: {path}")
    if np.isinf(numeric).any():
        raise ValueError(f"Infinite uncertainty values: {path}")
    return frame


def main() -> int:
    root = parse_args().root.expanduser().resolve()
    report = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("split") != "val":
        raise ValueError("Analysis manifest is not a completed validation result")
    if bool(report.get("test_data_loaded", True)):
        raise ValueError("Manifest reports test-data access")
    if set(report["primary_full_cohort_methods"]) != FULL_METHODS:
        raise ValueError("Full-cohort method contract changed")
    if set(report["strict_common_methods"]) != ALL_METHODS:
        raise ValueError("Strict-common method contract changed")
    if float(report["lamm_n3_baseline_speed_max_abs_difference"]) > 2.0e-5:
        raise ValueError("LAMM N3 implementation cross-check failed")

    visits = finite_table(root / "tables" / "per_visit_velocity.csv")
    if set(visits.method.astype(str)) != ALL_METHODS:
        raise ValueError("Per-visit method mismatch")
    counts = visits.groupby("method").scan_id.nunique().to_dict()
    for method in FULL_METHODS:
        if int(counts.get(method, 0)) != 269:
            raise ValueError(f"{method} does not contain 269 validation visits")
    if int(counts.get("latent_inr", 0)) != 100:
        raise ValueError("Latent INR does not contain its 100 validation visits")

    # Every full-cohort method must use exactly the same observed reference at a scan.
    full = visits[visits.method.isin(FULL_METHODS)]
    reference_spread = full.groupby("scan_id").observed_speed_mm_per_year.agg(lambda x: x.max() - x.min())
    # Direct-mesh CSV serialization originates from float32 Torch reductions;
    # the independent decoder paths use NumPy float64 reductions.
    if float(reference_spread.max()) > 2.0e-7:
        raise ValueError("Observed velocity differs across matched full-cohort methods")
    if not set(visits[visits.method.eq("latent_inr")].scan_id).issubset(set(full.scan_id)):
        raise ValueError("INR visit set is not a subset of the full validation cohort")

    full_summary = finite_table(root / "tables" / "full_cohort_age_summary.csv")
    common_summary = finite_table(root / "tables" / "strict_common_age_summary.csv")
    legacy_summary = finite_table(root / "tables" / "legacy_age_summary.csv")
    if set(full_summary.method.astype(str)) != FULL_METHODS or len(full_summary) != 36:
        raise ValueError("Full age-summary shape/method mismatch")
    if set(common_summary.method.astype(str)) != ALL_METHODS or len(common_summary) != 42:
        raise ValueError("Strict-common age-summary shape/method mismatch")
    if set(legacy_summary.method_label.astype(str)) != {"PCA Cocycle", "INR Cocycle", "Latent ODE", "BrainODE"}:
        raise ValueError("Historical comparator mismatch")

    required_figures = {
        "full_cohort_speed_by_age.png",
        "full_cohort_inward_normal_by_age.png",
        "full_cohort_normal_agreement_by_age.png",
        "strict_common_speed_by_age.png",
        "strict_common_inward_normal_by_age.png",
        "strict_common_normal_agreement_by_age.png",
        "legacy_ode_brainode_velocity_by_age.png",
    }
    missing = [name for name in sorted(required_figures) if not (root / "figures" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing figures: {missing}")
    small = [name for name in sorted(required_figures) if (root / "figures" / name).stat().st_size < 10_000]
    if small:
        raise ValueError(f"Suspiciously small figures: {small}")

    print(
        json.dumps(
            {
                "status": "passed",
                "split": "val",
                "methods": sorted(ALL_METHODS),
                "full_validation_visits": 269,
                "strict_common_visits": 100,
                "full_age_rows": len(full_summary),
                "strict_common_age_rows": len(common_summary),
                "legacy_age_rows": len(legacy_summary),
                "lamm_n3_crosscheck": report["lamm_n3_baseline_speed_max_abs_difference"],
                "test_data_loaded": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
