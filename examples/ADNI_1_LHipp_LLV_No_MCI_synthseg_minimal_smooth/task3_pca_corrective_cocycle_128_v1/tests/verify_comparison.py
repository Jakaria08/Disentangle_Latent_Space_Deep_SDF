#!/usr/bin/env python3
"""Fast, data-free contract tests for the five-model comparison aggregation."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare_five_models as F  # noqa: E402


def synthetic_rows() -> list[dict]:
    rows = []
    models = (*F.FIXED_MODELS, "inr256")
    for model_number, model in enumerate(models):
        for subject_number, (subject, diagnosis) in enumerate((("1", "CN"), ("2", "AD"))):
            value = 1.0 + 0.1 * model_number + 0.01 * subject_number
            rows.append({
                "representation": model,
                "subject_id": subject,
                "diagnosis": diagnosis,
                "prediction_assd_mm": value,
                "nochange_assd_mm": 2.0,
                "prediction_hd95_mm": 2.0 * value,
                "nochange_hd95_mm": 4.0,
                "prediction_chamfer_l2_squared_mm2": value * value,
                "nochange_chamfer_l2_squared_mm2": 4.0,
                "prediction_volume_absolute_error_mm3": 10.0 * value,
                "nochange_volume_absolute_error_mm3": 20.0,
                "prediction_volume_relative_error": 0.01 * value,
                "nochange_volume_relative_error": 0.02,
                "floor_assd_mm": 0.1,
                "floor_hd95_mm": 0.2,
                "floor_chamfer_l2_squared_mm2": 0.03,
                "floor_volume_relative_error": 0.001,
                "prediction_log_volume_rate_absolute_error_per_year": 0.01 * value,
                "nochange_log_volume_rate_absolute_error_per_year": 0.02,
                "observed_signed_log_volume_rate_per_year": -0.01 - 0.01 * subject_number,
                "prediction_signed_log_volume_rate_raw_anchor_per_year": -0.008 - 0.01 * subject_number,
                "observed_annualized_percent_change": -1.0 - subject_number,
                "prediction_annualized_percent_change_raw_anchor": -0.8 - subject_number,
                "prediction_atrophy_direction_agreement": 1.0,
                "source_true_volume_mm3": 3000.0,
                "target_true_volume_mm3": 2900.0,
                "prediction_volume_mm3": 2910.0,
                "observed_volume_change_mm3_per_year": -50.0,
                "prediction_volume_change_raw_anchor_mm3_per_year": -45.0,
            })
    return rows


def main() -> int:
    models = (*F.FIXED_MODELS, "inr256")
    rows = synthetic_rows()
    summary = F.summarize_rows(rows, "synthetic matched cohort", models)
    assert len(summary) == 3 * len(models)
    assert len(F.trend_gaps(summary)) == len(models)
    intervals = F.bootstrap_against_pca(rows, 100, 42, models)
    assert len(intervals) == (len(models) - 1) * len(F.EXACT_ERROR_METRICS)
    assert all(row["subjects"] == 2 for row in intervals)
    branch_rows = [dict(row, representation=F.CORRECTIVE_PCA_BRANCH) for row in rows if row["representation"] == "pca_corrective128"]
    decoder = F.decoder_ablation_bootstrap([*rows, *branch_rows], 100, 42)
    assert len(decoder) == len(F.EXACT_ERROR_METRICS) + 2
    assert all(row["corrective_minus_pca_decode_mean"] == 0.0 for row in decoder)
    print("comparison contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
