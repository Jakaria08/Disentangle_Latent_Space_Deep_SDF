from __future__ import annotations

import math

import numpy as np

import analyze_velocity_by_age as A


def _row(subject: str, diagnosis: str, age: float, observed: float, predicted: float, weight: float = 1.0):
    return {
        "scan_id": f"{subject}_{age}",
        "subject_id": subject,
        "diagnosis": diagnosis,
        "age_years": age,
        "reference_reliability": weight,
        "predicted_speed_mm_per_year": predicted,
        "observed_speed_mm_per_year": observed,
        "predicted_inward_normal_mm_per_year": 0.5 * predicted,
        "observed_inward_normal_mm_per_year": 0.5 * observed,
        "vector_rmse_mm_per_year": abs(predicted - observed),
        "zero_vector_rmse_mm_per_year": observed,
        "normal_rmse_mm_per_year": 0.5 * abs(predicted - observed),
        "zero_normal_rmse_mm_per_year": 0.5 * observed,
        "vector_cosine": 0.8,
        "normal_pearson": 0.7,
        "normal_sign_agreement": 0.75,
    }


def test_age_bin_boundaries_are_deterministic():
    edges = A.validate_age_edges([70, 75, 80, 95])
    assert A.age_bin_index(69.999, edges) is None
    assert A.age_bin_index(70.0, edges) == 0
    assert A.age_bin_index(75.0, edges) == 1
    assert A.age_bin_index(80.0, edges) == 2
    assert A.age_bin_index(95.0, edges) == 2
    assert A.age_bin_index(95.001, edges) is None
    assert A.age_bin_label(2, edges) == "[80, 95]"


def test_weighted_age_summary_uses_reliability():
    records = [
        _row("a", "CN", 72.0, observed=2.0, predicted=1.0, weight=1.0),
        _row("b", "CN", 73.0, observed=4.0, predicted=3.0, weight=3.0),
    ]
    result = A.summarize_records(records)
    assert math.isclose(result["observed_speed_mm_per_year"], 3.5)
    assert math.isclose(result["predicted_speed_mm_per_year"], 2.5)
    assert math.isclose(result["speed_ratio"], 2.5 / 3.5)
    assert result["subjects"] == 2
    assert result["visits"] == 2


def test_bootstrap_is_subject_clustered_and_repeatable():
    records = [
        _row("a", "AD", 72.0, observed=2.0, predicted=1.0),
        _row("a", "AD", 73.0, observed=2.2, predicted=1.1),
        _row("b", "AD", 72.0, observed=4.0, predicted=3.0),
        _row("c", "AD", 74.0, observed=6.0, predicted=4.0),
    ]
    left = A.subject_bootstrap_intervals(records, samples=200, seed=7)
    right = A.subject_bootstrap_intervals(records, samples=200, seed=7)
    assert left == right
    assert left["observed_speed_mm_per_year_ci95_low"] < left[
        "observed_speed_mm_per_year_ci95_high"
    ]
    assert left["speed_ratio_ci95_low"] <= left["speed_ratio_ci95_high"]


def test_age_summary_separates_diagnosis_and_interval():
    edges = np.asarray([70.0, 75.0, 80.0])
    records = [
        _row("c1", "CN", 72.0, 2.0, 1.0),
        _row("c2", "CN", 77.0, 3.0, 2.0),
        _row("a1", "AD", 72.0, 4.0, 3.0),
        _row("a2", "AD", 77.0, 5.0, 4.0),
    ]
    summary = A.age_summary(records, edges, bootstrap_samples=0, seed=3)
    assert [(row["diagnosis"], row["age_bin_index"]) for row in summary] == [
        ("CN", 0),
        ("CN", 1),
        ("AD", 0),
        ("AD", 1),
    ]
    assert all(math.isnan(row["vector_cosine_ci95_low"]) for row in summary)
