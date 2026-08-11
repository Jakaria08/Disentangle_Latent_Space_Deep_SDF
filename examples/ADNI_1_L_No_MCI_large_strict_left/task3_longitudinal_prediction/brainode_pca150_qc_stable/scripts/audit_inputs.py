#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from core_brainode_common import (
    SPLITS,
    TASK_DIR,
    as_float,
    as_int,
    continuous_age_years,
    count_by,
    group_rows_by_subject,
    load_clean_manifest,
    load_config,
    load_json,
    load_numpy_vector,
    load_pca_coefficients_csv,
    load_representation_manifest,
    normalize_age,
    resolve_repo_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Task 1 and Task 2 artifacts before building the core BrainODE dataset."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    return parser.parse_args()


def row_continuous_age(row: dict[str, str], baseline_age_years: float) -> float:
    value = str(row.get("continuous_age_years", "")).strip()
    if value:
        return as_float(value)
    return continuous_age_years(
        baseline_age_years,
        as_float(row["months_from_baseline"]),
    )


def row_continuous_norm(
    row: dict[str, str],
    continuous_year: float,
    age_min_train: float,
    age_max_train: float,
) -> float:
    value = str(row.get("continuous_age_norm", "")).strip()
    if value:
        return as_float(value)
    return normalize_age(continuous_year, age_min_train, age_max_train)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    errors: list[str] = []
    warnings: list[str] = []

    task1 = config["task1"]
    task2 = config["task2"]
    expected_counts = {
        split: int(config["expected_counts"][split]) for split in (*SPLITS, "total")
    }
    expected_subject_counts = {
        split: int(config["expected_subject_counts"][split])
        for split in (*SPLITS, "total")
    }
    available_components = int(config["brainode"]["available_components"])
    primary_components = int(config["brainode"]["primary_components"])

    required_paths = {
        "task1_clean_manifest": resolve_repo_path(task1["clean_manifest"]),
        "task1_trajectory_json": resolve_repo_path(task1["trajectory_json"]),
        "task1_split_summary": resolve_repo_path(task1["split_summary"]),
        "task1_age_norm_stats": resolve_repo_path(task1["age_norm_stats"]),
        "task1_validation_report": resolve_repo_path(task1["validation_report"]),
        "task2_representation_manifest": resolve_repo_path(
            task2["representation_manifest"]
        ),
        "task2_pca_coefficients_csv": resolve_repo_path(task2["pca_coefficients_csv"]),
        "task2_validation_report": resolve_repo_path(task2["validation_report"]),
        "task2_pca_mean": resolve_repo_path(task2["pca_model_dir"]) / "mean.npy",
        "task2_pca_components": resolve_repo_path(task2["pca_model_dir"])
        / "components_256.npy",
        "task2_pca_faces": resolve_repo_path(task2["pca_model_dir"]) / "faces.npy",
        "task2_pca_mean_coeff": resolve_repo_path(task2["pca_model_dir"])
        / "coefficient_mean_train.npy",
        "task2_pca_std_coeff": resolve_repo_path(task2["pca_model_dir"])
        / "coefficient_std_train.npy",
    }
    missing_paths = {
        name: str(path)
        for name, path in required_paths.items()
        if not path.is_file()
    }
    if missing_paths:
        errors.extend(
            f"Missing required input file: {name} -> {path}"
            for name, path in sorted(missing_paths.items())
        )
        report = {
            "status": "fail",
            "errors": errors,
            "warnings": warnings,
            "missing_paths": missing_paths,
        }
        write_json(TASK_DIR / "audit" / "input_audit.json", report)
        print(json.dumps({"status": "fail", "errors": len(errors)}, indent=2))
        return 1

    task1_validation = load_json(required_paths["task1_validation_report"])
    if task1_validation.get("status") != "pass":
        errors.append("Task 1 validation report status is not pass.")
    if task1_validation.get("errors"):
        errors.append("Task 1 validation report contains errors.")

    task2_validation = load_json(required_paths["task2_validation_report"])
    if task2_validation.get("status") != "pass":
        errors.append("Task 2 PCA validation report status is not pass.")
    if task2_validation.get("errors"):
        errors.append("Task 2 PCA validation report contains errors.")

    clean_rows = load_clean_manifest(required_paths["task1_clean_manifest"])
    representation_rows = load_representation_manifest(
        required_paths["task2_representation_manifest"]
    )
    pca_rows = load_pca_coefficients_csv(required_paths["task2_pca_coefficients_csv"])
    trajectories = load_json(required_paths["task1_trajectory_json"])
    split_summary = load_json(required_paths["task1_split_summary"])
    age_stats = load_json(required_paths["task1_age_norm_stats"])

    if len(clean_rows) != expected_counts["total"]:
        errors.append(
            f"Expected {expected_counts['total']} clean rows, found {len(clean_rows)}."
        )
    if len(representation_rows) != expected_counts["total"]:
        errors.append(
            "Representation manifest row count does not match expected total."
        )
    if len(pca_rows) != expected_counts["total"]:
        errors.append(f"Expected {expected_counts['total']} PCA rows, found {len(pca_rows)}.")

    clean_split_counts = count_by(clean_rows, "split")
    representation_split_counts = count_by(representation_rows, "split")
    pca_split_counts = count_by(pca_rows, "split")
    for split in SPLITS:
        expected = expected_counts[split]
        if clean_split_counts.get(split, 0) != expected:
            errors.append(
                f"Expected {expected} clean {split} rows, found {clean_split_counts.get(split, 0)}."
            )
        if representation_split_counts.get(split, 0) != expected:
            errors.append(
                f"Expected {expected} representation {split} rows, found {representation_split_counts.get(split, 0)}."
            )
        if pca_split_counts.get(split, 0) != expected:
            errors.append(
                f"Expected {expected} PCA {split} rows, found {pca_split_counts.get(split, 0)}."
            )

    clean_by_scan = {row["scan_id"]: row for row in clean_rows}
    representation_by_scan = {row["scan_id"]: row for row in representation_rows}
    pca_by_scan = {row["scan_id"]: row for row in pca_rows}
    if len(clean_by_scan) != len(clean_rows):
        errors.append("Duplicate scan_id values found in Task 1 clean manifest.")
    if len(representation_by_scan) != len(representation_rows):
        errors.append("Duplicate scan_id values found in Task 2 representation manifest.")
    if len(pca_by_scan) != len(pca_rows):
        errors.append("Duplicate scan_id values found in Task 2 PCA coefficient CSV.")

    clean_scan_ids = set(clean_by_scan)
    representation_scan_ids = set(representation_by_scan)
    pca_scan_ids = set(pca_by_scan)
    if clean_scan_ids != representation_scan_ids:
        errors.append(
            "Task 1 clean manifest and Task 2 representation manifest scan_id sets differ."
        )
    if clean_scan_ids != pca_scan_ids:
        errors.append(
            "Task 1 clean manifest and Task 2 PCA coefficient CSV scan_id sets differ."
        )

    subjects = group_rows_by_subject(clean_rows)
    if len(subjects) != expected_subject_counts["total"]:
        errors.append(
            f"Expected {expected_subject_counts['total']} subjects, found {len(subjects)}."
        )
    subject_split_counts = Counter()
    visit_count_distribution = Counter()
    continuous_norm_values: list[float] = []
    continuous_year_values: list[float] = []
    for subject_id, visits in sorted(subjects.items()):
        splits = {row["split"] for row in visits}
        diagnoses = {row["diagnosis"] for row in visits}
        labels = {row["label_ad"] for row in visits}
        if len(splits) != 1:
            errors.append(f"Subject appears in multiple splits: {subject_id}")
            continue
        split = next(iter(splits))
        subject_split_counts[split] += 1
        visit_count_distribution[len(visits)] += 1
        if len(diagnoses) != 1:
            errors.append(f"Subject has inconsistent diagnosis labels: {subject_id}")
        if len(labels) != 1:
            errors.append(f"Subject has inconsistent AD labels: {subject_id}")

        baseline_age = as_float(visits[0]["age_years"])
        previous_continuous = None
        for expected_order, row in enumerate(visits):
            visit_order = as_int(row["visit_order"])
            if visit_order != expected_order:
                errors.append(
                    f"Subject visit_order is not contiguous from zero: {subject_id}"
                )
            continuous_year = row_continuous_age(row, baseline_age)
            continuous_norm = row_continuous_norm(
                row,
                continuous_year,
                float(age_stats["age_min_train"]),
                float(age_stats["age_max_train"]),
            )
            continuous_year_values.append(continuous_year)
            continuous_norm_values.append(continuous_norm)
            if previous_continuous is not None and not continuous_year > previous_continuous:
                errors.append(
                    f"Continuous age is not strictly increasing for subject {subject_id}."
                )
            previous_continuous = continuous_year

            rep_row = representation_by_scan.get(row["scan_id"])
            pca_row = pca_by_scan.get(row["scan_id"])
            if rep_row is None or pca_row is None:
                errors.append(f"Missing Task 2 representation row for {row['scan_id']}")
                continue
            for key in ("subject_id", "split", "diagnosis", "label_ad", "visit_order"):
                if row[key] != rep_row[key]:
                    errors.append(
                        f"Task 1 / representation mismatch for {row['scan_id']} column {key}."
                    )
                if key in pca_row and row[key] != pca_row[key]:
                    errors.append(
                        f"Task 1 / PCA CSV mismatch for {row['scan_id']} column {key}."
                    )
            coeff_path = Path(rep_row["pca_coeff_path"])
            if not coeff_path.is_file():
                errors.append(f"Missing PCA coefficient file: {coeff_path}")
                continue
            try:
                coefficients = load_numpy_vector(coeff_path, available_components)
            except Exception as exc:
                errors.append(f"Invalid PCA coefficient vector for {row['scan_id']}: {exc}")
                continue
            if not np.isfinite(coefficients[:primary_components]).all():
                errors.append(f"Primary PCA coefficients contain non-finite values: {row['scan_id']}")

    for split in SPLITS:
        expected = expected_subject_counts[split]
        if subject_split_counts.get(split, 0) != expected:
            errors.append(
                f"Expected {expected} {split} subjects, found {subject_split_counts.get(split, 0)}."
            )

    if set(trajectories) != set(subjects):
        errors.append("Task 1 trajectory JSON subject set does not match clean manifest.")
    else:
        for subject_id, visits in subjects.items():
            trajectory = trajectories[subject_id]
            trajectory_visits = trajectory.get("visits", [])
            if len(trajectory_visits) != len(visits):
                errors.append(
                    f"Trajectory visit count mismatch for subject {subject_id}."
                )
                continue
            scan_ids_manifest = [row["scan_id"] for row in visits]
            scan_ids_trajectory = [visit["scan_id"] for visit in trajectory_visits]
            if scan_ids_manifest != scan_ids_trajectory:
                errors.append(
                    f"Trajectory visit ordering mismatch for subject {subject_id}."
                )

    components = np.load(required_paths["task2_pca_components"])
    mean = np.load(required_paths["task2_pca_mean"])
    faces = np.load(required_paths["task2_pca_faces"])
    coeff_mean = np.load(required_paths["task2_pca_mean_coeff"])
    coeff_std = np.load(required_paths["task2_pca_std_coeff"])
    if components.shape != (available_components, mean.shape[0]):
        errors.append(
            f"Unexpected PCA components shape: {components.shape} against mean shape {mean.shape}."
        )
    if mean.ndim != 1:
        errors.append(f"Unexpected PCA mean shape: {mean.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        errors.append(f"Unexpected PCA faces shape: {faces.shape}")
    if coeff_mean.shape != (available_components,):
        errors.append(f"Unexpected PCA coefficient mean shape: {coeff_mean.shape}")
    if coeff_std.shape != (available_components,):
        errors.append(f"Unexpected PCA coefficient std shape: {coeff_std.shape}")
    if np.any(coeff_std <= 0.0):
        errors.append("PCA coefficient std contains non-positive values.")

    summary_checks = {
        "clean_scan_count": int(split_summary["clean_scan_count"]),
        "clean_subject_count": int(split_summary["clean_subject_count"]),
    }
    if summary_checks["clean_scan_count"] != expected_counts["total"]:
        errors.append("Task 1 split_summary clean_scan_count is unexpected.")
    if summary_checks["clean_subject_count"] != expected_subject_counts["total"]:
        errors.append("Task 1 split_summary clean_subject_count is unexpected.")

    report = {
        "status": "pass" if not errors else "fail",
        "config": str(Path(args.config).resolve()),
        "task1_clean_manifest_sha256": sha256_file(required_paths["task1_clean_manifest"]),
        "task2_representation_manifest_sha256": sha256_file(
            required_paths["task2_representation_manifest"]
        ),
        "task2_pca_coefficients_csv_sha256": sha256_file(
            required_paths["task2_pca_coefficients_csv"]
        ),
        "counts": {
            "expected_scans": expected_counts,
            "expected_subjects": expected_subject_counts,
            "clean_split_counts": clean_split_counts,
            "representation_split_counts": representation_split_counts,
            "pca_split_counts": pca_split_counts,
            "subject_split_counts": dict(subject_split_counts),
            "visit_count_distribution": dict(sorted(visit_count_distribution.items())),
            "diagnosis_scan_counts": count_by(clean_rows, "diagnosis"),
        },
        "continuous_age_years": {
            "min": float(min(continuous_year_values)),
            "max": float(max(continuous_year_values)),
        },
        "continuous_age_norm": {
            "min": float(min(continuous_norm_values)),
            "max": float(max(continuous_norm_values)),
        },
        "pca_shapes": {
            "components": list(components.shape),
            "mean": list(mean.shape),
            "faces": list(faces.shape),
            "coefficient_mean": list(coeff_mean.shape),
            "coefficient_std": list(coeff_std.shape),
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(TASK_DIR / "audit" / "input_audit.json", report)
    print(json.dumps({"status": report["status"], "errors": len(errors)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
