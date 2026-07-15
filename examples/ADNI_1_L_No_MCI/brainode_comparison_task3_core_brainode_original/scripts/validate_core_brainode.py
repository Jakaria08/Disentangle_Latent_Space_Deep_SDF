#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from core_brainode_common import (
    SPLITS,
    TASK_DIR,
    load_config,
    load_json,
    read_csv,
    resolve_repo_path,
    write_json,
)


STAGES = ("inputs", "dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated core BrainODE dataset artifacts."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    parser.add_argument("--stage", choices=STAGES, default="dataset")
    return parser.parse_args()


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_npz(
    path: Path,
    expected_subjects: int,
    expected_visits: int,
    expected_primary_components: int,
    expected_available_components: int,
    errors: list[str],
    checks: dict[str, object],
    key_prefix: str,
) -> None:
    if not path.is_file():
        add_error(errors, f"Missing dataset archive: {path}")
        return

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "subject_ids",
            "subject_splits",
            "subject_diagnoses",
            "subject_label_ad",
            "subject_cognition",
            "subject_baseline_age_years",
            "subject_visit_offsets",
            "visit_scan_ids",
            "visit_subject_ids",
            "visit_splits",
            "visit_diagnoses",
            "visit_label_ad",
            "visit_cognition",
            "visit_orders",
            "visit_months_from_baseline",
            "visit_age_years",
            "visit_age_norm_from_manifest",
            "visit_continuous_age_years",
            "visit_continuous_age_norm",
            "visit_pca_150",
            "visit_pca_256",
            "train_coefficient_mean_150",
            "train_coefficient_std_150",
            "train_coefficient_mean_256",
            "train_coefficient_std_256",
        }
        missing = required.difference(archive.files)
        if missing:
            add_error(errors, f"{path.name} is missing arrays: {sorted(missing)}")
            return

        subject_ids = archive["subject_ids"]
        subject_visit_offsets = archive["subject_visit_offsets"]
        visit_scan_ids = archive["visit_scan_ids"]
        visit_splits = archive["visit_splits"]
        visit_label_ad = archive["visit_label_ad"]
        visit_cognition = archive["visit_cognition"]
        visit_continuous_age_years = archive["visit_continuous_age_years"]
        visit_continuous_age_norm = archive["visit_continuous_age_norm"]
        visit_pca_150 = archive["visit_pca_150"]
        visit_pca_256 = archive["visit_pca_256"]

        checks[f"{key_prefix}_subject_count"] = int(subject_ids.shape[0])
        checks[f"{key_prefix}_visit_count"] = int(visit_scan_ids.shape[0])
        checks[f"{key_prefix}_visit_pca_150_shape"] = list(visit_pca_150.shape)
        checks[f"{key_prefix}_visit_pca_256_shape"] = list(visit_pca_256.shape)

        if subject_ids.shape[0] != expected_subjects:
            add_error(
                errors,
                f"{path.name} expected {expected_subjects} subjects, found {subject_ids.shape[0]}.",
            )
        if visit_scan_ids.shape[0] != expected_visits:
            add_error(
                errors,
                f"{path.name} expected {expected_visits} visits, found {visit_scan_ids.shape[0]}.",
            )
        if subject_visit_offsets.shape != (expected_subjects + 1,):
            add_error(
                errors,
                f"{path.name} has invalid subject_visit_offsets shape {subject_visit_offsets.shape}.",
            )
        elif int(subject_visit_offsets[-1]) != expected_visits:
            add_error(errors, f"{path.name} last subject offset does not equal visit count.")
        if visit_pca_150.shape != (expected_visits, expected_primary_components):
            add_error(
                errors,
                f"{path.name} visit_pca_150 shape mismatch: {visit_pca_150.shape}.",
            )
        if visit_pca_256.shape != (expected_visits, expected_available_components):
            add_error(
                errors,
                f"{path.name} visit_pca_256 shape mismatch: {visit_pca_256.shape}.",
            )
        if not np.allclose(
            visit_pca_150, visit_pca_256[:, :expected_primary_components]
        ):
            add_error(errors, f"{path.name} visit_pca_150 does not match visit_pca_256 slice.")
        if not np.isfinite(visit_pca_150).all() or not np.isfinite(visit_pca_256).all():
            add_error(errors, f"{path.name} contains non-finite PCA coefficients.")
        if archive["train_coefficient_mean_150"].shape != (expected_primary_components,):
            add_error(errors, f"{path.name} train_coefficient_mean_150 shape is invalid.")
        if archive["train_coefficient_std_150"].shape != (expected_primary_components,):
            add_error(errors, f"{path.name} train_coefficient_std_150 shape is invalid.")
        if archive["train_coefficient_mean_256"].shape != (expected_available_components,):
            add_error(errors, f"{path.name} train_coefficient_mean_256 shape is invalid.")
        if archive["train_coefficient_std_256"].shape != (expected_available_components,):
            add_error(errors, f"{path.name} train_coefficient_std_256 shape is invalid.")
        if np.any(archive["train_coefficient_std_150"] <= 0.0) or np.any(
            archive["train_coefficient_std_256"] <= 0.0
        ):
            add_error(errors, f"{path.name} coefficient std contains non-positive values.")

        split_counts = Counter(str(value) for value in visit_splits.tolist())
        checks[f"{key_prefix}_visit_split_counts"] = dict(split_counts)
        for index in range(expected_subjects):
            start = int(subject_visit_offsets[index])
            end = int(subject_visit_offsets[index + 1])
            if not start < end:
                add_error(errors, f"{path.name} subject index {index} has empty visit slice.")
                continue
            times = visit_continuous_age_years[start:end]
            if not np.all(np.diff(times) > 0.0):
                add_error(errors, f"{path.name} subject index {index} has non-monotonic times.")
            labels = visit_label_ad[start:end]
            cognition = visit_cognition[start:end]
            if not np.all(labels == labels[0]):
                add_error(errors, f"{path.name} subject index {index} has mixed diagnosis labels.")
            if not np.allclose(cognition, float(labels[0])):
                add_error(errors, f"{path.name} subject index {index} has cognition/label mismatch.")
        if not np.isfinite(visit_continuous_age_norm).all():
            add_error(errors, f"{path.name} has non-finite normalized continuous ages.")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}

    expected_scans = {
        split: int(config["expected_counts"][split]) for split in (*SPLITS, "total")
    }
    expected_subjects = {
        split: int(config["expected_subject_counts"][split])
        for split in (*SPLITS, "total")
    }
    primary_components = int(config["brainode"]["primary_components"])
    available_components = int(config["brainode"]["available_components"])

    audit_path = TASK_DIR / "audit" / "input_audit.json"
    if not audit_path.is_file():
        add_error(errors, f"Missing input audit: {audit_path}")
    else:
        audit_report = load_json(audit_path)
        checks["input_audit_status"] = audit_report.get("status")
        if audit_report.get("status") != "pass":
            add_error(errors, "Input audit status is not pass.")

    if args.stage == "inputs":
        report = {
            "stage": args.stage,
            "status": "pass" if not errors else "fail",
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }
        write_json(TASK_DIR / "metadata" / "validation_inputs.json", report)
        print(json.dumps(report, indent=2))
        return 0 if not errors else 1

    required_files = [
        TASK_DIR / "metadata" / "core_brainode_scan_manifest.csv",
        TASK_DIR / "metadata" / "core_brainode_subject_sequences.json",
        TASK_DIR / "metadata" / "dataset_summary.json",
        TASK_DIR / "dataset" / "all_subject_sequences.npz",
        TASK_DIR / "dataset" / "train_subject_sequences.npz",
        TASK_DIR / "dataset" / "val_subject_sequences.npz",
        TASK_DIR / "dataset" / "test_subject_sequences.npz",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        errors.extend(f"Missing generated file: {path}" for path in missing_files)
        report = {
            "stage": args.stage,
            "status": "fail",
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }
        write_json(TASK_DIR / "metadata" / "validation_dataset.json", report)
        print(json.dumps(report, indent=2))
        return 1

    scan_manifest = read_csv(TASK_DIR / "metadata" / "core_brainode_scan_manifest.csv")
    subject_sequences = load_json(TASK_DIR / "metadata" / "core_brainode_subject_sequences.json")
    summary = load_json(TASK_DIR / "metadata" / "dataset_summary.json")
    source_clean = read_csv(resolve_repo_path(config["task1"]["clean_manifest"]))

    checks["scan_manifest_rows"] = len(scan_manifest)
    checks["subject_sequence_count"] = len(subject_sequences)
    checks["dataset_summary_status"] = summary.get("status")
    if len(scan_manifest) != expected_scans["total"]:
        add_error(errors, f"Expected {expected_scans['total']} scan manifest rows, found {len(scan_manifest)}.")
    if len(subject_sequences) != expected_subjects["total"]:
        add_error(errors, f"Expected {expected_subjects['total']} subject sequences, found {len(subject_sequences)}.")
    if summary.get("status") != "pass":
        add_error(errors, "dataset_summary.json status is not pass.")

    manifest_split_counts = Counter(row["split"] for row in scan_manifest)
    manifest_subject_split_counts = Counter(
        subject["split"] for subject in subject_sequences.values()
    )
    checks["scan_manifest_split_counts"] = dict(manifest_split_counts)
    checks["subject_sequence_split_counts"] = dict(manifest_subject_split_counts)
    for split in SPLITS:
        if manifest_split_counts.get(split, 0) != expected_scans[split]:
            add_error(errors, f"Scan manifest {split} count mismatch.")
        if manifest_subject_split_counts.get(split, 0) != expected_subjects[split]:
            add_error(errors, f"Subject sequence {split} count mismatch.")

    source_scan_ids = {row["scan_id"] for row in source_clean}
    manifest_scan_ids = {row["scan_id"] for row in scan_manifest}
    json_scan_ids = {
        visit["scan_id"]
        for subject in subject_sequences.values()
        for visit in subject["visits"]
    }
    if manifest_scan_ids != source_scan_ids:
        add_error(errors, "Scan manifest scan_id set does not match Task 1 clean manifest.")
    if json_scan_ids != source_scan_ids:
        add_error(errors, "Subject sequence JSON scan_id set does not match Task 1 clean manifest.")

    for subject_id, subject in sorted(subject_sequences.items()):
        visits = subject["visits"]
        if len(visits) != int(subject["visit_count"]):
            add_error(errors, f"Subject {subject_id} visit_count does not match visits length.")
        orders = [int(visit["visit_order"]) for visit in visits]
        if orders != list(range(len(orders))):
            add_error(errors, f"Subject {subject_id} visit_order is not contiguous.")
        times = [float(visit["continuous_age_years"]) for visit in visits]
        if any(second <= first for first, second in zip(times, times[1:])):
            add_error(errors, f"Subject {subject_id} continuous_age_years is not strictly increasing.")
        diagnosis = subject["diagnosis"]
        label_ad = int(subject["label_ad"])
        cognition_value = float(subject["cognition_value"])
        expected_cognition = 0.0 if diagnosis == "CN" else 1.0
        if cognition_value != expected_cognition or cognition_value != float(label_ad):
            add_error(errors, f"Subject {subject_id} has inconsistent diagnosis/cognition encoding.")

    validate_npz(
        TASK_DIR / "dataset" / "all_subject_sequences.npz",
        expected_subjects=expected_subjects["total"],
        expected_visits=expected_scans["total"],
        expected_primary_components=primary_components,
        expected_available_components=available_components,
        errors=errors,
        checks=checks,
        key_prefix="all",
    )
    for split in SPLITS:
        validate_npz(
            TASK_DIR / "dataset" / f"{split}_subject_sequences.npz",
            expected_subjects=expected_subjects[split],
            expected_visits=expected_scans[split],
            expected_primary_components=primary_components,
            expected_available_components=available_components,
            errors=errors,
            checks=checks,
            key_prefix=split,
        )

    report = {
        "stage": args.stage,
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    output_path = TASK_DIR / "metadata" / "validation_dataset.json"
    write_json(output_path, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
