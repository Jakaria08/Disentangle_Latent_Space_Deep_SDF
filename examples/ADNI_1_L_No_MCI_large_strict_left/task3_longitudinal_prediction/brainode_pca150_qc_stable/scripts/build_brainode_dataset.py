#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core_brainode_common import (
    SPLITS,
    SPLIT_ORDER,
    TASK_DIR,
    as_float,
    as_int,
    continuous_age_years,
    count_by,
    format_float,
    group_rows_by_subject,
    load_clean_manifest,
    load_config,
    load_json,
    load_numpy_vector,
    load_representation_manifest,
    normalize_age,
    resolve_repo_path,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scan-level and subject-level PCA sequence datasets for core BrainODE training."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    parser.add_argument(
        "--skip-audit-check",
        action="store_true",
        help="Allow building even if audit/input_audit.json is missing or failed.",
    )
    return parser.parse_args()


def package_subjects(
    ordered_subject_ids: list[str],
    subject_sequences: dict[str, dict[str, Any]],
    coeff_mean_256: np.ndarray,
    coeff_std_256: np.ndarray,
    primary_components: int,
) -> dict[str, np.ndarray]:
    subject_splits: list[str] = []
    subject_diagnoses: list[str] = []
    subject_label_ad: list[int] = []
    subject_cognition: list[float] = []
    subject_baseline_age_years: list[float] = []
    subject_visit_offsets = [0]

    visit_scan_ids: list[str] = []
    visit_subject_ids: list[str] = []
    visit_splits: list[str] = []
    visit_diagnoses: list[str] = []
    visit_label_ad: list[int] = []
    visit_cognition: list[float] = []
    visit_orders: list[int] = []
    visit_months_from_baseline: list[float] = []
    visit_age_years: list[float] = []
    visit_age_norm_from_manifest: list[float] = []
    visit_continuous_age_years: list[float] = []
    visit_continuous_age_norm: list[float] = []
    pca_150_rows: list[np.ndarray] = []
    pca_256_rows: list[np.ndarray] = []

    for subject_id in ordered_subject_ids:
        subject = subject_sequences[subject_id]
        subject_splits.append(subject["split"])
        subject_diagnoses.append(subject["diagnosis"])
        subject_label_ad.append(int(subject["label_ad"]))
        subject_cognition.append(float(subject["cognition_value"]))
        subject_baseline_age_years.append(float(subject["baseline_age_years"]))
        for visit in subject["visits"]:
            visit_scan_ids.append(visit["scan_id"])
            visit_subject_ids.append(subject_id)
            visit_splits.append(subject["split"])
            visit_diagnoses.append(subject["diagnosis"])
            visit_label_ad.append(int(subject["label_ad"]))
            visit_cognition.append(float(subject["cognition_value"]))
            visit_orders.append(int(visit["visit_order"]))
            visit_months_from_baseline.append(float(visit["months_from_baseline"]))
            visit_age_years.append(float(visit["age_years"]))
            visit_age_norm_from_manifest.append(float(visit["age_norm_manifest"]))
            visit_continuous_age_years.append(float(visit["continuous_age_years"]))
            visit_continuous_age_norm.append(float(visit["continuous_age_norm"]))
            pca_150_rows.append(np.asarray(visit["pca_coefficients_150"], dtype=np.float32))
            pca_256_rows.append(np.asarray(visit["pca_coefficients_256"], dtype=np.float32))
        subject_visit_offsets.append(len(visit_scan_ids))

    pca_256 = np.stack(pca_256_rows, axis=0).astype(np.float32)
    pca_150 = np.stack(pca_150_rows, axis=0).astype(np.float32)
    return {
        "subject_ids": np.asarray(ordered_subject_ids),
        "subject_splits": np.asarray(subject_splits),
        "subject_diagnoses": np.asarray(subject_diagnoses),
        "subject_label_ad": np.asarray(subject_label_ad, dtype=np.int64),
        "subject_cognition": np.asarray(subject_cognition, dtype=np.float32),
        "subject_baseline_age_years": np.asarray(
            subject_baseline_age_years, dtype=np.float32
        ),
        "subject_visit_offsets": np.asarray(subject_visit_offsets, dtype=np.int64),
        "visit_scan_ids": np.asarray(visit_scan_ids),
        "visit_subject_ids": np.asarray(visit_subject_ids),
        "visit_splits": np.asarray(visit_splits),
        "visit_diagnoses": np.asarray(visit_diagnoses),
        "visit_label_ad": np.asarray(visit_label_ad, dtype=np.int64),
        "visit_cognition": np.asarray(visit_cognition, dtype=np.float32),
        "visit_orders": np.asarray(visit_orders, dtype=np.int64),
        "visit_months_from_baseline": np.asarray(
            visit_months_from_baseline, dtype=np.float32
        ),
        "visit_age_years": np.asarray(visit_age_years, dtype=np.float32),
        "visit_age_norm_from_manifest": np.asarray(
            visit_age_norm_from_manifest, dtype=np.float32
        ),
        "visit_continuous_age_years": np.asarray(
            visit_continuous_age_years, dtype=np.float32
        ),
        "visit_continuous_age_norm": np.asarray(
            visit_continuous_age_norm, dtype=np.float32
        ),
        "visit_pca_150": pca_150,
        "visit_pca_256": pca_256,
        "train_coefficient_mean_150": coeff_mean_256[:primary_components].astype(
            np.float32
        ),
        "train_coefficient_std_150": coeff_std_256[:primary_components].astype(
            np.float32
        ),
        "train_coefficient_mean_256": coeff_mean_256.astype(np.float32),
        "train_coefficient_std_256": coeff_std_256.astype(np.float32),
    }


def summarize_subject_sequences(
    subject_sequences: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    subject_count_by_split = Counter()
    scan_count_by_split = Counter()
    subject_diagnosis_count = Counter()
    scan_diagnosis_count = Counter()
    visit_count_distribution = Counter()
    forward_pair_count_by_split = Counter()
    directed_pair_count_by_split = Counter()
    continuous_age_norm_values: list[float] = []

    for subject in subject_sequences.values():
        split = subject["split"]
        diagnosis = subject["diagnosis"]
        subject_count_by_split[split] += 1
        subject_diagnosis_count[diagnosis] += 1
        visit_count = len(subject["visits"])
        visit_count_distribution[visit_count] += 1
        forward_pairs = visit_count * (visit_count - 1) // 2
        forward_pair_count_by_split[split] += forward_pairs
        directed_pair_count_by_split[split] += 2 * forward_pairs
        for visit in subject["visits"]:
            scan_count_by_split[split] += 1
            scan_diagnosis_count[diagnosis] += 1
            continuous_age_norm_values.append(float(visit["continuous_age_norm"]))

    return {
        "subject_count": int(sum(subject_count_by_split.values())),
        "scan_count": int(sum(scan_count_by_split.values())),
        "subject_count_by_split": dict(subject_count_by_split),
        "scan_count_by_split": dict(scan_count_by_split),
        "subject_diagnosis_count": dict(subject_diagnosis_count),
        "scan_diagnosis_count": dict(scan_diagnosis_count),
        "visit_count_distribution_subjects": dict(sorted(visit_count_distribution.items())),
        "forward_pair_count_by_split": dict(forward_pair_count_by_split),
        "directed_pair_count_by_split": dict(directed_pair_count_by_split),
        "continuous_age_norm_min": float(min(continuous_age_norm_values)),
        "continuous_age_norm_max": float(max(continuous_age_norm_values)),
    }


def strip_coefficients_for_json(
    subject_sequences: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    serialized = copy.deepcopy(subject_sequences)
    for subject in serialized.values():
        for visit in subject["visits"]:
            visit.pop("pca_coefficients_150", None)
            visit.pop("pca_coefficients_256", None)
    return serialized


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
    audit_path = TASK_DIR / "audit" / "input_audit.json"
    if not args.skip_audit_check:
        if not audit_path.is_file():
            print(f"Missing input audit: {audit_path}", file=sys.stderr)
            return 1
        audit_report = load_json(audit_path)
        if audit_report.get("status") != "pass":
            print("Input audit did not pass. Fix inputs before building.", file=sys.stderr)
            return 1

    clean_manifest_path = resolve_repo_path(config["task1"]["clean_manifest"])
    representation_manifest_path = resolve_repo_path(
        config["task2"]["representation_manifest"]
    )
    age_stats_path = resolve_repo_path(config["task1"]["age_norm_stats"])
    pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])

    clean_rows = load_clean_manifest(clean_manifest_path)
    representation_rows = load_representation_manifest(representation_manifest_path)
    age_stats = load_json(age_stats_path)

    coeff_mean_256 = np.asarray(
        np.load(pca_model_dir / "coefficient_mean_train.npy"), dtype=np.float32
    )
    coeff_std_256 = np.asarray(
        np.load(pca_model_dir / "coefficient_std_train.npy"), dtype=np.float32
    )
    primary_components = int(config["brainode"]["primary_components"])
    available_components = int(config["brainode"]["available_components"])

    representation_by_scan = {row["scan_id"]: row for row in representation_rows}
    clean_by_subject = group_rows_by_subject(clean_rows)
    age_min_train = float(age_stats["age_min_train"])
    age_max_train = float(age_stats["age_max_train"])

    scan_manifest_rows: list[dict[str, Any]] = []
    subject_sequences: dict[str, dict[str, Any]] = {}
    for subject_id, subject_rows in sorted(
        clean_by_subject.items(),
        key=lambda item: (SPLIT_ORDER[item[1][0]["split"]], item[0]),
    ):
        first_row = subject_rows[0]
        baseline_age_years = as_float(first_row["age_years"])
        diagnosis = first_row["diagnosis"]
        label_ad = as_int(first_row["label_ad"])
        cognition_value = float(config["brainode"]["condition_values"][diagnosis])
        subject_entry = {
            "split": first_row["split"],
            "diagnosis": diagnosis,
            "label_ad": label_ad,
            "cognition_value": cognition_value,
            "baseline_age_years": baseline_age_years,
            "visit_count": len(subject_rows),
            "visits": [],
        }
        for row in subject_rows:
            rep_row = representation_by_scan.get(row["scan_id"])
            if rep_row is None:
                raise KeyError(f"Missing representation row for {row['scan_id']}")
            coefficients_256 = load_numpy_vector(
                rep_row["pca_coeff_path"], expected_dim=available_components
            )
            continuous_year = row_continuous_age(row, baseline_age_years)
            continuous_norm = row_continuous_norm(
                row, continuous_year, age_min_train, age_max_train
            )
            visit_entry = {
                "scan_id": row["scan_id"],
                "image_id": row["image_id"],
                "filename": row["filename"],
                "visit_order": as_int(row["visit_order"]),
                "visit_label_raw": row["visit_label_raw"],
                "visit_month_from_label": as_int(row["visit_month_from_label"]),
                "months_from_baseline": as_float(row["months_from_baseline"]),
                "age_years": as_float(row["age_years"]),
                "age_norm_manifest": as_float(row["age_norm"]),
                "continuous_age_years": continuous_year,
                "continuous_age_norm": continuous_norm,
                "mesh_path": row["mesh_path"],
                "sdf_npz_path": row["sdf_npz_path"],
                "pca_coeff_path": rep_row["pca_coeff_path"],
                "pca_components_available": available_components,
                "pca_components_used": primary_components,
                "pca_coefficients_150": coefficients_256[:primary_components].tolist(),
                "pca_coefficients_256": coefficients_256.tolist(),
            }
            subject_entry["visits"].append(visit_entry)
            scan_manifest_rows.append(
                {
                    "scan_id": row["scan_id"],
                    "image_id": row["image_id"],
                    "subject_id": subject_id,
                    "split": row["split"],
                    "diagnosis": diagnosis,
                    "label_ad": label_ad,
                    "cognition_value": format_float(cognition_value, decimals=1),
                    "gender": row["gender"],
                    "visit_label_raw": row["visit_label_raw"],
                    "visit_month_from_label": as_int(row["visit_month_from_label"]),
                    "visit_order": as_int(row["visit_order"]),
                    "months_from_baseline": format_float(
                        as_float(row["months_from_baseline"]), decimals=1
                    ),
                    "baseline_age_years": format_float(baseline_age_years, decimals=6),
                    "age_years": format_float(as_float(row["age_years"]), decimals=6),
                    "age_norm_manifest": format_float(as_float(row["age_norm"])),
                    "continuous_age_years": format_float(continuous_year, decimals=6),
                    "continuous_age_norm": format_float(continuous_norm),
                    "mesh_path": row["mesh_path"],
                    "sdf_npz_path": row["sdf_npz_path"],
                    "pca_coeff_path": rep_row["pca_coeff_path"],
                    "pca_components_available": available_components,
                    "pca_components_used": primary_components,
                }
            )
        subject_sequences[subject_id] = subject_entry

    ordered_all_subject_ids = sorted(
        subject_sequences,
        key=lambda subject_id: (
            SPLIT_ORDER[subject_sequences[subject_id]["split"]],
            subject_id,
        ),
    )
    (TASK_DIR / "dataset").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    all_dataset = package_subjects(
        ordered_all_subject_ids,
        subject_sequences,
        coeff_mean_256=coeff_mean_256,
        coeff_std_256=coeff_std_256,
        primary_components=primary_components,
    )
    np.savez_compressed(TASK_DIR / "dataset" / "all_subject_sequences.npz", **all_dataset)

    for split in SPLITS:
        split_subject_ids = [
            subject_id
            for subject_id in ordered_all_subject_ids
            if subject_sequences[subject_id]["split"] == split
        ]
        split_dataset = package_subjects(
            split_subject_ids,
            subject_sequences,
            coeff_mean_256=coeff_mean_256,
            coeff_std_256=coeff_std_256,
            primary_components=primary_components,
        )
        np.savez_compressed(
            TASK_DIR / "dataset" / f"{split}_subject_sequences.npz", **split_dataset
        )

    scan_manifest_path = TASK_DIR / "metadata" / "core_brainode_scan_manifest.csv"
    write_csv(
        scan_manifest_path,
        fieldnames=list(scan_manifest_rows[0].keys()),
        rows=scan_manifest_rows,
    )
    write_json(
        TASK_DIR / "metadata" / "core_brainode_subject_sequences.json",
        strip_coefficients_for_json(subject_sequences),
    )

    summary = summarize_subject_sequences(subject_sequences)
    summary.update(
        {
            "status": "pass",
            "brainode_primary_components": primary_components,
            "brainode_available_components": available_components,
            "source_task1_clean_manifest": str(clean_manifest_path),
            "source_task1_clean_manifest_sha256": sha256_file(clean_manifest_path),
            "source_task2_representation_manifest": str(representation_manifest_path),
            "source_task2_representation_manifest_sha256": sha256_file(
                representation_manifest_path
            ),
            "age_min_train": age_min_train,
            "age_max_train": age_max_train,
            "files": {
                "scan_manifest": str(scan_manifest_path),
                "subject_sequences_json": str(
                    TASK_DIR / "metadata" / "core_brainode_subject_sequences.json"
                ),
                "all_subject_sequences_npz": str(
                    TASK_DIR / "dataset" / "all_subject_sequences.npz"
                ),
                "train_subject_sequences_npz": str(
                    TASK_DIR / "dataset" / "train_subject_sequences.npz"
                ),
                "val_subject_sequences_npz": str(
                    TASK_DIR / "dataset" / "val_subject_sequences.npz"
                ),
                "test_subject_sequences_npz": str(
                    TASK_DIR / "dataset" / "test_subject_sequences.npz"
                ),
            },
        }
    )
    write_json(TASK_DIR / "metadata" / "dataset_summary.json", summary)

    print(
        json.dumps(
            {
                "status": "pass",
                "subjects": summary["subject_count"],
                "scans": summary["scan_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
