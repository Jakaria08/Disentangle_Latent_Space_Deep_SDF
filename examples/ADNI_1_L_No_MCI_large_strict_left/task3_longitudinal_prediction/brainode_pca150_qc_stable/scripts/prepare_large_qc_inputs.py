#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core_brainode_common import (
    SPLIT_ORDER,
    TASK_DIR,
    load_config,
    load_json,
    read_csv,
    resolve_repo_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare QC-clean large strict-left manifests for BrainODE."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode_large_qc.json"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_key_subject(subject_id: str) -> tuple[int, str]:
    return (int(subject_id), subject_id) if str(subject_id).isdigit() else (10**9, str(subject_id))


def sex_to_gender(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized.startswith("m"):
        return "M"
    if normalized.startswith("f"):
        return "F"
    return str(value).strip()


def format_float(value: Any, decimals: int = 10) -> str:
    return f"{float(value):.{decimals}f}"


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    large_qc = config["large_qc"]
    source_metadata_path = resolve_repo_path(large_qc["source_metadata"])
    source_representation_path = resolve_repo_path(
        large_qc["source_representation_manifest"]
    )
    source_pca_csv_path = resolve_repo_path(large_qc["source_pca_coefficients_csv"])
    source_age_stats_path = resolve_repo_path(large_qc["source_age_norm_stats"])

    metadata_rows = read_csv(source_metadata_path)
    representation_rows = read_csv(source_representation_path)
    pca_rows = read_csv(source_pca_csv_path)
    source_age_stats = load_json(source_age_stats_path)

    require_columns(
        metadata_rows,
        {
            "scan_id",
            "filename",
            "subject_id",
            "image_id",
            "split",
            "diagnosis",
            "label_ad",
            "visit_order",
            "visit_label",
            "visit_month",
            "months_from_baseline",
            "age_years",
            "age_norm",
            "continuous_age_years",
            "continuous_age_norm",
            "sex",
            "mesh_path",
            "sdf_npz_path",
        },
        "QC metadata",
    )
    require_columns(
        representation_rows,
        {
            "scan_id",
            "image_id",
            "subject_id",
            "split",
            "diagnosis",
            "label_ad",
            "visit_order",
            "age_norm",
            "ground_truth_mesh_path",
            "sdf_npz_path",
            "pca_coeff_path",
            "pca_150_mesh_path",
            "pca_256_mesh_path",
        },
        "representation manifest",
    )
    require_columns(
        pca_rows,
        {
            "scan_id",
            "image_id",
            "subject_id",
            "split",
            "diagnosis",
            "label_ad",
            "visit_order",
            "age_norm",
            "coefficient_dimension",
            "coefficient_path",
        },
        "PCA coefficient CSV",
    )

    representation_by_scan = {row["scan_id"]: row for row in representation_rows}
    pca_by_scan = {row["scan_id"]: row for row in pca_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        grouped[row["subject_id"]].append(row)

    errors: list[str] = []
    warnings: list[str] = []
    clean_rows: list[dict[str, Any]] = []
    representation_output_rows: list[dict[str, Any]] = []
    pca_output_rows: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, Any]] = {}
    skipped_subjects: list[dict[str, Any]] = []

    for subject_id, visits in sorted(
        grouped.items(),
        key=lambda item: (SPLIT_ORDER.get(item[1][0]["split"], 99), sort_key_subject(item[0])),
    ):
        visits = sorted(
            visits,
            key=lambda row: (
                float(row["continuous_age_norm"]),
                int(row["visit_order"]),
                row["scan_id"],
            ),
        )
        splits = {row["split"] for row in visits}
        diagnoses = {row["diagnosis"] for row in visits}
        labels = {row["label_ad"] for row in visits}
        if len(splits) != 1 or len(diagnoses) != 1 or len(labels) != 1:
            skipped_subjects.append(
                {
                    "subject_id": subject_id,
                    "splits": sorted(splits),
                    "diagnoses": sorted(diagnoses),
                    "labels": sorted(labels),
                    "scan_ids": [row["scan_id"] for row in visits],
                }
            )
            continue
        if len(visits) < 2:
            skipped_subjects.append(
                {
                    "subject_id": subject_id,
                    "reason": "fewer_than_two_visits",
                    "scan_ids": [row["scan_id"] for row in visits],
                }
            )
            continue

        split = visits[0]["split"]
        diagnosis = visits[0]["diagnosis"]
        label_ad = int(float(visits[0]["label_ad"]))
        if diagnosis not in {"CN", "AD"} or label_ad not in {0, 1}:
            errors.append(f"Unexpected diagnosis/label for subject {subject_id}")
            continue
        previous_time = None
        trajectory_visits = []
        for reindexed_order, row in enumerate(visits):
            scan_id = row["scan_id"]
            rep_row = representation_by_scan.get(scan_id)
            pca_row = pca_by_scan.get(scan_id)
            if rep_row is None:
                errors.append(f"Missing representation row for {scan_id}")
                continue
            if pca_row is None:
                errors.append(f"Missing PCA coefficient row for {scan_id}")
                continue
            time_value = float(row["continuous_age_norm"])
            if previous_time is not None and time_value <= previous_time:
                errors.append(f"Non-increasing time for subject {subject_id}")
            previous_time = time_value

            clean_row = {
                "scan_id": scan_id,
                "filename": row["filename"],
                "subject_id": subject_id,
                "image_id": row["image_id"],
                "split": split,
                "diagnosis": diagnosis,
                "label_ad": str(label_ad),
                "gender": sex_to_gender(row["sex"]),
                "visit_label_raw": row["visit_label"],
                "visit_month_from_label": str(int(float(row["visit_month"]))),
                "visit_order": str(reindexed_order),
                "age_years": format_float(row["age_years"], decimals=8),
                "age_norm": format_float(row["age_norm"]),
                "continuous_age_years": format_float(row["continuous_age_years"], decimals=8),
                "continuous_age_norm": format_float(row["continuous_age_norm"]),
                "months_from_baseline": format_float(row["months_from_baseline"], decimals=6),
                "mesh_path": row["mesh_path"],
                "sdf_npz_path": row["sdf_npz_path"],
                "has_duplicate_visit": "False",
                "clean_subject_visit_count": str(len(visits)),
            }
            clean_rows.append(clean_row)

            rep_out = dict(rep_row)
            rep_out.update(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "diagnosis": diagnosis,
                    "label_ad": str(label_ad),
                    "visit_order": str(reindexed_order),
                    "age_norm": clean_row["age_norm"],
                    "ground_truth_mesh_path": row["mesh_path"],
                    "sdf_npz_path": row["sdf_npz_path"],
                }
            )
            representation_output_rows.append(rep_out)

            pca_out = dict(pca_row)
            pca_out.update(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "diagnosis": diagnosis,
                    "label_ad": str(label_ad),
                    "visit_order": str(reindexed_order),
                    "age_norm": clean_row["age_norm"],
                }
            )
            pca_output_rows.append(pca_out)

            trajectory_visits.append(
                {
                    "scan_id": scan_id,
                    "visit_order": reindexed_order,
                    "visit_label_raw": row["visit_label"],
                    "months_from_baseline": float(row["months_from_baseline"]),
                    "age_years": float(row["age_years"]),
                    "continuous_age_years": float(row["continuous_age_years"]),
                    "continuous_age_norm": float(row["continuous_age_norm"]),
                }
            )

        trajectories[subject_id] = {
            "split": split,
            "diagnosis": diagnosis,
            "label_ad": label_ad,
            "visit_count": len(trajectory_visits),
            "visits": trajectory_visits,
        }

    clean_rows = sorted(
        clean_rows,
        key=lambda row: (
            SPLIT_ORDER.get(row["split"], 99),
            sort_key_subject(row["subject_id"]),
            int(row["visit_order"]),
            row["scan_id"],
        ),
    )

    expected_counts = config["expected_counts"]
    expected_subjects = config["expected_subject_counts"]
    scan_counts = Counter(row["split"] for row in clean_rows)
    subject_counts = Counter(value["split"] for value in trajectories.values())
    for split in ("train", "val", "test"):
        if scan_counts.get(split, 0) != int(expected_counts[split]):
            errors.append(
                f"{split} scan count mismatch: expected {expected_counts[split]}, "
                f"found {scan_counts.get(split, 0)}"
            )
        if subject_counts.get(split, 0) != int(expected_subjects[split]):
            errors.append(
                f"{split} subject count mismatch: expected {expected_subjects[split]}, "
                f"found {subject_counts.get(split, 0)}"
            )
    if len(clean_rows) != int(expected_counts["total"]):
        errors.append(
            f"Total scan count mismatch: expected {expected_counts['total']}, found {len(clean_rows)}"
        )
    if len(trajectories) != int(expected_subjects["total"]):
        errors.append(
            f"Total subject count mismatch: expected {expected_subjects['total']}, found {len(trajectories)}"
        )

    for row in representation_output_rows:
        if not Path(row["pca_coeff_path"]).is_file():
            errors.append(f"Missing PCA coefficient file: {row['pca_coeff_path']}")
            break
    for row in pca_output_rows:
        if not Path(row["coefficient_path"]).is_file():
            errors.append(f"Missing PCA coefficient file: {row['coefficient_path']}")
            break

    task1 = config["task1"]
    task2 = config["task2"]
    clean_manifest_path = resolve_repo_path(task1["clean_manifest"])
    trajectory_path = resolve_repo_path(task1["trajectory_json"])
    split_summary_path = resolve_repo_path(task1["split_summary"])
    age_stats_path = resolve_repo_path(task1["age_norm_stats"])
    validation_path = resolve_repo_path(task1["validation_report"])
    filtered_representation_path = resolve_repo_path(task2["representation_manifest"])
    filtered_pca_path = resolve_repo_path(task2["pca_coefficients_csv"])

    if clean_rows:
        write_csv(clean_manifest_path, clean_rows)
        write_csv(filtered_representation_path, representation_output_rows)
        write_csv(filtered_pca_path, pca_output_rows)
    write_json(trajectory_path, trajectories)

    age_min = float(source_age_stats.get("age_min_train", source_age_stats.get("age_min")))
    age_max = float(source_age_stats.get("age_max_train", source_age_stats.get("age_max")))
    age_stats = {
        "age_min_train": age_min,
        "age_max_train": age_max,
        "age_range_train": age_max - age_min,
        "formula": source_age_stats.get(
            "formula",
            "age_norm = (age_numeric - age_min_train) / (age_max_train - age_min_train)",
        ),
        "source": str(source_age_stats_path),
    }
    write_json(age_stats_path, age_stats)

    visit_distribution = Counter(
        int(value["visit_count"]) for value in trajectories.values()
    )
    split_summary = {
        "status": "pass" if not errors else "fail",
        "source_metadata": str(source_metadata_path),
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "clean_scan_count": len(clean_rows),
        "clean_subject_count": len(trajectories),
        "scan_count_by_split": dict(scan_counts),
        "subject_count_by_split": dict(subject_counts),
        "scan_diagnosis_count": dict(Counter(row["diagnosis"] for row in clean_rows)),
        "subject_diagnosis_count": dict(
            Counter(value["diagnosis"] for value in trajectories.values())
        ),
        "visit_count_distribution_subjects": dict(sorted(visit_distribution.items())),
        "skipped_subject_count": len(skipped_subjects),
        "skipped_subjects": skipped_subjects,
        "outputs": {
            "clean_manifest": str(clean_manifest_path),
            "trajectory_json": str(trajectory_path),
            "filtered_representation_manifest": str(filtered_representation_path),
            "filtered_pca_coefficients_csv": str(filtered_pca_path),
            "age_norm_stats": str(age_stats_path),
        },
    }
    write_json(split_summary_path, split_summary)

    validation = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "source_metadata_rows": len(metadata_rows),
            "clean_manifest_rows": len(clean_rows),
            "subject_count": len(trajectories),
            "scan_count_by_split": dict(scan_counts),
            "subject_count_by_split": dict(subject_counts),
            "filtered_representation_rows": len(representation_output_rows),
            "filtered_pca_rows": len(pca_output_rows),
            "age_min_train": age_min,
            "age_max_train": age_max,
        },
    }
    write_json(validation_path, validation)
    print(json.dumps(validation, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
