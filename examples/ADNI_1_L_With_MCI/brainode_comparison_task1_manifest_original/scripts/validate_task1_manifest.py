#!/usr/bin/env python3
"""Validate Task 1 ADNI With-MCI original manifest artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SPLITS = ("train", "val", "test")
FILE_PREFIX = "adni_with_mci_left_original"
ALLOWED_DIAGNOSES = {"CN", "MCI", "AD"}


def task_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated Task 1 manifest files.")
    parser.add_argument("--output-root", type=Path, default=task_root_from_script())
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def yes(value: str) -> bool:
    return str(value).strip().lower() == "yes"


def as_float(value: str) -> float:
    return float(str(value).strip())


def as_int(value: str) -> int:
    return int(float(str(value).strip()))


def add_error(errors: List[Dict[str, Any]], code: str, message: str, **details: Any) -> None:
    errors.append({"code": code, "message": message, "details": details})


def add_warning(warnings: List[Dict[str, Any]], code: str, message: str, **details: Any) -> None:
    warnings.append({"code": code, "message": message, "details": details})


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    return dict(Counter(row[key] for row in rows))


def validate_required_files(root: Path) -> Tuple[List[Path], List[Path]]:
    required = [
        root / f"metadata/{FILE_PREFIX}_master.csv",
        root / f"metadata/{FILE_PREFIX}_clean.csv",
        root / f"metadata/{FILE_PREFIX}_subject_trajectories.json",
        root / "metadata/age_norm_stats.json",
        root / "metadata/split_summary.json",
        root / "metadata/cleaning_report.json",
        root / "splits/train_clean.json",
        root / "splits/val_clean.json",
        root / "splits/test_clean.json",
        root / "pairs/pairs_train.csv",
        root / "pairs/pairs_val.csv",
        root / "pairs/pairs_test.csv",
        root / "pairs/pairs_all.csv",
        root / "triplets/triplets_train.csv",
        root / "triplets/triplets_val.csv",
        root / "triplets/triplets_test.csv",
        root / "triplets/triplets_all.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    return required, missing


def main() -> None:
    args = parse_args()
    root = args.output_root
    metadata_dir = root / "metadata"
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    required, missing = validate_required_files(root)
    if missing:
        for path in missing:
            add_error(errors, "missing_required_file", "Required generated file is missing.", path=str(path))
        report = {
            "status": "fail",
            "output_root": str(root),
            "required_file_count": len(required),
            "missing_file_count": len(missing),
            "errors": errors,
            "warnings": warnings,
        }
        metadata_dir.mkdir(parents=True, exist_ok=True)
        write_json(metadata_dir / "validation_report.json", report)
        print("Validation failed: required files missing.")
        raise SystemExit(1)

    master = read_csv(metadata_dir / f"{FILE_PREFIX}_master.csv")
    clean = read_csv(metadata_dir / f"{FILE_PREFIX}_clean.csv")
    pairs_all = read_csv(root / "pairs/pairs_all.csv")
    triplets_all = read_csv(root / "triplets/triplets_all.csv")
    trajectories = read_json(metadata_dir / f"{FILE_PREFIX}_subject_trajectories.json")
    age_stats = read_json(metadata_dir / "age_norm_stats.json")
    split_summary = read_json(metadata_dir / "split_summary.json")
    cleaning_report = read_json(metadata_dir / "cleaning_report.json")

    master_by_scan = {row["scan_id"]: row for row in master}
    clean_by_scan = {row["scan_id"]: row for row in clean}
    if len(master_by_scan) != len(master):
        add_error(errors, "duplicate_master_scan_id", "Master CSV has duplicate scan IDs.")
    if len(clean_by_scan) != len(clean):
        add_error(errors, "duplicate_clean_scan_id", "Clean CSV has duplicate scan IDs.")

    for row in master:
        if row["diagnosis"] not in ALLOWED_DIAGNOSES:
            add_error(
                errors,
                "unexpected_master_diagnosis",
                "Master row contains an unexpected diagnosis.",
                scan_id=row["scan_id"],
                diagnosis=row["diagnosis"],
            )
        if row["label_dx"] not in {"0", "1", "2"}:
            add_error(
                errors,
                "invalid_label_dx_master",
                "Master row label_dx is not 0, 1, or 2.",
                scan_id=row["scan_id"],
                label_dx=row["label_dx"],
            )
        if row["diagnosis"] == "MCI" and row["label_ad"] not in {"", "None"}:
            add_error(
                errors,
                "invalid_mci_label_ad_master",
                "MCI row should not carry a binary AD label.",
                scan_id=row["scan_id"],
                label_ad=row["label_ad"],
            )
        if row["diagnosis"] == "CN" and row["label_ad"] != "0":
            add_error(errors, "invalid_cn_label_ad_master", "CN row must have label_ad 0.", scan_id=row["scan_id"])
        if row["diagnosis"] == "AD" and row["label_ad"] != "1":
            add_error(errors, "invalid_ad_label_ad_master", "AD row must have label_ad 1.", scan_id=row["scan_id"])
        if not yes(row["mesh_exists"]):
            add_error(errors, "missing_mesh", "Master row mesh does not exist.", scan_id=row["scan_id"])
        if not yes(row["sdf_exists"]):
            add_error(errors, "missing_sdf", "Master row SDF sample does not exist.", scan_id=row["scan_id"])

    for row in clean:
        master_row = master_by_scan.get(row["scan_id"])
        if master_row is None:
            add_error(errors, "clean_not_in_master", "Clean scan is absent from master.", scan_id=row["scan_id"])
            continue
        if not yes(master_row["keep_scan"]):
            add_error(errors, "clean_not_marked_keep", "Clean scan is not marked keep_scan in master.", scan_id=row["scan_id"])
        if row["diagnosis"] not in ALLOWED_DIAGNOSES:
            add_error(errors, "unexpected_clean_diagnosis", "Clean row contains unexpected diagnosis.", scan_id=row["scan_id"])
        if row["label_dx"] not in {"0", "1", "2"}:
            add_error(errors, "invalid_label_dx", "Clean row label_dx is not 0, 1, or 2.", scan_id=row["scan_id"])
        if row["diagnosis"] == "MCI" and row["label_ad"] not in {"", "None"}:
            add_error(errors, "invalid_mci_label_ad", "MCI clean row should have empty label_ad.", scan_id=row["scan_id"])
        if row["diagnosis"] == "CN" and row["label_ad"] != "0":
            add_error(errors, "invalid_cn_label_ad", "CN clean row must have label_ad 0.", scan_id=row["scan_id"])
        if row["diagnosis"] == "AD" and row["label_ad"] != "1":
            add_error(errors, "invalid_ad_label_ad", "AD clean row must have label_ad 1.", scan_id=row["scan_id"])

    subject_splits: Dict[str, set] = defaultdict(set)
    subject_visit_months: Dict[str, List[int]] = defaultdict(list)
    subject_visit_orders: Dict[str, List[int]] = defaultdict(list)
    for row in clean:
        subject_splits[row["subject_id"]].add(row["split"])
        subject_visit_months[row["subject_id"]].append(as_int(row["visit_month_from_label"]))
        subject_visit_orders[row["subject_id"]].append(as_int(row["visit_order"]))
    leaked = {sid: sorted(values) for sid, values in subject_splits.items() if len(values) > 1}
    if leaked:
        add_error(errors, "subject_split_leakage", "Clean subjects appear in multiple splits.", leaked=leaked)
    for sid, months in subject_visit_months.items():
        if len(months) < 2:
            add_error(errors, "subject_too_few_clean_visits", "Clean subject has fewer than two visits.", subject_id=sid)
        if len(months) != len(set(months)):
            add_error(errors, "duplicate_clean_visit_month", "Clean subject has duplicate visit months.", subject_id=sid, months=months)
        orders = sorted(subject_visit_orders[sid])
        if orders != list(range(len(orders))):
            add_error(errors, "non_contiguous_visit_order", "Clean subject visit_order is not contiguous from zero.", subject_id=sid, orders=orders)

    for split in SPLITS:
        split_json = read_json(root / "splits" / f"{split}_clean.json")
        csv_names = sorted(row["filename"] for row in clean if row["split"] == split)
        if sorted(split_json) != csv_names:
            add_error(errors, "split_json_mismatch", "Clean split JSON does not match clean CSV.", split=split)

    train_norms = [as_float(row["age_norm"]) for row in clean if row["split"] == "train"]
    if train_norms and (min(train_norms) < -1e-8 or max(train_norms) > 1.0 + 1e-8):
        add_error(errors, "train_age_norm_out_of_range", "Train age_norm should be in [0, 1].")
    all_norms = [as_float(row["age_norm"]) for row in clean]
    if all_norms and (min(all_norms) < -1e-8 or max(all_norms) > 1.0 + 1e-8):
        add_warning(
            warnings,
            "nontrain_age_norm_out_of_range",
            "At least one val/test age_norm lies outside the train min/max age range.",
            min_age_norm=min(all_norms),
            max_age_norm=max(all_norms),
        )
    if age_stats["age_min_train"] > age_stats["age_max_train"]:
        add_error(errors, "invalid_age_stats", "age_min_train exceeds age_max_train.")

    for row in pairs_all:
        source = clean_by_scan.get(row["source_scan_id"])
        target = clean_by_scan.get(row["target_scan_id"])
        if source is None or target is None:
            add_error(errors, "pair_scan_missing", "Pair references a scan missing from clean CSV.", pair=row)
            continue
        if source["subject_id"] != target["subject_id"] or source["subject_id"] != row["subject_id"]:
            add_error(errors, "pair_subject_mismatch", "Pair source/target subject mismatch.", pair=row)
        if source["split"] != target["split"] or source["split"] != row["split"]:
            add_error(errors, "pair_split_mismatch", "Pair source/target split mismatch.", pair=row)
        if source["diagnosis"] != row["diagnosis"] or target["diagnosis"] != row["diagnosis"]:
            add_error(errors, "pair_diagnosis_mismatch", "Pair diagnosis mismatch.", pair=row)
        if as_int(row["source_visit_order"]) >= as_int(row["target_visit_order"]):
            add_error(errors, "pair_order_not_forward", "Pair visit order is not forward.", pair=row)
        if as_float(row["delta_months"]) < 3.0:
            add_error(errors, "pair_delta_too_small", "Pair delta_months is below 3.", pair=row)

    for split in SPLITS:
        split_rows = read_csv(root / "pairs" / f"pairs_{split}.csv")
        expected = [row for row in pairs_all if row["split"] == split]
        if len(split_rows) != len(expected):
            add_error(errors, "pair_split_count_mismatch", "Split pair CSV count mismatch.", split=split)

    for row in triplets_all:
        scans = [clean_by_scan.get(row[key]) for key in ("scan_s", "scan_r", "scan_t")]
        if any(scan is None for scan in scans):
            add_error(errors, "triplet_scan_missing", "Triplet references a scan missing from clean CSV.", triplet=row)
            continue
        subjects = {scan["subject_id"] for scan in scans if scan is not None}
        splits = {scan["split"] for scan in scans if scan is not None}
        diagnoses = {scan["diagnosis"] for scan in scans if scan is not None}
        if subjects != {row["subject_id"]}:
            add_error(errors, "triplet_subject_mismatch", "Triplet source/reference/target subject mismatch.", triplet=row)
        if splits != {row["split"]}:
            add_error(errors, "triplet_split_mismatch", "Triplet source/reference/target split mismatch.", triplet=row)
        if diagnoses != {row["diagnosis"]}:
            add_error(errors, "triplet_diagnosis_mismatch", "Triplet diagnosis mismatch.", triplet=row)
        orders = [as_int(row[key]) for key in ("visit_order_s", "visit_order_r", "visit_order_t")]
        if not (orders[0] < orders[1] < orders[2]):
            add_error(errors, "triplet_order_not_increasing", "Triplet visit order is not strictly increasing.", triplet=row)

    for split in SPLITS:
        split_rows = read_csv(root / "triplets" / f"triplets_{split}.csv")
        expected = [row for row in triplets_all if row["split"] == split]
        if len(split_rows) != len(expected):
            add_error(errors, "triplet_split_count_mismatch", "Split triplet CSV count mismatch.", split=split)

    if set(trajectories) != set(subject_visit_months):
        add_error(
            errors,
            "trajectory_subject_mismatch",
            "Trajectory JSON subject set does not match clean CSV.",
            trajectory_subject_count=len(trajectories),
            clean_subject_count=len(subject_visit_months),
        )

    computed_summary = {
        "master_scan_count": len(master),
        "clean_scan_count": len(clean),
        "clean_subject_count": len(subject_visit_months),
        "pair_count": len(pairs_all),
        "triplet_count": len(triplets_all),
        "master_scan_count_by_split": count_by(master, "split"),
        "clean_scan_count_by_split": count_by(clean, "split"),
        "pair_count_by_split": count_by(pairs_all, "split"),
        "triplet_count_by_split": count_by(triplets_all, "split"),
        "clean_diagnosis_scan_count": count_by(clean, "diagnosis"),
        "pair_task_count": count_by(pairs_all, "task"),
        "clean_visit_count_distribution_subjects": {
            str(key): int(value)
            for key, value in sorted(
                Counter(len(months) for months in subject_visit_months.values()).items()
            )
        },
    }

    distribution = computed_summary["clean_visit_count_distribution_subjects"]
    distribution_subject_total = sum(int(value) for value in distribution.values())
    distribution_scan_total = sum(int(key) * int(value) for key, value in distribution.items())
    if distribution_subject_total != computed_summary["clean_subject_count"]:
        add_error(
            errors,
            "visit_distribution_subject_total_mismatch",
            "Clean visit-count distribution does not sum to clean subject count.",
            distribution=distribution,
        )
    if distribution_scan_total != computed_summary["clean_scan_count"]:
        add_error(
            errors,
            "visit_distribution_scan_total_mismatch",
            "Clean visit-count distribution weighted total does not sum to clean scan count.",
            distribution=distribution,
        )

    for key in ("master_scan_count", "clean_scan_count", "clean_subject_count", "pair_count", "triplet_count"):
        if split_summary.get(key) != computed_summary[key]:
            add_error(
                errors,
                "summary_count_mismatch",
                "split_summary.json count does not match regenerated count.",
                key=key,
                summary=split_summary.get(key),
                computed=computed_summary[key],
            )
    if split_summary.get("clean_visit_count_distribution_subjects") != distribution:
        add_error(
            errors,
            "summary_visit_distribution_mismatch",
            "split_summary.json clean visit-count distribution does not match regenerated distribution.",
            summary=split_summary.get("clean_visit_count_distribution_subjects"),
            computed=distribution,
        )

    report = {
        "status": "pass" if not errors else "fail",
        "output_root": str(root),
        "required_file_count": len(required),
        "missing_file_count": len(missing),
        "computed_summary": computed_summary,
        "age_stats": age_stats,
        "dropped_scan_count": len(cleaning_report.get("dropped_scans", [])),
        "duplicate_visit_group_count": len(cleaning_report.get("duplicate_visit_groups", [])),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(metadata_dir / "validation_report.json", report)

    print(f"Validation status: {report['status']}")
    print(f"Master scans: {computed_summary['master_scan_count']}")
    print(f"Clean scans: {computed_summary['clean_scan_count']}")
    print(f"Clean subjects: {computed_summary['clean_subject_count']}")
    print(f"Pairs: {computed_summary['pair_count']}")
    print(f"Triplets: {computed_summary['triplet_count']}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
