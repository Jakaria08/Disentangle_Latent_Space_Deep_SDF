#!/usr/bin/env python3
"""Build cleaned ADNI No-MCI left original longitudinal manifests.

The script is intentionally read-only with respect to the source ADNI data and
the repository's existing split files. All generated artifacts are written
under the containing task folder.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ("train", "val", "test")


MASTER_FIELDS = [
    "scan_id",
    "filename",
    "subject_id_filename",
    "subject_id",
    "image_id",
    "split",
    "diagnosis",
    "label_ad",
    "gender",
    "visit_label_raw",
    "visit_month_from_label",
    "age_years",
    "mesh_path",
    "sdf_npz_path",
    "mesh_exists",
    "sdf_exists",
    "has_duplicate_visit",
    "duplicate_group_key",
    "duplicate_kept_scan_id",
    "keep_scan",
    "drop_reason",
    "visit_order",
    "months_from_baseline",
    "age_norm",
    "clean_subject_visit_count",
]


CLEAN_FIELDS = [
    "scan_id",
    "filename",
    "subject_id",
    "image_id",
    "split",
    "diagnosis",
    "label_ad",
    "gender",
    "visit_label_raw",
    "visit_month_from_label",
    "visit_order",
    "age_years",
    "age_norm",
    "months_from_baseline",
    "mesh_path",
    "sdf_npz_path",
    "has_duplicate_visit",
    "clean_subject_visit_count",
]


PAIR_FIELDS = [
    "subject_id",
    "split",
    "source_scan_id",
    "target_scan_id",
    "source_filename",
    "target_filename",
    "source_image_id",
    "target_image_id",
    "source_visit_label",
    "target_visit_label",
    "source_visit_order",
    "target_visit_order",
    "source_visit_month",
    "target_visit_month",
    "source_months_from_baseline",
    "target_months_from_baseline",
    "source_age_years",
    "target_age_years",
    "source_age_norm",
    "target_age_norm",
    "delta_months",
    "diagnosis",
    "label_ad",
    "task",
]


TRIPLET_FIELDS = [
    "subject_id",
    "split",
    "scan_s",
    "scan_r",
    "scan_t",
    "filename_s",
    "filename_r",
    "filename_t",
    "visit_order_s",
    "visit_order_r",
    "visit_order_t",
    "visit_month_s",
    "visit_month_r",
    "visit_month_t",
    "months_s",
    "months_r",
    "months_t",
    "age_s",
    "age_r",
    "age_t",
    "age_norm_s",
    "age_norm_r",
    "age_norm_t",
    "diagnosis",
    "label_ad",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def task_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    adni_root = Path("/home/jakaria/ADNI/ADNI_1/adni_processed")
    parser = argparse.ArgumentParser(
        description="Build ADNI No-MCI left original manifest, pairs, and triplets."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=repo_root / "examples/splits/splits_left_hippocampus_ADNI_No_MCI",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=adni_root / "adni_metadata_filtered.csv",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=adni_root
        / "left_hippocampus_correspondence/minimal_scaled_obj_files",
    )
    parser.add_argument(
        "--sdf-dir",
        type=Path,
        default=adni_root
        / "left_hippocampus_correspondence/sdf_data/SdfSamples/minimal_scaled_obj_files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=task_root_from_script(),
    )
    parser.add_argument(
        "--min-delta-months",
        type=float,
        default=3.0,
        help="Minimum time gap for generated forecasting pairs.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> None:
    for subdir in ("metadata", "splits", "pairs", "triplets"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})


def format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def read_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "diagnosis", "gender", "age", "visit", "image_data_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Metadata CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            image_id = row["image_data_id"].strip()
            if image_id:
                rows[image_id] = row
    return rows


def split_filename(split_dir: Path, split: str) -> Path:
    return split_dir / f"{split}_split_left_hippocampus_adni_no_mci.json"


def load_splits(split_dir: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for split in SPLITS:
        path = split_filename(split_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        values = read_json(path)
        if not isinstance(values, list):
            raise RuntimeError(f"Split file must contain a list: {path}")
        result[split] = [str(item) for item in values]
    return result


def parse_image_id(filename: str) -> str:
    match = re.search(r"_I(\d+)(?:_|\.|$)", filename)
    if match is None:
        raise RuntimeError(f"Could not parse image id from filename: {filename}")
    return "I" + match.group(1)


def parse_subject_id_from_filename(filename: str) -> str:
    parts = Path(filename).name.split("_")
    if len(parts) < 4 or parts[0] != "ADNI":
        raise RuntimeError(f"Could not parse ADNI subject id from filename: {filename}")
    return "_".join(parts[1:4])


def image_id_number(image_id: str) -> int:
    match = re.search(r"(\d+)", image_id)
    return int(match.group(1)) if match else 10**12


def visit_month(visit_label: str) -> Optional[int]:
    label = str(visit_label).strip().lower()
    if label in {"sc", "bl", "screening", "baseline"}:
        return 0
    match = re.fullmatch(r"m(\d+)", label)
    if match:
        return int(match.group(1))
    return None


def label_ad_from_diagnosis(diagnosis: str) -> Optional[int]:
    diag = diagnosis.strip().upper()
    if diag == "CN":
        return 0
    if diag == "AD":
        return 1
    return None


def make_master_rows(
    splits: Dict[str, List[str]],
    metadata: Dict[str, Dict[str, str]],
    mesh_dir: Path,
    sdf_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    seen_scan_ids: Dict[str, str] = {}
    for split, filenames in splits.items():
        for filename in filenames:
            scan_id = Path(filename).stem
            image_id = parse_image_id(filename)
            subject_id_filename = parse_subject_id_from_filename(filename)
            meta = metadata.get(image_id)
            if meta is None:
                warnings.append({"type": "missing_metadata", "filename": filename, "image_id": image_id})
                meta = {}
            diagnosis = str(meta.get("diagnosis", "")).strip()
            label_ad = label_ad_from_diagnosis(diagnosis)
            if label_ad is None:
                warnings.append(
                    {
                        "type": "unexpected_diagnosis",
                        "filename": filename,
                        "image_id": image_id,
                        "diagnosis": diagnosis,
                    }
                )
            visit_label = str(meta.get("visit", "")).strip()
            month = visit_month(visit_label)
            if month is None:
                warnings.append(
                    {
                        "type": "unknown_visit_label",
                        "filename": filename,
                        "image_id": image_id,
                        "visit": visit_label,
                    }
                )
            subject_id = str(meta.get("subject_id", "")).strip() or subject_id_filename
            if subject_id != subject_id_filename:
                warnings.append(
                    {
                        "type": "subject_id_mismatch",
                        "filename": filename,
                        "metadata_subject_id": subject_id,
                        "filename_subject_id": subject_id_filename,
                    }
                )
            if scan_id in seen_scan_ids:
                warnings.append(
                    {
                        "type": "duplicate_scan_id_in_splits",
                        "scan_id": scan_id,
                        "first_split": seen_scan_ids[scan_id],
                        "second_split": split,
                    }
                )
            seen_scan_ids[scan_id] = split

            mesh_path = mesh_dir / f"{scan_id}.obj"
            sdf_path = sdf_dir / f"{scan_id}.npz"
            row = {
                "scan_id": scan_id,
                "filename": filename,
                "subject_id_filename": subject_id_filename,
                "subject_id": subject_id,
                "image_id": image_id,
                "split": split,
                "diagnosis": diagnosis,
                "label_ad": label_ad,
                "gender": str(meta.get("gender", "")).strip(),
                "visit_label_raw": visit_label,
                "visit_month_from_label": month,
                "age_years": float(meta["age"]) if str(meta.get("age", "")).strip() else None,
                "mesh_path": str(mesh_path),
                "sdf_npz_path": str(sdf_path),
                "mesh_exists": mesh_path.is_file(),
                "sdf_exists": sdf_path.is_file(),
                "has_duplicate_visit": False,
                "duplicate_group_key": "",
                "duplicate_kept_scan_id": "",
                "keep_scan": False,
                "drop_reason": "",
                "visit_order": "",
                "months_from_baseline": "",
                "age_norm": "",
                "clean_subject_visit_count": "",
            }
            if not row["mesh_exists"]:
                warnings.append({"type": "missing_mesh", "scan_id": scan_id, "path": str(mesh_path)})
            if not row["sdf_exists"]:
                warnings.append({"type": "missing_sdf", "scan_id": scan_id, "path": str(sdf_path)})
            rows.append(row)
    return rows, warnings


def candidate_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
    has_files_rank = 0 if row["mesh_exists"] and row["sdf_exists"] else 1
    visit = str(row["visit_label_raw"]).lower()
    baseline_rank = 0 if visit == "bl" else 1 if visit == "sc" else 2
    return (has_files_rank, baseline_rank, image_id_number(row["image_id"]), row["scan_id"])


def clean_rows(master_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in master_rows:
        rows_by_subject[row["subject_id"]].append(row)

    clean: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "duplicate_visit_groups": [],
        "dropped_scans": [],
        "subjects_excluded_fewer_than_2_visits": [],
    }

    for subject_id, subject_rows in sorted(rows_by_subject.items()):
        split_values = sorted({row["split"] for row in subject_rows})
        if len(split_values) != 1:
            for row in subject_rows:
                row["drop_reason"] = "subject_appears_in_multiple_splits"
                report["dropped_scans"].append(drop_record(row))
            continue

        valid_rows = [
            row
            for row in subject_rows
            if row["mesh_exists"]
            and row["sdf_exists"]
            and row["visit_month_from_label"] is not None
            and row["age_years"] is not None
            and row["label_ad"] in (0, 1)
        ]
        invalid_ids = {row["scan_id"] for row in subject_rows}.difference(
            {row["scan_id"] for row in valid_rows}
        )
        for row in subject_rows:
            if row["scan_id"] in invalid_ids:
                reasons = []
                if not row["mesh_exists"]:
                    reasons.append("missing_mesh")
                if not row["sdf_exists"]:
                    reasons.append("missing_sdf")
                if row["visit_month_from_label"] is None:
                    reasons.append("unknown_visit_label")
                if row["age_years"] is None:
                    reasons.append("missing_age")
                if row["label_ad"] not in (0, 1):
                    reasons.append("unexpected_diagnosis")
                row["drop_reason"] = "|".join(reasons) or "invalid_scan"
                report["dropped_scans"].append(drop_record(row))

        by_month: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            by_month[int(row["visit_month_from_label"])].append(row)

        kept_candidates: List[Dict[str, Any]] = []
        for month, group in sorted(by_month.items()):
            group_sorted = sorted(group, key=candidate_sort_key)
            kept = group_sorted[0]
            group_key = f"{subject_id}:month_{month:03d}"
            if len(group_sorted) > 1:
                report["duplicate_visit_groups"].append(
                    {
                        "subject_id": subject_id,
                        "visit_month": month,
                        "kept_scan_id": kept["scan_id"],
                        "scan_ids": [row["scan_id"] for row in group_sorted],
                    }
                )
            for row in group_sorted:
                row["has_duplicate_visit"] = len(group_sorted) > 1
                row["duplicate_group_key"] = group_key if len(group_sorted) > 1 else ""
                row["duplicate_kept_scan_id"] = kept["scan_id"] if len(group_sorted) > 1 else ""
                if row is kept:
                    kept_candidates.append(row)
                else:
                    row["drop_reason"] = f"duplicate_visit_replaced_by:{kept['scan_id']}"
                    report["dropped_scans"].append(drop_record(row))

        kept_candidates.sort(
            key=lambda row: (
                int(row["visit_month_from_label"]),
                float(row["age_years"]),
                image_id_number(row["image_id"]),
            )
        )

        if len(kept_candidates) < 2:
            report["subjects_excluded_fewer_than_2_visits"].append(
                {
                    "subject_id": subject_id,
                    "split": split_values[0] if split_values else "",
                    "candidate_visit_count": len(kept_candidates),
                    "scan_ids": [row["scan_id"] for row in kept_candidates],
                }
            )
            for row in kept_candidates:
                row["drop_reason"] = "subject_has_fewer_than_2_clean_visits"
                report["dropped_scans"].append(drop_record(row))
            continue

        baseline_month = int(kept_candidates[0]["visit_month_from_label"])
        for order, row in enumerate(kept_candidates):
            row["keep_scan"] = True
            row["visit_order"] = order
            row["months_from_baseline"] = (
                int(row["visit_month_from_label"]) - baseline_month
            )
            row["clean_subject_visit_count"] = len(kept_candidates)
            clean.append(row)

    return clean, report


def drop_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scan_id": row["scan_id"],
        "filename": row["filename"],
        "subject_id": row["subject_id"],
        "image_id": row["image_id"],
        "split": row["split"],
        "visit_label_raw": row["visit_label_raw"],
        "visit_month_from_label": row["visit_month_from_label"],
        "drop_reason": row["drop_reason"],
    }


def apply_age_norm(clean_rows_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_ages = [
        float(row["age_years"])
        for row in clean_rows_list
        if row["split"] == "train" and row["age_years"] is not None
    ]
    if not train_ages:
        raise RuntimeError("No train ages available for age normalization.")
    age_min = min(train_ages)
    age_max = max(train_ages)
    denom = age_max - age_min
    if denom <= 0:
        raise RuntimeError("Train age min and max are identical; cannot normalize age.")
    for row in clean_rows_list:
        row["age_norm"] = (float(row["age_years"]) - age_min) / denom
    all_norms = [float(row["age_norm"]) for row in clean_rows_list]
    return {
        "age_min_train": age_min,
        "age_max_train": age_max,
        "age_norm_formula": "(age_years - age_min_train) / (age_max_train - age_min_train)",
        "train_age_count": len(train_ages),
        "clean_age_norm_min_all_splits": min(all_norms),
        "clean_age_norm_max_all_splits": max(all_norms),
    }


def clean_projection(row: Dict[str, Any]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in CLEAN_FIELDS}


def make_trajectories(clean_rows_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in clean_rows_list:
        by_subject[row["subject_id"]].append(row)

    trajectories: Dict[str, Any] = {}
    for subject_id, rows in sorted(by_subject.items()):
        rows = sorted(rows, key=lambda row: int(row["visit_order"]))
        first = rows[0]
        trajectories[subject_id] = {
            "split": first["split"],
            "diagnosis": first["diagnosis"],
            "label_ad": int(first["label_ad"]),
            "clean_visit_count": len(rows),
            "visits": [
                {
                    "visit_order": int(row["visit_order"]),
                    "scan_id": row["scan_id"],
                    "filename": row["filename"],
                    "image_id": row["image_id"],
                    "visit_label_raw": row["visit_label_raw"],
                    "visit_month_from_label": int(row["visit_month_from_label"]),
                    "age_years": float(row["age_years"]),
                    "age_norm": float(row["age_norm"]),
                    "months_from_baseline": float(row["months_from_baseline"]),
                    "mesh_path": row["mesh_path"],
                    "sdf_npz_path": row["sdf_npz_path"],
                }
                for row in rows
            ],
        }
    return trajectories


def pair_task(source_visit_month: int, target_visit_month: int) -> str:
    if source_visit_month == 0 and target_visit_month == 6:
        return "b_to_m06"
    if source_visit_month == 0 and target_visit_month == 12:
        return "b_to_m12"
    if source_visit_month == 6 and target_visit_month == 12:
        return "m06_to_m12"
    return f"m{source_visit_month:02d}_to_m{target_visit_month:02d}"


def build_pairs(clean_rows_list: List[Dict[str, Any]], min_delta_months: float) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in clean_rows_list:
        by_subject[row["subject_id"]].append(row)
    for subject_id, rows in sorted(by_subject.items()):
        rows = sorted(rows, key=lambda row: int(row["visit_order"]))
        for source, target in itertools.combinations(rows, 2):
            delta = float(target["months_from_baseline"]) - float(source["months_from_baseline"])
            if delta < float(min_delta_months):
                continue
            source_month = int(source["visit_month_from_label"])
            target_month = int(target["visit_month_from_label"])
            pairs.append(
                {
                    "subject_id": subject_id,
                    "split": source["split"],
                    "source_scan_id": source["scan_id"],
                    "target_scan_id": target["scan_id"],
                    "source_filename": source["filename"],
                    "target_filename": target["filename"],
                    "source_image_id": source["image_id"],
                    "target_image_id": target["image_id"],
                    "source_visit_label": source["visit_label_raw"],
                    "target_visit_label": target["visit_label_raw"],
                    "source_visit_order": int(source["visit_order"]),
                    "target_visit_order": int(target["visit_order"]),
                    "source_visit_month": source_month,
                    "target_visit_month": target_month,
                    "source_months_from_baseline": float(source["months_from_baseline"]),
                    "target_months_from_baseline": float(target["months_from_baseline"]),
                    "source_age_years": float(source["age_years"]),
                    "target_age_years": float(target["age_years"]),
                    "source_age_norm": float(source["age_norm"]),
                    "target_age_norm": float(target["age_norm"]),
                    "delta_months": delta,
                    "diagnosis": source["diagnosis"],
                    "label_ad": int(source["label_ad"]),
                    "task": pair_task(source_month, target_month),
                }
            )
    return pairs


def build_triplets(clean_rows_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    triplets: List[Dict[str, Any]] = []
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in clean_rows_list:
        by_subject[row["subject_id"]].append(row)
    for subject_id, rows in sorted(by_subject.items()):
        rows = sorted(rows, key=lambda row: int(row["visit_order"]))
        for s_row, r_row, t_row in itertools.combinations(rows, 3):
            triplets.append(
                {
                    "subject_id": subject_id,
                    "split": s_row["split"],
                    "scan_s": s_row["scan_id"],
                    "scan_r": r_row["scan_id"],
                    "scan_t": t_row["scan_id"],
                    "filename_s": s_row["filename"],
                    "filename_r": r_row["filename"],
                    "filename_t": t_row["filename"],
                    "visit_order_s": int(s_row["visit_order"]),
                    "visit_order_r": int(r_row["visit_order"]),
                    "visit_order_t": int(t_row["visit_order"]),
                    "visit_month_s": int(s_row["visit_month_from_label"]),
                    "visit_month_r": int(r_row["visit_month_from_label"]),
                    "visit_month_t": int(t_row["visit_month_from_label"]),
                    "months_s": float(s_row["months_from_baseline"]),
                    "months_r": float(r_row["months_from_baseline"]),
                    "months_t": float(t_row["months_from_baseline"]),
                    "age_s": float(s_row["age_years"]),
                    "age_r": float(r_row["age_years"]),
                    "age_t": float(t_row["age_years"]),
                    "age_norm_s": float(s_row["age_norm"]),
                    "age_norm_r": float(r_row["age_norm"]),
                    "age_norm_t": float(t_row["age_norm"]),
                    "diagnosis": s_row["diagnosis"],
                    "label_ad": int(s_row["label_ad"]),
                }
            )
    return triplets


def write_split_jsons(output_root: Path, clean_rows_list: List[Dict[str, Any]]) -> None:
    for split in SPLITS:
        filenames = [
            row["filename"]
            for row in sorted(
                clean_rows_list,
                key=lambda row: (row["subject_id"], int(row["visit_order"]), row["filename"]),
            )
            if row["split"] == split
        ]
        write_json(output_root / "splits" / f"{split}_clean.json", filenames)


def write_pairs(output_root: Path, pairs: List[Dict[str, Any]]) -> None:
    write_csv(output_root / "pairs" / "pairs_all.csv", pairs, PAIR_FIELDS)
    for split in SPLITS:
        split_pairs = [row for row in pairs if row["split"] == split]
        write_csv(output_root / "pairs" / f"pairs_{split}.csv", split_pairs, PAIR_FIELDS)


def write_triplets(output_root: Path, triplets: List[Dict[str, Any]]) -> None:
    write_csv(output_root / "triplets" / "triplets_all.csv", triplets, TRIPLET_FIELDS)
    for split in SPLITS:
        split_triplets = [row for row in triplets if row["split"] == split]
        write_csv(output_root / "triplets" / f"triplets_{split}.csv", split_triplets, TRIPLET_FIELDS)


def counts_by_split(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(row["split"] for row in rows)
    return {split: int(counter.get(split, 0)) for split in SPLITS}


def subject_counts_by_split(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    split_subjects: Dict[str, set] = {split: set() for split in SPLITS}
    for row in rows:
        split_subjects[row["split"]].add(row["subject_id"])
    return {split: len(split_subjects[split]) for split in SPLITS}


def make_split_summary(
    master_rows: List[Dict[str, Any]],
    clean_rows_list: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    triplets: List[Dict[str, Any]],
    cleaning_report: Dict[str, Any],
) -> Dict[str, Any]:
    visits_by_subject: Dict[str, int] = {}
    for row in clean_rows_list:
        visits_by_subject[row["subject_id"]] = int(row["clean_subject_visit_count"])
    clean_subject_visit_counts = Counter(visits_by_subject.values())
    diagnosis_clean = Counter(row["diagnosis"] for row in clean_rows_list)
    diagnosis_master = Counter(row["diagnosis"] for row in master_rows)
    pair_task_counts = Counter(row["task"] for row in pairs)
    pair_task_by_split: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        pair_task_by_split[split] = dict(Counter(row["task"] for row in pairs if row["split"] == split))
    return {
        "master_scan_count": len(master_rows),
        "master_subject_count": len({row["subject_id"] for row in master_rows}),
        "master_scan_count_by_split": counts_by_split(master_rows),
        "master_subject_count_by_split": subject_counts_by_split(master_rows),
        "master_diagnosis_scan_count": dict(diagnosis_master),
        "clean_scan_count": len(clean_rows_list),
        "clean_subject_count": len({row["subject_id"] for row in clean_rows_list}),
        "clean_scan_count_by_split": counts_by_split(clean_rows_list),
        "clean_subject_count_by_split": subject_counts_by_split(clean_rows_list),
        "clean_diagnosis_scan_count": dict(diagnosis_clean),
        "clean_visit_count_distribution_subjects": {
            str(key): int(value) for key, value in sorted(clean_subject_visit_counts.items())
        },
        "duplicate_visit_group_count": len(cleaning_report["duplicate_visit_groups"]),
        "dropped_scan_count": len(cleaning_report["dropped_scans"]),
        "subjects_excluded_fewer_than_2_visits": len(
            cleaning_report["subjects_excluded_fewer_than_2_visits"]
        ),
        "pair_count": len(pairs),
        "pair_count_by_split": counts_by_split(pairs),
        "pair_task_count": dict(pair_task_counts),
        "pair_task_count_by_split": pair_task_by_split,
        "triplet_count": len(triplets),
        "triplet_count_by_split": counts_by_split(triplets),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.output_root)

    splits = load_splits(args.split_dir)
    metadata = read_metadata(args.metadata_csv)
    master_rows, warnings = make_master_rows(
        splits=splits,
        metadata=metadata,
        mesh_dir=args.mesh_dir,
        sdf_dir=args.sdf_dir,
    )
    clean_rows_list, cleaning_report = clean_rows(master_rows)
    age_stats = apply_age_norm(clean_rows_list)
    clean_rows_list = sorted(
        clean_rows_list, key=lambda row: (row["split"], row["subject_id"], int(row["visit_order"]))
    )
    pairs = build_pairs(clean_rows_list, min_delta_months=args.min_delta_months)
    triplets = build_triplets(clean_rows_list)
    trajectories = make_trajectories(clean_rows_list)

    metadata_dir = args.output_root / "metadata"
    write_csv(metadata_dir / "adni_no_mci_left_original_master.csv", master_rows, MASTER_FIELDS)
    write_csv(
        metadata_dir / "adni_no_mci_left_original_clean.csv",
        [clean_projection(row) for row in clean_rows_list],
        CLEAN_FIELDS,
    )
    write_json(metadata_dir / "adni_no_mci_left_original_subject_trajectories.json", trajectories)
    write_json(metadata_dir / "age_norm_stats.json", age_stats)
    write_split_jsons(args.output_root, clean_rows_list)
    write_pairs(args.output_root, pairs)
    write_triplets(args.output_root, triplets)

    cleaning_report["warnings"] = warnings
    cleaning_report["inputs"] = {
        "split_dir": str(args.split_dir),
        "metadata_csv": str(args.metadata_csv),
        "mesh_dir": str(args.mesh_dir),
        "sdf_dir": str(args.sdf_dir),
        "min_delta_months": args.min_delta_months,
    }
    write_json(metadata_dir / "cleaning_report.json", cleaning_report)
    split_summary = make_split_summary(master_rows, clean_rows_list, pairs, triplets, cleaning_report)
    write_json(metadata_dir / "split_summary.json", split_summary)

    print("Task 1 manifest build complete.")
    print(f"Output root: {args.output_root}")
    print(f"Master scans: {split_summary['master_scan_count']}")
    print(f"Clean scans: {split_summary['clean_scan_count']}")
    print(f"Clean subjects: {split_summary['clean_subject_count']}")
    print(f"Pairs: {split_summary['pair_count']}")
    print(f"Triplets: {split_summary['triplet_count']}")
    print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
