#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]

DEFAULT_SELECTED_SCANS = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI_large_strict_left"
    / "manifests"
    / "selected_scans.csv"
)
DEFAULT_SPLIT_DIR = (
    REPO_ROOT
    / "examples"
    / "splits"
    / "splits_left_hippocampus_ADNI_Large_Strict_No_MCI"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/jakaria/ADNI/ADNI_1_GO_Large/left_hippocampus_strict_no_mci"
)
DEFAULT_OUTPUT = (
    TASK_DIR / "metadata" / "adni_large_strict_no_mci_left_manifest.csv"
)


FIELDNAMES = [
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
    "sex",
    "sex_numeric",
    "baseline_diagnosis",
    "left_mask_volume_mm3",
    "left_mesh_volume_mm3",
    "left_surface_area_mm2",
    "left_mask_voxels",
    "mesh_path",
    "sdf_npz_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Task2 INR manifest for the large strict no-MCI left "
            "hippocampus correspondence/SDF dataset."
        )
    )
    parser.add_argument(
        "--selected-scans",
        default=str(DEFAULT_SELECTED_SCANS),
        help="CSV created by the strict no-MCI Task1 pipeline.",
    )
    parser.add_argument(
        "--split-dir",
        default=str(DEFAULT_SPLIT_DIR),
        help="Directory containing train/val/test split JSON files.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root outside the repo.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output manifest CSV.",
    )
    parser.add_argument(
        "--require-sdf",
        action="store_true",
        help="Fail if any expected SDF npz file is missing.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def as_float(value: str | None, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def format_float(value: float | None, precision: int = 10) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}g}"


def load_split_map(split_dir: Path) -> dict[str, str]:
    split_files = {
        "train": split_dir / "train_split_left_hippocampus_adni_large.json",
        "val": split_dir / "val_split_left_hippocampus_adni_large.json",
        "test": split_dir / "test_split_left_hippocampus_adni_large.json",
    }
    mapping: dict[str, str] = {}
    for split, path in split_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"Split file must contain a list: {path}")
        for value in values:
            stem = Path(str(value)).stem
            if stem in mapping:
                raise ValueError(f"Duplicate split assignment for {stem}")
            mapping[stem] = split
    return mapping


def month_sort_key(row: dict[str, str]) -> tuple[float, str, str]:
    month = as_float(row.get("month_from_viscode"))
    if month is None:
        month = as_float(row.get("Month.bl"), 0.0)
    return (float(month or 0.0), row.get("EXAMDATE", ""), row.get("VISCODE", ""))


def add_visit_order(rows: list[dict[str, str]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["RID"]].append(row)

    visit_order_by_stem: dict[str, int] = {}
    for _rid, subject_rows in grouped.items():
        subject_rows = sorted(
            subject_rows,
            key=lambda row: (month_sort_key(row), row.get("left_stem", "")),
        )
        for index, row in enumerate(subject_rows):
            visit_order_by_stem[row["left_stem"]] = index
    return visit_order_by_stem


def label_ad(diagnosis: str) -> int:
    if diagnosis == "CN":
        return 0
    if diagnosis == "AD":
        return 1
    raise ValueError(f"Strict no-MCI manifest expected CN/AD only, got {diagnosis!r}")


def main() -> int:
    args = parse_args()
    selected_path = Path(args.selected_scans).expanduser().resolve()
    split_dir = Path(args.split_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not selected_path.is_file():
        raise FileNotFoundError(f"Missing selected scans CSV: {selected_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    obj_dir = (
        dataset_root
        / "left_hippocampus_correspondence"
        / "minimal_scaled_obj_files"
    )
    sdf_dir = (
        dataset_root
        / "left_hippocampus_correspondence"
        / "sdf_data"
        / "SdfSamples"
        / "minimal_scaled_obj_files"
    )
    split_by_stem = load_split_map(split_dir)
    selected_rows = read_csv(selected_path)
    by_stem = {row["left_stem"]: row for row in selected_rows if row.get("left_stem")}
    if len(by_stem) != len(selected_rows):
        raise ValueError("selected_scans.csv contains missing or duplicate left_stem values.")

    split_only = sorted(set(split_by_stem).difference(by_stem))
    selected_only = sorted(set(by_stem).difference(split_by_stem))
    if split_only or selected_only:
        raise ValueError(
            "Split files and selected_scans.csv do not describe the same scans: "
            f"split_only={len(split_only)}, selected_only={len(selected_only)}"
        )

    visit_order_by_stem = add_visit_order(selected_rows)
    train_ages = [
        as_float(by_stem[stem].get("age_numeric"))
        for stem, split in split_by_stem.items()
        if split == "train"
    ]
    train_ages = [age for age in train_ages if age is not None]
    if not train_ages:
        raise ValueError("No train ages found for age normalization.")
    age_min = min(train_ages)
    age_max = max(train_ages)
    if age_max <= age_min:
        raise ValueError("Train age range is degenerate.")

    subject_first_month: dict[str, float] = {}
    for row in selected_rows:
        month = as_float(row.get("month_from_viscode"))
        if month is None:
            month = as_float(row.get("Month.bl"), 0.0)
        rid = row["RID"]
        subject_first_month[rid] = min(subject_first_month.get(rid, month), month)

    missing_mesh: list[str] = []
    missing_sdf: list[str] = []
    output_rows: list[dict[str, str]] = []
    for stem in sorted(split_by_stem, key=lambda value: (split_by_stem[value], value)):
        source = by_stem[stem]
        age = as_float(source.get("age_numeric"))
        if age is None:
            raise ValueError(f"Missing age_numeric for {stem}")
        month = as_float(source.get("month_from_viscode"))
        if month is None:
            month = as_float(source.get("Month.bl"), 0.0)
        diagnosis = source.get("visit_dx_3class", "")
        mesh_path = obj_dir / f"{stem}.obj"
        sdf_path = sdf_dir / f"{stem}.npz"
        if not mesh_path.is_file():
            missing_mesh.append(stem)
        if not sdf_path.is_file():
            missing_sdf.append(stem)

        output_rows.append(
            {
                "scan_id": stem,
                "filename": f"{stem}.obj",
                "subject_id": source["RID"],
                "image_id": source.get("scan_id", stem.rsplit("_", 1)[0]),
                "split": split_by_stem[stem],
                "diagnosis": diagnosis,
                "label_ad": str(label_ad(diagnosis)),
                "visit_order": str(visit_order_by_stem[stem]),
                "visit_label": source.get("VISCODE", ""),
                "visit_month": format_float(month),
                "months_from_baseline": format_float(
                    float(month or 0.0) - subject_first_month[source["RID"]]
                ),
                "age_years": format_float(age),
                "age_norm": format_float((age - age_min) / (age_max - age_min), 12),
                "sex": source.get("PTGENDER", ""),
                "sex_numeric": source.get("gender_numeric", ""),
                "baseline_diagnosis": source.get("baseline_dx_3class", ""),
                "left_mask_volume_mm3": source.get("left_mask_volume_mm3", ""),
                "left_mesh_volume_mm3": source.get("left_mesh_volume_mm3", ""),
                "left_surface_area_mm2": source.get("left_surface_area_mm2", ""),
                "left_mask_voxels": source.get("left_mask_voxels", ""),
                "mesh_path": str(mesh_path),
                "sdf_npz_path": str(sdf_path),
            }
        )

    if missing_mesh:
        raise FileNotFoundError(
            f"Missing {len(missing_mesh)} mesh files; first few: {missing_mesh[:10]}"
        )
    if args.require_sdf and missing_sdf:
        raise FileNotFoundError(
            f"Missing {len(missing_sdf)} SDF files; first few: {missing_sdf[:10]}"
        )

    split_order = {"train": 0, "val": 1, "test": 2}
    output_rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            int(row["subject_id"]),
            int(row["visit_order"]),
            row["scan_id"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    split_counts = Counter(row["split"] for row in output_rows)
    diagnosis_counts = Counter(row["diagnosis"] for row in output_rows)
    subject_counts = {
        split: len({row["subject_id"] for row in output_rows if row["split"] == split})
        for split in ("train", "val", "test")
    }
    age_values = [float(row["age_years"]) for row in output_rows]
    age_norm_values = [float(row["age_norm"]) for row in output_rows]
    summary = {
        "manifest": str(output_path),
        "selected_scans": str(selected_path),
        "dataset_root": str(dataset_root),
        "mesh_dir": str(obj_dir),
        "sdf_dir": str(sdf_dir),
        "row_count": len(output_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "subject_counts": subject_counts,
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "age_years": {
            "min": min(age_values),
            "max": max(age_values),
            "train_min": age_min,
            "train_max": age_max,
        },
        "age_norm": {
            "min": min(age_norm_values),
            "max": max(age_norm_values),
            "train_min": 0.0,
            "train_max": 1.0,
        },
        "missing_mesh_count": len(missing_mesh),
        "missing_sdf_count": len(missing_sdf),
        "sdf_complete": len(missing_sdf) == 0,
        "first_missing_sdf": missing_sdf[:10],
    }
    write_json(TASK_DIR / "metadata" / "manifest_summary.json", summary)
    write_json(
        TASK_DIR / "metadata" / "age_norm_stats.json",
        {
            "source": "train split age_numeric",
            "age_min": age_min,
            "age_max": age_max,
            "formula": "age_norm = (age_numeric - age_min) / (age_max - age_min)",
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
