#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SPLITS = ("train", "val", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_task_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else TASK_DIR / path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(value: str | int | float) -> int:
    return int(float(str(value).strip()))


def as_float(value: str | int | float) -> float:
    return float(str(value).strip())


def count_by(rows: Iterable[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter(row[key] for row in rows))


def summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty collection.")
    return {
        "count": float(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def sort_scan_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_ORDER.get(row["split"], 99),
            row["subject_id"],
            as_int(row["visit_order"]),
            row["scan_id"],
        ),
    )


def load_clean_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {
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
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Clean manifest is missing columns: {sorted(missing)}")
    return sort_scan_rows(rows)


def load_representation_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {
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
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(
            f"Representation manifest is missing columns: {sorted(missing)}"
        )
    return sort_scan_rows(rows)


def load_pca_coefficients_csv(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {
        "scan_id",
        "split",
        "diagnosis",
        "label_ad",
        "coefficient_dimension",
        "coefficient_path",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"PCA coefficient CSV is missing columns: {sorted(missing)}")
    return sort_scan_rows(rows)


def group_rows_by_subject(
    rows: Iterable[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)
    for subject_id, subject_rows in grouped.items():
        grouped[subject_id] = sorted(
            subject_rows,
            key=lambda row: (as_int(row["visit_order"]), row["scan_id"]),
        )
    return dict(grouped)


def load_numpy_vector(path: str | Path, expected_dim: int) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ValueError(f"Expected shape ({expected_dim},), got {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Array contains non-finite values: {path}")
    return array


def format_float(value: float, decimals: int = 10) -> str:
    return f"{value:.{decimals}f}"


def continuous_age_years(baseline_age_years: float, months_from_baseline: float) -> float:
    return baseline_age_years + months_from_baseline / 12.0


def normalize_age(age_years: float, age_min_train: float, age_max_train: float) -> float:
    denominator = age_max_train - age_min_train
    if denominator <= 0.0:
        raise ValueError("age_max_train must exceed age_min_train.")
    return (age_years - age_min_train) / denominator
