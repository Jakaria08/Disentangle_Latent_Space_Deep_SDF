#!/usr/bin/env python3
"""Build a deterministic 173/15/15 CALSNIC-control left-cortex manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from calsnic_common import (
    DATASET_ROOT,
    DEFAULT_APPROX_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    load_mesh,
    require_bulk_path,
)


METADATA = Path("/home/jakaria/CALSNIC/Final_Data_sheet_April2025.csv")
ETIV_METADATA = Path("/home/jakaria/CALSNIC/full_volume_normalized_etiv.csv")
BALANCE_FIELDS = ("study", "site", "sex", "age_bin", "etiv_bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train", type=int, default=173)
    parser.add_argument("--val", type=int, default=15)
    parser.add_argument("--test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-bulk-output", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def stable_number(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def read_best_metadata(subjects: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, tuple[int, dict[str, str]]] = {}
    with METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            subject = row.get("Filename", "").strip()
            if subject not in subjects:
                continue
            score = sum(bool(row.get(key, "").strip()) for key in ("Study", "Site", "Sex", "Age"))
            if subject not in result or score > result[subject][0]:
                result[subject] = (score, row)
    missing = sorted(subjects.difference(result))
    if missing:
        raise ValueError(f"Metadata missing for subjects: {missing[:5]}")
    return {subject: row for subject, (_score, row) in result.items()}


def read_etiv(subjects: set[str]) -> tuple[dict[str, float], list[str], float | None]:
    result: dict[str, float] = {}
    missing_values: list[str] = []
    with ETIV_METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            subject = row["Subject_ID"].strip()
            if subject in subjects:
                value = row.get("eTIV_mm3", "").strip()
                if not value:
                    missing_values.append(subject)
                    continue
                try:
                    parsed = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid eTIV_mm3={value!r} for {subject} in {ETIV_METADATA}."
                    ) from error
                if not np.isfinite(parsed) or parsed <= 0.0:
                    raise ValueError(
                        f"Invalid non-positive/non-finite eTIV_mm3={value!r} for {subject} "
                        f"in {ETIV_METADATA}."
                    )
                result[subject] = parsed
    missing = sorted(subjects.difference(result))
    if missing != sorted(set(missing_values)):
        raise ValueError(f"eTIV metadata missing for subjects: {missing[:5]}")
    if missing:
        if not result:
            raise ValueError("No valid eTIV values are available for median imputation.")
        imputation_value = float(np.median(np.asarray(list(result.values()), dtype=np.float64)))
        for subject in missing:
            result[subject] = imputation_value
        return result, missing, imputation_value
    return result, [], None


def assign_quantile_bins(records: list[dict], key: str, output: str, bins: int = 4) -> None:
    values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    for row in records:
        row[output] = str(int(np.searchsorted(edges, float(row[key]), side="right")))


def select_balanced(records: list[dict], count: int, seed: int) -> list[dict]:
    """Greedily match all requested marginal distributions with deterministic ties."""
    if count >= len(records):
        return list(records)
    population = {field: Counter(str(row[field]) for row in records) for field in BALANCE_FIELDS}
    selected_counts = {field: Counter() for field in BALANCE_FIELDS}
    remaining = list(records)
    selected = []
    for step in range(count):
        fraction = (step + 1) / len(records)

        def cost(row: dict) -> tuple[float, int]:
            value = 0.0
            for field in BALANCE_FIELDS:
                category = str(row[field])
                target = population[field][category] * fraction
                after = selected_counts[field][category] + 1
                value += (after - target) ** 2 / max(target, 0.25)
            return value, stable_number(row["scan_id"], seed + step * 1009)

        chosen = min(remaining, key=cost)
        remaining.remove(chosen)
        selected.append(chosen)
        for field in BALANCE_FIELDS:
            selected_counts[field][str(chosen[field])] += 1
    return selected


def fit_scaled_from_mm(mm: np.ndarray, scaled: np.ndarray) -> tuple[float, np.ndarray, dict]:
    mm_centered = mm - mm.mean(axis=0, keepdims=True)
    scaled_centered = scaled - scaled.mean(axis=0, keepdims=True)
    denominator = float(np.sum(mm_centered * mm_centered))
    scale = float(np.sum(mm_centered * scaled_centered) / denominator)
    translation = scaled.mean(axis=0) - scale * mm.mean(axis=0)
    residual = np.linalg.norm(scale * mm + translation[None, :] - scaled, axis=1)
    return scale, translation, {
        "transform_residual_mean": float(residual.mean()),
        "transform_residual_p99": float(np.quantile(residual, 0.99)),
        "transform_residual_max": float(residual.max()),
    }


def main() -> None:
    args = parse_args()
    if min(args.train, args.val, args.test) < 1:
        raise ValueError("All split counts must be positive.")
    output_root = require_bulk_path(args.output_root, allow_non_bulk=args.allow_non_bulk_output)
    manifest_path = output_root / "manifests" / DEFAULT_APPROX_MANIFEST.name
    report_path = output_root / "selections" / "calsnic_control_L_split.json"
    for path in (manifest_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")

    sdf_dir = DATASET_ROOT / "sdf_data_control" / "SdfSamples" / "scaled_obj_files"
    scaled_dir = DATASET_ROOT / "scaled_obj_files"
    mm_dir = DATASET_ROOT / "obj_files"
    sdf_ids = {path.stem for path in sdf_dir.glob("*.npz")}
    scaled_ids = {path.stem for path in scaled_dir.glob("*.obj")}
    mm_ids = {path.stem for path in mm_dir.glob("*.obj")}
    subjects = sdf_ids & scaled_ids & mm_ids
    if len(subjects) != len(sdf_ids) or len(subjects) != len(scaled_ids) or len(subjects) != len(mm_ids):
        raise ValueError("SDF, scaled-mesh, and millimetre-mesh subject sets do not match.")
    if args.train + args.val + args.test != len(subjects):
        raise ValueError(
            f"Requested {args.train + args.val + args.test} rows, but found {len(subjects)} matched subjects."
        )

    metadata = read_best_metadata(subjects)
    etiv, etiv_imputed_subjects, etiv_imputation_value = read_etiv(subjects)
    records = []
    transform_reports = []
    population_min = np.full(3, np.inf)
    population_max = np.full(3, -np.inf)
    for number, subject in enumerate(sorted(subjects), start=1):
        scaled_path = scaled_dir / f"{subject}.obj"
        mm_path = mm_dir / f"{subject}.obj"
        scaled_mesh = load_mesh(scaled_path)
        mm_mesh = load_mesh(mm_path)
        if scaled_mesh.vertices.shape != mm_mesh.vertices.shape or not np.array_equal(
            scaled_mesh.faces, mm_mesh.faces
        ):
            raise ValueError(f"Scaled/mm topology mismatch for {subject}.")
        center = np.asarray(scaled_mesh.bounds, dtype=np.float64).mean(axis=0)
        sdf_vertices = np.asarray(scaled_mesh.vertices, dtype=np.float64) - center[None, :]
        population_min = np.minimum(population_min, sdf_vertices.min(axis=0))
        population_max = np.maximum(population_max, sdf_vertices.max(axis=0))
        scale, translation, transform_report = fit_scaled_from_mm(
            np.asarray(mm_mesh.vertices, dtype=np.float64),
            np.asarray(scaled_mesh.vertices, dtype=np.float64),
        )
        if scale <= 0.0 or transform_report["transform_residual_p99"] > 1.0e-4:
            raise ValueError(f"Scaled/mm transform fit failed for {subject}: {transform_report}")
        source = metadata[subject]
        records.append(
            {
                "scan_id": subject,
                "subject_id": subject,
                "split": "",
                "diagnosis": "Control",
                "study": source.get("Study", "").strip(),
                "site": source.get("Site", "").strip(),
                "sex": source.get("Sex", "").strip(),
                "age_years": float(source["Age"]),
                "etiv_mm3": float(etiv[subject]),
                "etiv_imputed": "1" if subject in etiv_imputed_subjects else "0",
                "age_bin": "",
                "etiv_bin": "",
                "mesh_path": str(scaled_path.resolve()),
                "mesh_path_mm": str(mm_path.resolve()),
                "source_sdf_npz_path": str((sdf_dir / f"{subject}.npz").resolve()),
                "sdf_npz_path": str((sdf_dir / f"{subject}.npz").resolve()),
                "sdf_label_kind": "approximate_preprocessmesh_bbox_centered",
                "mesh_center_x": float(center[0]),
                "mesh_center_y": float(center[1]),
                "mesh_center_z": float(center[2]),
                "scaled_from_mm_scale": scale,
                "scaled_from_mm_tx": float(translation[0]),
                "scaled_from_mm_ty": float(translation[1]),
                "scaled_from_mm_tz": float(translation[2]),
                "split_seed": int(args.seed),
            }
        )
        transform_reports.append({"scan_id": subject, **transform_report})
        if number % 25 == 0 or number == len(subjects):
            print(f"Validated geometry {number}/{len(subjects)}", flush=True)

    assign_quantile_bins(records, "age_years", "age_bin")
    assign_quantile_bins(records, "etiv_mm3", "etiv_bin")
    test = select_balanced(records, args.test, args.seed + 1)
    test_ids = {row["scan_id"] for row in test}
    remaining = [row for row in records if row["scan_id"] not in test_ids]
    val = select_balanced(remaining, args.val, args.seed + 2)
    val_ids = {row["scan_id"] for row in val}
    for row in records:
        row["split"] = "test" if row["scan_id"] in test_ids else "val" if row["scan_id"] in val_ids else "train"
    order = {"train": 0, "val": 1, "test": 2}
    records.sort(key=lambda row: (order[row["split"]], row["scan_id"]))

    atomic_write_csv(manifest_path, records, allow_non_bulk=args.allow_non_bulk_output)
    split_report = {}
    for split in ("train", "val", "test"):
        rows = [row for row in records if row["split"] == split]
        split_report[split] = {
            "count": len(rows),
            "scan_ids": [row["scan_id"] for row in rows],
            **{field: dict(sorted(Counter(str(row[field]) for row in rows).items())) for field in BALANCE_FIELDS},
        }
    report = {
        "manifest": str(manifest_path),
        "seed": int(args.seed),
        "selection_unit": "one_mesh_per_subject",
        "balance_fields": list(BALANCE_FIELDS),
        "split_report": split_report,
        "population_sdf_mesh_bounds": [population_min.tolist(), population_max.tolist()],
        "max_transform_residual_p99": max(row["transform_residual_p99"] for row in transform_reports),
        "preprocessmesh_transform": "subtract per-mesh AABB midpoint; no additional scaling",
        "etiv_imputation": {
            "method": "median of valid matched subjects",
            "value_mm3": etiv_imputation_value,
            "subjects": etiv_imputed_subjects,
            "reason": "blank eTIV_mm3 in the source metadata; eTIV is used only for split balancing",
        },
        "test_policy": "locked_until_architecture_and_checkpoint_are_selected_on_validation",
    }
    atomic_write_json(report_path, report, allow_non_bulk=args.allow_non_bulk_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
