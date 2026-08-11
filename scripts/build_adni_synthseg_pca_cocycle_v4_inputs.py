#!/usr/bin/env python3
"""Build and validate subject-disjoint PCA-Cocycle-v4 inputs for two SynthSeg shapes.

This script consumes the conservative strict-no-MCI QC manifest and creates a
reproducible, *metadata-only* PCA-Cocycle-v4 input bundle.  It never changes
the source correspondence meshes and it does not fit PCA or train a model.

The hippocampus and lateral ventricle have separate correspondence atlases and
very different vertex counts.  Therefore the downstream representation must be
two train-only PCA blocks in physical-mm correspondence space, rather than one
raw concatenated vertex PCA.  Selected PC scores will be standardised from the
training split before a shared Cocycle/Brain-ODE model, while unstandardised
scores remain available for mesh reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/home/jakaria/ADNI/ADNI_1_GO_Large/synthseg_minimal_correspondence/full")
DEFAULT_QC_ROOT = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI_synthseg_minimal_smooth_qc"
DEFAULT_QC_MANIFEST = DEFAULT_QC_ROOT / "manifests" / "strict_no_mci_longitudinal_keep_conservative.csv"
DEFAULT_SCAN_QC = DEFAULT_QC_ROOT / "mesh_qc" / "scan_qc.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_pca_cocycle_v4"
)

STRUCTURES: dict[str, dict[str, Any]] = {
    "left_hippocampus": {"short_name": "hippocampus", "vertex_count": 2746, "face_count": 5488},
    "left_lateral_ventricle": {"short_name": "lateral_ventricle", "vertex_count": 8346, "face_count": 16688},
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-manifest", type=Path, default=DEFAULT_QC_MANIFEST)
    parser.add_argument("--scan-qc", type=Path, default=DEFAULT_SCAN_QC)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_sizes(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("Each diagnosis needs at least three people for train/val/test.")
    if min(train_ratio, val_ratio, test_ratio) <= 0 or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must be positive and sum to one.")
    n_val = max(1, int(round(total * val_ratio)))
    n_test = max(1, int(round(total * test_ratio)))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Cannot create a non-empty training split from {total} subjects.")
    return n_train, n_val, n_test


def make_subject_split(subject_table: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for diagnosis in ("CN", "AD"):
        subjects = sorted(subject_table.loc[subject_table["diagnosis"] == diagnosis, "subject_id"].astype(str))
        random.Random(f"{seed}:{diagnosis}").shuffle(subjects)
        n_train, n_val, _n_test = split_sizes(len(subjects), *ratios)
        for subject in subjects[:n_train]:
            assignment[subject] = "train"
        for subject in subjects[n_train : n_train + n_val]:
            assignment[subject] = "val"
        for subject in subjects[n_train + n_val :]:
            assignment[subject] = "test"
    if len(assignment) != len(subject_table):
        raise RuntimeError("Every subject must receive exactly one split assignment.")
    return assignment


def mesh_path(source_root: Path, structure: str, scan_id: str) -> Path:
    return source_root / structure / "minimal_smooth_correspondence" / "final_ply_mm" / f"{scan_id}.ply"


def require_input_contract(scans: pd.DataFrame, scan_qc: pd.DataFrame, source_root: Path) -> dict[str, Any]:
    required = {
        "scan_id", "subject_id", "VISCODE", "visit_month", "age_years", "sex",
        "baseline_diagnosis", "visit_diagnosis", "strict_subject_no_mci",
    }
    missing = sorted(required.difference(scans.columns))
    if missing:
        raise KeyError(f"QC keep manifest lacks required fields: {missing}")
    if scans["scan_id"].duplicated().any():
        raise ValueError("QC keep manifest has duplicate scan IDs.")
    if not scans["baseline_diagnosis"].isin(["CN", "AD"]).all():
        raise ValueError("Only baseline CN/AD scans may enter the PCA-Cocycle input.")
    if not scans["visit_diagnosis"].eq(scans["baseline_diagnosis"]).all():
        raise ValueError("Diagnosis-stable cohort expected: a visit diagnosis differs from baseline.")
    strict_values = scans["strict_subject_no_mci"].astype(str).str.lower()
    if not strict_values.isin(["true", "1", "yes"]).all():
        raise ValueError("QC keep manifest is not strict no-MCI for every scan.")
    if scans[["visit_month", "age_years"]].isna().any().any():
        raise ValueError("Visit month and age must be available for every retained scan.")

    ordered = scans.sort_values(["subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable")
    counts = ordered.groupby("subject_id")["scan_id"].nunique()
    if counts.lt(2).any():
        raise ValueError("Every retained participant needs at least two scans.")
    duplicate_time = ordered.duplicated(["subject_id", "visit_month"], keep=False)
    if duplicate_time.any():
        raise ValueError("A subject has duplicate longitudinal time points.")
    for subject_id, group in ordered.groupby("subject_id", sort=False):
        if (group["visit_month"].diff().dropna() <= 0).any():
            raise ValueError(f"Non-increasing visits for subject {subject_id}.")

    required_qc = {"scan_id", "structure", "hard_mesh_qc_flag", "correspondence_topology_hash"}
    missing_qc = sorted(required_qc.difference(scan_qc.columns))
    if missing_qc:
        raise KeyError(f"Scan QC file lacks required fields: {missing_qc}")
    selected_ids = set(ordered["scan_id"].astype(str))
    selected_qc = scan_qc.loc[scan_qc["scan_id"].astype(str).isin(selected_ids)].copy()
    expected_records = len(selected_ids) * len(STRUCTURES)
    if len(selected_qc) != expected_records:
        raise ValueError(f"Expected {expected_records} structure QC records, found {len(selected_qc)}.")
    if selected_qc["hard_mesh_qc_flag"].astype(bool).any():
        raise ValueError("A hard mesh-QC failure reached the keep manifest.")

    path_missing: dict[str, list[str]] = {structure: [] for structure in STRUCTURES}
    for scan_id in selected_ids:
        for structure in STRUCTURES:
            path = mesh_path(source_root, structure, scan_id)
            if not path.is_file():
                path_missing[structure].append(scan_id)
    if any(path_missing.values()):
        summary = {name: values[:10] for name, values in path_missing.items() if values}
        raise FileNotFoundError(f"Missing physical-mm correspondence meshes: {summary}")

    topology: dict[str, dict[str, Any]] = {}
    for structure, expected in STRUCTURES.items():
        rows = selected_qc.loc[selected_qc["structure"] == structure]
        hashes = sorted(rows["correspondence_topology_hash"].dropna().astype(str).unique())
        vertex_counts = sorted(pd.to_numeric(rows.get("correspondence_vertices_actual"), errors="coerce").dropna().unique())
        face_counts = sorted(pd.to_numeric(rows.get("correspondence_faces_actual"), errors="coerce").dropna().unique())
        if len(hashes) != 1 or vertex_counts != [expected["vertex_count"]] or face_counts != [expected["face_count"]]:
            raise ValueError(
                f"Selected {structure} meshes do not have one valid correspondence topology: "
                f"hashes={len(hashes)}, vertices={vertex_counts}, faces={face_counts}."
            )
        topology[structure] = {
            "correspondence_topology_hash": hashes[0],
            "vertex_count": expected["vertex_count"],
            "face_count": expected["face_count"],
            "coordinate_space": "final_ply_mm (rigidly aligned + Deformetrica correspondence; physical mm volume restored)",
        }
    return {"topology": topology, "path_missing": path_missing}


def add_qc_measurements(scans: pd.DataFrame, scan_qc: pd.DataFrame) -> pd.DataFrame:
    fields = ["scan_id", "structure", "correspondence_volume_actual_mm3", "correspondence_surface_area_actual_mm2"]
    available = [field for field in fields if field in scan_qc.columns]
    piv = scan_qc.loc[:, available].pivot(index="scan_id", columns="structure")
    piv.columns = [f"{metric}_{structure}" for metric, structure in piv.columns]
    result = scans.merge(piv, left_on="scan_id", right_index=True, how="left", validate="one_to_one")
    rename = {
        "correspondence_volume_actual_mm3_left_hippocampus": "hippocampus_volume_mm3",
        "correspondence_surface_area_actual_mm2_left_hippocampus": "hippocampus_surface_area_mm2",
        "correspondence_volume_actual_mm3_left_lateral_ventricle": "lateral_ventricle_volume_mm3",
        "correspondence_surface_area_actual_mm2_left_lateral_ventricle": "lateral_ventricle_surface_area_mm2",
    }
    return result.rename(columns=rename)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    qc_manifest_path = args.qc_manifest.expanduser().resolve()
    scan_qc_path = args.scan_qc.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not qc_manifest_path.is_file() or not scan_qc_path.is_file():
        raise FileNotFoundError("Both the QC keep manifest and scan_qc.csv are required.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    print("=" * 88)
    print("Build PCA-Cocycle-v4 metadata only (no PCA fitting, no model training)")
    print(f"QC keep manifest: {qc_manifest_path}")
    print(f"Source meshes:    {source_root} (read only)")
    print(f"Output bundle:    {output}")
    print("=" * 88, flush=True)

    scans = pd.read_csv(qc_manifest_path, dtype={"scan_id": str, "subject_id": str, "RID": str, "VISCODE": str})
    scan_qc = pd.read_csv(scan_qc_path, dtype={"scan_id": str, "subject_id": str})
    validation = require_input_contract(scans, scan_qc, source_root)
    print(f"[1/3] Verified both physical-mm correspondence meshes for {len(scans):,} scans.", flush=True)

    scans = add_qc_measurements(scans, scan_qc)
    scans = scans.sort_values(["subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)
    scans["diagnosis"] = scans["baseline_diagnosis"]
    scans["label_ad"] = scans["diagnosis"].map({"CN": 0, "AD": 1}).astype(int)
    scans["visit_order"] = scans.groupby("subject_id").cumcount().astype(int)
    scans["months_from_baseline"] = scans["visit_month"] - scans.groupby("subject_id")["visit_month"].transform("min")

    subject_table = (
        scans.groupby("subject_id", as_index=False)
        .agg(diagnosis=("diagnosis", "first"), scan_count=("scan_id", "nunique"), baseline_age_years=("age_years", "min"))
    )
    assignment = make_subject_split(
        subject_table,
        int(args.seed),
        (float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)),
    )
    scans["split"] = scans["subject_id"].astype(str).map(assignment)
    subject_table["split"] = subject_table["subject_id"].astype(str).map(assignment)
    if scans["split"].isna().any() or subject_table["split"].isna().any():
        raise RuntimeError("Missing subject split assignment.")
    if scans.groupby("subject_id")["split"].nunique().gt(1).any():
        raise RuntimeError("Subject leakage: a person appears in more than one split.")

    train_ages = scans.loc[scans["split"] == "train", "age_years"]
    age_min, age_max = float(train_ages.min()), float(train_ages.max())
    if age_max <= age_min:
        raise ValueError("Training age range is degenerate.")
    scans["age_norm_train"] = (scans["age_years"] - age_min) / (age_max - age_min)

    for structure, detail in STRUCTURES.items():
        short = detail["short_name"]
        scans[f"{short}_mesh_path_mm"] = scans["scan_id"].map(
            lambda scan_id: str(mesh_path(source_root, structure, str(scan_id)))
        )
        scans[f"{short}_vertex_count"] = int(detail["vertex_count"])
        scans[f"{short}_face_count"] = int(detail["face_count"])

    split_order = {name: index for index, name in enumerate(SPLITS)}
    scans["_split_order"] = scans["split"].map(split_order)
    scans = scans.sort_values(["_split_order", "subject_id", "visit_order", "scan_id"], kind="stable").drop(columns="_split_order")
    subject_table["_split_order"] = subject_table["split"].map(split_order)
    subject_table = subject_table.sort_values(["_split_order", "diagnosis", "subject_id"], kind="stable").drop(columns="_split_order")

    metadata = output / "metadata"
    splits_dir = output / "splits"
    configs = output / "configs"
    manifest_path = metadata / "adni_synthseg_lhipp_llv_strict_no_mci_manifest.csv"
    subject_path = metadata / "subject_split_assignments.csv"
    manifest_fields = [
        "scan_id", "subject_id", "split", "diagnosis", "label_ad", "baseline_diagnosis", "visit_diagnosis",
        "VISCODE", "visit_month", "months_from_baseline", "visit_order", "age_years", "age_norm_train", "sex",
        "hippocampus_volume_mm3", "hippocampus_surface_area_mm2", "lateral_ventricle_volume_mm3", "lateral_ventricle_surface_area_mm2",
        "hippocampus_mesh_path_mm", "hippocampus_vertex_count", "hippocampus_face_count",
        "lateral_ventricle_mesh_path_mm", "lateral_ventricle_vertex_count", "lateral_ventricle_face_count",
    ]
    missing_fields = sorted(set(manifest_fields).difference(scans.columns))
    if missing_fields:
        raise RuntimeError(f"Could not construct required manifest columns: {missing_fields}")
    write_csv(manifest_path, scans.loc[:, manifest_fields].to_dict("records"), manifest_fields)
    write_csv(
        subject_path,
        subject_table.loc[:, ["subject_id", "split", "diagnosis", "scan_count", "baseline_age_years"]].to_dict("records"),
        ["subject_id", "split", "diagnosis", "scan_count", "baseline_age_years"],
    )
    for split in SPLITS:
        ids = subject_table.loc[subject_table["split"] == split, "subject_id"].astype(str).tolist()
        write_json(splits_dir / f"{split}_subjects.json", ids)

    # Re-open the generated bundle.  This validates the files actually written,
    # not just the in-memory data used to create them.
    written_manifest = pd.read_csv(manifest_path, dtype={"scan_id": str, "subject_id": str})
    if list(written_manifest.columns) != manifest_fields or len(written_manifest) != len(scans):
        raise RuntimeError("Written PCA-Cocycle manifest does not match its required schema/count.")
    if written_manifest["scan_id"].duplicated().any() or written_manifest.groupby("subject_id")["split"].nunique().gt(1).any():
        raise RuntimeError("Written PCA-Cocycle manifest has duplicate scans or subject split leakage.")
    for column in ("hippocampus_mesh_path_mm", "lateral_ventricle_mesh_path_mm"):
        missing = [value for value in written_manifest[column].astype(str) if not Path(value).is_file()]
        if missing:
            raise FileNotFoundError(f"Written manifest has {len(missing)} unavailable paths in {column}.")
    split_subject_sets = {
        split: set(json.loads((splits_dir / f"{split}_subjects.json").read_text(encoding="utf-8")))
        for split in SPLITS
    }
    if set().union(*split_subject_sets.values()) != set(written_manifest["subject_id"].astype(str)):
        raise RuntimeError("Written split JSON files do not cover all manifest subjects.")
    if any(split_subject_sets[left] & split_subject_sets[right] for left in SPLITS for right in SPLITS if left < right):
        raise RuntimeError("Written split JSON files overlap in subjects.")

    pca_contract = {
        "name": "adni_synthseg_lhipp_llv_strict_no_mci_pca_cocycle_v4_input",
        "status": "input_bundle_only; PCA and model training have not run",
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "qc_keep_manifest": str(qc_manifest_path),
        "qc_keep_manifest_sha256": sha256_file(qc_manifest_path),
        "fit_split": "train",
        "split_policy": {
            "unit": "subject",
            "stratified_by": "stable baseline diagnosis (CN/AD)",
            "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
            "seed": int(args.seed),
            "subject_leakage": False,
        },
        "longitudinal_policy": {
            "minimum_visits": 2,
            "strictly_increasing_visit_month": True,
            "direct_CN_AD_changers_excluded": True,
            "strict_no_mci": True,
        },
        "representation_policy": {
            "coordinate_space": "final_ply_mm",
            "pca_fit": "separate PCA per structure, fit on training scans only",
            "why_not_raw_joint_vertex_pca": "the two structures use separate atlases and the ventricle has 8,346 vertices versus 2,746 for hippocampus; raw concatenation would over-weight ventricle coordinates",
            "candidate_max_components_per_structure": 150,
            "candidate_report_dimensions_per_structure": [32, 64, 75, 100, 128, 150],
            "initial_shared_cocycle_dimension": 150,
            "initial_block_allocation_after_pca": {"left_hippocampus": 75, "left_lateral_ventricle": 75},
            "shared_model_features": "selected PCA scores standardised using training-split mean and standard deviation within each structure block",
            "reconstruction_features": "unstandardised PCA scores and both PCA bases",
        },
        "structures": validation["topology"],
    }
    write_json(configs / "pca_cocycle_v4_input_contract.json", pca_contract)

    split_summary = {
        split: {
            "scans": int((scans["split"] == split).sum()),
            "subjects": int((subject_table["split"] == split).sum()),
            "scans_by_diagnosis": {
                str(key): int(value)
                for key, value in scans.loc[scans["split"] == split, "diagnosis"].value_counts().sort_index().items()
            },
            "subjects_by_diagnosis": {
                str(key): int(value)
                for key, value in subject_table.loc[subject_table["split"] == split, "diagnosis"].value_counts().sort_index().items()
            },
        }
        for split in SPLITS
    }
    report = {
        "passed": True,
        "source_meshes_modified": False,
        "pca_fitted": False,
        "model_trained": False,
        "manifest": str(manifest_path),
        "subject_assignments": str(subject_path),
        "counts": {"scans": int(len(scans)), "subjects": int(len(subject_table))},
        "split_summary": split_summary,
        "age_norm_train": {"minimum": age_min, "maximum": age_max},
        "checks": {
            "all_manifest_mesh_paths_exist": True,
            "both_structures_per_scan": True,
            "hard_mesh_qc_failures_in_manifest": 0,
            "duplicate_scan_ids": 0,
            "subjects_with_fewer_than_two_visits": 0,
            "duplicate_subject_time_points": 0,
            "nonpositive_adjacent_time_intervals": 0,
            "subject_split_leakage": False,
            "topology": validation["topology"],
        },
    }
    write_json(metadata / "input_validation.json", report)

    print(f"[2/3] Assigned {len(subject_table):,} subjects to subject-disjoint, diagnosis-stratified splits.", flush=True)
    print(f"[3/3] Wrote validated metadata bundle: {output}", flush=True)
    print(json.dumps({"counts": report["counts"], "splits": split_summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
