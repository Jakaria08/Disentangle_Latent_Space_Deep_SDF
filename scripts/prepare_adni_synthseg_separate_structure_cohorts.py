#!/usr/bin/env python3
"""Audit generated data and build independent hippocampus/LV PCA input cohorts.

This is the non-destructive preparation stage for the current SynthSeg
correspondence data.  It preserves all source meshes and QC reports, marks
older generated artifacts as active/superseded rather than deleting them, and
creates one master subject split plus separate hippocampus and left-lateral-
ventricle manifests.

No PCA, Cocycle, or Brain-ODE model is fitted by this script.
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
QC_ROOT = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI_synthseg_minimal_smooth_qc"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
DEFAULT_SOURCE_ROOT = Path("/home/jakaria/ADNI/ADNI_1_GO_Large/synthseg_minimal_correspondence/full")
STRUCTURES: dict[str, dict[str, Any]] = {
    "left_hippocampus": {
        "short_name": "hippocampus",
        "directory": "hippocampus_pca_cocycle_v4",
        "vertex_count": 2746,
        "face_count": 5488,
    },
    "left_lateral_ventricle": {
        "short_name": "lateral_ventricle",
        "directory": "lateral_ventricle_pca_cocycle_v4",
        "vertex_count": 8346,
        "face_count": 16688,
    },
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Refresh only the non-destructive generated-data audit; do not rebuild cohorts.",
    )
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


def ensure_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing generated output: {path}. "
            "Choose a new --output-root or preserve the existing result."
        )


def truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def split_sizes(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    if total < 3 or min(ratios) <= 0 or abs(sum(ratios) - 1.0) > 1.0e-8:
        raise ValueError("Need at least three subjects per diagnosis and positive split ratios that sum to one.")
    n_val = max(1, int(round(total * val_ratio)))
    n_test = max(1, int(round(total * test_ratio)))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Cannot create a non-empty train split for {total} subjects.")
    return n_train, n_val, n_test


def make_master_assignment(master: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> dict[str, str]:
    subjects = master.drop_duplicates("subject_id")[["subject_id", "baseline_diagnosis"]].copy()
    assignment: dict[str, str] = {}
    for diagnosis in ("CN", "AD"):
        ids = sorted(subjects.loc[subjects["baseline_diagnosis"] == diagnosis, "subject_id"].astype(str))
        random.Random(f"{seed}:{diagnosis}").shuffle(ids)
        n_train, n_val, _n_test = split_sizes(len(ids), ratios)
        assignment.update({subject: "train" for subject in ids[:n_train]})
        assignment.update({subject: "val" for subject in ids[n_train : n_train + n_val]})
        assignment.update({subject: "test" for subject in ids[n_train + n_val :]})
    if len(assignment) != len(subjects):
        raise RuntimeError("Could not assign every master-cohort subject to exactly one split.")
    return assignment


def generated_data_audit(output_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify generated data; never remove or modify a listed artifact."""
    legacy_left = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI_large_strict_left" / "task2_representations"
    legacy_pca = legacy_left / "pca"
    legacy_original = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI" / "brainode_comparison_task2_representations"
    previous_joint = output_root / "task2_pca_cocycle_v4"
    expected = [
        {
            "artifact": "prior left-hippocampus Task2 representations",
            "path": legacy_left,
            "scope": "generated PCA/INR data",
            "status": "superseded",
            "reason": "legacy single-structure source has 2,206 hippocampus vertices, not the current 2,746-vertex correspondence topology",
        },
        {
            "artifact": "prior left-hippocampus PCA outputs",
            "path": legacy_pca,
            "scope": "generated PCA model/coefficients",
            "status": "superseded",
            "reason": "PCA basis, scores, and train split belong to the legacy topology/source and cannot be reused",
        },
        {
            "artifact": "older Brain-ODE comparison PCA outputs",
            "path": legacy_original,
            "scope": "generated PCA/Brain-ODE inputs",
            "status": "superseded",
            "reason": "older ADNI preparation is not the current strict no-MCI structure-specific correspondence cohort",
        },
        {
            "artifact": "joint hippocampus+LV PCA-Cocycle-v4 input bundle",
            "path": previous_joint,
            "scope": "generated manifest/splits/configuration",
            "status": "superseded",
            "reason": "uses the joint 2,550-scan cohort and a combined 75+75 feature contract; the new plan has independent structure-specific cohorts and PCA models",
        },
        {
            "artifact": "new master cohort",
            "path": output_root / "cohort_master",
            "scope": "generated manifest/splits",
            "status": "rebuild",
            "reason": "required new shared identity split for structure-specific experiments",
        },
        {
            "artifact": "new hippocampus pipeline inputs",
            "path": output_root / STRUCTURES["left_hippocampus"]["directory"],
            "scope": "generated manifests/PCA inputs",
            "status": "rebuild",
            "reason": "requires current hippocampus-only QC scan set and master split",
        },
        {
            "artifact": "new LV pipeline inputs",
            "path": output_root / STRUCTURES["left_lateral_ventricle"]["directory"],
            "scope": "generated manifests/PCA inputs",
            "status": "rebuild",
            "reason": "requires current LV-only QC scan set and master split",
        },
    ]
    rows = []
    preserved = []
    for item in expected:
        path = Path(item["path"])
        exists = path.exists()
        status = item["status"]
        if status == "rebuild" and exists:
            status = "active" if path.name != "cohort_master" else "active"
        rows.append(
            {
                "artifact": item["artifact"],
                "path": str(path),
                "exists": exists,
                "scope": item["scope"],
                "status": status,
                "reason": item["reason"],
                "action": "preserve; do not reference" if status == "superseded" else "create/use for active pipeline",
            }
        )
        if status == "superseded" and exists:
            preserved.append(str(path))
    return rows, preserved


def build_master(records: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scans = records.drop_duplicates("scan_id", keep="first").copy()
    required = {
        "scan_id", "subject_id", "VISCODE", "visit_month", "age_years", "baseline_diagnosis", "visit_diagnosis", "strict_subject_no_mci",
    }
    missing = sorted(required.difference(scans.columns))
    if missing:
        raise KeyError(f"Input QC records lack: {missing}")
    if not truthy(scans["strict_subject_no_mci"]).all():
        raise ValueError("Strict no-MCI QC input unexpectedly contains a non-strict subject.")
    if not scans["baseline_diagnosis"].isin(("CN", "AD")).all():
        raise ValueError("Strict QC input unexpectedly contains a non-CN/AD baseline diagnosis.")
    visit_sets = scans.groupby("subject_id", sort=True)["visit_diagnosis"].agg(
        lambda values: sorted({str(value) for value in values if pd.notna(value)})
    )
    stable_subjects = set(visit_sets.loc[visit_sets.map(len).eq(1)].index.astype(str))
    stable = scans.loc[scans["subject_id"].astype(str).isin(stable_subjects)].copy()
    visits = stable.groupby("subject_id", sort=True)["scan_id"].nunique()
    master = stable.loc[stable["subject_id"].astype(str).isin(set(visits.loc[visits.ge(2)].index.astype(str)))].copy()
    master = master.sort_values(["subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)
    if master.duplicated(["subject_id", "visit_month"], keep=False).any():
        raise ValueError("Master cohort has duplicate visit months within a subject.")
    if any((group["visit_month"].diff().dropna() <= 0).any() for _, group in master.groupby("subject_id", sort=False)):
        raise ValueError("Master cohort has non-increasing longitudinal visits.")
    assignment = make_master_assignment(master, seed, ratios)
    master["split"] = master["subject_id"].astype(str).map(assignment)
    master["diagnosis"] = master["baseline_diagnosis"]
    master["label_ad"] = master["diagnosis"].map({"CN": 0, "AD": 1}).astype(int)
    master["master_visit_order"] = master.groupby("subject_id").cumcount().astype(int)
    master["months_from_baseline"] = master["visit_month"] - master.groupby("subject_id")["visit_month"].transform("min")
    subject_table = (
        master.groupby("subject_id", as_index=False)
        .agg(
            split=("split", "first"), diagnosis=("diagnosis", "first"),
            scan_count=("scan_id", "nunique"), baseline_age_years=("age_years", "min"),
        )
        .sort_values(["split", "diagnosis", "subject_id"], kind="stable")
        .reset_index(drop=True)
    )
    direct_changers = visit_sets.loc[~visit_sets.index.astype(str).isin(stable_subjects)].rename("observed_visit_diagnoses").reset_index()
    direct_changers["reason"] = "direct_CN_AD_diagnosis_change"
    return master, subject_table, direct_changers


def mesh_path(source_root: Path, structure: str, scan_id: str) -> Path:
    return source_root / structure / "minimal_smooth_correspondence" / "final_ply_mm" / f"{scan_id}.ply"


def add_structure_qc_fields(frame: pd.DataFrame, selected_qc: pd.DataFrame) -> pd.DataFrame:
    values = selected_qc.set_index("scan_id")
    output = frame.copy()
    for source, destination in (
        ("correspondence_volume_actual_mm3", "correspondence_volume_mm3"),
        ("correspondence_surface_area_actual_mm2", "correspondence_surface_area_mm2"),
        ("correspondence_topology_hash", "correspondence_topology_hash"),
    ):
        if source not in values.columns:
            raise KeyError(f"QC field missing: {source}")
        output[destination] = output["scan_id"].astype(str).map(values[source])
    return output


def structure_cohort(
    *,
    structure: str,
    master: pd.DataFrame,
    scan_qc: pd.DataFrame,
    pair_qc: pd.DataFrame,
    source_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = STRUCTURES[structure]
    candidate_ids = set(master["scan_id"].astype(str))
    candidate_subjects = set(master["subject_id"].astype(str))
    q = scan_qc.loc[
        (scan_qc["structure"] == structure) & scan_qc["scan_id"].astype(str).isin(candidate_ids)
    ].copy()
    if len(q) != len(candidate_ids):
        raise ValueError(f"Expected one {structure} QC row per master scan, found {len(q)} for {len(candidate_ids)} scans.")
    hard = q.loc[truthy(q["hard_mesh_qc_flag"]), ["scan_id"]].copy()
    hard["reason"] = "hard_mesh_geometry_or_correspondence_failure"

    pairs = pair_qc.loc[
        (pair_qc["structure"] == structure) & pair_qc["subject_id"].astype(str).isin(candidate_subjects)
    ].copy()
    strong = pairs.loc[truthy(pairs["strong_pair_qc_flag"])].copy()
    pair_rows: list[dict[str, str]] = []
    for row in strong.itertuples(index=False):
        for scan_id in (str(row.source_scan_id), str(row.target_scan_id)):
            if scan_id in candidate_ids:
                pair_rows.append({"scan_id": scan_id, "reason": "strong_adjacent_shape_or_extreme_volume_failure"})
    pair_exclusions = pd.DataFrame(pair_rows, columns=["scan_id", "reason"])
    reasons = pd.concat([hard, pair_exclusions], ignore_index=True)
    reasons = (
        reasons.groupby("scan_id", sort=True)["reason"].agg(lambda values: ";".join(sorted(set(values)))).rename("qc_exclusion_reasons").reset_index()
        if not reasons.empty
        else pd.DataFrame(columns=["scan_id", "qc_exclusion_reasons"])
    )
    excluded_ids = set(reasons["scan_id"].astype(str))
    pre_prune = master.loc[~master["scan_id"].astype(str).isin(excluded_ids)].copy()
    counts = pre_prune.groupby("subject_id", sort=True)["scan_id"].nunique()
    singleton_subjects = set(counts.loc[counts.lt(2)].index.astype(str))
    singleton_rows = pre_prune.loc[pre_prune["subject_id"].astype(str).isin(singleton_subjects)].copy()
    singleton_rows["qc_exclusion_reasons"] = "fewer_than_two_visits_after_structure_qc"
    final = pre_prune.loc[~pre_prune["subject_id"].astype(str).isin(singleton_subjects)].copy()
    final = final.sort_values(["split", "subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)
    final["visit_order"] = final.groupby("subject_id").cumcount().astype(int)
    final["months_from_baseline"] = final["visit_month"] - final.groupby("subject_id")["visit_month"].transform("min")
    train_age = final.loc[final["split"] == "train", "age_years"]
    age_min, age_max = float(train_age.min()), float(train_age.max())
    if age_max <= age_min:
        raise ValueError(f"Degenerate training age range for {structure}.")
    final["age_norm_train"] = (final["age_years"] - age_min) / (age_max - age_min)
    final["structure"] = structure
    final["mesh_path_mm"] = final["scan_id"].astype(str).map(lambda scan_id: str(mesh_path(source_root, structure, scan_id)))
    final["vertex_count"] = int(spec["vertex_count"])
    final["face_count"] = int(spec["face_count"])
    final = add_structure_qc_fields(final, q)

    excluded = reasons.merge(master, on="scan_id", how="left", validate="one_to_one")
    excluded = pd.concat([excluded, singleton_rows], ignore_index=True, sort=False)
    excluded = excluded.sort_values(["subject_id", "visit_month", "scan_id"], kind="stable").reset_index(drop=True)

    selected_q = q.loc[q["scan_id"].astype(str).isin(set(final["scan_id"].astype(str)))].copy()
    selected_pair_endpoints = set()
    for row in strong.itertuples(index=False):
        selected_pair_endpoints.add(str(row.source_scan_id))
        selected_pair_endpoints.add(str(row.target_scan_id))
    missing_paths = [path for path in final["mesh_path_mm"].astype(str) if not Path(path).is_file()]
    topology_hashes = sorted(selected_q["correspondence_topology_hash"].dropna().astype(str).unique())
    vertex_counts = sorted(pd.to_numeric(selected_q["correspondence_vertices_actual"], errors="coerce").dropna().unique())
    face_counts = sorted(pd.to_numeric(selected_q["correspondence_faces_actual"], errors="coerce").dropna().unique())
    failures: list[str] = []
    if missing_paths:
        failures.append("missing_physical_mm_correspondence_paths")
    if truthy(selected_q["hard_mesh_qc_flag"]).any():
        failures.append("hard_qc_failure_in_keep_manifest")
    if set(final["scan_id"].astype(str)) & selected_pair_endpoints:
        failures.append("strong_pair_endpoint_in_keep_manifest")
    if final.groupby("subject_id")["scan_id"].nunique().lt(2).any():
        failures.append("subject_with_fewer_than_two_visits")
    if final.duplicated(["subject_id", "visit_month"], keep=False).any():
        failures.append("duplicate_subject_visit_month")
    if any((group["visit_month"].diff().dropna() <= 0).any() for _, group in final.groupby("subject_id", sort=False)):
        failures.append("non_increasing_visit_time")
    if final.groupby("subject_id")["split"].nunique().gt(1).any():
        failures.append("subject_split_leakage")
    if len(topology_hashes) != 1 or vertex_counts != [spec["vertex_count"]] or face_counts != [spec["face_count"]]:
        failures.append("topology_contract_failure")
    if not final["visit_diagnosis"].eq(final["baseline_diagnosis"]).all():
        failures.append("diagnosis_not_stable")
    if not truthy(final["strict_subject_no_mci"]).all():
        failures.append("non_strict_no_mci_subject")
    summary = {
        "passed": not failures,
        "structure": structure,
        "counts": {
            "scans": int(final["scan_id"].nunique()),
            "subjects": int(final["subject_id"].nunique()),
            "automatic_excluded_scans": int(len(excluded)),
            "subjects_removed_from_master": int(master["subject_id"].nunique() - final["subject_id"].nunique()),
        },
        "split_counts": {
            split: {
                "scans": int((final["split"] == split).sum()),
                "subjects": int(final.loc[final["split"] == split, "subject_id"].nunique()),
                "subjects_by_diagnosis": {
                    str(key): int(value)
                    for key, value in final.loc[final["split"] == split].drop_duplicates("subject_id")["diagnosis"].value_counts().sort_index().items()
                },
            }
            for split in SPLITS
        },
        "automatic_reasons": {
            "hard_mesh_scans": int(len(hard)),
            "strong_pair_endpoint_scans": int(len(pair_exclusions.drop_duplicates("scan_id"))),
            "post_qc_singleton_scans": int(len(singleton_rows)),
            "strong_pairs": int(len(strong)),
        },
        "checks": {
            "all_mesh_paths_exist": not missing_paths,
            "hard_qc_failures_in_keep_manifest": int(truthy(selected_q["hard_mesh_qc_flag"]).sum()),
            "strong_pair_endpoints_in_keep_manifest": int(len(set(final["scan_id"].astype(str)) & selected_pair_endpoints)),
            "topology_hash": topology_hashes[0] if len(topology_hashes) == 1 else None,
            "vertex_counts": [int(value) for value in vertex_counts],
            "face_counts": [int(value) for value in face_counts],
            "age_norm_train_min": age_min,
            "age_norm_train_max": age_max,
        },
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"Structure cohort validation failed for {structure}: {failures}")
    return final, excluded, summary


def main() -> int:
    args = parse_args()
    qc_root = args.qc_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ratios = (float(args.train_ratio), float(args.val_ratio), float(args.test_ratio))
    records_path = qc_root / "mesh_qc" / "input_records.csv"
    scan_qc_path = qc_root / "mesh_qc" / "scan_qc.csv"
    pair_qc_path = qc_root / "mesh_qc" / "adjacent_pair_qc.csv"
    for path in (records_path, scan_qc_path, pair_qc_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required QC input missing: {path}")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if args.audit_only:
        audit_rows, protected_superseded = generated_data_audit(output_root)
        master_root = output_root / "cohort_master"
        audit_fields = ["artifact", "path", "exists", "scope", "status", "reason", "action"]
        write_csv(master_root / "reports" / "generated_data_audit.csv", audit_rows, audit_fields)
        write_json(
            master_root / "reports" / "generated_data_audit.json",
            {
                "source_meshes_modified": False,
                "note": "Superseded generated artifacts are retained on disk and excluded from active inputs.",
                "superseded_paths_preserved": protected_superseded,
                "artifacts": audit_rows,
            },
        )
        print(json.dumps({"audit_only": True, "status_counts": dict(Counter(row["status"] for row in audit_rows))}, indent=2))
        return 0
    for directory in (output_root / "cohort_master", *[output_root / spec["directory"] for spec in STRUCTURES.values()]):
        ensure_new_output(directory)

    print("=" * 88)
    print("Prepare independent hippocampus and LV cohorts (no PCA/model fitting)")
    print(f"QC source:  {qc_root}")
    print(f"Mesh source: {source_root} (read only)")
    print(f"Output root: {output_root}")
    print("=" * 88, flush=True)
    records = pd.read_csv(records_path, dtype={"scan_id": str, "subject_id": str, "RID": str, "VISCODE": str})
    scan_qc = pd.read_csv(scan_qc_path, dtype={"scan_id": str, "subject_id": str})
    pair_qc = pd.read_csv(pair_qc_path, dtype={"source_scan_id": str, "target_scan_id": str, "subject_id": str})

    print("[1/4] Auditing generated data; superseded items are preserved, not deleted…", flush=True)
    audit_rows, protected_superseded = generated_data_audit(output_root)
    master_root = output_root / "cohort_master"
    audit_fields = ["artifact", "path", "exists", "scope", "status", "reason", "action"]
    write_csv(master_root / "reports" / "generated_data_audit.csv", audit_rows, audit_fields)
    write_json(
        master_root / "reports" / "generated_data_audit.json",
        {
            "source_meshes_modified": False,
            "note": "Superseded generated artifacts are retained on disk and excluded from active inputs.",
            "superseded_paths_preserved": protected_superseded,
            "artifacts": audit_rows,
        },
    )

    print("[2/4] Building strict no-MCI, diagnosis-stable master subject split…", flush=True)
    master, subject_table, direct_changers = build_master(records, int(args.seed), ratios)
    master_manifest = master_root / "metadata" / "master_strict_no_mci_stable_min2_manifest.csv"
    master_fields = [
        "scan_id", "subject_id", "split", "diagnosis", "label_ad", "baseline_diagnosis", "visit_diagnosis", "VISCODE",
        "visit_month", "months_from_baseline", "master_visit_order", "age_years", "sex", "strict_subject_no_mci",
    ]
    write_csv(master_manifest, master.loc[:, master_fields].to_dict("records"), master_fields)
    subject_fields = ["subject_id", "split", "diagnosis", "scan_count", "baseline_age_years"]
    write_csv(master_root / "metadata" / "master_subject_split_assignments.csv", subject_table.to_dict("records"), subject_fields)
    write_csv(
        master_root / "metadata" / "excluded_direct_CN_AD_changers.csv",
        direct_changers.to_dict("records"),
        list(direct_changers.columns),
    )
    for split in SPLITS:
        write_json(
            master_root / "splits" / f"{split}_subjects.json",
            subject_table.loc[subject_table["split"] == split, "subject_id"].astype(str).tolist(),
        )
    master_report = {
        "passed": True,
        "source_meshes_modified": False,
        "master_manifest": str(master_manifest),
        "master_manifest_sha256": sha256_file(master_manifest),
        "policy": "strict no-MCI, stable CN/AD diagnosis, minimum two visits, subject-level diagnosis-stratified split",
        "seed": int(args.seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {
            "scans": int(len(master)),
            "subjects": int(master["subject_id"].nunique()),
            "direct_CN_AD_changers_excluded": int(len(direct_changers)),
            "split_subjects": {split: int((subject_table["split"] == split).sum()) for split in SPLITS},
        },
        "subject_counts_by_split_diagnosis": {
            split: {
                str(key): int(value)
                for key, value in subject_table.loc[subject_table["split"] == split, "diagnosis"].value_counts().sort_index().items()
            }
            for split in SPLITS
        },
        "checks": {"subject_split_leakage": False, "minimum_visits": 2, "strict_no_mci": True},
    }
    write_json(master_root / "reports" / "master_cohort_validation.json", master_report)

    structure_summaries: dict[str, Any] = {}
    for step, structure in enumerate(STRUCTURES, start=3):
        spec = STRUCTURES[structure]
        print(f"[{step}/4] Building and validating {structure}-only cohort…", flush=True)
        final, excluded, summary = structure_cohort(
            structure=structure, master=master, scan_qc=scan_qc, pair_qc=pair_qc, source_root=source_root
        )
        target = output_root / spec["directory"]
        manifest = target / "metadata" / f"{spec['short_name']}_qc_keep_manifest.csv"
        manifest_fields = [
            "scan_id", "subject_id", "split", "diagnosis", "label_ad", "baseline_diagnosis", "visit_diagnosis", "VISCODE",
            "visit_month", "months_from_baseline", "visit_order", "age_years", "age_norm_train", "sex", "strict_subject_no_mci",
            "structure", "mesh_path_mm", "vertex_count", "face_count", "correspondence_volume_mm3", "correspondence_surface_area_mm2", "correspondence_topology_hash",
        ]
        write_csv(manifest, final.loc[:, manifest_fields].to_dict("records"), manifest_fields)
        excluded_fields = [field for field in ["scan_id", "subject_id", "split", "VISCODE", "visit_month", "baseline_diagnosis", "visit_diagnosis", "qc_exclusion_reasons"] if field in excluded]
        write_csv(target / "metadata" / f"{spec['short_name']}_auto_excluded_scans.csv", excluded.loc[:, excluded_fields].to_dict("records"), excluded_fields)
        for split in SPLITS:
            write_json(
                target / "splits" / f"{split}_subjects.json",
                sorted(final.loc[final["split"] == split, "subject_id"].astype(str).unique().tolist()),
            )
        summary["manifest"] = str(manifest)
        summary["manifest_sha256"] = sha256_file(manifest)
        summary["pca_policy"] = {
            "fit_split": "train",
            "coordinate_space": "final_ply_mm",
            "independent_structure_pca": True,
            "candidate_components": [32, 64, 100, 128, 150],
            "max_components": 150,
            "no_cross_structure_features": True,
        }
        write_json(target / "reports" / "input_validation.json", summary)
        write_json(target / "configs" / "pca_input_contract.json", {
            "name": f"adni_synthseg_{spec['short_name']}_strict_no_mci_pca_input",
            "status": "PCA input prepared; PCA/Cocycle/Brain-ODE not fitted",
            "master_manifest": str(master_manifest),
            "master_manifest_sha256": sha256_file(master_manifest),
            "structure_manifest": str(manifest),
            "structure_manifest_sha256": sha256_file(manifest),
            "structure": structure,
            "coordinate_space": "final_ply_mm",
            "fit_split": "train",
            "max_components": 150,
            "candidate_components": [32, 64, 100, 128, 150],
            "topology": {
                "vertex_count": spec["vertex_count"],
                "face_count": spec["face_count"],
                "correspondence_topology_hash": summary["checks"]["topology_hash"],
            },
            "source_meshes_modified": False,
        })
        structure_summaries[structure] = summary

    print("\nPreparation complete. No source mesh, QC report, PCA, or neural model was modified.")
    print(json.dumps({"master": master_report["counts"], "structures": {name: summary["counts"] for name, summary in structure_summaries.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
