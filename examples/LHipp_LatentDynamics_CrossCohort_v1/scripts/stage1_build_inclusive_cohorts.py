#!/usr/bin/env python3
"""Stage 1a: converter-keeping (inclusive) AIBL and OASIS hippocampus cohorts.

The strict cohorts keep only diagnosis-stable CN/AD subjects. The converter line (stage 5)
also needs CN->AD, MCI->AD, CN->MCI, stable-MCI and reverting subjects, with their per-visit
labels and conversion windows. This script builds those cohorts from the already-computed
inclusive QC (``--cohort-filter all --keep-diagnosis-changers``) using the *same* structure
QC rules as the strict builder: hard mesh failures are excluded, strong adjacent-pair
failures exclude the culprit endpoint, and subjects need two visits afterwards.

Splits: a subject that is in the strict cohort keeps its strict split, so a model trained on
strict train never sees an inclusive test subject. New subjects are split by the same seed,
stratified by trajectory group.

Nothing is written outside the experiment's bulk root; nothing in the source trees changes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import benchmark_common as bc

STRUCTURE = "left_hippocampus"


def cohort_builder():
    """The strict cohort builder, imported (not copied) so the QC rules cannot drift."""
    path = bc.REPO_ROOT / "scripts" / "prepare_adni_synthseg_separate_structure_cohorts.py"
    spec = importlib.util.spec_from_file_location("strict_cohort_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strict_cohort_builder"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="+", default=None, help="Default: every cohort with an inclusive QC root.")
    parser.add_argument("--pair-exclusion-policy", choices=("both_endpoints", "culprit"), default="culprit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_inclusive(cohort: str, sources: dict[str, Any], builder, policy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = sources["cohorts"][cohort]
    qc_root = bc.resolve(spec["inclusive_qc_root"])
    records = pd.read_csv(qc_root / "mesh_qc" / "input_records.csv",
                          dtype={"scan_id": str, "subject_id": str, "RID": str, "VISCODE": str}, low_memory=False)
    scan_qc = pd.read_csv(qc_root / "mesh_qc" / "scan_qc.csv", dtype={"scan_id": str, "subject_id": str}, low_memory=False)
    pair_qc = pd.read_csv(qc_root / "mesh_qc" / "adjacent_pair_qc.csv",
                          dtype={"source_scan_id": str, "target_scan_id": str, "subject_id": str}, low_memory=False)
    report: dict[str, Any] = {"cohort": cohort, "qc_root": str(qc_root), "pair_exclusion_policy": policy}

    scans = records.drop_duplicates("scan_id", keep="first").copy()
    report["input_scans"] = int(len(scans))
    labelled = scans["visit_diagnosis"].isin(bc.LABEL_ORDER)
    report["dropped_unlabelled_scans"] = int((~labelled).sum())
    scans = scans.loc[labelled].copy()
    scans = scans.sort_values(["subject_id", "visit_month", "scan_id"], kind="stable")
    duplicates = scans.duplicated(["subject_id", "visit_month"], keep="first")
    report["dropped_same_month_repeat_scans"] = sorted(scans.loc[duplicates, "scan_id"].astype(str))
    scans = scans.loc[~duplicates].copy()
    counts = scans.groupby("subject_id")["scan_id"].transform("nunique")
    scans = scans.loc[counts >= 2].copy()
    candidate_ids = set(scans["scan_id"].astype(str))
    candidate_subjects = set(scans["subject_id"].astype(str))

    # Structure QC, rule for rule as in the strict builder's structure_cohort().
    q = scan_qc.loc[(scan_qc["structure"] == STRUCTURE) & scan_qc["scan_id"].astype(str).isin(candidate_ids)].copy()
    if len(q) != len(candidate_ids):
        raise ValueError(f"{cohort}: expected one {STRUCTURE} QC row per candidate scan, got {len(q)} for {len(candidate_ids)}")
    hard = set(q.loc[builder.truthy(q["hard_mesh_qc_flag"]), "scan_id"].astype(str))
    pairs = pair_qc.loc[(pair_qc["structure"] == STRUCTURE) & pair_qc["subject_id"].astype(str).isin(candidate_subjects)]
    strong = pairs.loc[builder.truthy(pairs["strong_pair_qc_flag"])].copy()
    blamed = builder.blamed_pair_endpoints(strong, q, policy)
    pair_excluded = {scan for chosen in blamed.values() for scan in chosen if scan in candidate_ids}
    excluded = hard | pair_excluded
    report["excluded_hard_mesh_scans"] = int(len(hard))
    report["excluded_strong_pair_scans"] = int(len(pair_excluded - hard))
    final = scans.loc[~scans["scan_id"].astype(str).isin(excluded)].copy()
    visits = final.groupby("subject_id")["scan_id"].transform("nunique")
    report["excluded_singleton_after_qc_scans"] = int((visits < 2).sum())
    final = final.loc[visits >= 2].copy()
    final = final.sort_values(["subject_id", "visit_month", "scan_id"], kind="stable").reset_index(drop=True)

    final["cohort"] = cohort
    final["subject_key"] = [bc.qualify(cohort, value) for value in final["subject_id"]]
    final["scan_key"] = [bc.qualify(cohort, value) for value in final["scan_id"]]
    final["visit_label"] = final["visit_diagnosis"].astype(str)
    final["visit_order"] = final.groupby("subject_id").cumcount().astype(int)
    final["months_from_baseline"] = final["visit_month"] - final.groupby("subject_id")["visit_month"].transform("min")
    final["years_from_baseline"] = final["months_from_baseline"] / 12.0
    final["mesh_path_mm"] = [str(builder.mesh_path(Path(spec["mesh_root"]), STRUCTURE, scan)) for scan in final["scan_id"]]
    final["vertex_count"] = int(builder.STRUCTURES[STRUCTURE]["vertex_count"])
    final["face_count"] = int(builder.STRUCTURES[STRUCTURE]["face_count"])
    final = builder.add_structure_qc_fields(final, q)

    subject_rows = []
    for subject_key, group in final.groupby("subject_key", sort=True):
        labels = group["visit_label"].tolist()
        times = group["years_from_baseline"].tolist()
        ad_a, ad_b = bc.conversion_window(times, labels, "AD")
        mci_a, mci_b = bc.conversion_window(times, labels, "MCI")
        subject_rows.append({
            "subject_key": subject_key,
            "baseline_label": labels[0],
            "last_label": labels[-1],
            "trajectory_group": bc.trajectory_group(labels),
            "conv_to_ad_window_a_years": ad_a,
            "conv_to_ad_window_b_years": ad_b,
            "conv_to_mci_window_a_years": mci_a,
            "conv_to_mci_window_b_years": mci_b,
        })
    subjects = pd.DataFrame(subject_rows)

    strict = bc.read_strict_manifest(cohort, sources)
    strict_split = strict.drop_duplicates("subject_key").set_index("subject_key")["split"].to_dict()
    strict_groups = subjects.loc[subjects["subject_key"].isin(strict_split), "trajectory_group"]
    unexpected = sorted(set(strict_groups).difference({"CN-stable", "AD-stable"}))
    if unexpected:
        raise ValueError(f"{cohort}: strict subjects have non-stable inclusive trajectories {unexpected}")
    new = subjects.loc[~subjects["subject_key"].isin(strict_split)]
    ratios = tuple(float(sources["split_ratios"][split]) for split in bc.SPLITS)
    new_split = bc.stratified_split(dict(zip(new["subject_key"], new["trajectory_group"])), int(sources["seed"]), ratios, f"{cohort}:inclusive")
    subjects["split"] = subjects["subject_key"].map({**strict_split, **new_split})
    subjects["in_strict_cohort"] = subjects["subject_key"].isin(strict_split)
    if subjects["split"].isna().any():
        raise RuntimeError(f"{cohort}: unassigned inclusive subjects")
    final = final.merge(subjects, on="subject_key", how="left", validate="many_to_one")
    final["in_strict_manifest"] = final["scan_key"].isin(set(strict["scan_key"]))
    final["label"] = (final["visit_label"] == "AD").astype(int)

    strict_scans = set(strict["scan_key"])
    kept = set(final["scan_key"])
    report["strict_scan_coverage"] = {
        "strict_scans": len(strict_scans),
        "strict_scans_kept_inclusively": len(strict_scans & kept),
        "strict_scans_missing": sorted(strict_scans - kept)[:50],
        "strict_scans_missing_count": len(strict_scans - kept),
    }
    report["counts"] = {
        "scans": int(len(final)),
        "subjects": int(final["subject_key"].nunique()),
        "subjects_by_group": subjects["trajectory_group"].value_counts().sort_index().to_dict(),
        "subjects_by_split_group": {
            split: subjects.loc[subjects["split"] == split, "trajectory_group"].value_counts().sort_index().to_dict()
            for split in bc.SPLITS
        },
        "new_subjects_beyond_strict": int((~subjects["in_strict_cohort"]).sum()),
    }
    return final, report


def validate(final: pd.DataFrame, cohort: str, sources: dict[str, Any]) -> list[str]:
    failures = []
    topology = sources["topology"]
    if set(final["correspondence_topology_hash"].astype(str)) != {topology["correspondence_topology_hash"]}:
        failures.append("topology_hash")
    if not all(Path(path).is_file() for path in final["mesh_path_mm"]):
        failures.append("missing_mesh_paths")
    try:
        bc.validate_longitudinal_frame(final, f"{cohort} inclusive", require_stable_label=False)
    except ValueError as error:
        failures.append(str(error))
    windows = final.drop_duplicates("subject_key")
    converters = windows["trajectory_group"].isin(["CN->AD", "MCI->AD"])
    if windows.loc[converters, "conv_to_ad_window_b_years"].isna().any():
        failures.append("converter_without_ad_window")
    if (windows["conv_to_ad_window_a_years"] >= windows["conv_to_ad_window_b_years"]).any():
        failures.append("non_increasing_conversion_window")
    return failures


COLUMNS = [
    "cohort", "subject_key", "scan_key", "subject_id", "scan_id", "split", "VISCODE", "visit_month", "months_from_baseline",
    "years_from_baseline", "visit_order", "age_years", "sex", "visit_label", "label", "baseline_label", "last_label",
    "trajectory_group", "conv_to_ad_window_a_years", "conv_to_ad_window_b_years", "conv_to_mci_window_a_years",
    "conv_to_mci_window_b_years", "in_strict_cohort", "in_strict_manifest", "mesh_path_mm", "vertex_count", "face_count",
    "correspondence_volume_mm3", "correspondence_surface_area_mm2", "correspondence_topology_hash",
]


def main() -> int:
    args = parse_args()
    sources = bc.load_cohort_sources()
    builder = cohort_builder()
    names = args.cohorts or [name for name, spec in sources["cohorts"].items() if spec.get("inclusive_qc_root")]
    summary = {}
    for cohort in names:
        destination = bc.require_bulk(bc.STAGE1_ROOT / "cohorts" / cohort)
        manifest_path = destination / "inclusive_manifest.csv"
        if manifest_path.exists() and not args.overwrite:
            print(f"[{cohort}] inclusive manifest exists, skipping (pass --overwrite to rebuild): {manifest_path}", flush=True)
            continue
        final, report = build_inclusive(cohort, sources, builder, args.pair_exclusion_policy)
        failures = validate(final, cohort, sources)
        report["failures"] = failures
        report["passed"] = not failures
        if failures:
            print(json.dumps(report, indent=2, default=str))
            raise RuntimeError(f"{cohort}: inclusive cohort validation failed: {failures}")
        bc.atomic_csv(manifest_path, final.loc[:, COLUMNS])
        report["manifest"] = str(manifest_path)
        report["manifest_sha256"] = bc.sha256_file(manifest_path)
        bc.atomic_json(destination / "inclusive_report.json", report)
        summary[cohort] = report["counts"]
        print(f"[{cohort}] {report['counts']['scans']} scans / {report['counts']['subjects']} subjects; "
              f"groups {report['counts']['subjects_by_group']}; strict scans kept "
              f"{report['strict_scan_coverage']['strict_scans_kept_inclusively']}/{report['strict_scan_coverage']['strict_scans']}",
              flush=True)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
