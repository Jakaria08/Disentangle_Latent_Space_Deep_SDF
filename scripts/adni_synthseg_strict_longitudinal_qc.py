#!/usr/bin/env python3
"""Create a strict no-MCI, longitudinally consistent SynthSeg QC cohort.

This is a *read-only* QC driver for the two current correspondence structures:
left hippocampus and left lateral ventricle.  It first runs the mesh/topology
and adjacent-pair QC, then writes recommendation manifests.  It never removes,
renames, or edits a source mesh.

The recommended automatic cohort is deliberately conservative but not
over-aggressive:

* retain only baseline-CN or baseline-AD people with ``strict_subject_no_mci``;
* remove the few people whose available visits change directly between CN and AD;
* require at least two usable longitudinal visits;
* remove a scan if either final smooth/correspondence mesh has a hard geometry
  failure, or if it is an endpoint of a strong correspondence-shape failure;
* re-apply the two-visit requirement after scan removal.

Statistical volume/compactness outliers and ordinary (non-strong) trajectory
warnings are reported separately, rather than automatically excluded: those can
represent real atrophy or ventricular enlargement.  This preserves a useful,
less-biased cohort while still excluding objectively unusable meshes/shapes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE = Path("/home/jakaria/ADNI/ADNI_1_GO_Large/synthseg_minimal_correspondence/full")
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "examples" / "ADNI_1_L_No_MCI_synthseg_minimal_smooth_qc"
DEFAULT_MESH_QC_SCRIPT = HERE.parent / "examples" / "ADNI_1_L_With_MCI" / "synthseg_longitudinal_qc.py"
STRUCTURES = ("left_hippocampus", "left_lateral_ventricle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mesh-qc-script",
        type=Path,
        default=DEFAULT_MESH_QC_SCRIPT,
        help="Existing geometry/pair-QC implementation to run before cohort finalisation.",
    )
    parser.add_argument(
        "--skip-mesh-qc",
        action="store_true",
        help="Reuse output-dir/mesh_qc from a completed run.  Normal use should omit this.",
    )
    parser.add_argument(
        "--cohort-filter",
        choices=("strict_no_mci", "all"),
        default="strict_no_mci",
        help=(
            "Cohort selection passed to the mesh QC.  'strict_no_mci' keeps baseline-CN/AD subjects "
            "with no MCI-labelled visit, as for ADNI.  'all' keeps whatever the manifest contains, "
            "which is required for a cohort outside the CN/MCI/AD axis such as CALSNIC (Control/ALS)."
        ),
    )
    parser.add_argument(
        "--keep-diagnosis-changers",
        action="store_true",
        help="Retain subjects whose visit diagnosis changes, instead of excluding them as ADNI does.",
    )
    parser.add_argument(
        "--pair-exclusion-policy",
        choices=("both_endpoints", "culprit"),
        default="both_endpoints",
        help=(
            "Which scans a strong adjacent-pair failure excludes.  'both_endpoints' removes both "
            "scans, which discards a good scan whenever only one of the two is at fault.  'culprit' "
            "removes the scan shared by two consecutive failing pairs, falling back to whichever "
            "endpoint independently carries a scan-level review flag, and to the later scan when "
            "neither does."
        ),
    )
    parser.add_argument(
        "--multi-component-volume-tolerance-pct",
        type=float,
        default=0.0,
        help="Forwarded to the mesh QC: extra components below this volume share are review-only.",
    )
    parser.add_argument(
        "--pair-outlier-method",
        choices=("quantile", "stratified_mad"),
        default="quantile",
        help="Forwarded to the mesh QC: how adjacent-pair shape outliers are chosen.",
    )
    parser.add_argument(
        "--pair-outlier-mad-multiplier",
        type=float,
        default=8.0,
        help="Forwarded to the mesh QC when --pair-outlier-method=stratified_mad.",
    )
    return parser.parse_args()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read an optional bool CSV column safely (including textual booleans)."""
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].map(_truthy)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {path.name}: {len(frame):,} rows", flush=True)


def _scan_table(records: pd.DataFrame) -> pd.DataFrame:
    """Return one stable source-manifest row per scan and validate the duplication."""
    required = {"scan_id", "subject_id", "VISCODE", "visit_month", "baseline_diagnosis", "visit_diagnosis"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise KeyError(f"QC input records are missing required columns: {missing}")
    unique = records.drop_duplicates("scan_id", keep="first").copy()
    expected = set(STRUCTURES)
    found = records.groupby("scan_id")["structure"].agg(lambda s: set(s.astype(str)))
    bad = found.loc[found.map(lambda structures: structures != expected)]
    if not bad.empty:
        examples = ", ".join(map(str, bad.index[:5]))
        raise RuntimeError(
            "Each retained scan must have both structures for a joint-shape cohort; "
            f"failed for {len(bad)} scans (e.g. {examples})."
        )
    return unique.sort_values(["subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)


def _stable_diagnosis_scans(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove direct CN<->AD changers; MCI is already excluded by source strict flag."""
    scan_labels = scans[["subject_id", "visit_diagnosis"]].copy()
    labels = scan_labels.groupby("subject_id", sort=True)["visit_diagnosis"].agg(
        lambda values: sorted(set(str(v) for v in values if pd.notna(v)))
    )
    stable_ids = labels.loc[labels.map(lambda values: len(values) == 1)].index.astype(str)
    excluded = labels.loc[~labels.index.astype(str).isin(set(stable_ids))].reset_index(name="observed_visit_diagnoses")
    excluded["reason"] = "direct_CN_AD_diagnosis_change"
    return scans.loc[scans["subject_id"].astype(str).isin(set(stable_ids))].copy(), excluded


def _at_least_two_visits(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = scans.groupby("subject_id", sort=True)["scan_id"].nunique()
    eligible = set(count.loc[count.ge(2)].index.astype(str))
    dropped = scans.loc[~scans["subject_id"].astype(str).isin(eligible)].copy()
    return scans.loc[scans["subject_id"].astype(str).isin(eligible)].copy(), dropped


def _blamed_endpoints(
    strong_pairs: pd.DataFrame, scan_qc: pd.DataFrame, policy: str
) -> dict[int, tuple[str, ...]]:
    """Decide which scan of each failing pair is actually at fault.

    A pair failure says the transition is implausible, not which of the two scans caused
    it.  A scan shared by two consecutive failing pairs is the common factor, and failing
    that, an endpoint that independently trips a scan-level review flag is the better
    suspect than its neighbour.
    """

    endpoints = [
        (index, str(row.structure), str(row.source_scan_id), str(row.target_scan_id))
        for index, row in enumerate(strong_pairs.itertuples(index=False))
    ]
    if policy == "both_endpoints":
        return {index: (source, target) for index, _structure, source, target in endpoints}

    appearances: Counter[tuple[str, str]] = Counter()
    for _index, structure, source, target in endpoints:
        appearances[(structure, source)] += 1
        appearances[(structure, target)] += 1
    reviewed = {
        (str(row.structure), str(row.scan_id))
        for row in scan_qc.loc[_flag(scan_qc, "review_mesh_qc_flag")].itertuples(index=False)
    }
    blamed: dict[int, tuple[str, ...]] = {}
    for index, structure, source, target in endpoints:
        shared = [scan for scan in (source, target) if appearances[(structure, scan)] > 1]
        if shared:
            blamed[index] = tuple(dict.fromkeys(shared))
            continue
        suspect = [scan for scan in (source, target) if (structure, scan) in reviewed]
        blamed[index] = (suspect[0],) if len(suspect) == 1 else (target,)
    return blamed


def _reasons_by_scan(
    scan_qc: pd.DataFrame, pair_qc: pd.DataFrame, policy: str = "both_endpoints"
) -> pd.DataFrame:
    """Combine hard geometry flags and objective strong pair flags across structures."""
    hard = scan_qc.loc[_flag(scan_qc, "hard_mesh_qc_flag"), ["scan_id", "structure"]].copy()
    hard["reason"] = "hard_mesh_geometry_or_correspondence_failure"

    strong = _flag(pair_qc, "strong_pair_qc_flag")
    strong_pairs = pair_qc.loc[strong].copy()
    blamed = _blamed_endpoints(strong_pairs, scan_qc, policy)
    pair_rows: list[dict[str, str]] = []
    for position, row in enumerate(strong_pairs.itertuples(index=False)):
        pair_reasons: list[str] = []
        if _truthy(getattr(row, "flag_shape_rms_displacement_mm_per_year_outlier", False)) or _truthy(
            getattr(row, "flag_shape_p95_displacement_mm_per_year_outlier", False)
        ):
            pair_reasons.append("strong_adjacent_correspondence_shape_outlier")
        if _truthy(getattr(row, "flag_extreme_volume_jump", False)):
            pair_reasons.append("extreme_adjacent_volume_jump")
        if _truthy(getattr(row, "flag_raw_smooth_sign_disagreement", False)):
            pair_reasons.append("raw_smooth_volume_change_sign_disagreement")
        if not pair_reasons:
            pair_reasons.append("strong_adjacent_pair_qc_failure")
        for scan_id in blamed[position]:
            for reason in pair_reasons:
                pair_rows.append({"scan_id": scan_id, "structure": str(row.structure), "reason": reason})
    pair = pd.DataFrame(pair_rows, columns=["scan_id", "structure", "reason"])
    reasons = pd.concat([hard, pair], ignore_index=True)
    if reasons.empty:
        return pd.DataFrame(columns=["scan_id", "qc_exclusion_reasons"])
    return (
        reasons.groupby("scan_id", sort=True)["reason"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("qc_exclusion_reasons")
        .reset_index()
    )


def _warning_tables(scan_qc: pd.DataFrame, pair_qc: pd.DataFrame, keep_scan_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write non-exclusion warnings only for otherwise kept scans/pairs."""
    scan_warning = _flag(scan_qc, "review_mesh_qc_flag") & ~_flag(scan_qc, "hard_mesh_qc_flag")
    warnings_scan = scan_qc.loc[scan_warning & scan_qc["scan_id"].astype(str).isin(keep_scan_ids)].copy()
    warnings_pair = pair_qc.loc[
        _flag(pair_qc, "any_pair_qc_flag") & ~_flag(pair_qc, "strong_pair_qc_flag")
    ].copy()
    if not warnings_pair.empty:
        warnings_pair = warnings_pair.loc[
            warnings_pair["source_scan_id"].astype(str).isin(keep_scan_ids)
            & warnings_pair["target_scan_id"].astype(str).isin(keep_scan_ids)
        ].copy()
    return warnings_scan, warnings_pair


def _summary_counts(scans: pd.DataFrame) -> dict[str, object]:
    return {
        "scans": int(scans["scan_id"].nunique()),
        "subjects": int(scans["subject_id"].nunique()),
        "scans_by_baseline_group": {
            str(key): int(value)
            for key, value in scans["baseline_diagnosis"].value_counts(dropna=False).sort_index().items()
        },
        "subjects_by_baseline_group": {
            str(key): int(value)
            for key, value in (
                scans.drop_duplicates("subject_id")["baseline_diagnosis"].value_counts(dropna=False).sort_index().items()
            )
        },
    }


def _run_mesh_qc(args: argparse.Namespace, mesh_qc_dir: Path) -> None:
    script = args.mesh_qc_script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Mesh QC script not found: {script}")
    command = [
        sys.executable,
        str(script),
        "--input-root",
        str(args.source_root.expanduser().resolve()),
        "--output-dir",
        str(mesh_qc_dir),
        "--structures",
        ",".join(STRUCTURES),
        "--cohort-filter",
        args.cohort_filter,
        "--cohort-label",
        "baseline",
        "--no-html",
        "--multi-component-volume-tolerance-pct",
        str(args.multi_component_volume_tolerance_pct),
        "--pair-outlier-method",
        args.pair_outlier_method,
        "--pair-outlier-mad-multiplier",
        str(args.pair_outlier_mad_multiplier),
    ]
    print("\n[1/3] Running mesh, topology, and adjacent-pair QC (source meshes are read only)…", flush=True)
    print("      " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    mesh_qc_dir = output / "mesh_qc"
    manifests_dir = output / "manifests"
    reports_dir = output / "reports"
    if not (source / "manifests" / "selected_scans.csv").is_file():
        raise FileNotFoundError(f"Missing selected-scans manifest under source root: {source}")

    print("=" * 88)
    print("ADNI SynthSeg strict no-MCI longitudinal QC")
    print(f"Source meshes (read-only): {source}")
    print(f"QC reports/manifests:       {output}")
    print("Policy: strict no-MCI + stable CN/AD diagnosis + >=2 visits + hard-shape exclusions")
    print("No source mesh will be edited, renamed, or deleted.")
    print("=" * 88, flush=True)
    if not args.skip_mesh_qc:
        _run_mesh_qc(args, mesh_qc_dir)
    else:
        print("\n[1/3] Reusing existing mesh QC output (--skip-mesh-qc).", flush=True)

    input_records_path = mesh_qc_dir / "input_records.csv"
    scan_qc_path = mesh_qc_dir / "scan_qc.csv"
    pair_qc_path = mesh_qc_dir / "adjacent_pair_qc.csv"
    for path in (input_records_path, scan_qc_path, pair_qc_path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected QC result is missing: {path}")

    print("\n[2/3] Building strict, diagnosis-stable longitudinal manifest…", flush=True)
    records = pd.read_csv(input_records_path, dtype={"scan_id": str, "subject_id": str, "RID": str, "VISCODE": str})
    scan_qc = pd.read_csv(scan_qc_path, dtype={"scan_id": str, "subject_id": str})
    pair_qc = pd.read_csv(pair_qc_path, dtype={"source_scan_id": str, "target_scan_id": str, "subject_id": str})
    scans = _scan_table(records)

    if args.keep_diagnosis_changers:
        stable_scans = scans.copy()
        direct_changers = pd.DataFrame(columns=["subject_id", "observed_visit_diagnoses", "reason"])
    else:
        stable_scans, direct_changers = _stable_diagnosis_scans(scans)
    longitudinal_pre_qc, singleton_before_qc = _at_least_two_visits(stable_scans)
    _write_csv(stable_scans, manifests_dir / "strict_no_mci_diagnosis_stable_all_scans.csv")
    _write_csv(direct_changers, manifests_dir / "excluded_direct_CN_AD_changers.csv")
    _write_csv(singleton_before_qc, manifests_dir / "excluded_single_visit_before_mesh_qc.csv")

    candidate_ids = set(longitudinal_pre_qc["scan_id"].astype(str))
    reasons = _reasons_by_scan(scan_qc, pair_qc, args.pair_exclusion_policy)
    automatic_exclusions = reasons.loc[reasons["scan_id"].astype(str).isin(candidate_ids)].copy()
    automatic_exclusions = automatic_exclusions.merge(
        longitudinal_pre_qc.drop_duplicates("scan_id"), on="scan_id", how="left", validate="one_to_one"
    )
    accepted_before_final_visit_check = longitudinal_pre_qc.loc[
        ~longitudinal_pre_qc["scan_id"].astype(str).isin(set(automatic_exclusions["scan_id"].astype(str)))
    ].copy()
    final_keep, singleton_after_qc = _at_least_two_visits(accepted_before_final_visit_check)

    excluded_after_qc = singleton_after_qc.copy()
    excluded_after_qc["qc_exclusion_reasons"] = "fewer_than_two_visits_after_automatic_qc"
    automatic_exclusions = pd.concat([automatic_exclusions, excluded_after_qc], ignore_index=True, sort=False)
    automatic_exclusions = automatic_exclusions.sort_values(
        ["subject_id", "visit_month", "scan_id"], kind="stable"
    ).reset_index(drop=True)

    final_keep = final_keep.sort_values(["subject_id", "visit_month", "VISCODE", "scan_id"], kind="stable").reset_index(drop=True)
    final_keep["qc_cohort_action"] = "keep"
    final_keep["qc_policy"] = "strict_no_mci_stable_longitudinal_conservative"
    _write_csv(automatic_exclusions, manifests_dir / "auto_excluded_scans_conservative.csv")
    _write_csv(final_keep, manifests_dir / "strict_no_mci_longitudinal_keep_conservative.csv")

    keep_ids = set(final_keep["scan_id"].astype(str))
    warning_scans, warning_pairs = _warning_tables(scan_qc, pair_qc, keep_ids)
    _write_csv(warning_scans, manifests_dir / "kept_scan_warnings_not_auto_excluded.csv")
    _write_csv(warning_pairs, manifests_dir / "kept_pair_warnings_not_auto_excluded.csv")

    print("\n[3/3] Writing final report…", flush=True)
    summary = {
        "source_meshes_modified": False,
        "source_root": str(source),
        "structures": list(STRUCTURES),
        "policy": {
            "cohort_filter": args.cohort_filter,
            "keep_diagnosis_changers": bool(args.keep_diagnosis_changers),
            "pair_exclusion_policy": args.pair_exclusion_policy,
            "multi_component_volume_tolerance_pct": float(args.multi_component_volume_tolerance_pct),
            "pair_outlier_method": args.pair_outlier_method,
            "pair_outlier_mad_multiplier": float(args.pair_outlier_mad_multiplier),
            "clinical": "baseline CN/AD and strict_subject_no_mci from the source manifest; direct CN<->AD changers excluded",
            "longitudinal": "at least two visits before and after automatic QC",
            "automatic_mesh_exclusions": "hard final smooth/correspondence geometry failure in either structure, or endpoint of a strong adjacent correspondence-shape failure",
            "not_automatically_excluded": "distributional volume/compactness outliers and ordinary trajectory warnings; retained in separate warning manifests because they can be biological",
        },
        "counts": {
            "strict_no_mci_input": _summary_counts(scans),
            "after_direct_CN_AD_change_exclusion": _summary_counts(stable_scans),
            "longitudinal_before_mesh_qc": _summary_counts(longitudinal_pre_qc),
            "recommended_keep": _summary_counts(final_keep),
            "direct_CN_AD_changer_subjects": int(direct_changers["subject_id"].nunique()),
            "single_visit_scans_before_mesh_qc": int(len(singleton_before_qc)),
            "automatic_scan_exclusions": int(len(automatic_exclusions)),
            "subjects_lost_after_mesh_qc": int(
                longitudinal_pre_qc["subject_id"].nunique() - final_keep["subject_id"].nunique()
            ),
            "kept_scan_warnings": int(len(warning_scans)),
            "kept_pair_warnings": int(len(warning_pairs)),
        },
        "files": {
            "recommended_manifest": str(manifests_dir / "strict_no_mci_longitudinal_keep_conservative.csv"),
            "automatic_exclusions": str(manifests_dir / "auto_excluded_scans_conservative.csv"),
            "mesh_qc_summary": str(mesh_qc_dir / "qc_summary.json"),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2), flush=True)
    print("\nQC complete. The source mesh directory was not modified.", flush=True)
    print(f"Recommended cohort: {summary['counts']['recommended_keep']['scans']:,} scans / "
          f"{summary['counts']['recommended_keep']['subjects']:,} subjects", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
