#!/usr/bin/env python3
"""Build QC-aware longitudinal volume and shape analysis tables for SynthSeg.

This command reads the raw/smoothed/correspondence mesh provenance and never
modifies mesh files.  It prepares compact CSV/NPZ artifacts for the companion
notebook, including strict-no-MCI CN-versus-AD analyses and local shape maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthseg_longitudinal_common import (
    DEFAULT_SYNTHSEG_ROOT,
    DIAGNOSES,
    COHORT_FILTERS,
    STRUCTURES,
    build_adjacent_pairs,
    ensure_directory,
    input_validation_summary,
    load_analysis_records,
    pairwise_rate_statistics,
    parse_structures,
    save_shape_maps,
    subject_rate_summary,
    to_builtin,
    write_json,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_SYNTHSEG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=here / "synthseg_longitudinal_analysis")
    parser.add_argument("--structures", default=",".join(STRUCTURES))
    parser.add_argument(
        "--cohort-filter",
        choices=COHORT_FILTERS,
        default="strict_no_mci",
        help="Default: baseline-CN/AD subjects with no MCI-labelled meshed visit.",
    )
    parser.add_argument("--cohort-label", choices=("baseline", "visit"), default="baseline")
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=here / "synthseg_mesh_qc",
        help="Directory produced by synthseg_longitudinal_qc.py.",
    )
    parser.add_argument(
        "--qc-policy",
        choices=("all", "drop-review-scans", "drop-excluded-subjects"),
        default="all",
        help="Filtering affects only derived analysis tables, never source meshes.",
    )
    parser.add_argument("--age-bin-width-years", type=float, default=2.0)
    parser.add_argument("--skip-shape-maps", action="store_true")
    parser.add_argument(
        "--reuse-existing-shape-maps",
        action="store_true",
        help="Reuse an existing local_shape_speed_maps.npz and summary instead of recomputing correspondence maps.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def apply_qc_policy(records: pd.DataFrame, qc_dir: Path, policy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {"qc_policy": policy, "qc_dir": str(qc_dir), "rows_before": int(len(records))}
    if policy == "all":
        info["rows_after"] = int(len(records))
        info["qc_applied"] = False
        return records.copy(), info
    scan_path = qc_dir / "scan_qc.csv"
    subject_path = qc_dir / "subject_qc.csv"
    if not scan_path.is_file() or not subject_path.is_file():
        raise FileNotFoundError(
            f"QC policy {policy!r} needs scan_qc.csv and subject_qc.csv in {qc_dir}. Run synthseg_longitudinal_qc.py first."
        )
    scan_qc = pd.read_csv(scan_path, dtype={"scan_id": str, "subject_id": str})
    subject_qc = pd.read_csv(subject_path, dtype={"subject_id": str})
    result = records.copy()
    if policy == "drop-review-scans":
        allowed = {
            (str(row.structure), str(row.scan_id))
            for row in scan_qc.loc[scan_qc["final_scan_qc_action"].eq("pass")].itertuples(index=False)
        }
        keep = [
            (str(structure), str(scan_id)) in allowed
            for structure, scan_id in zip(result["structure"], result["scan_id"])
        ]
        result = result.loc[keep].copy()
        info["excluded_scan_records"] = int(len(records) - len(result))
    else:
        allowed_subjects = {
            (str(row.structure), str(row.subject_id))
            for row in subject_qc.loc[~subject_qc["subject_qc_action"].eq("exclude_subject")].itertuples(index=False)
        }
        keep = [
            (str(structure), str(subject_id)) in allowed_subjects
            for structure, subject_id in zip(result["structure"], result["subject_id"])
        ]
        result = result.loc[keep].copy()
        info["excluded_scan_records"] = int(len(records) - len(result))
    info["rows_after"] = int(len(result))
    info["qc_applied"] = True
    return result, info


def add_visit_metrics(records: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    result = result.sort_values(["structure", "subject_id", "visit_month", "scan_id"], kind="stable")
    result["baseline_volume_mm3"] = result.groupby(["structure", "subject_id"])["smooth_mesh_volume_mm3"].transform("first")
    result["relative_volume_to_baseline"] = result["smooth_mesh_volume_mm3"] / result["baseline_volume_mm3"]
    result["volume_change_from_baseline_pct"] = 100.0 * (result["relative_volume_to_baseline"] - 1.0)
    result["years_from_baseline"] = result["visit_month"] / 12.0
    return result


def visit_summary(visits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("smooth_mesh_volume_mm3", "relative_volume_to_baseline", "volume_change_from_baseline_pct")
    for keys, frame in visits.groupby(["structure", "cohort_diagnosis", "visit_month"], dropna=False, sort=True):
        structure, diagnosis, month = keys
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            sem = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
            rows.append(
                {
                    "structure": structure,
                    "cohort_diagnosis": diagnosis,
                    "visit_month": month,
                    "years_from_baseline": float(month) / 12.0 if pd.notna(month) else np.nan,
                    "metric": metric,
                    "count": count,
                    "mean": mean,
                    "median": float(values.median()) if count else np.nan,
                    "std": float(values.std(ddof=1)) if count > 1 else np.nan,
                    "ci95_low": mean - 1.96 * sem if np.isfinite(sem) else np.nan,
                    "ci95_high": mean + 1.96 * sem if np.isfinite(sem) else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows)


def age_bin_summary(pairs: pd.DataFrame, width: float) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    if width <= 0.0:
        raise ValueError("--age-bin-width-years must be positive")
    result = pairs.copy()
    ages = result["midpoint_age_years"].dropna()
    if ages.empty:
        return pd.DataFrame()
    start = np.floor(float(ages.min()) / width) * width
    stop = np.ceil(float(ages.max()) / width) * width + width
    result["age_bin"] = pd.cut(result["midpoint_age_years"], bins=np.arange(start, stop + width, width), right=False)
    rows: list[dict[str, Any]] = []
    for keys, frame in result.groupby(["structure", "cohort_diagnosis", "age_bin"], observed=True, sort=True):
        structure, diagnosis, interval = keys
        values = frame["signed_volume_change_pct_per_year"].dropna()
        rows.append(
            {
                "structure": structure,
                "cohort_diagnosis": diagnosis,
                "age_bin_start": float(interval.left),
                "age_bin_end": float(interval.right),
                "age_bin_center": 0.5 * (float(interval.left) + float(interval.right)),
                "count": int(len(values)),
                "mean_signed_volume_change_pct_per_year": float(values.mean()) if len(values) else np.nan,
                "median_signed_volume_change_pct_per_year": float(values.median()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def within_subject_trend_summary(visits: pd.DataFrame) -> pd.DataFrame:
    """Estimate time trends after removing each participant's volume intercept.

    Unlike visit-wise cohort means, this estimator is not distorted when only a
    selected subset of participants reaches later follow-up visits.  The
    log-volume coefficient is reported as approximate percent change per year.
    """

    rows: list[dict[str, Any]] = []
    for (structure, diagnosis), frame in visits.groupby(["structure", "cohort_diagnosis"], sort=True):
        frame = frame.loc[
            frame["years_from_baseline"].notna() & frame["smooth_mesh_volume_mm3"].gt(0.0)
        ].copy()
        counts = frame.groupby("subject_id").size()
        frame = frame.loc[frame["subject_id"].isin(counts.index[counts.ge(2)])].copy()
        if frame.empty:
            continue
        groups = frame.groupby("subject_id")
        time_centered = frame["years_from_baseline"] - groups["years_from_baseline"].transform("mean")
        log_volume = np.log(frame["smooth_mesh_volume_mm3"])
        log_volume_centered = log_volume - log_volume.groupby(frame["subject_id"]).transform("mean")
        volume_centered = frame["smooth_mesh_volume_mm3"] - groups["smooth_mesh_volume_mm3"].transform("mean")
        denominator = float(np.sum(time_centered.to_numpy(dtype=float) ** 2))
        if denominator <= 0.0:
            continue

        def fit(response: pd.Series) -> tuple[float, float]:
            slope = float(np.sum(time_centered.to_numpy(dtype=float) * response.to_numpy(dtype=float)) / denominator)
            residual = response.to_numpy(dtype=float) - slope * time_centered.to_numpy(dtype=float)
            degrees_of_freedom = max(1, len(frame) - frame["subject_id"].nunique() - 1)
            standard_error = float(np.sqrt(np.sum(residual**2) / degrees_of_freedom / denominator))
            return slope, standard_error

        log_slope, log_se = fit(log_volume_centered)
        volume_slope, volume_se = fit(volume_centered)
        rows.append(
            {
                "structure": structure,
                "cohort_diagnosis": diagnosis,
                "longitudinal_subject_count": int(frame["subject_id"].nunique()),
                "observation_count": int(len(frame)),
                "max_observed_years": float(frame["years_from_baseline"].max()),
                "annual_log_volume_change_pct": 100.0 * log_slope,
                "annual_log_volume_change_ci95_low": 100.0 * (log_slope - 1.96 * log_se),
                "annual_log_volume_change_ci95_high": 100.0 * (log_slope + 1.96 * log_se),
                "within_subject_volume_slope_mm3_per_year": volume_slope,
                "within_subject_volume_slope_ci95_low": volume_slope - 1.96 * volume_se,
                "within_subject_volume_slope_ci95_high": volume_slope + 1.96 * volume_se,
            }
        )
    return pd.DataFrame.from_records(rows)


def trajectory_transitions(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby(
            ["structure", "cohort_diagnosis", "source_visit_diagnosis", "target_visit_diagnosis"], dropna=False
        )
        .agg(pair_count=("subject_id", "size"), subject_count=("subject_id", "nunique"))
        .reset_index()
        .sort_values(["structure", "cohort_diagnosis", "pair_count"], ascending=[True, True, False])
    )


def cohort_counts(visits: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    scan_counts = (
        visits.groupby(["structure", "cohort_diagnosis"], dropna=False)
        .agg(scan_count=("scan_id", "size"), subject_count=("subject_id", "nunique"), mean_age_years=("age_years", "mean"))
        .reset_index()
    )
    pair_counts = (
        pairs.groupby(["structure", "cohort_diagnosis"], dropna=False)
        .agg(adjacent_pair_count=("subject_id", "size"), longitudinal_subject_count=("subject_id", "nunique"))
        .reset_index()
    )
    return scan_counts.merge(pair_counts, on=["structure", "cohort_diagnosis"], how="left")


def add_welch_p_values(rate_stats: pd.DataFrame, subject_rates: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        rate_stats["welch_t_p_value"] = np.nan
        return rate_stats
    values: list[float] = []
    for row in rate_stats.to_dict("records"):
        frame = subject_rates.loc[subject_rates["structure"].eq(row["structure"])]
        first = frame.loc[frame["cohort_diagnosis"].eq(row["group_a"]), row["metric"]].dropna()
        second = frame.loc[frame["cohort_diagnosis"].eq(row["group_b"]), row["metric"]].dropna()
        values.append(float(ttest_ind(second, first, equal_var=False, nan_policy="omit").pvalue) if len(first) > 1 and len(second) > 1 else np.nan)
    rate_stats = rate_stats.copy()
    rate_stats["welch_t_p_value"] = values
    return rate_stats


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = ensure_directory(args.output_dir.expanduser().resolve())
    structures = parse_structures(args.structures)
    print("=" * 80)
    print("SynthSeg longitudinal analysis (read-only source meshes)")
    print(f"Input root: {input_root}")
    print(f"Analysis directory: {output_dir}")
    print(
        f"Cohort filter: {args.cohort_filter}; diagnosis label: {args.cohort_label}; QC policy: {args.qc_policy}"
    )
    print("=" * 80, flush=True)

    records = load_analysis_records(
        input_root,
        structures,
        cohort_label=args.cohort_label,
        cohort_filter=args.cohort_filter,
    )
    records = records.loc[records["cohort_diagnosis"].isin(DIAGNOSES)].copy()
    filtered, qc_info = apply_qc_policy(records, args.qc_dir.expanduser().resolve(), args.qc_policy)
    visits = add_visit_metrics(filtered)
    pairs = build_adjacent_pairs(visits)
    subjects = subject_rate_summary(pairs)
    rates = add_welch_p_values(
        pairwise_rate_statistics(subjects, bootstrap_draws=args.bootstrap_draws), subjects
    )
    summaries = visit_summary(visits)
    ages = age_bin_summary(pairs, args.age_bin_width_years)
    transitions = trajectory_transitions(pairs)
    counts = cohort_counts(visits, pairs)
    within_subject_trends = within_subject_trend_summary(visits)

    visits.to_csv(output_dir / "visit_metrics.csv", index=False)
    pairs.to_csv(output_dir / "adjacent_pair_metrics.csv", index=False)
    subjects.to_csv(output_dir / "subject_rate_summary.csv", index=False)
    rates.to_csv(output_dir / "diagnosis_rate_contrasts.csv", index=False)
    summaries.to_csv(output_dir / "visit_trajectory_summary.csv", index=False)
    ages.to_csv(output_dir / "age_bin_rate_summary.csv", index=False)
    transitions.to_csv(output_dir / "clinical_transition_summary.csv", index=False)
    counts.to_csv(output_dir / "cohort_counts.csv", index=False)
    within_subject_trends.to_csv(output_dir / "within_subject_trend_summary.csv", index=False)

    shape_summary = pd.DataFrame()
    shape_map_path = output_dir / "local_shape_speed_maps.npz"
    shape_summary_path = output_dir / "local_shape_speed_summary.csv"
    if args.reuse_existing_shape_maps:
        if not shape_map_path.is_file() or not shape_summary_path.is_file():
            raise FileNotFoundError(
                "--reuse-existing-shape-maps requires local_shape_speed_maps.npz and local_shape_speed_summary.csv in the output directory."
            )
        shape_summary = pd.read_csv(shape_summary_path)
        print(f"[analysis] Reusing existing local shape maps: {shape_map_path}", flush=True)
    elif not args.skip_shape_maps:
        print(f"\n[analysis] Building correspondence-based local shape-speed maps for {len(pairs):,} adjacent pairs.", flush=True)
        shape_summary, _ = save_shape_maps(pairs, shape_map_path, every=100)
        shape_summary.to_csv(shape_summary_path, index=False)
    else:
        print("[analysis] Skipped local shape maps by request.", flush=True)

    summary = {
        "source_meshes_modified": False,
        "cohort_definition": {
            "selected": args.cohort_label,
            "meaning": (
                "CN/AD are participant baseline cohorts after excluding every baseline-MCI, unknown-baseline, and MCI-visit subject."
                if args.cohort_label == "baseline" and args.cohort_filter == "strict_no_mci"
                else "Diagnosis grouping follows the requested cohort label and filter."
            ),
        },
        "cohort_filter": args.cohort_filter,
        "input": input_validation_summary(input_root, records),
        "qc": qc_info,
        "analysis_records": int(len(visits)),
        "adjacent_pairs": int(len(pairs)),
        "subject_rate_rows": int(len(subjects)),
        "within_subject_trend_rows": int(len(within_subject_trends)),
        "cohort_counts": counts.to_dict("records"),
        "shape_maps_written": str(shape_map_path) if shape_map_path.is_file() else None,
        "shape_summary_rows": int(len(shape_summary)),
        "volume_definition": "Physical mm³ volume from the minimally smoothed mesh, whose volume is preserved by final_ply_mm correspondence meshes.",
        "trajectory_safeguard": "Use within_subject_trend_summary.csv for the primary longitudinal direction/rate. Visit-wise cohort means are descriptive and can change composition at later follow-up.",
        "interpretation": {
            "left_hippocampus": "Negative signed log-volume rate indicates shrinkage; positive values indicate enlargement.",
            "left_lateral_ventricle": "Positive signed log-volume rate indicates ventricular enlargement; do not relabel this as atrophy.",
        },
    }
    write_json(output_dir / "analysis_summary.json", summary)
    import json

    print("\nAnalysis tables complete. Source meshes were not modified.")
    print(json.dumps(to_builtin(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
