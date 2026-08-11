#!/usr/bin/env python3
"""Read-only quality control for longitudinal SynthSeg correspondence meshes.

The script never repairs, replaces, moves, or deletes a mesh.  It writes a
separate review report containing scan-, adjacent-pair-, and subject-level
recommendations for the left hippocampus and left lateral ventricle.
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
    add_pair_shape_measurements,
    build_adjacent_pairs,
    ensure_directory,
    input_validation_summary,
    load_analysis_records,
    load_mesh,
    median_mad_bounds,
    mesh_topology_hash,
    parse_structures,
    safe_percent_error,
    terminal_progress,
    to_builtin,
    write_json,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_SYNTHSEG_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=here / "synthseg_mesh_qc", help="New report directory; source meshes are read only."
    )
    parser.add_argument("--structures", default=",".join(STRUCTURES))
    parser.add_argument(
        "--cohort-filter",
        choices=COHORT_FILTERS,
        default="strict_no_mci",
        help="Default: baseline-CN/AD subjects with no MCI-labelled meshed visit. Use 'all' only for a separate MCI study.",
    )
    parser.add_argument(
        "--cohort-label",
        choices=("baseline", "visit"),
        default="baseline",
        help="Diagnosis grouping after cohort filtering. Baseline is the default for CN-vs-AD trajectories.",
    )
    parser.add_argument("--volume-rate-warning-pct-per-year", type=float, default=20.0)
    parser.add_argument("--volume-rate-extreme-pct-per-year", type=float, default=50.0)
    parser.add_argument("--surface-area-jump-warning-pct", type=float, default=20.0)
    parser.add_argument("--shape-rate-outlier-quantile", type=float, default=0.995)
    parser.add_argument("--scan-outlier-mad-multiplier", type=float, default=5.0)
    parser.add_argument("--correspondence-volume-error-warning-pct", type=float, default=1.0)
    parser.add_argument("--top-overlays", type=int, default=12)
    parser.add_argument("--no-html", action="store_true")
    return parser.parse_args()


def inspect_one_mesh(path_value: object, prefix: str) -> dict[str, Any]:
    path = Path(str(path_value))
    result: dict[str, Any] = {
        f"{prefix}_mesh_exists": path.is_file(),
        f"{prefix}_mesh_load_ok": False,
        f"{prefix}_mesh_load_error": "",
        f"{prefix}_vertices_actual": np.nan,
        f"{prefix}_faces_actual": np.nan,
        f"{prefix}_components_actual": np.nan,
        f"{prefix}_watertight_actual": False,
        f"{prefix}_winding_consistent_actual": False,
        f"{prefix}_finite_vertices_actual": False,
        f"{prefix}_volume_actual_mm3": np.nan,
        f"{prefix}_surface_area_actual_mm2": np.nan,
    }
    if not path.is_file():
        result[f"{prefix}_mesh_load_error"] = "missing mesh file"
        return result
    try:
        mesh = load_mesh(path)
        result.update(
            {
                f"{prefix}_mesh_load_ok": True,
                f"{prefix}_vertices_actual": int(len(mesh.vertices)),
                f"{prefix}_faces_actual": int(len(mesh.faces)),
                f"{prefix}_components_actual": int(mesh.component_count),
                f"{prefix}_watertight_actual": bool(mesh.watertight),
                f"{prefix}_winding_consistent_actual": bool(mesh.winding_consistent),
                f"{prefix}_finite_vertices_actual": bool(mesh.finite_vertices),
                f"{prefix}_volume_actual_mm3": float(mesh.volume),
                f"{prefix}_surface_area_actual_mm2": float(mesh.area),
            }
        )
        if prefix == "correspondence":
            result["correspondence_topology_hash"] = mesh_topology_hash(mesh.faces)
    except Exception as exc:
        result[f"{prefix}_mesh_load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def scan_mesh_quality(records: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(records)
    print(f"\n[scan QC] Inspecting raw, minimal-smooth, and correspondence meshes for {total:,} records.", flush=True)
    for index, row in enumerate(records.to_dict("records"), start=1):
        terminal_progress("scan QC", index, total, every=100)
        result = {
            "scan_id": row["scan_id"],
            "subject_id": row["subject_id"],
            "RID": row.get("RID"),
            "VISCODE": row.get("VISCODE"),
            "structure": row["structure"],
            "structure_display": row["structure_display"],
            "cohort_diagnosis": row.get("cohort_diagnosis"),
            "baseline_diagnosis": row.get("baseline_diagnosis"),
            "visit_diagnosis": row.get("visit_diagnosis"),
            "visit_month": row.get("visit_month"),
            "age_years": row.get("age_years"),
            "pipeline_status": row.get("mesh_pipeline_status"),
            "mask_volume_mm3": row.get("mask_volume_mm3"),
            "raw_volume_reported_mm3": row.get("raw_mesh_volume_mm3"),
            "smooth_volume_reported_mm3": row.get("smooth_mesh_volume_mm3"),
            "raw_mesh_path": row.get("raw_mesh_path"),
            "smooth_mesh_path": row.get("smooth_mesh_path"),
            "correspondence_mesh_path": row.get("correspondence_mesh_path"),
        }
        for prefix, column in (
            ("raw", "raw_mesh_path"),
            ("smooth", "smooth_mesh_path"),
            ("correspondence", "correspondence_mesh_path"),
        ):
            result.update(inspect_one_mesh(row[column], prefix))
        result["raw_vs_reported_volume_pct"] = safe_percent_error(
            result["raw_volume_actual_mm3"], float(row.get("raw_mesh_volume_mm3", np.nan))
        )
        result["smooth_vs_reported_volume_pct"] = safe_percent_error(
            result["smooth_volume_actual_mm3"], float(row.get("smooth_mesh_volume_mm3", np.nan))
        )
        result["correspondence_vs_smooth_volume_pct"] = safe_percent_error(
            result["correspondence_volume_actual_mm3"], float(row.get("smooth_mesh_volume_mm3", np.nan))
        )
        result["raw_vs_mask_volume_pct"] = safe_percent_error(
            result["raw_volume_actual_mm3"], float(row.get("mask_volume_mm3", np.nan))
        )
        result["smooth_vs_mask_volume_pct_actual"] = safe_percent_error(
            result["smooth_volume_actual_mm3"], float(row.get("mask_volume_mm3", np.nan))
        )
        volume = result["smooth_volume_actual_mm3"]
        area = result["smooth_surface_area_actual_mm2"]
        result["smooth_surface_area_volume_ratio"] = area / (volume ** (2.0 / 3.0)) if volume > 0 else np.nan
        results.append(result)

    quality = pd.DataFrame.from_records(results)
    thresholds: dict[str, Any] = {}
    if quality.empty:
        return quality, thresholds
    quality["expected_correspondence_topology_hash"] = quality.groupby("structure")[
        "correspondence_topology_hash"
    ].transform(lambda values: values.dropna().mode().iloc[0] if not values.dropna().empty else "")
    quality["flag_correspondence_topology_mismatch"] = (
        quality["correspondence_topology_hash"].notna()
        & quality["correspondence_topology_hash"].ne(quality["expected_correspondence_topology_hash"])
    )
    hard_prefixes = ("raw", "smooth", "correspondence")
    for prefix in hard_prefixes:
        quality[f"flag_{prefix}_mesh_missing"] = ~quality[f"{prefix}_mesh_exists"].astype(bool)
        quality[f"flag_{prefix}_mesh_load_failed"] = ~quality[f"{prefix}_mesh_load_ok"].astype(bool)
        quality[f"flag_{prefix}_nonfinite_vertices"] = ~quality[f"{prefix}_finite_vertices_actual"].astype(bool)
    # Raw meshes can legitimately have more than one segmentation component; the
    # mesh pipeline intentionally retains raw geometry and uses a single-component
    # minimal-smooth mesh for correspondence.  Thus raw-component flags are review-only.
    quality["flag_raw_multiple_components_review"] = quality["raw_components_actual"].fillna(0).gt(1)
    quality["flag_raw_not_watertight_review"] = ~quality["raw_watertight_actual"].astype(bool)
    for prefix in ("smooth", "correspondence"):
        quality[f"flag_{prefix}_not_watertight"] = ~quality[f"{prefix}_watertight_actual"].astype(bool)
        quality[f"flag_{prefix}_bad_winding"] = ~quality[f"{prefix}_winding_consistent_actual"].astype(bool)
        quality[f"flag_{prefix}_multiple_components"] = quality[f"{prefix}_components_actual"].fillna(0).ne(1)
    quality["flag_correspondence_volume_error"] = quality[
        "correspondence_vs_smooth_volume_pct"
    ].abs().gt(float(args.correspondence_volume_error_warning_pct))

    for structure, index in quality.groupby("structure").groups.items():
        for metric in ("raw_vs_mask_volume_pct", "smooth_vs_mask_volume_pct_actual", "smooth_surface_area_volume_ratio"):
            median, mad, low, high = median_mad_bounds(quality.loc[index, metric], args.scan_outlier_mad_multiplier)
            slug = metric.replace("_actual", "")
            thresholds[f"{structure}.{metric}"] = {"median": median, "mad": mad, "low": low, "high": high}
            quality.loc[index, f"flag_{slug}_outlier"] = quality.loc[index, metric].lt(low) | quality.loc[index, metric].gt(high)

    quality["flag_pipeline_status_not_ok"] = ~quality["pipeline_status"].eq("ok")
    hard_flags = [
        *[f"flag_{prefix}_mesh_missing" for prefix in hard_prefixes],
        *[f"flag_{prefix}_mesh_load_failed" for prefix in hard_prefixes],
        *[f"flag_{prefix}_nonfinite_vertices" for prefix in hard_prefixes],
        *[f"flag_{prefix}_not_watertight" for prefix in ("smooth", "correspondence")],
        *[f"flag_{prefix}_bad_winding" for prefix in ("smooth", "correspondence")],
        *[f"flag_{prefix}_multiple_components" for prefix in ("smooth", "correspondence")],
        "flag_correspondence_topology_mismatch",
        "flag_correspondence_volume_error",
        "flag_pipeline_status_not_ok",
    ]
    # Distributional raw/smooth volume and surface-ratio metrics remain review warnings.
    quality["hard_mesh_qc_flag"] = quality[hard_flags].any(axis=1)
    warning_flags = [column for column in quality if column.startswith("flag_") and column not in hard_flags]
    quality["review_mesh_qc_flag"] = quality[warning_flags].any(axis=1)
    quality["scan_qc_action"] = np.select(
        [quality["hard_mesh_qc_flag"], quality["review_mesh_qc_flag"]],
        ["review_scan", "review_scan"],
        default="pass",
    )
    return quality, thresholds


def _annualized_log_rate(source: pd.Series, target: pd.Series, years: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=source.index, dtype=float)
    valid = source.gt(0) & target.gt(0) & years.gt(0)
    result.loc[valid] = 100.0 * np.log(target.loc[valid] / source.loc[valid]) / years.loc[valid]
    return result


def pair_quality(records: pd.DataFrame, scan_qc: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = build_adjacent_pairs(records)
    if pairs.empty:
        return pairs, {}
    scan_fields = scan_qc[
        ["structure", "scan_id", "scan_qc_action", "hard_mesh_qc_flag", "review_mesh_qc_flag"]
    ].copy()
    pairs = pairs.merge(
        scan_fields.rename(columns={column: f"source_{column}" for column in scan_fields if column not in {"structure", "scan_id"}}),
        left_on=["structure", "source_scan_id"],
        right_on=["structure", "scan_id"],
        how="left",
    ).drop(columns="scan_id")
    pairs = pairs.merge(
        scan_fields.rename(columns={column: f"target_{column}" for column in scan_fields if column not in {"structure", "scan_id"}}),
        left_on=["structure", "target_scan_id"],
        right_on=["structure", "scan_id"],
        how="left",
    ).drop(columns="scan_id")
    pairs["raw_signed_volume_change_pct_per_year"] = _annualized_log_rate(
        pairs["source_raw_volume_mm3"], pairs["target_raw_volume_mm3"], pairs["delta_years"]
    )
    pairs["mask_signed_volume_change_pct_per_year"] = _annualized_log_rate(
        pairs["source_mask_volume_mm3"], pairs["target_mask_volume_mm3"], pairs["delta_years"]
    )
    pairs["surface_area_change_pct_per_year"] = _annualized_log_rate(
        pairs["source_surface_area_mm2"], pairs["target_surface_area_mm2"], pairs["delta_years"]
    )
    pairs["smooth_raw_rate_difference_pct_per_year"] = (
        pairs["signed_volume_change_pct_per_year"] - pairs["raw_signed_volume_change_pct_per_year"]
    )
    pairs = add_pair_shape_measurements(pairs, progress_name="pair shape QC")
    pairs["flag_pair_shape_measurement_failed"] = pairs["shape_measurement_error"].astype(str).ne("")
    pairs["flag_large_volume_jump"] = pairs["absolute_volume_change_pct_per_year"].gt(
        float(args.volume_rate_warning_pct_per_year)
    )
    pairs["flag_extreme_volume_jump"] = pairs["absolute_volume_change_pct_per_year"].gt(
        float(args.volume_rate_extreme_pct_per_year)
    )
    pairs["flag_surface_area_jump"] = pairs["surface_area_change_pct_per_year"].abs().gt(
        float(args.surface_area_jump_warning_pct)
    )
    pairs["flag_raw_smooth_sign_disagreement"] = (
        np.sign(pairs["signed_volume_change_pct_per_year"])
        != np.sign(pairs["raw_signed_volume_change_pct_per_year"])
    ) & pairs["signed_volume_change_pct_per_year"].abs().gt(5.0) & pairs["raw_signed_volume_change_pct_per_year"].abs().gt(5.0)
    thresholds: dict[str, Any] = {}
    quantile = float(args.shape_rate_outlier_quantile)
    for structure, index in pairs.groupby("structure").groups.items():
        for metric in ("shape_rms_displacement_mm_per_year", "shape_p95_displacement_mm_per_year"):
            finite = pairs.loc[index, metric].replace([np.inf, -np.inf], np.nan).dropna()
            threshold = float(finite.quantile(quantile)) if not finite.empty else np.nan
            thresholds[f"{structure}.{metric}_q{quantile}"] = threshold
            pairs.loc[index, f"flag_{metric}_outlier"] = pairs.loc[index, metric].gt(threshold)
    shape_flags = [column for column in pairs if column.startswith("flag_shape_")]
    pairs["flag_pair_involves_review_scan"] = (
        pairs["source_review_mesh_qc_flag"].fillna(True).astype(bool)
        | pairs["target_review_mesh_qc_flag"].fillna(True).astype(bool)
    )
    pairs["strong_pair_qc_flag"] = pairs[
        ["flag_extreme_volume_jump", "flag_raw_smooth_sign_disagreement", *shape_flags]
    ].any(axis=1)
    pairs["any_pair_qc_flag"] = pairs[
        ["flag_large_volume_jump", "flag_surface_area_jump", "flag_pair_involves_review_scan", *shape_flags]
    ].any(axis=1)
    pairs["qc_score"] = (
        pairs["absolute_volume_change_pct_per_year"].fillna(0.0)
        + pairs["surface_area_change_pct_per_year"].abs().fillna(0.0)
        + pairs["shape_p95_displacement_mm_per_year"].fillna(0.0)
    )
    return pairs, thresholds


def subject_quality(scan_qc: pd.DataFrame, pair_qc: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (structure, subject_id), scans in scan_qc.groupby(["structure", "subject_id"], sort=True):
        pairs = pair_qc.loc[(pair_qc["structure"] == structure) & (pair_qc["subject_id"] == subject_id)]
        flagged = int(pairs["strong_pair_qc_flag"].sum()) if not pairs.empty else 0
        pair_count = int(len(pairs))
        fraction = flagged / pair_count if pair_count else 0.0
        max_rate = float(pairs["absolute_volume_change_pct_per_year"].max()) if pair_count else np.nan
        review_scans = int(scans["review_mesh_qc_flag"].sum())
        exclude = bool(flagged >= 2 or fraction >= 0.5 or (np.isfinite(max_rate) and max_rate > args.volume_rate_extreme_pct_per_year))
        action = "exclude_subject" if exclude else ("review_subject" if flagged or review_scans else "pass")
        rows.append(
            {
                "structure": structure,
                "subject_id": subject_id,
                "cohort_diagnosis": scans["cohort_diagnosis"].iloc[0],
                "scan_count": int(len(scans)),
                "review_scan_count": review_scans,
                "adjacent_pair_count": pair_count,
                "strong_flagged_pair_count": flagged,
                "strong_flagged_pair_fraction": fraction,
                "max_absolute_volume_change_pct_per_year": max_rate,
                "subject_qc_action": action,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_pair_overlays(output_dir: Path, pairs: pd.DataFrame, count: int) -> list[str]:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return []
    written: list[str] = []
    overlay_dir = ensure_directory(output_dir / "top_pair_overlays")
    candidates = pairs.sort_values("qc_score", ascending=False).head(max(0, count))
    for index, row in enumerate(candidates.to_dict("records"), start=1):
        try:
            source = load_mesh(row["source_correspondence_mesh_path"])
            target = load_mesh(row["target_correspondence_mesh_path"])
            figure = go.Figure()
            for name, mesh, color in (("source", source, "#2E86DE"), ("target", target, "#C0392B")):
                figure.add_trace(
                    go.Mesh3d(
                        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                        name=name, color=color, opacity=0.45,
                    )
                )
            figure.update_layout(
                title=(f"QC pair {index}: {row['structure']} | {row['source_scan_id']} → {row['target_scan_id']}"),
                template="plotly_white", scene={"aspectmode": "data"},
            )
            path = overlay_dir / f"{index:02d}_{row['structure']}_{row['source_scan_id']}_to_{row['target_scan_id']}.html"
            figure.write_html(path, include_plotlyjs="cdn")
            written.append(str(path))
        except Exception as exc:
            print(f"[QC overlays] skipped overlay {index}: {exc}", flush=True)
    return written


def write_html_index(output_dir: Path, summary: dict[str, Any], scan_qc: pd.DataFrame, pair_qc: pd.DataFrame, subject_qc: pd.DataFrame) -> Path:
    def table(frame: pd.DataFrame, limit: int = 25) -> str:
        return frame.head(limit).to_html(index=False, escape=True, border=0) if not frame.empty else "<p>None.</p>"

    html = """<!doctype html><html><head><meta charset='utf-8'><title>SynthSeg mesh QC</title>
    <style>body{font-family:Arial,sans-serif;margin:28px}table{border-collapse:collapse;font-size:12px}th,td{padding:5px;border:1px solid #ddd}th{background:#f3f3f3}</style>
    </head><body><h1>SynthSeg longitudinal mesh QC</h1>"""
    html += f"<pre>{json_pretty(summary)}</pre>"
    html += "<h2>Scans requiring review</h2>" + table(scan_qc.loc[scan_qc["scan_qc_action"] != "pass"])
    html += "<h2>Strongly flagged adjacent pairs</h2>" + table(pair_qc.loc[pair_qc["strong_pair_qc_flag"]].sort_values("qc_score", ascending=False))
    html += "<h2>Subjects recommended for exclusion</h2>" + table(subject_qc.loc[subject_qc["subject_qc_action"] == "exclude_subject"])
    html += "</body></html>"
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def json_pretty(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(to_builtin(payload), indent=2)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = ensure_directory(args.output_dir.expanduser().resolve())
    structures = parse_structures(args.structures)
    print("=" * 80)
    print("SynthSeg mesh quality control (read-only source meshes)")
    print(f"Input root: {input_root}")
    print(f"Report directory: {output_dir}")
    print(f"Cohort filter: {args.cohort_filter}; diagnosis label: {args.cohort_label}")
    print("=" * 80, flush=True)
    records = load_analysis_records(
        input_root,
        structures,
        cohort_label=args.cohort_label,
        cohort_filter=args.cohort_filter,
    )
    records.to_csv(output_dir / "input_records.csv", index=False)
    scan_qc, scan_thresholds = scan_mesh_quality(records, args)
    pair_qc, pair_thresholds = pair_quality(records, scan_qc, args)
    subject_qc = subject_quality(scan_qc, pair_qc, args)

    involved_pair_scans = set(pair_qc.loc[pair_qc["strong_pair_qc_flag"], "source_scan_id"].astype(str))
    involved_pair_scans.update(pair_qc.loc[pair_qc["strong_pair_qc_flag"], "target_scan_id"].astype(str))
    scan_qc["flagged_by_strong_pair"] = scan_qc["scan_id"].astype(str).isin(involved_pair_scans)
    scan_qc["final_scan_qc_action"] = np.where(
        scan_qc["flagged_by_strong_pair"] & scan_qc["scan_qc_action"].eq("pass"),
        "review_pair_context",
        scan_qc["scan_qc_action"],
    )
    excluded_subject_keys = {
        (str(row.structure), str(row.subject_id))
        for row in subject_qc.loc[subject_qc["subject_qc_action"] == "exclude_subject"].itertuples(index=False)
    }
    review_scan_keys = {
        (str(row.structure), str(row.scan_id))
        for row in scan_qc.loc[scan_qc["final_scan_qc_action"] != "pass"].itertuples(index=False)
    }
    scan_action_lookup = {
        (str(row.structure), str(row.scan_id)): row.final_scan_qc_action
        for row in scan_qc[["structure", "scan_id", "final_scan_qc_action"]].itertuples(index=False)
    }
    subject_action_lookup = {
        (str(row.structure), str(row.subject_id)): row.subject_qc_action
        for row in subject_qc[["structure", "subject_id", "subject_qc_action"]].itertuples(index=False)
    }
    records.loc[:, "qc_scan_action"] = [
        scan_action_lookup.get((str(structure), str(scan_id)), "missing_qc")
        for structure, scan_id in zip(records["structure"], records["scan_id"])
    ]
    records.loc[:, "qc_subject_action"] = [
        subject_action_lookup.get((str(structure), str(subject_id)), "missing_qc")
        for structure, subject_id in zip(records["structure"], records["subject_id"])
    ]
    # The two filtered tables are recommendations only; mesh files remain untouched.
    records.loc[
        [
            (str(structure), str(scan_id)) not in review_scan_keys
            for structure, scan_id in zip(records["structure"], records["scan_id"])
        ]
    ].to_csv(
        output_dir / "records_after_drop_review_scans.csv", index=False
    )
    records.loc[
        [
            (str(structure), str(subject_id)) not in excluded_subject_keys
            for structure, subject_id in zip(records["structure"], records["subject_id"])
        ]
    ].to_csv(
        output_dir / "records_after_drop_excluded_subjects.csv", index=False
    )

    scan_qc.to_csv(output_dir / "scan_qc.csv", index=False)
    pair_qc.to_csv(output_dir / "adjacent_pair_qc.csv", index=False)
    pair_qc.loc[pair_qc["strong_pair_qc_flag"]].sort_values("qc_score", ascending=False).to_csv(
        output_dir / "bad_adjacent_pairs.csv", index=False
    )
    scan_qc.loc[scan_qc["final_scan_qc_action"] != "pass"].to_csv(output_dir / "review_scans.csv", index=False)
    subject_qc.to_csv(output_dir / "subject_qc.csv", index=False)
    subject_qc.loc[subject_qc["subject_qc_action"] != "pass"].to_csv(output_dir / "review_subjects.csv", index=False)

    summary = {
        "source_meshes_modified": False,
        "input": input_validation_summary(input_root, records),
        "cohort_filter": args.cohort_filter,
        "cohort_label": args.cohort_label,
        "structures": list(structures),
        "scan_records": int(len(scan_qc)),
        "adjacent_pairs": int(len(pair_qc)),
        "scan_actions": scan_qc["final_scan_qc_action"].value_counts().to_dict(),
        "strong_flagged_pairs": int(pair_qc["strong_pair_qc_flag"].sum()),
        "subject_actions": subject_qc["subject_qc_action"].value_counts().to_dict(),
        "scan_thresholds": scan_thresholds,
        "pair_thresholds": pair_thresholds,
        "notes": [
            "Raw multiple-component meshes are review-only because the raw segmentation is preserved and the correspondence pipeline uses a one-component smooth mesh.",
            "The QC recommendation tables do not delete or modify source meshes.",
            "The default strict_no_mci filter retains only baseline-CN/AD subjects with no MCI-labelled meshed visit; it excludes MCI converters/reverters and unknown baseline groups.",
        ],
    }
    overlay_paths: list[str] = []
    if not args.no_html:
        overlay_paths = write_pair_overlays(output_dir, pair_qc.loc[pair_qc["any_pair_qc_flag"]], args.top_overlays)
        summary["top_pair_overlays"] = overlay_paths
        summary["html_index"] = str(write_html_index(output_dir, summary, scan_qc, pair_qc, subject_qc))
    write_json(output_dir / "qc_summary.json", summary)
    print("\nQC complete. Source meshes were not modified.")
    print(json_pretty(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
