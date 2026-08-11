#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import html

import numpy as np
import pandas as pd


MappingLike = Dict[str, Any]


NUMERIC_COLUMNS = (
    "visit_order",
    "visit_month",
    "months_from_baseline",
    "age_years",
    "continuous_age_years",
    "left_mask_volume_mm3",
    "left_mesh_volume_mm3",
    "left_surface_area_mm2",
    "left_mask_voxels",
)


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "QC mask-derived longitudinal hippocampus meshes using per-scan "
            "volume/shape checks, adjacent follow-up consistency checks, and "
            "optional actual mesh displacement checks."
        )
    )
    parser.add_argument(
        "--metadata",
        default=str(
            experiment
            / "metadata"
            / "adni_large_strict_no_mci_left_direct_flow_records.csv"
        ),
        help="Metadata CSV with scan_id, subject_id, visit_order, volumes, and mesh_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(experiment / "analysis" / "mesh_longitudinal_qc"),
    )
    parser.add_argument(
        "--skip-mesh-shape-metrics",
        action="store_true",
        help="Skip loading OBJ/PLY files and only use metadata volume/area columns.",
    )
    parser.add_argument("--scan-outlier-quantile", type=float, default=0.005)
    parser.add_argument("--mesh-rate-threshold-pct-per-year", type=float, default=20.0)
    parser.add_argument("--mask-rate-threshold-pct-per-year", type=float, default=20.0)
    parser.add_argument("--mesh-mask-ratio-jump-threshold-pct", type=float, default=8.0)
    parser.add_argument("--surface-area-jump-threshold-pct", type=float, default=20.0)
    parser.add_argument("--sign-disagreement-min-delta-pct", type=float, default=5.0)
    parser.add_argument(
        "--shape-rate-outlier-quantile",
        type=float,
        default=0.995,
        help="Quantile threshold for pairwise shape displacement rate flags.",
    )
    parser.add_argument(
        "--subject-many-bad-pair-fraction",
        type=float,
        default=0.5,
        help="Flag a subject for exclusion if this fraction of adjacent pairs is bad.",
    )
    parser.add_argument(
        "--subject-extreme-rate-threshold-pct-per-year",
        type=float,
        default=50.0,
        help="Flag subject for exclusion if any adjacent mesh jump exceeds this rate.",
    )
    parser.add_argument(
        "--min-scans-after-filter",
        type=int,
        default=2,
        help="Minimum remaining scans required in filtered bad-scan metadata.",
    )
    parser.add_argument("--top-cases", type=int, default=40)
    parser.add_argument(
        "--export-top-pair-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write 3D source/target overlays for top bad adjacent pairs.",
    )
    parser.add_argument("--top-pair-html-count", type=int, default=20)
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip Plotly/HTML output and write only CSV/JSON files.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists() or not path.exists():
        return candidate.resolve()
    return path.resolve()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "scan_id",
        "subject_id",
        "split",
        "diagnosis",
        "visit_order",
        "mesh_path",
        "left_mask_volume_mm3",
        "left_mesh_volume_mm3",
        "left_surface_area_mm2",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Metadata is missing required columns: {missing}")
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["scan_id"] = frame["scan_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["diagnosis"] = frame["diagnosis"].astype(str)
    frame["mesh_path"] = frame["mesh_path"].astype(str)
    return frame


def add_scan_metadata_metrics(frame: pd.DataFrame, scan_outlier_quantile: float) -> Tuple[pd.DataFrame, MappingLike]:
    result = frame.copy()
    result["mesh_mask_ratio"] = (
        result["left_mesh_volume_mm3"] / result["left_mask_volume_mm3"]
    )
    result["surface_area_volume_ratio"] = result["left_surface_area_mm2"] / (
        result["left_mesh_volume_mm3"].clip(lower=1.0e-9) ** (2.0 / 3.0)
    )
    q = float(scan_outlier_quantile)
    if not (0.0 < q < 0.5):
        raise ValueError("--scan-outlier-quantile must be in (0, 0.5)")
    thresholds: MappingLike = {}
    for metric in ("mesh_mask_ratio", "surface_area_volume_ratio"):
        finite = result[metric].replace([np.inf, -np.inf], np.nan).dropna()
        low = float(finite.quantile(q)) if not finite.empty else float("nan")
        high = float(finite.quantile(1.0 - q)) if not finite.empty else float("nan")
        thresholds[f"{metric}_low_q{q}"] = low
        thresholds[f"{metric}_high_q{1.0 - q}"] = high
        result[f"flag_scan_{metric}_outlier"] = (
            (result[metric] < low) | (result[metric] > high)
        )
    result["flag_scan_missing_required_volume"] = (
        result[["left_mask_volume_mm3", "left_mesh_volume_mm3", "left_surface_area_mm2"]]
        .isna()
        .any(axis=1)
    )
    result["flag_scan_nonpositive_volume"] = (
        (result["left_mask_volume_mm3"] <= 0.0)
        | (result["left_mesh_volume_mm3"] <= 0.0)
    )
    return result, thresholds


def load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def mesh_quality_records(scan_qc: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in scan_qc.to_dict("records"):
        path = Path(str(row["mesh_path"]))
        record: Dict[str, Any] = {
            "scan_id": row["scan_id"],
            "mesh_path": str(path),
            "mesh_file_exists": path.is_file(),
            "mesh_load_ok": False,
            "mesh_load_error": "",
            "mesh_vertex_count": np.nan,
            "mesh_face_count": np.nan,
            "mesh_watertight": False,
            "mesh_winding_consistent": False,
            "mesh_component_count": np.nan,
            "mesh_nonfinite_vertices": np.nan,
            "mesh_scaled_volume_abs": np.nan,
            "mesh_scaled_surface_area": np.nan,
            "mesh_bbox_diag": np.nan,
            "mesh_centroid_x": np.nan,
            "mesh_centroid_y": np.nan,
            "mesh_centroid_z": np.nan,
        }
        if not path.is_file():
            record["mesh_load_error"] = "missing mesh file"
            records.append(record)
            continue
        try:
            mesh = load_mesh(path)
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)
            extents = np.asarray(mesh.extents, dtype=float)
            centroid = np.asarray(mesh.centroid, dtype=float)
            record.update(
                {
                    "mesh_load_ok": True,
                    "mesh_vertex_count": int(len(vertices)),
                    "mesh_face_count": int(len(faces)),
                    "mesh_watertight": bool(mesh.is_watertight),
                    "mesh_winding_consistent": bool(mesh.is_winding_consistent),
                    "mesh_component_count": int(len(mesh.split(only_watertight=False))),
                    "mesh_nonfinite_vertices": int((~np.isfinite(vertices)).any(axis=1).sum()),
                    "mesh_scaled_volume_abs": abs(float(mesh.volume)),
                    "mesh_scaled_surface_area": float(mesh.area),
                    "mesh_bbox_diag": float(np.linalg.norm(extents)),
                    "mesh_centroid_x": float(centroid[0]),
                    "mesh_centroid_y": float(centroid[1]),
                    "mesh_centroid_z": float(centroid[2]),
                }
            )
        except Exception as exc:
            record["mesh_load_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    quality = pd.DataFrame.from_records(records)
    if quality.empty:
        return quality
    vertex_mode = int(quality["mesh_vertex_count"].dropna().mode().iloc[0])
    face_mode = int(quality["mesh_face_count"].dropna().mode().iloc[0])
    quality["flag_mesh_file_missing"] = ~quality["mesh_file_exists"].astype(bool)
    quality["flag_mesh_load_failed"] = ~quality["mesh_load_ok"].astype(bool)
    quality["flag_mesh_topology_outlier"] = (
        (quality["mesh_vertex_count"] != vertex_mode)
        | (quality["mesh_face_count"] != face_mode)
    )
    quality["flag_mesh_not_watertight"] = ~quality["mesh_watertight"].astype(bool)
    quality["flag_mesh_bad_winding"] = ~quality["mesh_winding_consistent"].astype(bool)
    quality["flag_mesh_nonfinite_vertices"] = quality["mesh_nonfinite_vertices"].fillna(1) > 0
    return quality


def kabsch_align_points(target_vertices: np.ndarray, source_vertices: np.ndarray) -> np.ndarray:
    source_center = source_vertices.mean(axis=0)
    target_center = target_vertices.mean(axis=0)
    source0 = source_vertices - source_center
    target0 = target_vertices - target_center
    covariance = target0.T @ source0
    u, _s, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return target0 @ rotation + source_center


def adjacent_pair_records(scan_qc: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    sort_cols = ["split", "subject_id", "visit_order", "months_from_baseline", "age_years"]
    for (split, subject_id), group in scan_qc.sort_values(sort_cols).groupby(
        ["split", "subject_id"],
        sort=False,
    ):
        rows = list(group.to_dict("records"))
        for source, target in zip(rows[:-1], rows[1:]):
            source_month = float(source.get("months_from_baseline", np.nan))
            target_month = float(target.get("months_from_baseline", np.nan))
            if np.isfinite(source_month) and np.isfinite(target_month):
                dt_years = (target_month - source_month) / 12.0
            else:
                dt_years = float(target.get("continuous_age_years", np.nan)) - float(
                    source.get("continuous_age_years", np.nan)
                )
            if not np.isfinite(dt_years) or dt_years <= 0.0:
                continue
            record = {
                "split": str(split),
                "subject_id": str(subject_id),
                "diagnosis": str(source["diagnosis"]),
                "source_scan_id": str(source["scan_id"]),
                "target_scan_id": str(target["scan_id"]),
                "source_visit_order": int(source["visit_order"]),
                "target_visit_order": int(target["visit_order"]),
                "source_months_from_baseline": source_month,
                "target_months_from_baseline": target_month,
                "dt_years": dt_years,
                "source_age_years": float(source.get("continuous_age_years", source.get("age_years", np.nan))),
                "target_age_years": float(target.get("continuous_age_years", target.get("age_years", np.nan))),
                "source_mesh_path": str(source["mesh_path"]),
                "target_mesh_path": str(target["mesh_path"]),
                "source_mask_volume_mm3": float(source["left_mask_volume_mm3"]),
                "target_mask_volume_mm3": float(target["left_mask_volume_mm3"]),
                "source_mesh_volume_mm3": float(source["left_mesh_volume_mm3"]),
                "target_mesh_volume_mm3": float(target["left_mesh_volume_mm3"]),
                "source_surface_area_mm2": float(source["left_surface_area_mm2"]),
                "target_surface_area_mm2": float(target["left_surface_area_mm2"]),
                "source_mesh_mask_ratio": float(source["mesh_mask_ratio"]),
                "target_mesh_mask_ratio": float(target["mesh_mask_ratio"]),
            }
            for kind in ("mask", "mesh"):
                source_volume = record[f"source_{kind}_volume_mm3"]
                target_volume = record[f"target_{kind}_volume_mm3"]
                record[f"{kind}_delta_mm3"] = target_volume - source_volume
                record[f"{kind}_delta_pct"] = (
                    100.0 * (target_volume - source_volume) / source_volume
                    if source_volume > 0.0
                    else np.nan
                )
                record[f"{kind}_atrophy_pct_per_year"] = (
                    100.0 * np.log(source_volume / target_volume) / dt_years
                    if source_volume > 0.0 and target_volume > 0.0
                    else np.nan
                )
                record[f"{kind}_abs_log_rate_pct_per_year"] = (
                    100.0 * abs(np.log(target_volume / source_volume)) / dt_years
                    if source_volume > 0.0 and target_volume > 0.0
                    else np.nan
                )
            record["mesh_mask_delta_pct_disagreement"] = (
                record["mesh_delta_pct"] - record["mask_delta_pct"]
            )
            record["mesh_mask_sign_disagree"] = (
                bool(np.sign(record["mesh_delta_pct"]) != np.sign(record["mask_delta_pct"]))
                if np.isfinite(record["mesh_delta_pct"])
                and np.isfinite(record["mask_delta_pct"])
                and abs(record["mesh_delta_pct"]) > 1.0e-9
                and abs(record["mask_delta_pct"]) > 1.0e-9
                else False
            )
            record["mesh_mask_ratio_delta_pct"] = (
                100.0
                * (record["target_mesh_mask_ratio"] - record["source_mesh_mask_ratio"])
                / record["source_mesh_mask_ratio"]
                if record["source_mesh_mask_ratio"] > 0.0
                else np.nan
            )
            record["surface_area_delta_pct"] = (
                100.0
                * (record["target_surface_area_mm2"] - record["source_surface_area_mm2"])
                / record["source_surface_area_mm2"]
                if record["source_surface_area_mm2"] > 0.0
                else np.nan
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def add_pair_shape_metrics(pair_qc: pd.DataFrame) -> pd.DataFrame:
    if pair_qc.empty:
        return pair_qc
    mesh_cache: Dict[str, Any] = {}

    def cached(path_value: str):
        if path_value not in mesh_cache:
            mesh_cache[path_value] = load_mesh(Path(path_value))
        return mesh_cache[path_value]

    rows: List[Dict[str, Any]] = []
    for record in pair_qc.to_dict("records"):
        out = dict(record)
        out.update(
            {
                "shape_load_ok": False,
                "shape_error": "",
                "shape_rms_displacement": np.nan,
                "shape_p95_displacement": np.nan,
                "shape_max_displacement": np.nan,
                "shape_mean_abs_normal_displacement": np.nan,
                "shape_p95_abs_normal_displacement": np.nan,
                "shape_rms_displacement_per_year": np.nan,
                "shape_p95_displacement_per_year": np.nan,
                "shape_p95_abs_normal_displacement_per_year": np.nan,
                "shape_icp_like_mean_distance": np.nan,
            }
        )
        try:
            source_mesh = cached(str(record["source_mesh_path"]))
            target_mesh = cached(str(record["target_mesh_path"]))
            source_vertices = np.asarray(source_mesh.vertices, dtype=float)
            target_vertices = np.asarray(target_mesh.vertices, dtype=float)
            if source_vertices.shape != target_vertices.shape:
                raise ValueError(
                    f"topology mismatch: {source_vertices.shape} vs {target_vertices.shape}"
                )
            aligned_target = kabsch_align_points(target_vertices, source_vertices)
            displacement = aligned_target - source_vertices
            distances = np.linalg.norm(displacement, axis=1)
            normals = np.asarray(source_mesh.vertex_normals, dtype=float)
            signed_normal = np.einsum("ij,ij->i", displacement, normals)
            dt = max(float(record["dt_years"]), 1.0e-9)
            out.update(
                {
                    "shape_load_ok": True,
                    "shape_rms_displacement": float(np.sqrt(np.mean(distances**2))),
                    "shape_p95_displacement": float(np.quantile(distances, 0.95)),
                    "shape_max_displacement": float(np.max(distances)),
                    "shape_mean_abs_normal_displacement": float(np.mean(np.abs(signed_normal))),
                    "shape_p95_abs_normal_displacement": float(
                        np.quantile(np.abs(signed_normal), 0.95)
                    ),
                    "shape_rms_displacement_per_year": float(
                        np.sqrt(np.mean(distances**2)) / dt
                    ),
                    "shape_p95_displacement_per_year": float(
                        np.quantile(distances, 0.95) / dt
                    ),
                    "shape_p95_abs_normal_displacement_per_year": float(
                        np.quantile(np.abs(signed_normal), 0.95) / dt
                    ),
                    "shape_icp_like_mean_distance": float(np.mean(distances)),
                }
            )
        except Exception as exc:
            out["shape_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(out)
    return pd.DataFrame.from_records(rows)


def add_pair_flags(pair_qc: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, MappingLike]:
    result = pair_qc.copy()
    thresholds: MappingLike = {
        "mesh_rate_threshold_pct_per_year": float(args.mesh_rate_threshold_pct_per_year),
        "mask_rate_threshold_pct_per_year": float(args.mask_rate_threshold_pct_per_year),
        "mesh_mask_ratio_jump_threshold_pct": float(args.mesh_mask_ratio_jump_threshold_pct),
        "surface_area_jump_threshold_pct": float(args.surface_area_jump_threshold_pct),
        "sign_disagreement_min_delta_pct": float(args.sign_disagreement_min_delta_pct),
    }
    result["flag_huge_adjacent_mesh_jump"] = (
        result["mesh_abs_log_rate_pct_per_year"]
        > float(args.mesh_rate_threshold_pct_per_year)
    )
    result["flag_huge_adjacent_mask_jump"] = (
        result["mask_abs_log_rate_pct_per_year"]
        > float(args.mask_rate_threshold_pct_per_year)
    )
    min_delta = float(args.sign_disagreement_min_delta_pct)
    result["flag_mesh_mask_disagree_large"] = (
        result["mesh_mask_sign_disagree"].astype(bool)
        & (result["mesh_delta_pct"].abs() > min_delta)
        & (result["mask_delta_pct"].abs() > min_delta)
    )
    result["flag_mesh_mask_ratio_jump"] = (
        result["mesh_mask_ratio_delta_pct"].abs()
        > float(args.mesh_mask_ratio_jump_threshold_pct)
    )
    result["flag_surface_area_jump"] = (
        result["surface_area_delta_pct"].abs()
        > float(args.surface_area_jump_threshold_pct)
    )
    result["flag_pair_missing_volume"] = (
        result[
            [
                "source_mask_volume_mm3",
                "target_mask_volume_mm3",
                "source_mesh_volume_mm3",
                "target_mesh_volume_mm3",
            ]
        ]
        .isna()
        .any(axis=1)
    )
    shape_flag_cols: List[str] = []
    for metric in ("shape_p95_displacement_per_year", "shape_rms_displacement_per_year"):
        if metric in result.columns:
            finite = result[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite.empty:
                threshold = float(finite.quantile(float(args.shape_rate_outlier_quantile)))
                thresholds[f"{metric}_q{args.shape_rate_outlier_quantile}"] = threshold
                flag_col = f"flag_{metric}_outlier"
                result[flag_col] = result[metric] > threshold
                shape_flag_cols.append(flag_col)
    base_flag_cols = [
        "flag_huge_adjacent_mesh_jump",
        "flag_huge_adjacent_mask_jump",
        "flag_mesh_mask_disagree_large",
        "flag_mesh_mask_ratio_jump",
        "flag_surface_area_jump",
        "flag_pair_missing_volume",
    ]
    result["any_pair_qc_flag"] = result[base_flag_cols + shape_flag_cols].any(axis=1)
    result["strong_pair_qc_flag"] = result[
        [
            "flag_huge_adjacent_mesh_jump",
            "flag_huge_adjacent_mask_jump",
            "flag_mesh_mask_disagree_large",
            "flag_mesh_mask_ratio_jump",
            "flag_surface_area_jump",
        ]
        + shape_flag_cols
    ].any(axis=1)
    result["qc_score"] = (
        result["mesh_abs_log_rate_pct_per_year"].fillna(0.0)
        + result["mask_abs_log_rate_pct_per_year"].fillna(0.0)
        + result["mesh_mask_ratio_delta_pct"].abs().fillna(0.0)
        + 0.5 * result["surface_area_delta_pct"].abs().fillna(0.0)
        + 20.0 * result["mesh_mask_sign_disagree"].astype(float)
    )
    if "shape_p95_displacement_per_year" in result.columns:
        result["qc_score"] = result["qc_score"] + 20.0 * (
            result["shape_p95_displacement_per_year"].fillna(0.0)
            / max(
                float(
                    thresholds.get(
                        f"shape_p95_displacement_per_year_q{args.shape_rate_outlier_quantile}",
                        1.0,
                    )
                ),
                1.0e-9,
            )
        )
    return result, thresholds


def build_subject_qc(scan_qc: pd.DataFrame, pair_qc: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    subject_records: List[Dict[str, Any]] = []
    scan_flags = [column for column in scan_qc.columns if column.startswith("flag_scan_")]
    mesh_flags = [column for column in scan_qc.columns if column.startswith("flag_mesh_")]
    for (split, subject_id), scans in scan_qc.groupby(["split", "subject_id"], sort=False):
        scans = scans.sort_values(["visit_order", "months_from_baseline", "age_years"])
        pairs = pair_qc.loc[
            (pair_qc["split"] == split) & (pair_qc["subject_id"] == subject_id)
        ].copy()
        first = scans.iloc[0]
        last = scans.iloc[-1]
        scan_flag_count = int(scans[scan_flags + mesh_flags].any(axis=1).sum()) if scan_flags or mesh_flags else 0
        adjacent_pairs = int(len(pairs))
        flagged_pairs = int(pairs["strong_pair_qc_flag"].sum()) if adjacent_pairs else 0
        flag_fraction = flagged_pairs / adjacent_pairs if adjacent_pairs else 0.0
        max_mesh_rate = float(pairs["mesh_abs_log_rate_pct_per_year"].max()) if adjacent_pairs else np.nan
        max_mask_rate = float(pairs["mask_abs_log_rate_pct_per_year"].max()) if adjacent_pairs else np.nan
        max_ratio_jump = float(pairs["mesh_mask_ratio_delta_pct"].abs().max()) if adjacent_pairs else np.nan
        max_area_jump = float(pairs["surface_area_delta_pct"].abs().max()) if adjacent_pairs else np.nan
        max_shape_p95_rate = (
            float(pairs["shape_p95_displacement_per_year"].max())
            if adjacent_pairs and "shape_p95_displacement_per_year" in pairs.columns
            else np.nan
        )
        flag_any_pair = flagged_pairs > 0
        flag_many_bad_pairs = (
            flag_fraction >= float(args.subject_many_bad_pair_fraction)
            or flagged_pairs >= 2
        )
        flag_extreme_rate = (
            np.isfinite(max_mesh_rate)
            and max_mesh_rate > float(args.subject_extreme_rate_threshold_pct_per_year)
        )
        if flag_many_bad_pairs or flag_extreme_rate:
            suggested_action = "exclude_subject"
        elif flag_any_pair:
            suggested_action = "review_or_drop_bad_scan"
        elif scan_flag_count > 0:
            suggested_action = "review_scan_outlier"
        else:
            suggested_action = "pass"
        subject_records.append(
            {
                "split": str(split),
                "subject_id": str(subject_id),
                "diagnosis": str(first["diagnosis"]),
                "scan_count": int(len(scans)),
                "adjacent_pair_count": adjacent_pairs,
                "scan_flag_count": scan_flag_count,
                "flagged_pair_count": flagged_pairs,
                "flagged_pair_fraction": flag_fraction,
                "first_scan_id": str(first["scan_id"]),
                "last_scan_id": str(last["scan_id"]),
                "first_age_years": float(first.get("continuous_age_years", first.get("age_years", np.nan))),
                "last_age_years": float(last.get("continuous_age_years", last.get("age_years", np.nan))),
                "span_years": float(last.get("continuous_age_years", last.get("age_years", np.nan)))
                - float(first.get("continuous_age_years", first.get("age_years", np.nan))),
                "first_mesh_volume_mm3": float(first["left_mesh_volume_mm3"]),
                "last_mesh_volume_mm3": float(last["left_mesh_volume_mm3"]),
                "first_mask_volume_mm3": float(first["left_mask_volume_mm3"]),
                "last_mask_volume_mm3": float(last["left_mask_volume_mm3"]),
                "max_mesh_abs_log_rate_pct_per_year": max_mesh_rate,
                "max_mask_abs_log_rate_pct_per_year": max_mask_rate,
                "max_abs_mesh_mask_ratio_jump_pct": max_ratio_jump,
                "max_abs_surface_area_jump_pct": max_area_jump,
                "max_shape_p95_displacement_per_year": max_shape_p95_rate,
                "mesh_mask_sign_disagreement_count": int(pairs["mesh_mask_sign_disagree"].sum())
                if adjacent_pairs
                else 0,
                "flag_subject_any_bad_pair": bool(flag_any_pair),
                "flag_subject_many_bad_pairs": bool(flag_many_bad_pairs),
                "flag_subject_extreme_mesh_rate": bool(flag_extreme_rate),
                "suggested_action": suggested_action,
            }
        )
    return pd.DataFrame.from_records(subject_records)


def bad_scan_table(scan_qc: pd.DataFrame, pair_qc: pd.DataFrame) -> pd.DataFrame:
    scan_flags = [column for column in scan_qc.columns if column.startswith("flag_scan_")]
    mesh_flags = [column for column in scan_qc.columns if column.startswith("flag_mesh_")]
    scan_bad = scan_qc.copy()
    if scan_flags or mesh_flags:
        scan_bad["direct_scan_flag"] = scan_bad[scan_flags + mesh_flags].any(axis=1)
    else:
        scan_bad["direct_scan_flag"] = False
    bad_pair_scan_ids = set(
        pair_qc.loc[pair_qc["strong_pair_qc_flag"].astype(bool), "source_scan_id"].astype(str)
    ) | set(
        pair_qc.loc[pair_qc["strong_pair_qc_flag"].astype(bool), "target_scan_id"].astype(str)
    )
    scan_bad["participates_in_bad_adjacent_pair"] = scan_bad["scan_id"].isin(bad_pair_scan_ids)
    scan_bad["bad_scan"] = scan_bad["direct_scan_flag"] | scan_bad["participates_in_bad_adjacent_pair"]
    return scan_bad.loc[scan_bad["bad_scan"]].copy()


def filtered_metadata_outputs(
    metadata: pd.DataFrame,
    bad_scans: pd.DataFrame,
    subject_qc: pd.DataFrame,
    output_dir: Path,
    min_scans_after_filter: int,
) -> MappingLike:
    bad_scan_ids = set(bad_scans["scan_id"].astype(str))
    bad_subject_ids = set(
        subject_qc.loc[
            subject_qc["flag_subject_any_bad_pair"].astype(bool),
            "subject_id",
        ].astype(str)
    )
    exclude_subject_ids = set(
        subject_qc.loc[
            subject_qc["suggested_action"].isin(["exclude_subject"]),
            "subject_id",
        ].astype(str)
    )
    drop_any_bad_subject = metadata.loc[
        ~metadata["subject_id"].astype(str).isin(bad_subject_ids)
    ].copy()
    drop_exclude_subject = metadata.loc[
        ~metadata["subject_id"].astype(str).isin(exclude_subject_ids)
    ].copy()
    drop_bad_scans = metadata.loc[
        ~metadata["scan_id"].astype(str).isin(bad_scan_ids)
    ].copy()
    counts = drop_bad_scans.groupby(["split", "subject_id"], sort=False).size()
    keep_subjects = set(counts.loc[counts >= int(min_scans_after_filter)].index)
    drop_bad_scans_min_visits = (
        drop_bad_scans.set_index(["split", "subject_id"])
        .loc[list(keep_subjects)]
        .reset_index()
        if keep_subjects
        else drop_bad_scans.iloc[0:0].copy()
    )
    paths = {
        "metadata_drop_any_bad_subject": output_dir / "metadata_drop_any_bad_subject.csv",
        "metadata_drop_exclude_subject": output_dir / "metadata_drop_exclude_subject.csv",
        "metadata_drop_bad_scans": output_dir / "metadata_drop_bad_scans.csv",
        "metadata_drop_bad_scans_min_visits": output_dir
        / f"metadata_drop_bad_scans_min{int(min_scans_after_filter)}.csv",
    }
    drop_any_bad_subject.to_csv(paths["metadata_drop_any_bad_subject"], index=False)
    drop_exclude_subject.to_csv(paths["metadata_drop_exclude_subject"], index=False)
    drop_bad_scans.to_csv(paths["metadata_drop_bad_scans"], index=False)
    drop_bad_scans_min_visits.to_csv(paths["metadata_drop_bad_scans_min_visits"], index=False)
    return {
        "bad_scan_count": len(bad_scan_ids),
        "bad_subject_any_bad_pair_count": len(bad_subject_ids),
        "exclude_subject_count": len(exclude_subject_ids),
        "metadata_drop_any_bad_subject_rows": int(len(drop_any_bad_subject)),
        "metadata_drop_exclude_subject_rows": int(len(drop_exclude_subject)),
        "metadata_drop_bad_scans_rows": int(len(drop_bad_scans)),
        "metadata_drop_bad_scans_min_visits_rows": int(len(drop_bad_scans_min_visits)),
        "paths": paths,
    }


def try_import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except Exception:
        return None, None


def write_distribution_html(output_dir: Path, scan_qc: pd.DataFrame, pair_qc: pd.DataFrame) -> Optional[Path]:
    go, make_subplots = try_import_plotly()
    if go is None or make_subplots is None:
        return None
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "mesh/mask volume ratio",
            "surface area / volume^(2/3)",
            "adjacent mesh abs log-rate",
            "adjacent mask abs log-rate",
            "mesh/mask ratio jump",
            "surface-area jump",
        ),
    )
    figure.add_trace(go.Histogram(x=scan_qc["mesh_mask_ratio"], nbinsx=80), row=1, col=1)
    figure.add_trace(go.Histogram(x=scan_qc["surface_area_volume_ratio"], nbinsx=80), row=1, col=2)
    figure.add_trace(go.Histogram(x=pair_qc["mesh_abs_log_rate_pct_per_year"], nbinsx=80), row=1, col=3)
    figure.add_trace(go.Histogram(x=pair_qc["mask_abs_log_rate_pct_per_year"], nbinsx=80), row=2, col=1)
    figure.add_trace(go.Histogram(x=pair_qc["mesh_mask_ratio_delta_pct"], nbinsx=80), row=2, col=2)
    figure.add_trace(go.Histogram(x=pair_qc["surface_area_delta_pct"], nbinsx=80), row=2, col=3)
    figure.update_xaxes(title_text="ratio", row=1, col=1)
    figure.update_xaxes(title_text="shape compactness", row=1, col=2)
    figure.update_xaxes(title_text="%/year", row=1, col=3)
    figure.update_xaxes(title_text="%/year", row=2, col=1)
    figure.update_xaxes(title_text="%", row=2, col=2)
    figure.update_xaxes(title_text="%", row=2, col=3)
    figure.update_layout(
        title="Longitudinal mesh QC distributions",
        template="plotly_white",
        width=1400,
        height=850,
        showlegend=False,
    )
    path = output_dir / "qc_distributions.html"
    figure.write_html(path, include_plotlyjs=True)
    return path


def mesh_trace(mesh, name: str, color: str, opacity: float):
    go, _ = try_import_plotly()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=False,
        showscale=False,
    )


def write_top_pair_html(output_dir: Path, pair_qc: pd.DataFrame, count: int) -> List[Path]:
    go, make_subplots = try_import_plotly()
    if go is None or make_subplots is None:
        return []
    paths: List[Path] = []
    top = pair_qc.loc[pair_qc["strong_pair_qc_flag"].astype(bool)].sort_values(
        "qc_score",
        ascending=False,
    ).head(int(count))
    case_dir = output_dir / "top_bad_pair_html"
    case_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(top.to_dict("records"), start=1):
        try:
            source_mesh = load_mesh(Path(row["source_mesh_path"]))
            target_mesh = load_mesh(Path(row["target_mesh_path"]))
            figure = go.Figure()
            figure.add_trace(mesh_trace(source_mesh, "source", "#7f7f7f", 0.72))
            figure.add_trace(mesh_trace(target_mesh, "target", "#d62728", 0.45))
            title = (
                f"{idx:03d} {row['split']} subject {row['subject_id']} "
                f"{row['diagnosis']} | {row['source_scan_id']} -> {row['target_scan_id']} | "
                f"mesh delta {float(row['mesh_delta_pct']):.1f}% | "
                f"mask delta {float(row['mask_delta_pct']):.1f}%"
            )
            figure.update_layout(
                title=title,
                template="plotly_white",
                width=950,
                height=780,
                scene={
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "zaxis": {"visible": False},
                    "aspectmode": "data",
                },
            )
            path = case_dir / f"top_bad_pair_{idx:03d}_{row['source_scan_id']}_to_{row['target_scan_id']}.html"
            figure.write_html(path, include_plotlyjs=True)
            paths.append(path)
        except Exception:
            continue
    return paths


def dataframe_to_html_table(frame: pd.DataFrame, max_rows: int) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    return frame.head(max_rows).to_html(index=False, escape=True, border=0)


def write_index(
    output_dir: Path,
    *,
    summary: MappingLike,
    scan_qc: pd.DataFrame,
    pair_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    bad_scans: pd.DataFrame,
    distribution_html: Optional[Path],
    top_pair_paths: Sequence[Path],
    top_cases: int,
) -> Path:
    bad_pairs = pair_qc.loc[pair_qc["strong_pair_qc_flag"].astype(bool)].sort_values(
        "qc_score",
        ascending=False,
    )
    bad_subjects = subject_qc.loc[
        subject_qc["flag_subject_any_bad_pair"].astype(bool)
    ].sort_values(["suggested_action", "flagged_pair_count"], ascending=[True, False])
    min_visit_filtered_name = Path(
        summary["filtered_outputs"]["paths"]["metadata_drop_bad_scans_min_visits"]
    ).name
    links: List[str] = []
    for name in [
        "scan_qc.csv",
        "adjacent_pair_qc.csv",
        "bad_adjacent_pairs.csv",
        "bad_scans.csv",
        "subject_qc.csv",
        "bad_subjects.csv",
        "metadata_drop_any_bad_subject.csv",
        "metadata_drop_exclude_subject.csv",
        "metadata_drop_bad_scans.csv",
        min_visit_filtered_name,
    ]:
        links.append(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>")
    if distribution_html is not None:
        links.append(
            f"<li><a href='{html.escape(distribution_html.name)}'>{html.escape(distribution_html.name)}</a></li>"
        )
    for path in top_pair_paths:
        rel = path.relative_to(output_dir)
        links.append(f"<li><a href='{html.escape(str(rel))}'>{html.escape(path.name)}</a></li>")

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Longitudinal Mesh QC</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1280px; margin: 32px auto; color: #222; line-height: 1.45; }}
table {{ border-collapse: collapse; font-size: 12px; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid #ddd; padding: 4px 6px; white-space: nowrap; }}
th {{ background: #f2f2f2; }}
code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
.metric {{ border: 1px solid #ddd; padding: 10px; }}
.metric b {{ display: block; font-size: 18px; }}
</style>
</head>
<body>
<h1>Longitudinal Mesh QC</h1>
<div class="summary">
<div class="metric"><b>{summary['scan_count']}</b> scans</div>
<div class="metric"><b>{summary['subject_count']}</b> subjects</div>
<div class="metric"><b>{summary['adjacent_pair_count']}</b> adjacent pairs</div>
<div class="metric"><b>{summary['bad_adjacent_pair_count']}</b> bad adjacent pairs</div>
<div class="metric"><b>{summary['bad_scan_count']}</b> bad scans</div>
<div class="metric"><b>{summary['bad_subject_any_pair_count']}</b> subjects with bad pair</div>
<div class="metric"><b>{summary['exclude_subject_count']}</b> suggested exclude subjects</div>
<div class="metric"><b>{summary['mesh_shape_metrics_enabled']}</b> mesh-shape metrics</div>
</div>
<h2>Files</h2>
<ul>
{''.join(links)}
</ul>
<h2>Worst Adjacent Pairs</h2>
{dataframe_to_html_table(bad_pairs, top_cases)}
<h2>Bad Subjects</h2>
{dataframe_to_html_table(bad_subjects, top_cases)}
<h2>Bad Scans</h2>
{dataframe_to_html_table(bad_scans, top_cases)}
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def summarize_counts(
    scan_qc: pd.DataFrame,
    pair_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    bad_scans: pd.DataFrame,
    filtered_summary: MappingLike,
    thresholds: MappingLike,
    *,
    mesh_shape_metrics_enabled: bool,
) -> MappingLike:
    return {
        "scan_count": int(len(scan_qc)),
        "subject_count": int(scan_qc["subject_id"].nunique()),
        "adjacent_pair_count": int(len(pair_qc)),
        "bad_adjacent_pair_count": int(pair_qc["strong_pair_qc_flag"].sum())
        if not pair_qc.empty
        else 0,
        "bad_scan_count": int(len(bad_scans)),
        "bad_subject_any_pair_count": int(
            subject_qc["flag_subject_any_bad_pair"].sum()
        )
        if not subject_qc.empty
        else 0,
        "exclude_subject_count": int(
            (subject_qc["suggested_action"] == "exclude_subject").sum()
        )
        if not subject_qc.empty
        else 0,
        "mesh_shape_metrics_enabled": bool(mesh_shape_metrics_enabled),
        "scan_counts_by_split": scan_qc.groupby("split").size().to_dict(),
        "subject_counts_by_split": scan_qc.groupby("split")["subject_id"].nunique().to_dict(),
        "bad_pair_counts_by_split_diagnosis": pair_qc.loc[
            pair_qc["strong_pair_qc_flag"].astype(bool)
        ]
        .groupby(["split", "diagnosis"])
        .size()
        .to_dict()
        if not pair_qc.empty
        else {},
        "bad_scan_counts_by_split_diagnosis": bad_scans.groupby(["split", "diagnosis"])
        .size()
        .to_dict()
        if not bad_scans.empty
        else {},
        "filtered_outputs": filtered_summary,
        "thresholds": thresholds,
    }


def main() -> int:
    args = parse_args()
    experiment = Path(__file__).resolve().parents[1]
    metadata_path = resolve_path(args.metadata, base=experiment)
    output_dir = resolve_path(args.output_dir, base=experiment)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(metadata_path)
    scan_qc, scan_thresholds = add_scan_metadata_metrics(
        metadata,
        scan_outlier_quantile=float(args.scan_outlier_quantile),
    )

    mesh_shape_metrics_enabled = not bool(args.skip_mesh_shape_metrics)
    if mesh_shape_metrics_enabled:
        mesh_quality = mesh_quality_records(scan_qc)
        scan_qc = scan_qc.merge(mesh_quality, on=["scan_id", "mesh_path"], how="left")
    pair_qc = adjacent_pair_records(scan_qc)
    if mesh_shape_metrics_enabled:
        pair_qc = add_pair_shape_metrics(pair_qc)
    pair_qc, pair_thresholds = add_pair_flags(pair_qc, args)
    subject_qc = build_subject_qc(scan_qc, pair_qc, args)
    bad_scans = bad_scan_table(scan_qc, pair_qc)
    bad_pairs = pair_qc.loc[pair_qc["strong_pair_qc_flag"].astype(bool)].copy()
    bad_subjects = subject_qc.loc[
        subject_qc["flag_subject_any_bad_pair"].astype(bool)
    ].copy()

    scan_qc.to_csv(output_dir / "scan_qc.csv", index=False)
    pair_qc.to_csv(output_dir / "adjacent_pair_qc.csv", index=False)
    bad_pairs.sort_values("qc_score", ascending=False).to_csv(
        output_dir / "bad_adjacent_pairs.csv",
        index=False,
    )
    bad_scans.to_csv(output_dir / "bad_scans.csv", index=False)
    subject_qc.to_csv(output_dir / "subject_qc.csv", index=False)
    bad_subjects.sort_values(
        ["suggested_action", "flagged_pair_count"],
        ascending=[True, False],
    ).to_csv(output_dir / "bad_subjects.csv", index=False)

    filtered_summary = filtered_metadata_outputs(
        metadata,
        bad_scans,
        subject_qc,
        output_dir,
        min_scans_after_filter=int(args.min_scans_after_filter),
    )
    thresholds = {**scan_thresholds, **pair_thresholds}
    summary = summarize_counts(
        scan_qc,
        pair_qc,
        subject_qc,
        bad_scans,
        filtered_summary,
        thresholds,
        mesh_shape_metrics_enabled=mesh_shape_metrics_enabled,
    )
    write_json(output_dir / "qc_summary.json", summary)

    distribution_html = None
    top_pair_paths: List[Path] = []
    if not bool(args.no_html):
        distribution_html = write_distribution_html(output_dir, scan_qc, pair_qc)
        if args.export_top_pair_html and mesh_shape_metrics_enabled:
            top_pair_paths = write_top_pair_html(
                output_dir,
                pair_qc,
                count=int(args.top_pair_html_count),
            )
        index_path = write_index(
            output_dir,
            summary=summary,
            scan_qc=scan_qc,
            pair_qc=pair_qc,
            subject_qc=subject_qc,
            bad_scans=bad_scans,
            distribution_html=distribution_html,
            top_pair_paths=top_pair_paths,
            top_cases=int(args.top_cases),
        )
    else:
        index_path = None

    print(json.dumps(to_builtin(summary), indent=2))
    if index_path is not None:
        print(f"HTML index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
