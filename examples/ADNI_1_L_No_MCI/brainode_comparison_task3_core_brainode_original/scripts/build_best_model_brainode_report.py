#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import trimesh
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/best_model_brainode_comparison"
)
DEFAULT_FUTURE_MESH_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/future_mesh_forecast_comparison"
)

SPLITS = ("train", "val", "test")

FLOW_MODELS = {
    "old_pca32_siren": {
        "dataset": "old_adni",
        "family": "SIREN PCA32 flow",
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_siren_optimized",
        "role": "best old SIREN by test SDF improvement",
    },
    "old_smallnet_deepsdf": {
        "dataset": "old_adni",
        "family": "DeepSDF small-net flow",
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_smallnet_deepsdf_optimized",
        "role": "best old DeepSDF by test SDF error",
    },
    "old_siren_full256_direct": {
        "dataset": "old_adni",
        "family": "SIREN full256 direct flow",
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized",
        "role": "old rich visualization reference",
        "rich_dir": (
            "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
            "analysis/notebook_best"
        ),
        "volume_trend_csv": (
            "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
            "analysis/notebook_best/batch_observed_age_volume_trend.csv"
        ),
        "subject_volume_csv": (
            "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
            "analysis/notebook_best/batch_observed_age_subject_summary.csv"
        ),
    },
    "qc_siren_drop_bad_min2": {
        "dataset": "qc_large",
        "family": "SIREN QC direct/composed",
        "dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2"
        ),
        "role": "best QC SIREN by pair SDF and composed rollout",
        "rich_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/"
            "analysis/notebook_best"
        ),
        "speed_summary_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/"
            "analysis/notebook_best/selected_speed_summary.csv"
        ),
        "subject_volume_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/"
            "analysis/notebook_best/selected_subject_volume_summary.csv"
        ),
    },
    "qc_siren_local_decomp_volume": {
        "dataset": "qc_large",
        "family": "SIREN QC local decomposed volume",
        "dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1"
        ),
        "role": "QC volume-trend specialist",
        "rich_dir": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1/analysis/notebook_best"
        ),
        "speed_summary_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1/analysis/notebook_best/"
            "selected_speed_summary.csv"
        ),
        "subject_volume_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1/analysis/notebook_best/"
            "selected_subject_volume_summary.csv"
        ),
    },
}

BRAINODE_MODELS = {
    "old_brainode_pca150": {
        "dataset": "old_adni",
        "family": "BrainODE PCA150",
        "role": "old BrainODE baseline",
        "metrics_csv": (
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
            "training/core_attention_pca150/evaluation/best_pairwise/all_trajectory_metrics.csv"
        ),
    },
    "qc_brainode_pca150": {
        "dataset": "qc_large",
        "family": "BrainODE PCA150",
        "role": "QC BrainODE baseline",
        "metrics_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "brainode_pca150_qc_stable/training/core_attention_pca150_qc_stable/"
            "evaluation/best/all_trajectory_metrics.csv"
        ),
    },
}

RECONSTRUCTION_SOURCES = {
    "old_adni_task2": {
        "dataset": "old_adni",
        "kind": "full",
        "manifest_csv": (
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations_original/"
            "metadata/representation_manifest.csv"
        ),
        "metrics_csv": (
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations_original/"
            "evaluation/metrics/reconstruction_per_scan.csv"
        ),
    },
    "qc_large_siren_subset": {
        "dataset": "qc_large",
        "kind": "siren_subset",
        "manifest_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/"
            "metadata/representation_manifest.csv"
        ),
        "siren_chamfer_csv": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/inr/"
            "siren_naisr_5x512_warmstart_no_skip/chamfer_20_per_split_best/"
            "per_scan_chamfer.csv"
        ),
    },
}

BRAINODE_FUTURE_VOLUME_CSV = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/old_vs_qc_longitudinal_comparison/brainode_conditional_future_volume.csv"
)
BRAINODE_FUTURE_VOLUME_SUMMARY_CSV = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/old_vs_qc_longitudinal_comparison/brainode_conditional_future_volume_summary.csv"
)

ARTIFACT_PATTERNS = (
    "*ood*",
    "*counterfactual*",
    "*volume_forecast*",
    "*volume_trend*",
    "*direct_vs_composed*",
    "*interpolation*",
    "*pair_volume*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified best-model longitudinal BrainODE/SIREN/DeepSDF report."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument(
        "--skip-surface-area",
        action="store_true",
        help="Skip mesh surface-area calculations. Chamfer summaries are still reused.",
    )
    parser.add_argument(
        "--skip-qc-pca-chamfer",
        action="store_true",
        help="Skip QC PCA Chamfer computation on the QC SIREN 20-per-split subset.",
    )
    parser.add_argument(
        "--future-mesh-dir",
        default=DEFAULT_FUTURE_MESH_DIR,
        help=(
            "Directory produced by evaluate_future_mesh_forecasts.py. If the CSVs are "
            "missing, the report keeps the previous results and records the gap."
        ),
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def finite_count(values: list[float]) -> int:
    return sum(1 for value in values if math.isfinite(value))


def summary_stats(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def relative_path(target: str | Path, base: Path) -> str:
    target_path = Path(target)
    try:
        return os.path.relpath(target_path, start=base)
    except ValueError:
        return str(target_path)


def first_overall_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("grouping") == "overall":
            return row
    return rows[0] if rows else None


def flow_split_metrics() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, spec in FLOW_MODELS.items():
        run_dir = repo_path(spec["dir"])
        for split in SPLITS:
            summary_path = run_dir / "analysis" / "checkpoint_best" / f"{split}_summary.csv"
            summary_row = first_overall_row(read_csv(summary_path))
            base = {
                "dataset": spec["dataset"],
                "method": method,
                "family": spec["family"],
                "role": spec["role"],
                "split": split,
                "metric_space": "decoder_sdf_l1",
                "source_file": str(summary_path),
                "note": "SIREN/DeepSDF prediction error is decoder SDF L1.",
            }
            if not summary_row:
                output.append(
                    {
                        **base,
                        "status": "missing",
                        "rows": "",
                        "model_error_mean": "",
                        "no_change_error_mean": "",
                        "improvement_mean": "",
                        "beats_no_change_fraction": "",
                        "composed_error_mean": "",
                        "composed_improvement_mean": "",
                        "composed_beats_no_change_fraction": "",
                        "composed_beats_direct_fraction": "",
                    }
                )
                continue
            output.append(
                {
                    **base,
                    "status": "ok",
                    "rows": summary_row.get("rows", ""),
                    "model_error_mean": summary_row.get("model_target_sdf_l1_mean", ""),
                    "no_change_error_mean": summary_row.get("no_change_target_sdf_l1_mean", ""),
                    "improvement_mean": summary_row.get("sdf_l1_improvement_mean", ""),
                    "beats_no_change_fraction": summary_row.get(
                        "model_beats_no_change_fraction", ""
                    ),
                    "composed_error_mean": summary_row.get("composed_target_sdf_l1_mean", ""),
                    "composed_improvement_mean": summary_row.get(
                        "composed_sdf_l1_improvement_mean", ""
                    ),
                    "composed_beats_no_change_fraction": summary_row.get(
                        "composed_beats_no_change_fraction", ""
                    ),
                    "composed_beats_direct_fraction": summary_row.get(
                        "composed_beats_direct_fraction", ""
                    ),
                }
            )
    return output


def brainode_split_metrics() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, spec in BRAINODE_MODELS.items():
        metric_path = repo_path(spec["metrics_csv"])
        rows = read_csv(metric_path)
        for split in SPLITS:
            split_rows = [row for row in rows if row.get("split") == split]
            base = {
                "dataset": spec["dataset"],
                "method": method,
                "family": spec["family"],
                "role": spec["role"],
                "split": split,
                "metric_space": "pca_inverse_vertex_mae",
                "source_file": str(metric_path),
                "note": "BrainODE prediction error is PCA inverse-transform vertex MAE.",
            }
            if not split_rows:
                output.append(
                    {
                        **base,
                        "status": "missing",
                        "rows": "",
                        "model_error_mean": "",
                        "no_change_error_mean": "",
                        "improvement_mean": "",
                        "beats_no_change_fraction": "",
                        "composed_error_mean": "",
                        "composed_improvement_mean": "",
                        "composed_beats_no_change_fraction": "",
                        "composed_beats_direct_fraction": "",
                    }
                )
                continue
            improvements = [as_float(row.get("endpoint_vertex_mae_improvement")) for row in split_rows]
            output.append(
                {
                    **base,
                    "status": "ok",
                    "rows": len(split_rows),
                    "model_error_mean": safe_mean(
                        [as_float(row.get("endpoint_vertex_mae")) for row in split_rows]
                    ),
                    "no_change_error_mean": safe_mean(
                        [as_float(row.get("no_change_vertex_mae")) for row in split_rows]
                    ),
                    "improvement_mean": safe_mean(improvements),
                    "beats_no_change_fraction": safe_mean(
                        [1.0 if value > 0.0 else 0.0 for value in improvements if math.isfinite(value)]
                    ),
                    "composed_error_mean": "",
                    "composed_improvement_mean": "",
                    "composed_beats_no_change_fraction": "",
                    "composed_beats_direct_fraction": "",
                }
            )
    return output


def direct_vs_composed_rows(split_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in split_metrics:
        if row.get("metric_space") != "decoder_sdf_l1" or row.get("status") != "ok":
            continue
        direct = as_float(row.get("model_error_mean"))
        composed = as_float(row.get("composed_error_mean"))
        if not math.isfinite(composed):
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "split": row["split"],
                "direct_sdf_l1_mean": direct,
                "composed_sdf_l1_mean": composed,
                "composed_minus_direct_sdf_l1": composed - direct,
                "direct_improvement_mean": row.get("improvement_mean", ""),
                "composed_improvement_mean": row.get("composed_improvement_mean", ""),
                "composed_beats_no_change_fraction": row.get(
                    "composed_beats_no_change_fraction", ""
                ),
                "composed_beats_direct_fraction": row.get("composed_beats_direct_fraction", ""),
                "source_file": row.get("source_file", ""),
            }
        )
    return rows


def volume_rates_from_trend(
    dataset: str,
    method: str,
    source_file: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(source_file)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("split", ""),
            row.get("subject_id", ""),
            row.get("diagnosis", ""),
            row.get("transport_method", ""),
        )
        grouped[key].append(row)

    rates: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    subject_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (split, subject_id, diagnosis, transport), visits in grouped.items():
        visits.sort(key=lambda row: as_float(row.get("years_from_baseline")))
        for previous, current in zip(visits, visits[1:]):
            v0 = as_float(previous.get("volume"))
            v1 = as_float(current.get("volume"))
            t0 = as_float(previous.get("years_from_baseline"))
            t1 = as_float(current.get("years_from_baseline"))
            if not (
                math.isfinite(v0)
                and math.isfinite(v1)
                and math.isfinite(t0)
                and math.isfinite(t1)
                and abs(v0) > 0.0
                and t1 > t0
            ):
                continue
            atrophy_pct_per_year = ((v0 - v1) / abs(v0)) * 100.0 / (t1 - t0)
            key = (split, diagnosis, transport)
            rates[key].append(atrophy_pct_per_year)
            subject_sets[key].add(subject_id)

    output: list[dict[str, Any]] = []
    for key, values in sorted(rates.items()):
        split, diagnosis, transport = key
        stats = summary_stats(values)
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": transport,
                "intervals": stats["count"],
                "subjects": len(subject_sets[key]),
                "mean_atrophy_pct_per_year": stats["mean"],
                "median_atrophy_pct_per_year": stats["median"],
                "source_file": str(source_file),
                "status": "ok",
            }
        )
    return output


def volume_trend_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, spec in FLOW_MODELS.items():
        if "volume_trend_csv" in spec:
            path = repo_path(spec["volume_trend_csv"])
            if path.is_file():
                output.extend(volume_rates_from_trend(spec["dataset"], method, path))
        elif "speed_summary_csv" in spec:
            path = repo_path(spec["speed_summary_csv"])
            for row in read_csv(path):
                output.append(
                    {
                        "dataset": spec["dataset"],
                        "method": method,
                        "split": row.get("split", ""),
                        "diagnosis": row.get("diagnosis", ""),
                        "transport_method": row.get("transport_method", ""),
                        "intervals": row.get("intervals", ""),
                        "subjects": row.get("subjects", ""),
                        "mean_atrophy_pct_per_year": row.get("mean_atrophy_pct_per_year", ""),
                        "median_atrophy_pct_per_year": "",
                        "source_file": str(path),
                        "status": "ok",
                    }
                )
    return output


def subject_volume_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, spec in FLOW_MODELS.items():
        path_value = spec.get("subject_volume_csv")
        if not path_value:
            continue
        path = repo_path(path_value)
        for row in read_csv(path):
            output.append(
                {
                    "dataset": spec["dataset"],
                    "method": method,
                    "split": row.get("split", ""),
                    "subject_id": row.get("subject_id", ""),
                    "diagnosis": row.get("diagnosis", ""),
                    "num_scans": row.get("num_scans", ""),
                    "base_age_years": row.get("base_age_years", ""),
                    "final_age_years": row.get("final_age_years", ""),
                    "span_years": row.get("span_years", ""),
                    "base_real_volume": row.get("base_real_volume", ""),
                    "base_recon_volume": row.get("base_recon_volume", ""),
                    "final_real_volume": row.get("final_real_volume", ""),
                    "final_predicted_volume": row.get("final_predicted_volume", ""),
                    "final_composed_volume": row.get("final_composed_volume", ""),
                    "real_delta_pct": row.get("real_delta_pct", ""),
                    "predicted_delta_pct": row.get("predicted_delta_pct", ""),
                    "composed_delta_pct": row.get("composed_delta_pct", ""),
                    "final_abs_volume_error": row.get("final_abs_volume_error", ""),
                    "final_composed_abs_volume_error": row.get(
                        "final_composed_abs_volume_error", ""
                    ),
                    "final_delta_sign_match": row.get("final_delta_sign_match", ""),
                    "composed_final_delta_sign_match": row.get(
                        "composed_final_delta_sign_match", ""
                    ),
                    "decode_success_fraction": row.get("decode_success_fraction", ""),
                    "source_file": str(path),
                    "status": "ok",
                }
            )
    return output


def load_mesh(path: str | Path) -> trimesh.Trimesh | None:
    if not path:
        return None
    mesh_path = Path(path)
    if not mesh_path.is_file():
        return None
    try:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
            return None
        return mesh
    except Exception:
        return None


def mesh_area(path: str | Path, cache: dict[str, float]) -> float:
    key = str(path)
    if key in cache:
        return cache[key]
    mesh = load_mesh(path)
    value = float(mesh.area) if mesh is not None else float("nan")
    cache[key] = value
    return value


def deterministic_seed(*parts: str) -> int:
    text = "|".join(parts)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**31 - 1)


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def surface_chamfer(
    gt_mesh_path: str | Path,
    pred_mesh_path: str | Path,
    sample_count: int,
    seed_key: str,
) -> dict[str, float]:
    gt_mesh = load_mesh(gt_mesh_path)
    pred_mesh = load_mesh(pred_mesh_path)
    if gt_mesh is None or pred_mesh is None:
        return {
            "gt_to_pred_mean": float("nan"),
            "pred_to_gt_mean": float("nan"),
            "chamfer_l1": float("nan"),
            "chamfer_l2_squared": float("nan"),
            "assd": float("nan"),
        }
    gt_points = sample_surface(gt_mesh, sample_count, deterministic_seed(seed_key, "gt"))
    pred_points = sample_surface(pred_mesh, sample_count, deterministic_seed(seed_key, "pred"))
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    gt_mean = float(np.mean(gt_to_pred))
    pred_mean = float(np.mean(pred_to_gt))
    return {
        "gt_to_pred_mean": gt_mean,
        "pred_to_gt_mean": pred_mean,
        "chamfer_l1": gt_mean + pred_mean,
        "chamfer_l2_squared": float(np.mean(gt_to_pred**2) + np.mean(pred_to_gt**2)),
        "assd": 0.5 * (gt_mean + pred_mean),
    }


def old_reconstruction_rows(skip_surface_area: bool) -> list[dict[str, Any]]:
    source = RECONSTRUCTION_SOURCES["old_adni_task2"]
    manifest_path = repo_path(source["manifest_csv"])
    metrics_path = repo_path(source["metrics_csv"])
    manifest = {row["scan_id"]: row for row in read_csv(manifest_path)}
    area_cache: dict[str, float] = {}
    output: list[dict[str, Any]] = []
    for row in read_csv(metrics_path):
        scan_id = row.get("scan_id", "")
        manifest_row = manifest.get(scan_id, {})
        gt_path = manifest_row.get("ground_truth_mesh_path", "")
        pred_path = row.get("predicted_mesh_path", "")
        gt_area = float("nan")
        pred_area = float("nan")
        if not skip_surface_area:
            gt_area = mesh_area(gt_path, area_cache)
            pred_area = mesh_area(pred_path, area_cache)
        output.append(
            {
                "dataset": "old_adni",
                "evaluation_set": "full_727_scans",
                "method": row.get("method", ""),
                "scan_id": scan_id,
                "subject_id": row.get("subject_id", ""),
                "split": row.get("split", ""),
                "diagnosis": row.get("diagnosis", ""),
                "visit_order": row.get("visit_order", ""),
                "chamfer_l2_squared": row.get("chamfer_l2_squared", ""),
                "assd": row.get("assd", ""),
                "hd95": row.get("hd95", ""),
                "ground_truth_volume": row.get("ground_truth_volume", ""),
                "predicted_volume": row.get("predicted_volume", ""),
                "volume_absolute_error": row.get("volume_absolute_error", ""),
                "volume_relative_error": row.get("volume_relative_error", ""),
                "ground_truth_surface_area": gt_area,
                "predicted_surface_area": pred_area,
                "surface_area_absolute_error": (
                    abs(pred_area - gt_area)
                    if math.isfinite(gt_area) and math.isfinite(pred_area)
                    else float("nan")
                ),
                "surface_area_relative_error": (
                    abs(pred_area - gt_area) / abs(gt_area)
                    if math.isfinite(gt_area) and math.isfinite(pred_area) and gt_area != 0.0
                    else float("nan")
                ),
                "ground_truth_mesh_path": gt_path,
                "predicted_mesh_path": pred_path,
                "status": "ok",
                "note": "Chamfer/ASSD were reused from Task2 sampled-surface reconstruction metrics.",
            }
        )
    return output


def qc_reconstruction_rows(skip_surface_area: bool, skip_qc_pca_chamfer: bool, samples: int) -> list[dict[str, Any]]:
    source = RECONSTRUCTION_SOURCES["qc_large_siren_subset"]
    manifest_path = repo_path(source["manifest_csv"])
    chamfer_path = repo_path(source["siren_chamfer_csv"])
    manifest = {row["scan_id"]: row for row in read_csv(manifest_path)}
    area_cache: dict[str, float] = {}
    output: list[dict[str, Any]] = []
    siren_rows = read_csv(chamfer_path)

    for row in siren_rows:
        scan_id = row.get("scan_id", "")
        gt_path = row.get("ground_truth_mesh_path", "")
        pred_path = row.get("predicted_mesh_path", "")
        gt_area = float("nan")
        pred_area = float("nan")
        if not skip_surface_area:
            gt_area = mesh_area(gt_path, area_cache)
            pred_area = mesh_area(pred_path, area_cache)
        output.append(
            {
                "dataset": "qc_large",
                "evaluation_set": "siren_20_per_split_subset",
                "method": "qc_siren_task2_no_skip_subset",
                "scan_id": scan_id,
                "subject_id": row.get("subject_id", ""),
                "split": row.get("split", ""),
                "diagnosis": row.get("diagnosis", ""),
                "visit_order": row.get("visit_order", ""),
                "chamfer_l2_squared": row.get("chamfer_l2_squared", ""),
                "assd": row.get("assd", ""),
                "hd95": row.get("hd95", ""),
                "ground_truth_volume": row.get("ground_truth_volume", ""),
                "predicted_volume": row.get("predicted_volume", ""),
                "volume_absolute_error": row.get("volume_absolute_error", ""),
                "volume_relative_error": row.get("volume_relative_error", ""),
                "ground_truth_surface_area": gt_area,
                "predicted_surface_area": pred_area,
                "surface_area_absolute_error": (
                    abs(pred_area - gt_area)
                    if math.isfinite(gt_area) and math.isfinite(pred_area)
                    else float("nan")
                ),
                "surface_area_relative_error": (
                    abs(pred_area - gt_area) / abs(gt_area)
                    if math.isfinite(gt_area) and math.isfinite(pred_area) and gt_area != 0.0
                    else float("nan")
                ),
                "ground_truth_mesh_path": gt_path,
                "predicted_mesh_path": pred_path,
                "status": "ok",
                "note": "QC SIREN Chamfer reused from the saved 20-per-split sampled-surface subset.",
            }
        )

    if skip_qc_pca_chamfer:
        return output

    for row in siren_rows:
        scan_id = row.get("scan_id", "")
        manifest_row = manifest.get(scan_id, {})
        gt_path = manifest_row.get("ground_truth_mesh_path", row.get("ground_truth_mesh_path", ""))
        for method, path_key in (
            ("qc_pca_150_subset", "pca_150_mesh_path"),
            ("qc_pca_256_subset", "pca_256_mesh_path"),
        ):
            pred_path = manifest_row.get(path_key, "")
            distances = surface_chamfer(gt_path, pred_path, samples, f"{method}:{scan_id}")
            gt_area = float("nan")
            pred_area = float("nan")
            if not skip_surface_area:
                gt_area = mesh_area(gt_path, area_cache)
                pred_area = mesh_area(pred_path, area_cache)
            gt_volume = as_float(row.get("ground_truth_volume"))
            pred_mesh = load_mesh(pred_path)
            pred_volume = float(abs(pred_mesh.volume)) if pred_mesh is not None else float("nan")
            volume_abs = abs(pred_volume - gt_volume) if math.isfinite(pred_volume) else float("nan")
            output.append(
                {
                    "dataset": "qc_large",
                    "evaluation_set": "siren_20_per_split_subset",
                    "method": method,
                    "scan_id": scan_id,
                    "subject_id": row.get("subject_id", ""),
                    "split": row.get("split", ""),
                    "diagnosis": row.get("diagnosis", ""),
                    "visit_order": row.get("visit_order", ""),
                    "chamfer_l2_squared": distances["chamfer_l2_squared"],
                    "assd": distances["assd"],
                    "hd95": "",
                    "ground_truth_volume": gt_volume,
                    "predicted_volume": pred_volume,
                    "volume_absolute_error": volume_abs,
                    "volume_relative_error": (
                        volume_abs / abs(gt_volume)
                        if math.isfinite(volume_abs) and gt_volume != 0.0
                        else float("nan")
                    ),
                    "ground_truth_surface_area": gt_area,
                    "predicted_surface_area": pred_area,
                    "surface_area_absolute_error": (
                        abs(pred_area - gt_area)
                        if math.isfinite(gt_area) and math.isfinite(pred_area)
                        else float("nan")
                    ),
                    "surface_area_relative_error": (
                        abs(pred_area - gt_area) / abs(gt_area)
                        if math.isfinite(gt_area) and math.isfinite(pred_area) and gt_area != 0.0
                        else float("nan")
                    ),
                    "ground_truth_mesh_path": gt_path,
                    "predicted_mesh_path": pred_path,
                    "status": "ok",
                    "note": (
                        "QC PCA Chamfer computed on the same 60 scans as the saved QC SIREN "
                        f"subset using {samples} sampled surface points per mesh."
                    ),
                }
            )
    return output


def reconstruction_summary(per_scan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_scan_rows:
        grouped[
            (
                str(row.get("dataset", "")),
                str(row.get("evaluation_set", "")),
                str(row.get("method", "")),
                str(row.get("split", "")),
                str(row.get("diagnosis", "")),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, evaluation_set, method, split, diagnosis), rows in sorted(grouped.items()):
        metric_values = {
            name: [as_float(row.get(name)) for row in rows]
            for name in (
                "chamfer_l2_squared",
                "assd",
                "hd95",
                "volume_relative_error",
                "surface_area_relative_error",
            )
        }
        out = {
            "dataset": dataset,
            "evaluation_set": evaluation_set,
            "method": method,
            "split": split,
            "diagnosis": diagnosis,
            "rows": len(rows),
        }
        for name, values in metric_values.items():
            stats = summary_stats(values)
            out[f"{name}_count"] = stats["count"]
            out[f"{name}_mean"] = stats["mean"]
            out[f"{name}_median"] = stats["median"]
            out[f"{name}_std"] = stats["std"]
        output.append(out)
    return output


def rich_artifact_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, spec in FLOW_MODELS.items():
        rich = spec.get("rich_dir")
        if not rich:
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "method": method,
                    "artifact_group": "rich_visualization",
                    "artifact_name": "",
                    "path": "",
                    "relative_link": "",
                    "status": "missing",
                    "note": "No rich visualization directory is configured for this model.",
                }
            )
            continue
        rich_dir = repo_path(rich)
        if not rich_dir.is_dir():
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "method": method,
                    "artifact_group": "rich_visualization",
                    "artifact_name": "",
                    "path": str(rich_dir),
                    "relative_link": "",
                    "status": "missing",
                    "note": "Configured rich visualization directory does not exist.",
                }
            )
            continue
        seen: set[Path] = set()
        for pattern in ARTIFACT_PATTERNS:
            for artifact in sorted(rich_dir.glob(pattern)):
                if artifact in seen or not artifact.is_file():
                    continue
                seen.add(artifact)
                rows.append(
                    {
                        "dataset": spec["dataset"],
                        "method": method,
                        "artifact_group": "forecast_ood_volume",
                        "artifact_name": artifact.name,
                        "path": str(artifact),
                        "relative_link": relative_path(artifact, output_dir),
                        "status": "ok",
                        "note": "Existing rich HTML/CSV artifact reused by this report.",
                    }
                )
    return rows


def brainode_future_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_csv(repo_path(BRAINODE_FUTURE_VOLUME_CSV)), read_csv(
        repo_path(BRAINODE_FUTURE_VOLUME_SUMMARY_CSV)
    )


def future_mesh_rows(future_mesh_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    per_pair_path = future_mesh_dir / "future_mesh_per_pair.csv"
    summary_path = future_mesh_dir / "future_mesh_summary.csv"
    per_pair = read_csv(per_pair_path)
    summary = read_csv(summary_path)
    return per_pair, summary, bool(per_pair and summary)


def coverage_gap_rows(future_mesh_available: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, spec in FLOW_MODELS.items():
        if "rich_dir" not in spec:
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "method": method,
                    "gap": "no_rich_volume_ood_html",
                    "impact": (
                        "Pair metrics exist, but selected subject forecast, OOD, volume-trend, "
                        "and shape visualization pages were not generated for this model."
                    ),
                    "suggested_action": (
                        "Run the model-specific rich visualization analysis if these plots are needed."
                    ),
                }
            )
    if not future_mesh_available:
        rows.extend(
            [
                {
                    "dataset": "old_adni/qc_large",
                    "method": "brainode_and_longitudinal_flow_models",
                    "gap": "future_mesh_real_target_evaluation_not_run",
                    "impact": (
                        "Static reconstruction Chamfer exists, but the fair future-shape test "
                        "against true target meshes has not been run yet."
                    ),
                    "suggested_action": (
                        "Run scripts/evaluate_future_mesh_forecasts.py, then rerun this report "
                        "builder to import future_mesh_summary.csv."
                    ),
                },
                {
                    "dataset": "old_adni/qc_large",
                    "method": "all_longitudinal_flow_models",
                    "gap": "predicted_future_surface_area_change",
                    "impact": (
                        "Existing rich longitudinal outputs save volume trends but not meshes or "
                        "surface-area values for each predicted future shape."
                    ),
                    "suggested_action": (
                        "Run evaluate_future_mesh_forecasts.py to decode predicted future meshes "
                        "and compute Chamfer, volume, and triangle surface area."
                    ),
                },
                {
                    "dataset": "brainode",
                    "method": "old_brainode_pca150/qc_brainode_pca150",
                    "gap": "split_wise_endpoint_volume_not_saved",
                    "impact": (
                        "BrainODE pairwise train/val/test endpoint errors exist, but endpoint "
                        "predicted volumes are not saved in the original evaluator."
                    ),
                    "suggested_action": (
                        "Run evaluate_future_mesh_forecasts.py to write endpoint inverse-PCA "
                        "mesh volumes and surface metrics."
                    ),
                },
            ]
        )
    rows.append(
        {
            "dataset": "qc_large",
            "method": "deepsdf",
            "gap": "qc_deepsdf_reconstruction_missing",
            "impact": "No QC-large DeepSDF reconstructed mesh/Chamfer result was found.",
            "suggested_action": "Train or reconstruct QC DeepSDF before adding it to QC Chamfer comparison.",
        }
    )
    return rows


def html_table(df: pd.DataFrame, max_rows: int = 30, precision: int = 5) -> str:
    if df.empty:
        return "<p class='muted'>No rows available.</p>"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{precision}g}"
            )
    suffix = ""
    if len(df) > max_rows:
        suffix = f"<p class='muted'>Showing {max_rows} of {len(df)} rows.</p>"
    return view.to_html(index=False, escape=True, classes="data-table") + suffix


def fig_to_html(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=True if include_js else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def make_metric_figures(
    split_metrics: pd.DataFrame,
    volume_trends: pd.DataFrame,
    recon_summary: pd.DataFrame,
    future_mesh_summary: pd.DataFrame,
) -> list[str]:
    parts: list[str] = []
    include_js = True

    numeric = split_metrics.copy()
    numeric["improvement_mean"] = pd.to_numeric(numeric["improvement_mean"], errors="coerce")
    numeric = numeric[numeric["status"] == "ok"]
    if not numeric.empty:
        fig = px.bar(
            numeric,
            x="method",
            y="improvement_mean",
            color="split",
            facet_row="metric_space",
            barmode="group",
            title="Prediction Improvement Over No-Change Baseline",
            labels={"improvement_mean": "mean improvement"},
        )
        fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=140))
        parts.append(fig_to_html(fig, include_js))
        include_js = False

    vol = volume_trends.copy()
    if not vol.empty:
        vol["mean_atrophy_pct_per_year"] = pd.to_numeric(
            vol["mean_atrophy_pct_per_year"], errors="coerce"
        )
        vol = vol[vol["status"] == "ok"]
        fig = px.bar(
            vol,
            x="transport_method",
            y="mean_atrophy_pct_per_year",
            color="diagnosis",
            facet_col="split",
            facet_row="method",
            title="Volume Trend by Split and Diagnosis",
            labels={"mean_atrophy_pct_per_year": "mean atrophy %/year"},
        )
        fig.update_layout(height=max(520, 190 * max(1, vol["method"].nunique())), margin=dict(l=40, r=20, t=70, b=120))
        parts.append(fig_to_html(fig, include_js))
        include_js = False

    recon = recon_summary.copy()
    if not recon.empty:
        recon["chamfer_l2_squared_mean"] = pd.to_numeric(
            recon["chamfer_l2_squared_mean"], errors="coerce"
        )
        plot_recon = recon[recon["diagnosis"].isin(["AD", "CN"])].copy()
        fig = px.bar(
            plot_recon,
            x="method",
            y="chamfer_l2_squared_mean",
            color="diagnosis",
            facet_col="split",
            facet_row="dataset",
            title="Reconstruction Chamfer From Sampled Surfaces",
            labels={"chamfer_l2_squared_mean": "mean squared symmetric Chamfer"},
        )
        fig.update_layout(height=650, margin=dict(l=40, r=20, t=70, b=150))
        parts.append(fig_to_html(fig, include_js))
        include_js = False

        area_col = "surface_area_relative_error_mean"
        if area_col in recon:
            recon[area_col] = pd.to_numeric(recon[area_col], errors="coerce")
            area_recon = recon[recon["diagnosis"].isin(["AD", "CN"])].copy()
            fig = px.bar(
                area_recon,
                x="method",
                y=area_col,
                color="diagnosis",
                facet_col="split",
                facet_row="dataset",
                title="Reconstruction Surface-Area Relative Error",
                labels={area_col: "mean relative area error"},
            )
            fig.update_layout(height=650, margin=dict(l=40, r=20, t=70, b=150))
            parts.append(fig_to_html(fig, include_js))

    future = future_mesh_summary.copy()
    if not future.empty:
        for column in (
            "chamfer_l2_squared_mean",
            "assd_mean",
            "hd95_mean",
            "volume_relative_error_mean",
            "surface_area_relative_error_mean",
            "decode_success_fraction",
        ):
            if column in future:
                future[column] = pd.to_numeric(future[column], errors="coerce")
        future_plot = future[
            future["transport_method"].isin(
                ["brainode_endpoint", "direct", "composed", "model_no_change"]
            )
        ].copy()
        if not future_plot.empty:
            fig = px.bar(
                future_plot,
                x="transport_method",
                y="chamfer_l2_squared_mean",
                color="diagnosis",
                facet_col="split",
                facet_row="model",
                title="Future Forecast Real-Mesh Chamfer Against True Target",
                labels={"chamfer_l2_squared_mean": "mean squared symmetric Chamfer"},
            )
            fig.update_layout(
                height=max(620, 185 * max(1, future_plot["model"].nunique())),
                margin=dict(l=40, r=20, t=70, b=130),
            )
            parts.append(fig_to_html(fig, include_js))
            include_js = False

            fig = px.bar(
                future_plot,
                x="transport_method",
                y="surface_area_relative_error_mean",
                color="diagnosis",
                facet_col="split",
                facet_row="model",
                title="Future Forecast Surface-Area Error Against True Target",
                labels={"surface_area_relative_error_mean": "mean relative area error"},
            )
            fig.update_layout(
                height=max(620, 185 * max(1, future_plot["model"].nunique())),
                margin=dict(l=40, r=20, t=70, b=130),
            )
            parts.append(fig_to_html(fig, include_js))
    return parts


def make_artifact_link_table(artifacts: pd.DataFrame, output_dir: Path) -> str:
    if artifacts.empty:
        return "<p class='muted'>No reusable artifacts found.</p>"
    rows = []
    html_rows = []
    for _, row in artifacts[artifacts["status"] == "ok"].head(120).iterrows():
        link = str(row.get("relative_link", ""))
        name = html.escape(str(row.get("artifact_name", "")))
        method = html.escape(str(row.get("method", "")))
        dataset = html.escape(str(row.get("dataset", "")))
        if link:
            name_html = f"<a href='{html.escape(link)}'>{name}</a>"
        else:
            name_html = name
        html_rows.append(
            f"<tr><td>{dataset}</td><td>{method}</td><td>{name_html}</td></tr>"
        )
        rows.append(row)
    if not html_rows:
        return "<p class='muted'>No artifact links available.</p>"
    extra = ""
    if len(artifacts[artifacts["status"] == "ok"]) > len(html_rows):
        extra = (
            f"<p class='muted'>Showing {len(html_rows)} of "
            f"{len(artifacts[artifacts['status'] == 'ok'])} reusable artifact links.</p>"
        )
    return (
        "<table class='data-table'><thead><tr><th>dataset</th><th>method</th>"
        "<th>artifact</th></tr></thead><tbody>"
        + "\n".join(html_rows)
        + "</tbody></table>"
        + extra
    )


def build_html_report(
    output_dir: Path,
    split_metrics: pd.DataFrame,
    direct_composed: pd.DataFrame,
    volume_trends: pd.DataFrame,
    subject_volume: pd.DataFrame,
    recon_summary: pd.DataFrame,
    future_mesh_summary: pd.DataFrame,
    brainode_future_summary: pd.DataFrame,
    artifacts: pd.DataFrame,
    gaps: pd.DataFrame,
) -> str:
    figures = make_metric_figures(
        split_metrics,
        volume_trends,
        recon_summary,
        future_mesh_summary,
    )
    best_numeric = split_metrics.copy()
    best_numeric["improvement_mean"] = pd.to_numeric(best_numeric["improvement_mean"], errors="coerce")
    best_numeric = best_numeric[best_numeric["status"] == "ok"].sort_values(
        ["dataset", "metric_space", "split", "improvement_mean"],
        ascending=[True, True, True, False],
    )

    files = {
        "split_numeric_metrics.csv": "Train/val/test pair prediction metrics.",
        "direct_vs_composed_metrics.csv": "Direct versus composed rollout metrics.",
        "volume_trend_by_split.csv": "Volume trend by split/diagnosis/transport method.",
        "subject_volume_forecasts.csv": "Selected subject final volume deltas.",
        "reconstruction_chamfer_area_summary.csv": "Chamfer, volume, and surface-area reconstruction summaries.",
        "reconstruction_chamfer_area_per_scan.csv": "Per-scan reconstruction geometry rows.",
        "future_mesh_forecast_summary.csv": "Optional real-mesh future forecast Chamfer/ASSD/volume/area summary.",
        "future_mesh_forecast_per_pair.csv": "Optional per-pair real-mesh future forecast metrics.",
        "ood_future_artifact_manifest.csv": "Existing OOD/forecast/conditional visualization artifacts.",
        "coverage_gaps.csv": "Missing or long-run-required coverage.",
    }
    file_items = "\n".join(
        f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a>: {html.escape(desc)}</li>"
        for name, desc in files.items()
    )

    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 28px; color: #1f2933; }
      h1 { margin-bottom: 0.2rem; }
      h2 { margin-top: 2.2rem; border-bottom: 1px solid #d9e2ec; padding-bottom: 0.35rem; }
      h3 { margin-top: 1.4rem; }
      .muted { color: #64748b; }
      .callout { background: #f5f7fa; border-left: 4px solid #486581; padding: 12px 16px; margin: 16px 0; }
      .data-table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }
      .data-table th, .data-table td { border: 1px solid #d9e2ec; padding: 6px 8px; vertical-align: top; }
      .data-table th { background: #eef2f7; text-align: left; position: sticky; top: 0; }
      .table-wrap { overflow-x: auto; }
      a { color: #1d4ed8; }
      code { background: #eef2f7; padding: 1px 4px; border-radius: 3px; }
    </style>
    """
    figure_html = "\n".join(figures)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Best Model vs BrainODE Longitudinal Comparison</title>
  {style}
</head>
<body>
  <h1>Best Model vs BrainODE Longitudinal Comparison</h1>
  <p class="muted">Generated from existing completed outputs in {html.escape(str(REPO_ROOT))}.</p>

  <div class="callout">
    <p><strong>Metric separation:</strong> SIREN/DeepSDF longitudinal prediction uses decoder SDF L1;
    BrainODE longitudinal prediction uses PCA inverse-transform vertex MAE. Reconstruction quality uses
    sampled-surface Chamfer, ASSD, volume error, and triangle surface-area error where meshes are available.</p>
  </div>

  <h2>Generated Files</h2>
  <ul>{file_items}</ul>

  <h2>Key Numeric Prediction Metrics</h2>
  <div class="table-wrap">{html_table(best_numeric, max_rows=80)}</div>

  <h2>Direct vs Composed Prediction</h2>
  <div class="table-wrap">{html_table(direct_composed, max_rows=80)}</div>

  <h2>Volume Trend by Split</h2>
  <div class="table-wrap">{html_table(volume_trends, max_rows=120)}</div>

  <h2>Selected Subject Final Volume Forecasts</h2>
  <div class="table-wrap">{html_table(subject_volume, max_rows=80)}</div>

  <h2>Reconstruction Chamfer and Surface Area</h2>
  <p class="muted">Old ADNI uses the full Task2 reconstruction table. QC-large uses the available SIREN
  20-per-split subset and PCA-150/PCA-256 computed on that same subset.</p>
  <div class="table-wrap">{html_table(recon_summary, max_rows=120)}</div>

  <h2>Future Forecast Real-Mesh Chamfer</h2>
  <p class="muted">This optional long-run table compares predicted future endpoint meshes against the
  true target mesh using the same sampled-surface metrics for BrainODE, direct flow, composed flow, and
  each model's no-change source reconstruction. If empty, run
  <code>scripts/evaluate_future_mesh_forecasts.py</code> first.</p>
  <div class="table-wrap">{html_table(future_mesh_summary, max_rows=160)}</div>

  <h2>BrainODE Conditional Future Volume</h2>
  <div class="table-wrap">{html_table(brainode_future_summary, max_rows=80)}</div>

  <h2>Reusable Forecast and OOD Visualizations</h2>
  {make_artifact_link_table(artifacts, output_dir)}

  <h2>Coverage Gaps</h2>
  <div class="table-wrap">{html_table(gaps, max_rows=80)}</div>

  <h2>Plots</h2>
  {figure_html}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    future_mesh_dir = repo_path(args.future_mesh_dir)

    split_rows = flow_split_metrics() + brainode_split_metrics()
    direct_rows = direct_vs_composed_rows(split_rows)
    volume_rows = volume_trend_rows()
    subject_rows = subject_volume_rows()
    artifact_rows = rich_artifact_rows(output_dir)
    future_rows, future_summary_rows = brainode_future_rows()
    future_mesh_per_pair_rows, future_mesh_summary_rows, future_mesh_available = future_mesh_rows(
        future_mesh_dir
    )
    gap_rows = coverage_gap_rows(future_mesh_available)

    recon_rows = old_reconstruction_rows(args.skip_surface_area)
    recon_rows.extend(
        qc_reconstruction_rows(
            args.skip_surface_area,
            args.skip_qc_pca_chamfer,
            args.surface_samples,
        )
    )
    recon_summary_rows = reconstruction_summary(recon_rows)

    write_csv(output_dir / "split_numeric_metrics.csv", split_rows)
    write_csv(output_dir / "direct_vs_composed_metrics.csv", direct_rows)
    write_csv(output_dir / "volume_trend_by_split.csv", volume_rows)
    write_csv(output_dir / "subject_volume_forecasts.csv", subject_rows)
    write_csv(output_dir / "ood_future_artifact_manifest.csv", artifact_rows)
    write_csv(output_dir / "brainode_conditional_future_volume.csv", future_rows)
    write_csv(output_dir / "brainode_conditional_future_volume_summary.csv", future_summary_rows)
    write_csv(output_dir / "future_mesh_forecast_per_pair.csv", future_mesh_per_pair_rows)
    write_csv(output_dir / "future_mesh_forecast_summary.csv", future_mesh_summary_rows)
    write_csv(output_dir / "coverage_gaps.csv", gap_rows)
    write_csv(output_dir / "reconstruction_chamfer_area_per_scan.csv", recon_rows)
    write_csv(output_dir / "reconstruction_chamfer_area_summary.csv", recon_summary_rows)

    split_df = pd.DataFrame(split_rows)
    direct_df = pd.DataFrame(direct_rows)
    volume_df = pd.DataFrame(volume_rows)
    subject_df = pd.DataFrame(subject_rows)
    recon_summary_df = pd.DataFrame(recon_summary_rows)
    future_summary_df = pd.DataFrame(future_summary_rows)
    future_mesh_summary_df = pd.DataFrame(future_mesh_summary_rows)
    artifact_df = pd.DataFrame(artifact_rows)
    gap_df = pd.DataFrame(gap_rows)

    html_text = build_html_report(
        output_dir=output_dir,
        split_metrics=split_df,
        direct_composed=direct_df,
        volume_trends=volume_df,
        subject_volume=subject_df,
        recon_summary=recon_summary_df,
        future_mesh_summary=future_mesh_summary_df,
        brainode_future_summary=future_summary_df,
        artifacts=artifact_df,
        gaps=gap_df,
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")

    summary = {
        "output_dir": str(output_dir),
        "split_metric_rows": len(split_rows),
        "direct_vs_composed_rows": len(direct_rows),
        "volume_trend_rows": len(volume_rows),
        "subject_volume_rows": len(subject_rows),
        "reconstruction_per_scan_rows": len(recon_rows),
        "reconstruction_summary_rows": len(recon_summary_rows),
        "future_mesh_per_pair_rows": len(future_mesh_per_pair_rows),
        "future_mesh_summary_rows": len(future_mesh_summary_rows),
        "future_mesh_available": future_mesh_available,
        "future_mesh_dir": str(future_mesh_dir),
        "artifact_rows": len(artifact_rows),
        "coverage_gap_rows": len(gap_rows),
        "surface_area_computed": not args.skip_surface_area,
        "qc_pca_chamfer_computed": not args.skip_qc_pca_chamfer,
        "surface_samples": args.surface_samples,
    }
    write_json(output_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
