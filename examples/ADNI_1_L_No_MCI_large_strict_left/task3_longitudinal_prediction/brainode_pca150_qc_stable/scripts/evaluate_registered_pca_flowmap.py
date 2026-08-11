#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from registered_pca_flowmap_model import RegisteredPCAFlowMap
from registered_pca_flowmap_utils import (
    DEFAULT_EXPERIMENT_NAME,
    SPLITS,
    TASK_DIR,
    PairRecord,
    build_pair_records,
    decode_pca_np,
    experiment_root,
    finite_mean,
    finite_median,
    load_pca_model,
    load_split_archive,
    mesh_area_np,
    mesh_volume_np,
    pca_latents,
    vertex_normals_np,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate registered PCA flow-map checkpoints on matched registered-mesh metrics."
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default="e05_dynpca_k16_registered_flowmap_seed42")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint", default="best_pareto")
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--transport-methods", nargs="+", default=["direct", "composed_observed"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--ood-min-gap-years", type=float, default=2.0)
    parser.add_argument("--top-change-fraction", type=float, default=0.20)
    parser.add_argument(
        "--current-pca-flow-csv",
        default=(
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "pca150_direct_cocycle_flow_qc_v1/analysis/checkpoint_best_val_endpoint_vertex_mae/"
            "pca_flow_per_pair.csv"
        ),
    )
    parser.add_argument(
        "--brainode-trajectory-csv",
        default=(
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "brainode_pca150_qc_stable/training/core_attention_pca150_qc_stable/"
            "evaluation/best/all_trajectory_metrics.csv"
        ),
    )
    parser.add_argument("--skip-baseline-comparison", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def experiment_dir_for(args: argparse.Namespace) -> Path:
    if args.experiment_dir:
        return Path(args.experiment_dir).expanduser()
    return experiment_root(args.experiment_name) / "runs" / str(args.run_name)


def checkpoint_path_for(args: argparse.Namespace, experiment_dir: Path) -> Path:
    path = Path(args.checkpoint).expanduser()
    if path.is_file():
        return path
    name = path.name if path.suffix == ".pth" else f"{path.name}.pth"
    candidate = experiment_dir / "checkpoints" / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Missing checkpoint: {candidate}")


def output_dir_for(args: argparse.Namespace, experiment_dir: Path, checkpoint_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    return experiment_dir / "analysis" / f"checkpoint_{checkpoint_path.stem}"


def make_model_from_checkpoint(payload: dict[str, Any], device: torch.device) -> RegisteredPCAFlowMap:
    config = payload["model_config"]
    model = RegisteredPCAFlowMap(
        latent_dim=int(config["latent_dim"]),
        dynamic_basis=payload["dynamic_basis"],
        coefficient_mean=payload["coefficient_mean"],
        coefficient_std=payload["coefficient_std"],
        cn_velocity_mean=payload["cn_velocity_mean"],
        ad_velocity_mean=payload["ad_velocity_mean"],
        hidden_dims=[int(value) for value in config["hidden_dims"]],
        activation=str(config["activation"]),
        dropout=float(config["dropout"]),
        latent_condition_dim=int(config["latent_condition_dim"]),
        residual_scale=float(config["residual_scale"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_direct(
    model: RegisteredPCAFlowMap,
    source: np.ndarray,
    record: PairRecord,
    device: torch.device,
) -> np.ndarray:
    latent = torch.from_numpy(source.astype(np.float32)).view(1, -1).to(device)
    pred = model.transport(
        latent,
        torch.tensor([record.source_age_norm], dtype=torch.float32, device=device),
        torch.tensor([record.target_age_norm], dtype=torch.float32, device=device),
        torch.tensor([record.source_age_years], dtype=torch.float32, device=device),
        torch.tensor([record.target_age_years], dtype=torch.float32, device=device),
        torch.tensor([float(record.label_ad)], dtype=torch.float32, device=device),
    )
    return pred[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_composed_observed(
    model: RegisteredPCAFlowMap,
    archive: dict[str, np.ndarray],
    latents: np.ndarray,
    record: PairRecord,
    device: torch.device,
) -> np.ndarray:
    current = torch.from_numpy(latents[record.source_index].astype(np.float32)).view(1, -1).to(device)
    condition = torch.tensor([float(record.label_ad)], dtype=torch.float32, device=device)
    for index in range(record.source_index, record.target_index):
        source_norm = float(archive["visit_continuous_age_norm"][index])
        target_norm = float(archive["visit_continuous_age_norm"][index + 1])
        source_years = float(archive["visit_continuous_age_years"][index])
        target_years = float(archive["visit_continuous_age_years"][index + 1])
        current = model.transport(
            current,
            torch.tensor([source_norm], dtype=torch.float32, device=device),
            torch.tensor([target_norm], dtype=torch.float32, device=device),
            torch.tensor([source_years], dtype=torch.float32, device=device),
            torch.tensor([target_years], dtype=torch.float32, device=device),
            condition,
        )
    return current[0].detach().cpu().numpy().astype(np.float32)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size == 0 or right.size == 0:
        return float("nan")
    if float(left.std()) <= 1.0e-12 or float(right.std()) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def top_dice(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    count = max(1, int(round(float(left.size) * float(fraction))))
    left_idx = set(np.argsort(np.abs(left))[-count:].tolist())
    right_idx = set(np.argsort(np.abs(right))[-count:].tolist())
    return float(2 * len(left_idx.intersection(right_idx)) / max(len(left_idx) + len(right_idx), 1))


def prediction_metrics(
    *,
    record: PairRecord,
    split: str,
    transport_method: str,
    source: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    template_normals: np.ndarray,
    top_fraction: float,
    checkpoint: str,
) -> dict[str, Any]:
    source_vertices = decode_pca_np(source[None, :], mean_flat, components)[0]
    target_vertices = decode_pca_np(target[None, :], mean_flat, components)[0]
    pred_vertices = decode_pca_np(predicted[None, :], mean_flat, components)[0]
    source_volume = float(mesh_volume_np(source_vertices, faces)[0])
    target_volume = float(mesh_volume_np(target_vertices, faces)[0])
    predicted_volume = float(mesh_volume_np(pred_vertices, faces)[0])
    source_area = float(mesh_area_np(source_vertices, faces)[0])
    target_area = float(mesh_area_np(target_vertices, faces)[0])
    predicted_area = float(mesh_area_np(pred_vertices, faces)[0])
    delta_years = max(float(record.delta_years), 1.0e-6)

    vertex_dist = np.linalg.norm(pred_vertices - target_vertices, axis=1)
    no_change_vertex_dist = np.linalg.norm(source_vertices - target_vertices, axis=1)
    vertex_coord_abs = np.abs(pred_vertices - target_vertices)
    no_change_coord_abs = np.abs(source_vertices - target_vertices)
    normal_true = np.sum(((target_vertices - source_vertices) / delta_years) * template_normals, axis=1)
    normal_pred = np.sum(((pred_vertices - source_vertices) / delta_years) * template_normals, axis=1)
    true_log_rate = (math.log(max(target_volume, 1.0e-8)) - math.log(max(source_volume, 1.0e-8))) / delta_years
    pred_log_rate = (math.log(max(predicted_volume, 1.0e-8)) - math.log(max(source_volume, 1.0e-8))) / delta_years

    return {
        "dataset": "qc_large",
        "model": "pca150_registered_flowmap_qc_v2",
        "family": "PCA150 registered flow map",
        "split": split,
        "subject_id": record.subject_id,
        "diagnosis": record.diagnosis,
        "label_ad": record.label_ad,
        "source_scan_id": record.source_scan_id,
        "target_scan_id": record.target_scan_id,
        "source_visit_order": record.source_visit_order,
        "target_visit_order": record.target_visit_order,
        "pair_type": record.pair_type,
        "gap_bin": record.gap_bin,
        "source_age_years": record.source_age_years,
        "target_age_years": record.target_age_years,
        "gap_years": record.delta_years,
        "source_age_norm": record.source_age_norm,
        "target_age_norm": record.target_age_norm,
        "transport_method": transport_method,
        "checkpoint": checkpoint,
        "endpoint_pca_mse": float(np.mean((predicted - target) ** 2)),
        "no_change_pca_mse": float(np.mean((source - target) ** 2)),
        "endpoint_vertex_euclidean": float(vertex_dist.mean()),
        "no_change_vertex_euclidean": float(no_change_vertex_dist.mean()),
        "endpoint_vertex_euclidean_improvement": float(
            no_change_vertex_dist.mean() - vertex_dist.mean()
        ),
        "endpoint_vertex_hd95": float(np.quantile(vertex_dist, 0.95)),
        "no_change_vertex_hd95": float(np.quantile(no_change_vertex_dist, 0.95)),
        "endpoint_vertex_hd95_improvement": float(
            np.quantile(no_change_vertex_dist, 0.95) - np.quantile(vertex_dist, 0.95)
        ),
        "endpoint_vertex_mae": float(vertex_coord_abs.mean()),
        "no_change_vertex_mae": float(no_change_coord_abs.mean()),
        "endpoint_vertex_mae_improvement": float(no_change_coord_abs.mean() - vertex_coord_abs.mean()),
        "endpoint_vertex_rmse": float(np.sqrt(np.mean((pred_vertices - target_vertices) ** 2))),
        "source_volume": source_volume,
        "target_volume": target_volume,
        "predicted_volume": predicted_volume,
        "volume_abs_error": float(abs(predicted_volume - target_volume)),
        "volume_relative_error": float(abs(predicted_volume - target_volume) / max(target_volume, 1.0e-8)),
        "true_log_volume_rate_per_year": true_log_rate,
        "predicted_log_volume_rate_per_year": pred_log_rate,
        "log_volume_rate_error": float(pred_log_rate - true_log_rate),
        "log_volume_rate_abs_error": float(abs(pred_log_rate - true_log_rate)),
        "source_surface_area": source_area,
        "target_surface_area": target_area,
        "predicted_surface_area": predicted_area,
        "surface_area_relative_error": float(abs(predicted_area - target_area) / max(target_area, 1.0e-8)),
        "local_normal_rate_mae": float(np.mean(np.abs(normal_pred - normal_true))),
        "local_normal_rate_pearson": pearson(normal_pred, normal_true),
        "local_normal_top_change_dice": top_dice(normal_pred, normal_true, top_fraction),
    }


def evaluate_split(
    *,
    model: RegisteredPCAFlowMap,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    template_normals: np.ndarray,
    checkpoint: str,
) -> list[dict[str, Any]]:
    archive = load_split_archive(split)
    latents = pca_latents(archive, int(args.components))
    records = build_pair_records(archive)
    if int(args.max_pairs_per_split) > 0:
        records = records[: int(args.max_pairs_per_split)]
    rows: list[dict[str, Any]] = []
    for record in records:
        source = latents[record.source_index]
        target = latents[record.target_index]
        for method in args.transport_methods:
            if method == "direct":
                predicted = predict_direct(model, source, record, device)
            elif method == "composed_observed":
                predicted = predict_composed_observed(model, archive, latents, record, device)
            else:
                raise ValueError(f"Unknown transport method: {method}")
            rows.append(
                prediction_metrics(
                    record=record,
                    split=split,
                    transport_method=method,
                    source=source,
                    target=target,
                    predicted=predicted,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    template_normals=template_normals,
                    top_fraction=float(args.top_change_fraction),
                    checkpoint=checkpoint,
                )
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "endpoint_pca_mse",
        "endpoint_vertex_euclidean",
        "endpoint_vertex_hd95",
        "endpoint_vertex_mae",
        "endpoint_vertex_rmse",
        "endpoint_vertex_euclidean_improvement",
        "endpoint_vertex_hd95_improvement",
        "volume_relative_error",
        "log_volume_rate_abs_error",
        "local_normal_rate_mae",
        "local_normal_rate_pearson",
        "local_normal_top_change_dice",
    ]
    group_specs = [
        ("overall", []),
        ("split", ["split"]),
        ("diagnosis", ["diagnosis"]),
        ("transport", ["transport_method"]),
        ("split_diagnosis_transport", ["split", "diagnosis", "transport_method"]),
        ("gap_bin_transport", ["gap_bin", "transport_method"]),
    ]
    summaries: list[dict[str, Any]] = []
    for name, fields in group_specs:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[field] for field in fields)
            groups.setdefault(key, []).append(row)
        for key, group_rows in groups.items():
            summary: dict[str, Any] = {"grouping": name, "rows": len(group_rows)}
            for field, value in zip(fields, key):
                summary[field] = value
            for metric in metrics:
                values = [float(row[metric]) for row in group_rows]
                summary[f"{metric}_mean"] = finite_mean(values)
                summary[f"{metric}_median"] = finite_median(values)
            summaries.append(summary)
    return summaries


def build_volume_trends(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [row for row in rows if int(row["source_visit_order"]) == 0]
    trend_rows: list[dict[str, Any]] = []
    by_subject_method: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        key = (row["split"], row["diagnosis"], row["subject_id"], row["transport_method"])
        by_subject_method.setdefault(key, []).append(row)
    for (split, diagnosis, subject_id, method), group_rows in by_subject_method.items():
        if len(group_rows) < 2:
            continue
        group_rows = sorted(group_rows, key=lambda row: float(row["gap_years"]))
        x = np.asarray([float(row["gap_years"]) for row in group_rows], dtype=np.float64)
        source_volume = float(group_rows[0]["source_volume"])
        observed = np.asarray(
            [(float(row["target_volume"]) - source_volume) / max(source_volume, 1.0e-8) for row in group_rows],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [
                (float(row["predicted_volume"]) - source_volume) / max(source_volume, 1.0e-8)
                for row in group_rows
            ],
            dtype=np.float64,
        )
        if np.unique(x).size < 2:
            continue
        observed_slope = float(np.polyfit(x, observed, deg=1)[0])
        predicted_slope = float(np.polyfit(x, predicted, deg=1)[0])
        trend_rows.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "transport_method": method,
                "visits_from_baseline": len(group_rows),
                "observed_relative_volume_slope_per_year": observed_slope,
                "predicted_relative_volume_slope_per_year": predicted_slope,
                "relative_volume_slope_error": predicted_slope - observed_slope,
                "relative_volume_slope_abs_error": abs(predicted_slope - observed_slope),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in trend_rows:
        groups.setdefault((row["split"], row["diagnosis"], row["transport_method"]), []).append(row)
    for (split, diagnosis, method), group_rows in groups.items():
        summary_rows.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": method,
                "subjects": len(group_rows),
                "observed_relative_volume_slope_per_year_mean": finite_mean(
                    float(row["observed_relative_volume_slope_per_year"]) for row in group_rows
                ),
                "predicted_relative_volume_slope_per_year_mean": finite_mean(
                    float(row["predicted_relative_volume_slope_per_year"]) for row in group_rows
                ),
                "relative_volume_slope_abs_error_mean": finite_mean(
                    float(row["relative_volume_slope_abs_error"]) for row in group_rows
                ),
                "relative_volume_slope_abs_error_median": finite_median(
                    float(row["relative_volume_slope_abs_error"]) for row in group_rows
                ),
            }
        )
    return trend_rows, summary_rows


def build_ood_gt_summary(rows: list[dict[str, Any]], min_gap_years: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ood_rows = [row for row in rows if float(row["gap_years"]) >= float(min_gap_years)]
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ood_rows:
        groups.setdefault((row["split"], row["diagnosis"], row["transport_method"]), []).append(row)
    for (split, diagnosis, method), group_rows in groups.items():
        summary.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": method,
                "min_gap_years": float(min_gap_years),
                "rows": len(group_rows),
                "endpoint_vertex_euclidean_mean": finite_mean(
                    float(row["endpoint_vertex_euclidean"]) for row in group_rows
                ),
                "volume_relative_error_mean": finite_mean(
                    float(row["volume_relative_error"]) for row in group_rows
                ),
                "log_volume_rate_abs_error_mean": finite_mean(
                    float(row["log_volume_rate_abs_error"]) for row in group_rows
                ),
                "local_normal_top_change_dice_mean": finite_mean(
                    float(row["local_normal_top_change_dice"]) for row in group_rows
                ),
            }
        )
    return ood_rows, summary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_baseline_comparison(rows: list[dict[str, Any]], current_pca_flow_csv: Path) -> list[dict[str, Any]]:
    baseline_rows = read_csv_rows(current_pca_flow_csv)
    if not baseline_rows:
        return []
    baseline_by_key = {
        (
            row["split"],
            row["source_scan_id"],
            row["target_scan_id"],
            row["transport_method"],
        ): row
        for row in baseline_rows
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row["split"]),
            str(row["source_scan_id"]),
            str(row["target_scan_id"]),
            str(row["transport_method"]),
        )
        base = baseline_by_key.get(key)
        if base is None:
            continue
        for metric in ("endpoint_vertex_mae", "endpoint_pca_mse", "volume_relative_error"):
            if metric not in base:
                continue
            current = float(row[metric])
            baseline = float(base[metric])
            comparisons.append(
                {
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "transport_method": row["transport_method"],
                    "source_scan_id": row["source_scan_id"],
                    "target_scan_id": row["target_scan_id"],
                    "metric": metric,
                    "registered_flowmap": current,
                    "current_pca_flow": baseline,
                    "improvement": baseline - current,
                    "relative_improvement": (baseline - current) / abs(baseline)
                    if abs(baseline) > 1.0e-12
                    else float("nan"),
                }
            )
    return comparisons


def summarize_baseline_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["split"], row["diagnosis"], row["transport_method"], row["metric"]),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (split, diagnosis, method, metric), group_rows in groups.items():
        improvements = [float(row["improvement"]) for row in group_rows]
        summary.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": method,
                "metric": metric,
                "rows": len(group_rows),
                "registered_flowmap_mean": finite_mean(float(row["registered_flowmap"]) for row in group_rows),
                "current_pca_flow_mean": finite_mean(float(row["current_pca_flow"]) for row in group_rows),
                "improvement_mean": finite_mean(improvements),
                "improvement_median": finite_median(improvements),
                "beats_current_fraction": finite_mean(1.0 if value > 0.0 else 0.0 for value in improvements),
            }
        )
    return summary


def build_brainode_comparison(rows: list[dict[str, Any]], brainode_csv: Path) -> list[dict[str, Any]]:
    brainode_rows = read_csv_rows(brainode_csv)
    if not brainode_rows:
        return []
    brainode_by_key: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in brainode_rows:
        if row.get("record_set") != "all_pairs" or row.get("direction") != "forward":
            continue
        start_order = int(float(row["start_visit_order"]))
        target_order = start_order + int(float(row["length"])) - 1
        brainode_by_key[(row["split"], row["subject_id"], start_order, target_order)] = row

    comparisons: list[dict[str, Any]] = []
    metric_map = {
        "endpoint_vertex_mae": "endpoint_vertex_mae",
        "endpoint_pca_mse": "endpoint_pca_mse",
    }
    for row in rows:
        key = (
            str(row["split"]),
            str(row["subject_id"]),
            int(row["source_visit_order"]),
            int(row["target_visit_order"]),
        )
        base = brainode_by_key.get(key)
        if base is None:
            continue
        for ours_metric, brainode_metric in metric_map.items():
            if brainode_metric not in base:
                continue
            ours = float(row[ours_metric])
            brainode = float(base[brainode_metric])
            comparisons.append(
                {
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "transport_method": row["transport_method"],
                    "subject_id": row["subject_id"],
                    "source_visit_order": row["source_visit_order"],
                    "target_visit_order": row["target_visit_order"],
                    "metric": ours_metric,
                    "registered_flowmap": ours,
                    "brainode": brainode,
                    "improvement": brainode - ours,
                    "relative_improvement": (brainode - ours) / abs(brainode)
                    if abs(brainode) > 1.0e-12
                    else float("nan"),
                }
            )
    return comparisons


def summarize_brainode_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["split"], row["diagnosis"], row["transport_method"], row["metric"]),
            [],
        ).append(row)
    summary: list[dict[str, Any]] = []
    for (split, diagnosis, method, metric), group_rows in groups.items():
        improvements = [float(row["improvement"]) for row in group_rows]
        summary.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": method,
                "metric": metric,
                "rows": len(group_rows),
                "registered_flowmap_mean": finite_mean(float(row["registered_flowmap"]) for row in group_rows),
                "brainode_mean": finite_mean(float(row["brainode"]) for row in group_rows),
                "improvement_mean": finite_mean(improvements),
                "improvement_median": finite_median(improvements),
                "beats_brainode_fraction": finite_mean(1.0 if value > 0.0 else 0.0 for value in improvements),
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    device = resolve_device(str(args.device))
    experiment_dir = experiment_dir_for(args)
    checkpoint_path = checkpoint_path_for(args, experiment_dir)
    output_dir = output_dir_for(args, experiment_dir, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location=device)
    model = make_model_from_checkpoint(payload, device)
    _, pca_model_dir, mean_flat, components, faces = load_pca_model(args.config, int(args.components))
    template_normals = vertex_normals_np(mean_flat.reshape(-1, 3), faces)

    pair_rows: list[dict[str, Any]] = []
    for split in args.splits:
        pair_rows.extend(
            evaluate_split(
                model=model,
                split=split,
                args=args,
                device=device,
                mean_flat=mean_flat,
                components=components,
                faces=faces,
                template_normals=template_normals,
                checkpoint=str(checkpoint_path),
            )
        )
    summary_rows = summarize_rows(pair_rows)
    volume_rows, volume_summary_rows = build_volume_trends(pair_rows)
    ood_rows, ood_summary_rows = build_ood_gt_summary(pair_rows, float(args.ood_min_gap_years))

    write_csv(output_dir / "registered_flow_per_pair.csv", pair_rows)
    write_csv(output_dir / "registered_flow_summary.csv", summary_rows)
    write_csv(output_dir / "registered_flow_volume_trends.csv", volume_rows)
    write_csv(output_dir / "registered_flow_volume_slope_summary.csv", volume_summary_rows)
    write_csv(output_dir / "registered_flow_ood_gt_per_pair.csv", ood_rows)
    write_csv(output_dir / "registered_flow_ood_gt_summary.csv", ood_summary_rows)

    comparison_rows: list[dict[str, Any]] = []
    comparison_summary_rows: list[dict[str, Any]] = []
    brainode_comparison_rows: list[dict[str, Any]] = []
    brainode_comparison_summary_rows: list[dict[str, Any]] = []
    if not bool(args.skip_baseline_comparison):
        comparison_rows = build_baseline_comparison(
            pair_rows,
            Path(args.current_pca_flow_csv).expanduser(),
        )
        comparison_summary_rows = summarize_baseline_comparison(comparison_rows)
        write_csv(output_dir / "baseline_comparison_matched.csv", comparison_rows)
        write_csv(output_dir / "baseline_comparison_summary.csv", comparison_summary_rows)
        brainode_comparison_rows = build_brainode_comparison(
            pair_rows,
            Path(args.brainode_trajectory_csv).expanduser(),
        )
        brainode_comparison_summary_rows = summarize_brainode_comparison(
            brainode_comparison_rows
        )
        write_csv(output_dir / "brainode_comparison_matched.csv", brainode_comparison_rows)
        write_csv(output_dir / "brainode_comparison_summary.csv", brainode_comparison_summary_rows)

    run = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "pca_model_dir": str(pca_model_dir),
        "splits": list(args.splits),
        "transport_methods": list(args.transport_methods),
        "pair_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "volume_trend_rows": len(volume_rows),
        "ood_gt_rows": len(ood_rows),
        "baseline_comparison_rows": len(comparison_rows),
        "baseline_comparison_summary_rows": len(comparison_summary_rows),
        "brainode_comparison_rows": len(brainode_comparison_rows),
        "brainode_comparison_summary_rows": len(brainode_comparison_summary_rows),
    }
    write_json(output_dir / "run.json", run)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
