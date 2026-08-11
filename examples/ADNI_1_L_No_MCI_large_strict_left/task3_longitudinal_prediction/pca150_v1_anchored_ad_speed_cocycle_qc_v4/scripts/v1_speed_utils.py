#!/usr/bin/env python3
"""Shared utilities for the v1-anchored AD speed-calibration experiment.

The experiment intentionally reuses the QC-stable BrainODE splits and the
completed v1 checkpoint.  All metadata functions use training records only
unless a caller explicitly requests another split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "deep_sdf").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
TASK_PARENT = EXPERIMENT_DIR.parent
TASK_DIR = TASK_PARENT / "brainode_pca150_qc_stable"
V1_DIR = TASK_PARENT / "pca150_direct_cocycle_flow_qc_v1"
V3_DIR = TASK_PARENT / "pca150_registered_flowmap_bidir_qc_v3"
REPO_ROOT = find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXPERIMENT_NAME = "pca150_v1_anchored_ad_speed_cocycle_qc_v4"
SPLITS = ("train", "val", "test")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    return config


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return float("nan")
    if float(left.std()) <= 1.0e-12 or float(right.std()) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def experiment_root(name: str = DEFAULT_EXPERIMENT_NAME) -> Path:
    return EXPERIMENT_DIR if str(name) == DEFAULT_EXPERIMENT_NAME else TASK_PARENT / str(name)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_split_archive(split: str) -> dict[str, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return load_npz(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")


def pca_latents(archive: dict[str, np.ndarray], components: int) -> np.ndarray:
    direct_key = f"visit_pca_{int(components)}"
    if direct_key in archive:
        return archive[direct_key].astype(np.float32)
    fallback_key = "visit_pca_256"
    if int(components) <= 256 and fallback_key in archive:
        return archive[fallback_key][:, : int(components)].astype(np.float32)
    raise KeyError(f"No PCA coefficients for {components} components")


def coefficient_stats(archive: dict[str, np.ndarray], components: int) -> tuple[np.ndarray, np.ndarray]:
    mean_key = f"train_coefficient_mean_{int(components)}"
    std_key = f"train_coefficient_std_{int(components)}"
    if mean_key in archive and std_key in archive:
        mean = archive[mean_key].astype(np.float32)
        std = archive[std_key].astype(np.float32)
    else:
        mean = archive["train_coefficient_mean_256"][: int(components)].astype(np.float32)
        std = archive["train_coefficient_std_256"][: int(components)].astype(np.float32)
    return mean, np.maximum(std, np.float32(1.0e-6))


def load_pca_model(components: int) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    core_config = load_json(TASK_DIR / "configs" / "core_brainode.json")
    pca_model_dir = resolve_repo_path(core_config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    pca_components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[: int(components)]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
    return pca_model_dir, mean_flat, pca_components, faces


def decode_pca_np(coefficients: np.ndarray, mean_flat: np.ndarray, components: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float32)
    flattened = coefficients.reshape(-1, coefficients.shape[-1]) @ components
    flattened = flattened + mean_flat
    return flattened.reshape(*coefficients.shape[:-1], -1, 3)


def decode_pca_torch(coefficients: torch.Tensor, mean_flat: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    flattened = coefficients.reshape(-1, coefficients.shape[-1]) @ components
    flattened = flattened + mean_flat
    return flattened.reshape(*coefficients.shape[:-1], -1, 3)


def mesh_volume_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    face_index = faces.astype(np.int64)
    v0 = vertices[:, face_index[:, 0], :]
    v1 = vertices[:, face_index[:, 1], :]
    v2 = vertices[:, face_index[:, 2], :]
    signed = np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2))
    return np.abs(signed.sum(axis=1) / 6.0)


def mesh_area_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    face_index = faces.astype(np.int64)
    v0 = vertices[:, face_index[:, 0], :]
    v1 = vertices[:, face_index[:, 1], :]
    v2 = vertices[:, face_index[:, 2], :]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=2).sum(axis=1)


def mesh_volume_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    face_index = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[:, face_index[:, 0], :]
    v1 = vertices[:, face_index[:, 1], :]
    v2 = vertices[:, face_index[:, 2], :]
    signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
    return torch.clamp(torch.abs(signed), min=1.0e-8)


def vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    face_index = faces.astype(np.int64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    v0 = vertices[face_index[:, 0]]
    v1 = vertices[face_index[:, 1]]
    v2 = vertices[face_index[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for corner in range(3):
        np.add.at(normals, face_index[:, corner], face_normals)
    return (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def top_change_dice(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    count = max(1, int(round(left.size * float(fraction))))
    left_ids = set(np.argsort(np.abs(left))[-count:].tolist())
    right_ids = set(np.argsort(np.abs(right))[-count:].tolist())
    return float(2.0 * len(left_ids.intersection(right_ids)) / max(len(left_ids) + len(right_ids), 1))


@dataclass(frozen=True)
class PairRecord:
    source_index: int
    target_index: int
    intermediate_index: int
    subject_id: str
    diagnosis: str
    label_ad: int
    source_scan_id: str
    target_scan_id: str
    source_visit_order: int
    target_visit_order: int
    source_age_norm: float
    target_age_norm: float
    source_age_years: float
    target_age_years: float
    delta_years: float
    abs_gap_years: float
    pair_type: str
    gap_bin: str
    direction: str


def gap_bin(delta_years: float, pair_type: str) -> str:
    if pair_type == "adjacent":
        return "adjacent"
    return "short" if abs(float(delta_years)) <= 2.0 else "long"


def build_pair_records(
    archive: dict[str, np.ndarray],
    *,
    pair_type: str = "all",
    include_backward: bool = False,
    diagnosis: str | None = None,
    max_gap_years: float = 0.0,
) -> list[PairRecord]:
    records: list[PairRecord] = []
    offsets = archive["subject_visit_offsets"]
    for subject_number in range(len(offsets) - 1):
        start, end = int(offsets[subject_number]), int(offsets[subject_number + 1])
        subject_diagnosis = str(archive["visit_diagnoses"][start])
        if diagnosis is not None and subject_diagnosis != diagnosis:
            continue
        for source_index in range(start, end - 1):
            for target_index in range(source_index + 1, end):
                source_order = int(archive["visit_orders"][source_index])
                target_order = int(archive["visit_orders"][target_index])
                current_type = "adjacent" if target_order - source_order == 1 else "nonadjacent"
                if pair_type != "all" and pair_type != current_type:
                    continue
                source_years = float(archive["visit_continuous_age_years"][source_index])
                target_years = float(archive["visit_continuous_age_years"][target_index])
                delta = target_years - source_years
                if delta <= 1.0e-8 or (max_gap_years > 0.0 and delta > max_gap_years):
                    continue
                intermediate = -1
                if target_index - source_index > 1:
                    intermediate = source_index + (target_index - source_index) // 2
                common = dict(
                    subject_id=str(archive["visit_subject_ids"][source_index]),
                    diagnosis=subject_diagnosis,
                    label_ad=int(archive["visit_label_ad"][source_index]),
                    pair_type=current_type,
                    gap_bin=gap_bin(delta, current_type),
                    abs_gap_years=abs(delta),
                    intermediate_index=intermediate,
                )
                records.append(
                    PairRecord(
                        source_index=source_index,
                        target_index=target_index,
                        source_scan_id=str(archive["visit_scan_ids"][source_index]),
                        target_scan_id=str(archive["visit_scan_ids"][target_index]),
                        source_visit_order=source_order,
                        target_visit_order=target_order,
                        source_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                        target_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                        source_age_years=source_years,
                        target_age_years=target_years,
                        delta_years=delta,
                        direction="forward",
                        **common,
                    )
                )
                if include_backward:
                    records.append(
                        PairRecord(
                            source_index=target_index,
                            target_index=source_index,
                            source_scan_id=str(archive["visit_scan_ids"][target_index]),
                            target_scan_id=str(archive["visit_scan_ids"][source_index]),
                            source_visit_order=target_order,
                            target_visit_order=source_order,
                            source_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                            target_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                            source_age_years=target_years,
                            target_age_years=source_years,
                            delta_years=-delta,
                            direction="backward",
                            **common,
                        )
                    )
    return records


def limit_records_stratified(records: list[PairRecord], maximum: int) -> list[PairRecord]:
    """Limit a mixed-diagnosis set without accidentally dropping AD or CN.

    The archives are ordered by subject, so a plain ``records[:N]`` can contain
    only one diagnosis on a small smoke run. Full experiments pass ``0`` and
    therefore retain every record.
    """
    maximum = int(maximum)
    if maximum <= 0 or len(records) <= maximum:
        return records
    groups = {
        "CN": [record for record in records if record.diagnosis == "CN"],
        "AD": [record for record in records if record.diagnosis == "AD"],
    }
    selected: list[PairRecord] = []
    group_order = ("CN", "AD")
    cursor = {name: 0 for name in group_order}
    while len(selected) < maximum:
        added = False
        for name in group_order:
            if len(selected) >= maximum:
                break
            index = cursor[name]
            if index < len(groups[name]):
                selected.append(groups[name][index])
                cursor[name] += 1
                added = True
        if not added:
            break
    return selected


def first_visit_sequence_starts(archive: dict[str, np.ndarray], diagnosis: str | None = None) -> list[int]:
    starts: list[int] = []
    offsets = archive["subject_visit_offsets"]
    for subject_number in range(len(offsets) - 1):
        start, end = int(offsets[subject_number]), int(offsets[subject_number + 1])
        if end - start < 3:
            continue
        if diagnosis is not None and str(archive["visit_diagnoses"][start]) != diagnosis:
            continue
        starts.append(start)
    return starts


def subject_end(archive: dict[str, np.ndarray], start: int) -> int:
    offsets = archive["subject_visit_offsets"]
    return int(offsets[np.searchsorted(offsets, int(start), side="right")])


def load_v1_flow(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from longitudinal_direct_flow import DirectAgeFlow

    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    flow_config = dict(payload["flow_config"])
    model = DirectAgeFlow(
        latent_size=int(flow_config["latent_size"]),
        hidden_dims=[int(value) for value in flow_config["hidden_dims"]],
        condition_dim=int(flow_config["condition_dim"]),
        activation=str(flow_config["activation"]),
        dropout=float(flow_config["dropout"]),
        include_delta_time_input=bool(flow_config["include_delta_time_input"]),
        latent_condition_mode=str(flow_config["latent_condition_mode"]),
        latent_condition_dim=int(flow_config["latent_condition_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


@torch.no_grad()
def v1_transport(
    model: torch.nn.Module,
    source: torch.Tensor,
    source_age_norm: torch.Tensor,
    target_age_norm: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    return model.transport(source, source_age_norm, target_age_norm, condition)


def row_metadata(record: PairRecord, split: str, transport_method: str, checkpoint: str) -> dict[str, Any]:
    return {
        "dataset": "qc_large",
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
        "direction": record.direction,
        "source_age_years": record.source_age_years,
        "target_age_years": record.target_age_years,
        "gap_years": record.abs_gap_years,
        "signed_delta_years": record.delta_years,
        "source_age_norm": record.source_age_norm,
        "target_age_norm": record.target_age_norm,
        "transport_method": transport_method,
        "checkpoint": checkpoint,
    }


def prediction_metrics(
    *,
    record: PairRecord,
    split: str,
    transport_method: str,
    checkpoint: str,
    model_name: str,
    family: str,
    source: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    template_normals: np.ndarray,
    top_fraction: float,
    ad_speed: float | None = None,
) -> dict[str, Any]:
    source_vertices = decode_pca_np(source[None, :], mean_flat, components)[0]
    target_vertices = decode_pca_np(target[None, :], mean_flat, components)[0]
    predicted_vertices = decode_pca_np(predicted[None, :], mean_flat, components)[0]
    source_volume = float(mesh_volume_np(source_vertices, faces)[0])
    target_volume = float(mesh_volume_np(target_vertices, faces)[0])
    predicted_volume = float(mesh_volume_np(predicted_vertices, faces)[0])
    source_area = float(mesh_area_np(source_vertices, faces)[0])
    target_area = float(mesh_area_np(target_vertices, faces)[0])
    predicted_area = float(mesh_area_np(predicted_vertices, faces)[0])
    delta = float(record.delta_years) if abs(float(record.delta_years)) > 1.0e-6 else 1.0e-6
    vertex_distance = np.linalg.norm(predicted_vertices - target_vertices, axis=1)
    no_change_distance = np.linalg.norm(source_vertices - target_vertices, axis=1)
    normal_true = np.sum(((target_vertices - source_vertices) / delta) * template_normals, axis=1)
    normal_pred = np.sum(((predicted_vertices - source_vertices) / delta) * template_normals, axis=1)
    true_rate = (math.log(max(target_volume, 1.0e-8)) - math.log(max(source_volume, 1.0e-8))) / delta
    pred_rate = (math.log(max(predicted_volume, 1.0e-8)) - math.log(max(source_volume, 1.0e-8))) / delta
    values = row_metadata(record, split, transport_method, checkpoint)
    values.update(
        {
            "model": model_name,
            "family": family,
            "endpoint_pca_mse": float(np.mean((predicted - target) ** 2)),
            "no_change_pca_mse": float(np.mean((source - target) ** 2)),
            "endpoint_vertex_euclidean": float(vertex_distance.mean()),
            "no_change_vertex_euclidean": float(no_change_distance.mean()),
            "endpoint_vertex_mae": float(np.abs(predicted_vertices - target_vertices).mean()),
            "no_change_vertex_mae": float(np.abs(source_vertices - target_vertices).mean()),
            "endpoint_vertex_rmse": float(np.sqrt(np.mean((predicted_vertices - target_vertices) ** 2))),
            "endpoint_vertex_hd95": float(np.quantile(vertex_distance, 0.95)),
            "volume_abs_error": float(abs(predicted_volume - target_volume)),
            "volume_relative_error": float(abs(predicted_volume - target_volume) / max(target_volume, 1.0e-8)),
            "source_volume": source_volume,
            "target_volume": target_volume,
            "predicted_volume": predicted_volume,
            "true_log_volume_rate_per_year": true_rate,
            "predicted_log_volume_rate_per_year": pred_rate,
            "log_volume_rate_error": float(pred_rate - true_rate),
            "log_volume_rate_abs_error": float(abs(pred_rate - true_rate)),
            "source_surface_area": source_area,
            "target_surface_area": target_area,
            "predicted_surface_area": predicted_area,
            "surface_area_relative_error": float(abs(predicted_area - target_area) / max(target_area, 1.0e-8)),
            "local_normal_rate_mae": float(np.abs(normal_pred - normal_true).mean()),
            "local_normal_rate_pearson": pearson(normal_pred, normal_true),
            "local_normal_top_change_dice": top_change_dice(normal_pred, normal_true, top_fraction),
            "ad_speed": float(ad_speed) if ad_speed is not None else float("nan"),
        }
    )
    values["endpoint_vertex_euclidean_improvement"] = values["no_change_vertex_euclidean"] - values["endpoint_vertex_euclidean"]
    values["endpoint_vertex_mae_improvement"] = values["no_change_vertex_mae"] - values["endpoint_vertex_mae"]
    values["endpoint_pca_improvement"] = values["no_change_pca_mse"] - values["endpoint_pca_mse"]
    return values


def summarize_prediction_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "endpoint_pca_mse",
        "endpoint_vertex_euclidean",
        "endpoint_vertex_mae",
        "endpoint_vertex_rmse",
        "endpoint_vertex_hd95",
        "volume_relative_error",
        "log_volume_rate_abs_error",
        "local_normal_rate_mae",
        "local_normal_rate_pearson",
        "local_normal_top_change_dice",
        "ad_speed",
    ]
    specs = [
        ("overall", []),
        ("split", ["split"]),
        ("diagnosis", ["diagnosis"]),
        ("split_diagnosis_transport", ["split", "diagnosis", "transport_method"]),
        ("gap_bin_transport", ["gap_bin", "transport_method"]),
    ]
    output: list[dict[str, Any]] = []
    for name, fields in specs:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[field] for field in fields)
            grouped.setdefault(key, []).append(row)
        for key, grouped_rows in grouped.items():
            row: dict[str, Any] = {"grouping": name, "rows": len(grouped_rows)}
            row.update(dict(zip(fields, key)))
            for metric in metrics:
                values = [float(item[metric]) for item in grouped_rows]
                row[f"{metric}_mean"] = finite_mean(values)
                row[f"{metric}_median"] = finite_median(values)
            output.append(row)
    return output


def build_volume_trends(
    rows: Sequence[dict[str, Any]],
    *,
    transport_method: str = "direct",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = [
        row for row in rows
        if row["direction"] == "forward" and int(row["source_visit_order"]) == 0 and row["transport_method"] == transport_method
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        groups.setdefault((row["split"], row["diagnosis"], row["subject_id"]), []).append(row)
    trends: list[dict[str, Any]] = []
    for (split, diagnosis, subject_id), group in groups.items():
        group = sorted(group, key=lambda value: float(value["gap_years"]))
        if len(group) < 2:
            continue
        x = np.asarray([float(value["gap_years"]) for value in group], dtype=np.float64)
        if np.unique(x).size < 2:
            continue
        observed = np.log(np.asarray([float(value["target_volume"]) for value in group], dtype=np.float64))
        predicted = np.log(np.asarray([float(value["predicted_volume"]) for value in group], dtype=np.float64))
        observed_slope = float(np.polyfit(x, observed, 1)[0])
        predicted_slope = float(np.polyfit(x, predicted, 1)[0])
        trends.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "transport_method": transport_method,
                "targets": len(group),
                "observed_log_volume_slope_per_year": observed_slope,
                "predicted_log_volume_slope_per_year": predicted_slope,
                "log_volume_slope_error": predicted_slope - observed_slope,
                "log_volume_slope_abs_error": abs(predicted_slope - observed_slope),
            }
        )
    summaries: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trends:
        summary_groups.setdefault((row["split"], row["diagnosis"]), []).append(row)
    for (split, diagnosis), group in summary_groups.items():
        observed = [float(value["observed_log_volume_slope_per_year"]) for value in group]
        predicted = [float(value["predicted_log_volume_slope_per_year"]) for value in group]
        summaries.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": transport_method,
                "subjects": len(group),
                "observed_log_volume_slope_per_year_mean": finite_mean(observed),
                "predicted_log_volume_slope_per_year_mean": finite_mean(predicted),
                "log_volume_slope_abs_error_mean": finite_mean(float(value["log_volume_slope_abs_error"]) for value in group),
                "log_volume_slope_abs_error_median": finite_median(float(value["log_volume_slope_abs_error"]) for value in group),
                "slope_pearson": pearson(np.asarray(predicted), np.asarray(observed)),
                "observed_slope_std": float(np.std(observed)),
                "predicted_slope_std": float(np.std(predicted)),
            }
        )
    return trends, summaries


def subject_bootstrap_improvement(
    rows: Sequence[dict[str, Any]],
    *,
    current_key: str,
    baseline_key: str,
    subject_key: str = "subject_id",
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    by_subject: dict[str, list[float]] = {}
    for row in rows:
        current = float(row[current_key])
        baseline = float(row[baseline_key])
        if math.isfinite(current) and math.isfinite(baseline):
            by_subject.setdefault(str(row[subject_key]), []).append(baseline - current)
    subject_means = np.asarray([np.mean(values) for values in by_subject.values()], dtype=np.float64)
    if subject_means.size == 0:
        return {"subjects": 0.0, "improvement_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    samples = subject_means[rng.integers(0, subject_means.size, size=(int(replicates), subject_means.size))].mean(axis=1)
    return {
        "subjects": float(subject_means.size),
        "improvement_mean": float(subject_means.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }
