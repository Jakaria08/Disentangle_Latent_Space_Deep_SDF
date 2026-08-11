#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from core_brainode_common import SPLITS, TASK_DIR, load_config, resolve_repo_path, write_json
from longitudinal_direct_flow import DirectAgeFlow
from train_pca_cocycle_flow import DEFAULT_RUN_NAME, load_npz, pca_latents


LOWER_IS_BETTER_GEOMETRY = (
    "gt_to_pred_mean",
    "pred_to_gt_mean",
    "chamfer_l1",
    "chamfer_l2_squared",
    "assd",
    "hd95",
    "volume_abs_error",
    "volume_relative_error",
    "surface_area_abs_error",
    "surface_area_relative_error",
)

NUMERIC_METRICS = (
    "endpoint_pca_mse",
    "endpoint_pca_improvement",
    "endpoint_vertex_mae",
    "endpoint_vertex_rmse",
    "endpoint_vertex_mae_improvement",
    "endpoint_vertex_rmse_improvement",
    *LOWER_IS_BETTER_GEOMETRY,
)


@dataclass(frozen=True)
class PairSpec:
    split: str
    subject_id: str
    diagnosis: str
    label_ad: int
    source_index: int
    target_index: int
    source_scan_id: str
    target_scan_id: str
    source_visit_order: int
    target_visit_order: int
    source_age_norm: float
    target_age_norm: float
    source_age_years: float
    target_age_years: float
    source_mesh_path: str
    target_mesh_path: str

    @property
    def pair_type(self) -> str:
        return "adjacent" if self.target_visit_order - self.source_visit_order == 1 else "nonadjacent"

    @property
    def gap_years(self) -> float:
        return float(self.target_age_years - self.source_age_years)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained PCA-space conditional cocycle flow against true "
            "future hippocampus meshes, with direct/composed rollout metrics, "
            "volume trends, OOD conditional forecasts, and an HTML report."
        )
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint", default="best_val_endpoint_vertex_mae")
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--surface-samples", type=int, default=30000)
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--pair-type", choices=("all", "adjacent", "nonadjacent"), default="all")
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--ood-horizons-years", type=float, nargs="+", default=[1.0, 2.0, 4.0, 6.0])
    parser.add_argument("--ood-composed-step-years", type=float, default=0.5)
    parser.add_argument("--max-ood-subjects-per-split", type=int, default=0)
    parser.add_argument(
        "--baseline-per-pair-csv",
        default=(
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
            "analysis/future_mesh_forecast_comparison/future_mesh_per_pair.csv"
        ),
    )
    parser.add_argument("--skip-baseline-comparison", action="store_true")
    parser.add_argument("--skip-volume-trends", action="store_true")
    parser.add_argument("--skip-ood", action="store_true")
    return parser.parse_args()


def resolve_device(value: str | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def experiment_dir_for(args: argparse.Namespace) -> Path:
    if args.experiment_dir:
        return Path(args.experiment_dir).expanduser()
    return TASK_DIR.parent / str(args.run_name)


def output_dir_for(args: argparse.Namespace, experiment_dir: Path, checkpoint_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    return experiment_dir / "analysis" / f"checkpoint_{checkpoint_path.stem}"


def checkpoint_path_for(args: argparse.Namespace, experiment_dir: Path) -> Path:
    path = Path(args.checkpoint).expanduser()
    if path.is_file():
        return path
    name = path.name if path.suffix == ".pth" else f"{path.name}.pth"
    candidate = experiment_dir / "checkpoints" / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Missing checkpoint: {candidate}")


def make_flow_from_checkpoint(payload: dict[str, Any], components: int) -> DirectAgeFlow:
    flow_config = dict(payload.get("flow_config", {}))
    latent_size = int(flow_config.get("latent_size", components))
    if latent_size != int(components):
        raise ValueError(
            f"Checkpoint latent size {latent_size} does not match --components {components}."
        )
    return DirectAgeFlow(
        latent_size=latent_size,
        hidden_dims=[int(value) for value in flow_config.get("hidden_dims", [128, 128])],
        condition_dim=int(flow_config.get("condition_dim", 1)),
        activation=str(flow_config.get("activation", "silu")),
        dropout=float(flow_config.get("dropout", 0.05)),
        zero_initialize_output=True,
        latent_condition_mode=str(flow_config.get("latent_condition_mode", "full")),
        latent_condition_dim=int(flow_config.get("latent_condition_dim", latent_size)),
        include_delta_time_input=bool(flow_config.get("include_delta_time_input", True)),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def safe_stem(value: str, max_chars: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned[:max_chars].strip("_") or "scan"


@lru_cache(maxsize=2048)
def load_mesh_from_path(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh scene at {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh at {path}, got {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Empty mesh at {path}")
    return loaded


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def finite_relative(abs_error: float, target_value: float) -> float:
    if not math.isfinite(abs_error) or not math.isfinite(target_value) or abs(target_value) <= 0.0:
        return float("nan")
    return float(abs_error / abs(target_value))


def empty_geometry_metrics(error: str) -> dict[str, Any]:
    return {
        "decode_success": False,
        "decode_error": error,
        "gt_to_pred_mean": float("nan"),
        "pred_to_gt_mean": float("nan"),
        "chamfer_l1": float("nan"),
        "chamfer_l2_squared": float("nan"),
        "assd": float("nan"),
        "hd95": float("nan"),
        "predicted_volume": float("nan"),
        "target_volume": float("nan"),
        "volume_abs_error": float("nan"),
        "volume_signed_error": float("nan"),
        "volume_relative_error": float("nan"),
        "predicted_surface_area": float("nan"),
        "target_surface_area": float("nan"),
        "surface_area_abs_error": float("nan"),
        "surface_area_signed_error": float("nan"),
        "surface_area_relative_error": float("nan"),
    }


def geometry_metrics(
    *,
    predicted_mesh: trimesh.Trimesh | None,
    target_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    sample_count: int,
    seed_key: str,
    decode_error: str | None = None,
) -> dict[str, Any]:
    if predicted_mesh is None:
        return empty_geometry_metrics(decode_error or "prediction mesh is missing")
    pred_points = sample_surface(
        predicted_mesh,
        sample_count,
        deterministic_seed(seed_key, "predicted_surface_points"),
    )
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    gt_to_pred, _ = pred_tree.query(target_points, k=1)
    pred_to_gt, _ = target_tree.query(pred_points, k=1)
    gt_mean = float(np.mean(gt_to_pred))
    pred_mean = float(np.mean(pred_to_gt))
    all_distances = np.concatenate([gt_to_pred, pred_to_gt])
    predicted_volume = float(abs(predicted_mesh.volume))
    target_volume = float(abs(target_mesh.volume))
    volume_signed = predicted_volume - target_volume
    volume_abs = abs(volume_signed)
    predicted_area = float(predicted_mesh.area)
    target_area = float(target_mesh.area)
    area_signed = predicted_area - target_area
    area_abs = abs(area_signed)
    return {
        "decode_success": True,
        "decode_error": "",
        "gt_to_pred_mean": gt_mean,
        "pred_to_gt_mean": pred_mean,
        "chamfer_l1": gt_mean + pred_mean,
        "chamfer_l2_squared": float(np.mean(gt_to_pred**2) + np.mean(pred_to_gt**2)),
        "assd": 0.5 * (gt_mean + pred_mean),
        "hd95": float(np.quantile(all_distances, 0.95)),
        "predicted_volume": predicted_volume,
        "target_volume": target_volume,
        "volume_abs_error": volume_abs,
        "volume_signed_error": volume_signed,
        "volume_relative_error": finite_relative(volume_abs, target_volume),
        "predicted_surface_area": predicted_area,
        "target_surface_area": target_area,
        "surface_area_abs_error": area_abs,
        "surface_area_signed_error": area_signed,
        "surface_area_relative_error": finite_relative(area_abs, target_area),
    }


def pca_mesh(
    coefficients: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
) -> trimesh.Trimesh:
    flat = coefficients.astype(np.float32) @ components.astype(np.float32) + mean_flat.astype(np.float32)
    vertices = flat.reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def pca_vertices(
    coefficients: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    flat = coefficients.astype(np.float32) @ components.astype(np.float32) + mean_flat.astype(np.float32)
    return flat.reshape(-1, 3)


def pca_endpoint_metrics(
    *,
    predicted: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
) -> dict[str, float]:
    predicted = predicted.astype(np.float32)
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    endpoint_pca_mse = float(np.mean((predicted - target) ** 2))
    no_change_pca_mse = float(np.mean((source - target) ** 2))
    predicted_vertices = pca_vertices(predicted, mean_flat, components)
    source_vertices = pca_vertices(source, mean_flat, components)
    target_vertices = pca_vertices(target, mean_flat, components)
    endpoint_vertex_mae = float(np.mean(np.abs(predicted_vertices - target_vertices)))
    no_change_vertex_mae = float(np.mean(np.abs(source_vertices - target_vertices)))
    endpoint_vertex_rmse = float(np.sqrt(np.mean((predicted_vertices - target_vertices) ** 2)))
    no_change_vertex_rmse = float(np.sqrt(np.mean((source_vertices - target_vertices) ** 2)))
    return {
        "endpoint_pca_mse": endpoint_pca_mse,
        "no_change_pca_mse": no_change_pca_mse,
        "endpoint_pca_improvement": no_change_pca_mse - endpoint_pca_mse,
        "endpoint_vertex_mae": endpoint_vertex_mae,
        "no_change_vertex_mae": no_change_vertex_mae,
        "endpoint_vertex_mae_improvement": no_change_vertex_mae - endpoint_vertex_mae,
        "endpoint_vertex_rmse": endpoint_vertex_rmse,
        "no_change_vertex_rmse": no_change_vertex_rmse,
        "endpoint_vertex_rmse_improvement": no_change_vertex_rmse - endpoint_vertex_rmse,
    }


def row_uid(parts: Iterable[object]) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def export_mesh(
    *,
    output_dir: Path,
    save_meshes: bool,
    mesh: trimesh.Trimesh | None,
    split: str,
    transport_method: str,
    source_scan_id: str,
    target_scan_id: str,
    uid: str,
) -> str:
    if not save_meshes or mesh is None:
        return ""
    mesh_dir = output_dir / "predicted_meshes" / split / transport_method
    mesh_dir.mkdir(parents=True, exist_ok=True)
    path = mesh_dir / f"{safe_stem(source_scan_id)}__to__{safe_stem(target_scan_id)}__{uid}.ply"
    mesh.export(path)
    return str(path)


def make_prediction_row(
    *,
    pair: PairSpec,
    transport_method: str,
    predicted_latent: np.ndarray,
    predicted_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    source_gt_metrics: dict[str, Any],
    model_no_change_metrics: dict[str, Any],
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
    sample_count: int,
    output_dir: Path,
    save_meshes: bool,
    checkpoint: str,
    precomputed_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uid = row_uid(
        (
            "qc_large",
            "pca_flow",
            pair.split,
            transport_method,
            pair.source_scan_id,
            pair.target_scan_id,
        )
    )
    metrics = (
        dict(precomputed_metrics)
        if precomputed_metrics is not None
        else geometry_metrics(
            predicted_mesh=predicted_mesh,
            target_mesh=target_mesh,
            target_points=target_points,
            sample_count=sample_count,
            seed_key=f"{uid}|{transport_method}",
        )
    )
    row: dict[str, Any] = {
        "row_uid": uid,
        "dataset": "qc_large",
        "model": "pca150_direct_cocycle_flow",
        "family": "PCA150 conditional cocycle flow",
        "split": pair.split,
        "subject_id": pair.subject_id,
        "diagnosis": pair.diagnosis,
        "label_ad": pair.label_ad,
        "source_scan_id": pair.source_scan_id,
        "target_scan_id": pair.target_scan_id,
        "source_visit_order": pair.source_visit_order,
        "target_visit_order": pair.target_visit_order,
        "pair_type": pair.pair_type,
        "source_age_years": pair.source_age_years,
        "target_age_years": pair.target_age_years,
        "gap_years": pair.gap_years,
        "source_age_norm": pair.source_age_norm,
        "target_age_norm": pair.target_age_norm,
        "source_mesh_path": pair.source_mesh_path,
        "target_mesh_path": pair.target_mesh_path,
        "transport_method": transport_method,
        "sampled_surface_points": sample_count,
        "checkpoint": checkpoint,
    }
    row.update(
        pca_endpoint_metrics(
            predicted=predicted_latent,
            source=source_latent,
            target=target_latent,
            mean_flat=mean_flat,
            components=components,
        )
    )
    row.update(metrics)
    row["predicted_mesh_path"] = export_mesh(
        output_dir=output_dir,
        save_meshes=save_meshes,
        mesh=predicted_mesh,
        split=pair.split,
        transport_method=transport_method,
        source_scan_id=pair.source_scan_id,
        target_scan_id=pair.target_scan_id,
        uid=uid,
    )
    for metric in LOWER_IS_BETTER_GEOMETRY:
        source_value = source_gt_metrics.get(metric, float("nan"))
        model_value = model_no_change_metrics.get(metric, float("nan"))
        value = metrics.get(metric, float("nan"))
        row[f"source_gt_no_change_{metric}"] = source_value
        row[f"model_no_change_{metric}"] = model_value
        row[f"{metric}_improvement_vs_source_gt_no_change"] = (
            float(source_value) - float(value)
            if math.isfinite(float(source_value)) and math.isfinite(float(value))
            else float("nan")
        )
        row[f"{metric}_improvement_vs_model_no_change"] = (
            float(model_value) - float(value)
            if math.isfinite(float(model_value)) and math.isfinite(float(value))
            else float("nan")
        )
    return row


def iter_pairs(
    *,
    archive: dict[str, np.ndarray],
    split: str,
    scan_manifest: dict[str, dict[str, str]],
    components: int,
    pair_type: str,
    max_pairs: int,
) -> Iterable[tuple[PairSpec, np.ndarray, np.ndarray, np.ndarray]]:
    latents = pca_latents(archive, components)
    offsets = archive["subject_visit_offsets"]
    yielded = 0
    for subject_index in range(len(offsets) - 1):
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        for source_index in range(start, end - 1):
            for target_index in range(source_index + 1, end):
                source_order = int(archive["visit_orders"][source_index])
                target_order = int(archive["visit_orders"][target_index])
                current_pair_type = "adjacent" if target_order - source_order == 1 else "nonadjacent"
                if pair_type != "all" and current_pair_type != pair_type:
                    continue
                source_scan_id = str(archive["visit_scan_ids"][source_index])
                target_scan_id = str(archive["visit_scan_ids"][target_index])
                source_manifest = scan_manifest[source_scan_id]
                target_manifest = scan_manifest[target_scan_id]
                time_slice = slice(source_index, target_index + 1)
                pair = PairSpec(
                    split=split,
                    subject_id=str(archive["visit_subject_ids"][source_index]),
                    diagnosis=str(archive["visit_diagnoses"][source_index]),
                    label_ad=int(archive["visit_label_ad"][source_index]),
                    source_index=source_index,
                    target_index=target_index,
                    source_scan_id=source_scan_id,
                    target_scan_id=target_scan_id,
                    source_visit_order=source_order,
                    target_visit_order=target_order,
                    source_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                    target_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                    source_age_years=float(archive["visit_continuous_age_years"][source_index]),
                    target_age_years=float(archive["visit_continuous_age_years"][target_index]),
                    source_mesh_path=source_manifest["mesh_path"],
                    target_mesh_path=target_manifest["mesh_path"],
                )
                yield (
                    pair,
                    archive["visit_continuous_age_norm"][time_slice].astype(np.float32).copy(),
                    latents[time_slice].astype(np.float32).copy(),
                    archive["visit_cognition"][time_slice].astype(np.float32).copy(),
                )
                yielded += 1
                if max_pairs > 0 and yielded >= int(max_pairs):
                    return


@torch.no_grad()
def predict_direct(
    flow: DirectAgeFlow,
    source_latent: np.ndarray,
    source_time: float,
    target_time: float,
    condition: float,
    device: torch.device,
) -> np.ndarray:
    source = torch.from_numpy(source_latent.astype(np.float32)).view(1, -1).to(device)
    source_t = torch.tensor([source_time], dtype=torch.float32, device=device)
    target_t = torch.tensor([target_time], dtype=torch.float32, device=device)
    cond = torch.tensor([condition], dtype=torch.float32, device=device)
    predicted = flow.transport(source, source_t, target_t, cond)
    return predicted[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_composed_observed(
    flow: DirectAgeFlow,
    latents: np.ndarray,
    times: np.ndarray,
    condition: float,
    device: torch.device,
) -> np.ndarray:
    current = torch.from_numpy(latents[0].astype(np.float32)).view(1, -1).to(device)
    cond = torch.tensor([condition], dtype=torch.float32, device=device)
    for index in range(len(times) - 1):
        source_t = torch.tensor([float(times[index])], dtype=torch.float32, device=device)
        target_t = torch.tensor([float(times[index + 1])], dtype=torch.float32, device=device)
        current = flow.transport(current, source_t, target_t, cond)
    return current[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_composed_year_steps(
    flow: DirectAgeFlow,
    source_latent: np.ndarray,
    source_age_years: float,
    target_age_years: float,
    condition: float,
    age_min_years: float,
    age_range_years: float,
    step_years: float,
    device: torch.device,
) -> np.ndarray:
    current = torch.from_numpy(source_latent.astype(np.float32)).view(1, -1).to(device)
    cond = torch.tensor([condition], dtype=torch.float32, device=device)
    current_age = float(source_age_years)
    target_age = float(target_age_years)
    step = max(float(step_years), 1.0e-6)
    while current_age < target_age - 1.0e-8:
        next_age = min(current_age + step, target_age)
        source_t = torch.tensor(
            [(current_age - age_min_years) / age_range_years],
            dtype=torch.float32,
            device=device,
        )
        target_t = torch.tensor(
            [(next_age - age_min_years) / age_range_years],
            dtype=torch.float32,
            device=device,
        )
        current = flow.transport(current, source_t, target_t, cond)
        current_age = next_age
    return current[0].detach().cpu().numpy().astype(np.float32)


def evaluate_pairs(
    *,
    flow: DirectAgeFlow,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    checkpoint_path: Path,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    scan_manifest: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for split in args.splits:
        archive = load_npz(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
        for pair, times_np, latents_np, conditions_np in iter_pairs(
            archive=archive,
            split=split,
            scan_manifest=scan_manifest,
            components=int(args.components),
            pair_type=str(args.pair_type),
            max_pairs=int(args.max_pairs_per_split),
        ):
            try:
                source_gt_mesh = load_mesh_from_path(pair.source_mesh_path)
                target_mesh = load_mesh_from_path(pair.target_mesh_path)
                pair_seed = f"qc_large|{pair.split}|{pair.source_scan_id}|{pair.target_scan_id}"
                target_points = sample_surface(
                    target_mesh,
                    int(args.surface_samples),
                    deterministic_seed(pair_seed, "target_surface_points"),
                )
                source_gt_metrics = geometry_metrics(
                    predicted_mesh=source_gt_mesh,
                    target_mesh=target_mesh,
                    target_points=target_points,
                    sample_count=int(args.surface_samples),
                    seed_key=f"{pair_seed}|source_gt_no_change",
                )
                source_latent = latents_np[0]
                target_latent = latents_np[-1]
                source_pca_mesh = pca_mesh(source_latent, mean_flat, components, faces)
                model_no_change_metrics = geometry_metrics(
                    predicted_mesh=source_pca_mesh,
                    target_mesh=target_mesh,
                    target_points=target_points,
                    sample_count=int(args.surface_samples),
                    seed_key=f"{pair_seed}|pca_flow|model_no_change",
                )
                direct_latent = predict_direct(
                    flow,
                    source_latent,
                    float(times_np[0]),
                    float(times_np[-1]),
                    float(conditions_np[0]),
                    device,
                )
                composed_latent = predict_composed_observed(
                    flow,
                    latents_np,
                    times_np,
                    float(conditions_np[0]),
                    device,
                )
                for method, latent in (
                    ("direct", direct_latent),
                    ("composed_observed", composed_latent),
                    ("model_no_change", source_latent),
                ):
                    mesh = source_pca_mesh if method == "model_no_change" else pca_mesh(latent, mean_flat, components, faces)
                    rows.append(
                        make_prediction_row(
                            pair=pair,
                            transport_method=method,
                            predicted_latent=latent,
                            predicted_mesh=mesh,
                            target_mesh=target_mesh,
                            target_points=target_points,
                            source_gt_metrics=source_gt_metrics,
                            model_no_change_metrics=model_no_change_metrics,
                            source_latent=source_latent,
                            target_latent=target_latent,
                            mean_flat=mean_flat,
                            components=components,
                            sample_count=int(args.surface_samples),
                            output_dir=output_dir,
                            save_meshes=bool(args.save_meshes),
                            checkpoint=str(checkpoint_path),
                            precomputed_metrics=model_no_change_metrics
                            if method == "model_no_change"
                            else None,
                        )
                    )
                direct_composed_mse = float(np.mean((direct_latent - composed_latent) ** 2))
                for row in rows[-3:]:
                    row["direct_composed_pca_mse"] = direct_composed_mse
            except Exception as exc:
                failures.append(
                    {
                        "split": pair.split,
                        "subject_id": pair.subject_id,
                        "source_scan_id": pair.source_scan_id,
                        "target_scan_id": pair.target_scan_id,
                        "error": repr(exc),
                    }
                )
    return rows, failures


def finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def grouped(rows: Sequence[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    result: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        result.setdefault(key, []).append(row)
    return result


def summarize_group(
    rows: list[dict[str, Any]],
    grouping: str,
    *,
    split: str = "",
    diagnosis: str = "",
    pair_type: str = "",
    transport_method: str = "",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "grouping": grouping,
        "split": split,
        "diagnosis": diagnosis,
        "pair_type": pair_type,
        "transport_method": transport_method,
        "rows": len(rows),
        "decode_success_fraction": mean(
            [1.0 if bool(row.get("decode_success", False)) else 0.0 for row in rows]
        ),
    }
    for metric in NUMERIC_METRICS:
        values = finite_values(rows, metric)
        summary[f"{metric}_mean"] = mean(values)
        summary[f"{metric}_median"] = median(values)
        summary[f"{metric}_std"] = std(values)
    for metric in LOWER_IS_BETTER_GEOMETRY:
        for suffix in (
            "improvement_vs_model_no_change",
            "improvement_vs_source_gt_no_change",
        ):
            key = f"{metric}_{suffix}"
            values = finite_values(rows, key)
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_median"] = median(values)
            summary[f"{key}_std"] = std(values)
    improvement_values = finite_values(rows, "chamfer_l2_squared_improvement_vs_model_no_change")
    summary["chamfer_l2_beats_model_no_change_fraction"] = mean(
        [1.0 if value > 0.0 else 0.0 for value in improvement_values]
    )
    vertex_improvements = finite_values(rows, "endpoint_vertex_mae_improvement")
    summary["vertex_mae_beats_no_change_fraction"] = mean(
        [1.0 if value > 0.0 else 0.0 for value in vertex_improvements]
    )
    return summary


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = [summarize_group(rows, "overall")]
    for (split,), group_rows in sorted(grouped(rows, ("split",)).items()):
        summaries.append(summarize_group(group_rows, "split", split=split))
    for (method,), group_rows in sorted(grouped(rows, ("transport_method",)).items()):
        summaries.append(summarize_group(group_rows, "transport", transport_method=method))
    for (split, method), group_rows in sorted(grouped(rows, ("split", "transport_method")).items()):
        summaries.append(
            summarize_group(group_rows, "split_transport", split=split, transport_method=method)
        )
    for (split, diagnosis, method), group_rows in sorted(
        grouped(rows, ("split", "diagnosis", "transport_method")).items()
    ):
        summaries.append(
            summarize_group(
                group_rows,
                "split_diagnosis_transport",
                split=split,
                diagnosis=diagnosis,
                transport_method=method,
            )
        )
    for (split, diagnosis, pair_type, method), group_rows in sorted(
        grouped(rows, ("split", "diagnosis", "pair_type", "transport_method")).items()
    ):
        summaries.append(
            summarize_group(
                group_rows,
                "split_diagnosis_pair_type_transport",
                split=split,
                diagnosis=diagnosis,
                pair_type=pair_type,
                transport_method=method,
            )
        )
    return summaries


def mesh_volume_area(mesh: trimesh.Trimesh) -> tuple[float, float]:
    return float(abs(mesh.volume)), float(mesh.area)


def build_volume_trends(
    *,
    flow: DirectAgeFlow,
    args: argparse.Namespace,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
    scan_manifest: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        archive = load_npz(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
        latents = pca_latents(archive, int(args.components))
        offsets = archive["subject_visit_offsets"]
        for subject_index in range(len(offsets) - 1):
            start = int(offsets[subject_index])
            end = int(offsets[subject_index + 1])
            source_latent = latents[start]
            source_time = float(archive["visit_continuous_age_norm"][start])
            condition = float(archive["visit_cognition"][start])
            composed_current = source_latent.copy()
            for visit_index in range(start, end):
                scan_id = str(archive["visit_scan_ids"][visit_index])
                manifest_row = scan_manifest[scan_id]
                real_mesh = load_mesh_from_path(manifest_row["mesh_path"])
                observed_volume, observed_area = mesh_volume_area(real_mesh)
                pca_target_mesh = pca_mesh(latents[visit_index], mean_flat, components, faces)
                pca_target_volume, pca_target_area = mesh_volume_area(pca_target_mesh)
                target_time = float(archive["visit_continuous_age_norm"][visit_index])
                if visit_index == start:
                    direct_latent = source_latent.copy()
                    composed_current = source_latent.copy()
                else:
                    direct_latent = predict_direct(
                        flow,
                        source_latent,
                        source_time,
                        target_time,
                        condition,
                        device,
                    )
                    previous_time = float(archive["visit_continuous_age_norm"][visit_index - 1])
                    composed_current = predict_direct(
                        flow,
                        composed_current,
                        previous_time,
                        target_time,
                        condition,
                        device,
                    )
                for method, latent in (
                    ("direct_from_baseline", direct_latent),
                    ("composed_observed_from_baseline", composed_current),
                    ("model_no_change", source_latent),
                ):
                    predicted_mesh = pca_mesh(latent, mean_flat, components, faces)
                    predicted_volume, predicted_area = mesh_volume_area(predicted_mesh)
                    rows.append(
                        {
                            "split": split,
                            "subject_id": str(archive["visit_subject_ids"][visit_index]),
                            "diagnosis": str(archive["visit_diagnoses"][visit_index]),
                            "label_ad": int(archive["visit_label_ad"][visit_index]),
                            "scan_id": scan_id,
                            "visit_order": int(archive["visit_orders"][visit_index]),
                            "age_years": float(archive["visit_continuous_age_years"][visit_index]),
                            "age_norm": target_time,
                            "transport_method": method,
                            "observed_volume": observed_volume,
                            "observed_surface_area": observed_area,
                            "target_pca_volume": pca_target_volume,
                            "target_pca_surface_area": pca_target_area,
                            "predicted_volume": predicted_volume,
                            "predicted_surface_area": predicted_area,
                            "volume_signed_error": predicted_volume - observed_volume,
                            "volume_abs_error": abs(predicted_volume - observed_volume),
                            "volume_relative_error": finite_relative(abs(predicted_volume - observed_volume), observed_volume),
                            "surface_area_signed_error": predicted_area - observed_area,
                            "surface_area_abs_error": abs(predicted_area - observed_area),
                            "surface_area_relative_error": finite_relative(abs(predicted_area - observed_area), observed_area),
                        }
                    )
    return rows, summarize_volume_slopes(rows)


def slope(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) < 2:
        return float("nan")
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if np.allclose(x, x[0]):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def summarize_volume_slopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subject_method_rows = grouped(rows, ("split", "diagnosis", "subject_id", "transport_method"))
    per_subject: list[dict[str, Any]] = []
    for (split, diagnosis, subject_id, method), group_rows in sorted(subject_method_rows.items()):
        ordered = sorted(group_rows, key=lambda row: float(row["age_years"]))
        ages = [float(row["age_years"]) for row in ordered]
        observed = [float(row["observed_volume"]) for row in ordered]
        predicted = [float(row["predicted_volume"]) for row in ordered]
        observed_slope = slope(ages, observed)
        predicted_slope = slope(ages, predicted)
        per_subject.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "subject_id": subject_id,
                "transport_method": method,
                "observed_volume_slope_per_year": observed_slope,
                "predicted_volume_slope_per_year": predicted_slope,
                "volume_slope_error": predicted_slope - observed_slope
                if math.isfinite(observed_slope) and math.isfinite(predicted_slope)
                else float("nan"),
            }
        )

    summary: list[dict[str, Any]] = []
    for (split, diagnosis, method), group_rows in sorted(
        grouped(per_subject, ("split", "diagnosis", "transport_method")).items()
    ):
        summary.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "transport_method": method,
                "subjects": len(group_rows),
                "observed_volume_slope_per_year_mean": mean(
                    finite_values(group_rows, "observed_volume_slope_per_year")
                ),
                "predicted_volume_slope_per_year_mean": mean(
                    finite_values(group_rows, "predicted_volume_slope_per_year")
                ),
                "volume_slope_error_mean": mean(finite_values(group_rows, "volume_slope_error")),
                "volume_slope_error_median": median(finite_values(group_rows, "volume_slope_error")),
            }
        )
    return [*per_subject, *summary]


def build_ood_forecasts(
    *,
    flow: DirectAgeFlow,
    args: argparse.Namespace,
    device: torch.device,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_archive = load_npz(TASK_DIR / "dataset" / "train_subject_sequences.npz")
    age_min = float(np.min(train_archive["visit_continuous_age_years"]))
    age_max = float(np.max(train_archive["visit_continuous_age_years"]))
    age_range = age_max - age_min
    if age_range <= 0.0:
        raise ValueError("Training age range is invalid.")

    rows: list[dict[str, Any]] = []
    for split in args.splits:
        archive = load_npz(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
        latents = pca_latents(archive, int(args.components))
        offsets = archive["subject_visit_offsets"]
        yielded_subjects = 0
        for subject_index in range(len(offsets) - 1):
            start = int(offsets[subject_index])
            source_latent = latents[start]
            source_age = float(archive["visit_continuous_age_years"][start])
            source_age_norm = float(archive["visit_continuous_age_norm"][start])
            source_mesh = pca_mesh(source_latent, mean_flat, components, faces)
            source_volume, source_area = mesh_volume_area(source_mesh)
            for horizon in [float(value) for value in args.ood_horizons_years]:
                future_age = source_age + horizon
                future_age_norm = (future_age - age_min) / age_range
                for rollout_condition, rollout_label in ((0.0, "CN"), (1.0, "AD")):
                    direct_latent = predict_direct(
                        flow,
                        source_latent,
                        source_age_norm,
                        future_age_norm,
                        rollout_condition,
                        device,
                    )
                    composed_latent = predict_composed_year_steps(
                        flow,
                        source_latent,
                        source_age,
                        future_age,
                        rollout_condition,
                        age_min,
                        age_range,
                        float(args.ood_composed_step_years),
                        device,
                    )
                    for method, latent in (
                        ("direct", direct_latent),
                        ("composed_fixed_year_steps", composed_latent),
                    ):
                        mesh = pca_mesh(latent, mean_flat, components, faces)
                        volume, area = mesh_volume_area(mesh)
                        rows.append(
                            {
                                "split": split,
                                "subject_id": str(archive["subject_ids"][subject_index]),
                                "source_scan_id": str(archive["visit_scan_ids"][start]),
                                "source_diagnosis": str(archive["visit_diagnoses"][start]),
                                "source_label_ad": int(archive["visit_label_ad"][start]),
                                "rollout_condition": rollout_label,
                                "rollout_condition_value": rollout_condition,
                                "transport_method": method,
                                "source_age_years": source_age,
                                "future_age_years": future_age,
                                "horizon_years": horizon,
                                "source_age_norm": source_age_norm,
                                "future_age_norm": future_age_norm,
                                "future_age_is_ood": bool(future_age < age_min or future_age > age_max),
                                "source_volume": source_volume,
                                "predicted_volume": volume,
                                "predicted_volume_delta": volume - source_volume,
                                "predicted_volume_relative_delta": finite_relative(volume - source_volume, source_volume),
                                "source_surface_area": source_area,
                                "predicted_surface_area": area,
                                "predicted_surface_area_delta": area - source_area,
                                "predicted_surface_area_relative_delta": finite_relative(area - source_area, source_area),
                            }
                        )
            yielded_subjects += 1
            if int(args.max_ood_subjects_per_split) > 0 and yielded_subjects >= int(args.max_ood_subjects_per_split):
                break

    summary: list[dict[str, Any]] = []
    for (split, source_dx, rollout_condition, method, horizon), group_rows in sorted(
        grouped(
            rows,
            (
                "split",
                "source_diagnosis",
                "rollout_condition",
                "transport_method",
                "horizon_years",
            ),
        ).items()
    ):
        summary.append(
            {
                "split": split,
                "source_diagnosis": source_dx,
                "rollout_condition": rollout_condition,
                "transport_method": method,
                "horizon_years": horizon,
                "rows": len(group_rows),
                "ood_fraction": mean(
                    [1.0 if bool(row["future_age_is_ood"]) else 0.0 for row in group_rows]
                ),
                "predicted_volume_delta_mean": mean(
                    finite_values(group_rows, "predicted_volume_delta")
                ),
                "predicted_volume_relative_delta_mean": mean(
                    finite_values(group_rows, "predicted_volume_relative_delta")
                ),
                "predicted_surface_area_delta_mean": mean(
                    finite_values(group_rows, "predicted_surface_area_delta")
                ),
            }
        )
    return rows, summary


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped(rows, group_keys).items()):
        row = {name: value for name, value in zip(group_keys, key)}
        row["rows"] = len(group_rows)
        for metric in metric_names:
            row[f"{metric}_mean"] = mean(finite_values(group_rows, metric))
            row[f"{metric}_median"] = median(finite_values(group_rows, metric))
        output.append(row)
    return output


def build_baseline_comparison(
    *,
    current_rows: list[dict[str, Any]],
    baseline_csv: Path,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in current_rows:
        if row.get("transport_method") not in {"direct", "composed_observed", "model_no_change"}:
            continue
        rows.append(
            {
                "source": "current_pca_flow",
                "dataset": "qc_large",
                "model": row["model"],
                "family": row["family"],
                "split": row["split"],
                "transport_method": row["transport_method"],
                "chamfer_l2_squared": row["chamfer_l2_squared"],
                "chamfer_l1": row["chamfer_l1"],
                "volume_relative_error": row["volume_relative_error"],
                "surface_area_relative_error": row["surface_area_relative_error"],
                "endpoint_vertex_mae": row["endpoint_vertex_mae"],
            }
        )

    baseline_models = {
        ("qc_brainode_pca150", "brainode_endpoint"),
        ("qc_siren_drop_bad_min2", "direct"),
        ("qc_siren_drop_bad_min2", "composed"),
        ("qc_siren_local_decomp_volume", "direct"),
        ("qc_siren_local_decomp_volume", "composed"),
    }
    if baseline_csv.is_file():
        with baseline_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("dataset") != "qc_large":
                    continue
                if row.get("split") not in set(splits):
                    continue
                if (row.get("model", ""), row.get("transport_method", "")) not in baseline_models:
                    continue
                rows.append(
                    {
                        "source": "existing_future_mesh_eval",
                        "dataset": row.get("dataset", ""),
                        "model": row.get("model", ""),
                        "family": row.get("family", ""),
                        "split": row.get("split", ""),
                        "transport_method": row.get("transport_method", ""),
                        "chamfer_l2_squared": row.get("chamfer_l2_squared", ""),
                        "chamfer_l1": row.get("chamfer_l1", ""),
                        "volume_relative_error": row.get("volume_relative_error", ""),
                        "surface_area_relative_error": row.get("surface_area_relative_error", ""),
                        "endpoint_vertex_mae": "",
                    }
                )

    return aggregate_rows(
        rows,
        ("source", "dataset", "model", "family", "split", "transport_method"),
        (
            "chamfer_l2_squared",
            "chamfer_l1",
            "volume_relative_error",
            "surface_area_relative_error",
            "endpoint_vertex_mae",
        ),
    )


def table_html(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    max_rows: int = 100,
) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_parts = []
    for row in list(rows)[:max_rows]:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                text = f"{value:.6g}" if math.isfinite(value) else ""
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def write_html_report(
    *,
    output_dir: Path,
    run_summary: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    volume_slope_rows: list[dict[str, Any]],
    ood_summary_rows: list[dict[str, Any]],
) -> None:
    split_transport_rows = [
        row for row in summary_rows if row.get("grouping") == "split_transport"
    ]
    test_baselines = [row for row in baseline_rows if row.get("split") == "test"]
    volume_summary_only = [
        row for row in volume_slope_rows if "subjects" in row
    ]
    css = """
body { font-family: Arial, sans-serif; margin: 24px; color: #1d2733; background: #fafafa; }
h1, h2 { margin: 0 0 12px; }
section { margin: 0 0 28px; padding: 18px; background: white; border: 1px solid #d9dee7; border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; table-layout: auto; }
th, td { border-bottom: 1px solid #e3e7ef; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #eef2f7; position: sticky; top: 0; }
.table-wrap { overflow-x: auto; max-width: 100%; }
.files a { display: inline-block; margin: 0 14px 8px 0; }
.note { color: #52606d; max-width: 980px; line-height: 1.45; }
code { background: #eef2f7; padding: 2px 4px; border-radius: 4px; }
"""
    files = [
        "pca_flow_per_pair.csv",
        "pca_flow_summary.csv",
        "baseline_comparison.csv",
        "pca_flow_volume_trends.csv",
        "pca_flow_volume_trend_slope_summary.csv",
        "pca_flow_ood_conditional_forecasts.csv",
        "pca_flow_ood_summary.csv",
        "pca_flow_failures.csv",
        "run.json",
    ]
    file_links = "".join(
        f'<a href="{html.escape(name)}">{html.escape(name)}</a>'
        for name in files
        if (output_dir / name).exists()
    )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PCA150 Conditional Cocycle Flow Evaluation</title>
<style>{css}</style>
</head>
<body>
<h1>PCA150 Conditional Cocycle Flow Evaluation</h1>
<section>
<h2>Run</h2>
<p class="note">This report evaluates the PCA-space flow on the QC-controlled large ADNI left hippocampus dataset. Lower is better for Chamfer, ASSD, HD95, volume error, area error, PCA MSE, and vertex error. Positive improvement means the prediction beats the PCA model no-change baseline.</p>
<pre>{html.escape(json.dumps(run_summary, indent=2, sort_keys=True))}</pre>
<div class="files">{file_links}</div>
</section>
<section>
<h2>Test Baseline Comparison</h2>
<div class="table-wrap">
{table_html(test_baselines, ["source", "model", "family", "split", "transport_method", "rows", "chamfer_l2_squared_mean", "volume_relative_error_mean", "surface_area_relative_error_mean", "endpoint_vertex_mae_mean"])}
</div>
</section>
<section>
<h2>PCA-Flow Split Summary</h2>
<div class="table-wrap">
{table_html(split_transport_rows, ["split", "transport_method", "rows", "endpoint_pca_mse_mean", "endpoint_pca_improvement_mean", "endpoint_vertex_mae_mean", "endpoint_vertex_mae_improvement_mean", "chamfer_l2_squared_mean", "chamfer_l2_squared_improvement_vs_model_no_change_mean", "volume_relative_error_mean", "surface_area_relative_error_mean", "vertex_mae_beats_no_change_fraction", "chamfer_l2_beats_model_no_change_fraction"])}
</div>
</section>
<section>
<h2>Volume Trend Slopes</h2>
<div class="table-wrap">
{table_html(volume_summary_only, ["split", "diagnosis", "transport_method", "subjects", "observed_volume_slope_per_year_mean", "predicted_volume_slope_per_year_mean", "volume_slope_error_mean", "volume_slope_error_median"])}
</div>
</section>
<section>
<h2>OOD Conditional Forecasts</h2>
<div class="table-wrap">
{table_html(ood_summary_rows, ["split", "source_diagnosis", "rollout_condition", "transport_method", "horizon_years", "rows", "ood_fraction", "predicted_volume_delta_mean", "predicted_volume_relative_delta_mean", "predicted_surface_area_delta_mean"], max_rows=160)}
</div>
</section>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if int(args.surface_samples) <= 0:
        raise ValueError("--surface-samples must be positive.")
    config = load_config(args.config)
    experiment_dir = experiment_dir_for(args)
    checkpoint_path = checkpoint_path_for(args, experiment_dir)
    output_dir = output_dir_for(args, experiment_dir, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_components = int(payload.get("components", args.components))
    if checkpoint_components != int(args.components):
        raise ValueError(
            f"Checkpoint has {checkpoint_components} components but --components={args.components}."
        )
    flow = make_flow_from_checkpoint(payload, int(args.components)).to(device)
    flow.load_state_dict(payload["model_state_dict"])
    flow.eval()

    pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[: int(args.components)]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
    scan_manifest = {
        row["scan_id"]: row for row in read_csv(TASK_DIR / "metadata" / "core_brainode_scan_manifest.csv")
    }

    pair_rows, failure_rows = evaluate_pairs(
        flow=flow,
        args=args,
        device=device,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        mean_flat=mean_flat,
        components=components,
        faces=faces,
        scan_manifest=scan_manifest,
    )
    summary_rows = build_summary(pair_rows)
    write_csv(output_dir / "pca_flow_per_pair.csv", pair_rows)
    write_csv(output_dir / "pca_flow_summary.csv", summary_rows)
    write_csv(output_dir / "pca_flow_failures.csv", failure_rows)

    if bool(args.skip_volume_trends):
        volume_rows: list[dict[str, Any]] = []
        volume_slope_rows: list[dict[str, Any]] = []
    else:
        volume_rows, volume_slope_rows = build_volume_trends(
            flow=flow,
            args=args,
            device=device,
            mean_flat=mean_flat,
            components=components,
            faces=faces,
            scan_manifest=scan_manifest,
        )
        write_csv(output_dir / "pca_flow_volume_trends.csv", volume_rows)
        write_csv(output_dir / "pca_flow_volume_trend_slope_summary.csv", volume_slope_rows)

    if bool(args.skip_ood):
        ood_rows: list[dict[str, Any]] = []
        ood_summary_rows: list[dict[str, Any]] = []
    else:
        ood_rows, ood_summary_rows = build_ood_forecasts(
            flow=flow,
            args=args,
            device=device,
            mean_flat=mean_flat,
            components=components,
            faces=faces,
        )
        write_csv(output_dir / "pca_flow_ood_conditional_forecasts.csv", ood_rows)
        write_csv(output_dir / "pca_flow_ood_summary.csv", ood_summary_rows)

    if bool(args.skip_baseline_comparison):
        baseline_rows = []
    else:
        baseline_rows = build_baseline_comparison(
            current_rows=pair_rows,
            baseline_csv=resolve_repo_path(args.baseline_per_pair_csv),
            splits=args.splits,
        )
        write_csv(output_dir / "baseline_comparison.csv", baseline_rows)

    run_summary = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "components": int(args.components),
        "splits": list(args.splits),
        "pair_type": str(args.pair_type),
        "surface_samples": int(args.surface_samples),
        "pair_rows": len(pair_rows),
        "summary_rows": len(summary_rows),
        "failure_rows": len(failure_rows),
        "volume_trend_rows": len(volume_rows),
        "ood_rows": len(ood_rows),
        "baseline_rows": len(baseline_rows),
        "device": str(device),
    }
    write_json(output_dir / "run.json", run_summary)
    write_html_report(
        output_dir=output_dir,
        run_summary=run_summary,
        summary_rows=summary_rows,
        baseline_rows=baseline_rows,
        volume_slope_rows=volume_slope_rows,
        ood_summary_rows=ood_summary_rows,
    )
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
