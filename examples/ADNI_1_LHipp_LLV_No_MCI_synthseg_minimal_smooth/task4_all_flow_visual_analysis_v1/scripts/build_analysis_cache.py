#!/usr/bin/env python3
"""Build all expensive data for the load-only all-flow results notebook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_ROOT.parents[2]
REGISTRY_PATH = TASK_ROOT / "configs" / "model_registry.json"
METRICS = {
    "assd_mm": False,
    "hd95_mm": False,
    "chamfer_l2_squared_mm2": False,
    "dice": True,
    "iou": True,
    "volume_relative_error": False,
    "rate_absolute_error_per_year": False,
    "normal_change_mae_mm_per_year": False,
    "normal_change_pearson": True,
    "normal_change_spearman": True,
    "hotspot_dice": True,
}
CURRENT_ORDER = ("pca", "spiral", "adaptive", "inr")
LEGACY_MODELS = {
    "pca150_direct_cocycle_flow": ("PCA Cocycle", "direct"),
    "qc_siren_drop_bad_min2": ("INR Cocycle", "direct"),
    "qc_siren_latent_ode": ("Latent ODE", "direct"),
    "qc_brainode_pca150": ("BrainODE", "brainode_endpoint"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-name", default="cache_v1")
    parser.add_argument("--surface-resolution", type=int, default=256)
    parser.add_argument("--surface-samples", type=int, default=5000)
    parser.add_argument("--voxel-pitch-mm", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--velocity-eps-years", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 3 or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return 0.0
    value = pearsonr(left, right).statistic if method == "pearson" else spearmanr(left, right).statistic
    return float(value) if math.isfinite(float(value)) else 0.0


def hotspot_dice(left: np.ndarray, right: np.ndarray, fraction: float = 0.20) -> float:
    count = max(1, int(math.ceil(len(left) * fraction)))
    first = set(np.argpartition(np.abs(left), -count)[-count:].tolist())
    second = set(np.argpartition(np.abs(right), -count)[-count:].tolist())
    return float(len(first & second) / count)


def mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def surface_distances(predicted: trimesh.Trimesh, target: trimesh.Trimesh, samples: int, seed: int) -> dict[str, float]:
    predicted_points, _ = trimesh.sample.sample_surface(predicted, samples, seed=seed)
    target_points, _ = trimesh.sample.sample_surface(target, samples, seed=seed + 1)
    pred_to_target = cKDTree(target_points).query(predicted_points, workers=-1)[0]
    target_to_pred = cKDTree(predicted_points).query(target_points, workers=-1)[0]
    return {
        "assd_mm": float(0.5 * (pred_to_target.mean() + target_to_pred.mean())),
        "hd95_mm": float(max(np.quantile(pred_to_target, 0.95), np.quantile(target_to_pred, 0.95))),
        "chamfer_l2_squared_mm2": float(np.mean(pred_to_target**2) + np.mean(target_to_pred**2)),
    }


def voxel_keys(value: trimesh.Trimesh, pitch: float) -> set[tuple[int, int, int]]:
    points = value.voxelized(float(pitch)).fill().points
    indices = np.rint(points / float(pitch)).astype(np.int64)
    return set(map(tuple, indices.tolist()))


def overlap(predicted: trimesh.Trimesh, target: trimesh.Trimesh, pitch: float) -> dict[str, float]:
    left, right = voxel_keys(predicted, pitch), voxel_keys(target, pitch)
    intersection = len(left & right)
    return {
        "dice": float(2.0 * intersection / max(len(left) + len(right), 1)),
        "iou": float(intersection / max(len(left | right), 1)),
    }


def normal_rate(source_vertices: np.ndarray, predicted: trimesh.Trimesh | np.ndarray, faces: np.ndarray, years: float) -> np.ndarray:
    source = mesh(source_vertices, faces)
    normals = np.asarray(source.vertex_normals)
    if isinstance(predicted, trimesh.Trimesh):
        sampled, _ = trimesh.sample.sample_surface(predicted, 10000, seed=1729)
        points = sampled[cKDTree(sampled).query(np.asarray(source_vertices), workers=-1)[1]]
    else:
        points = np.asarray(predicted)
    return np.sum((points - np.asarray(source_vertices)) * normals, axis=1) / float(years)


def endpoint_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    source_volume: float,
    target_volume: float,
    years: float,
    predicted_rate: np.ndarray,
    observed_rate: np.ndarray,
    samples: int,
    pitch: float,
    seed: int,
) -> dict[str, float]:
    predicted_volume = abs(float(predicted.volume))
    observed_log_rate = math.log(target_volume / source_volume) / years
    predicted_log_rate = math.log(predicted_volume / source_volume) / years
    return {
        **surface_distances(predicted, target, samples, seed),
        **overlap(predicted, target, pitch),
        "predicted_volume_mm3": predicted_volume,
        "volume_relative_error": abs(predicted_volume - target_volume) / target_volume,
        "observed_signed_log_volume_rate_per_year": observed_log_rate,
        "predicted_signed_log_volume_rate_per_year": predicted_log_rate,
        "rate_absolute_error_per_year": abs(predicted_log_rate - observed_log_rate),
        "atrophy_direction_agreement": float((predicted_log_rate < 0.0) == (observed_log_rate < 0.0)),
        "normal_change_mae_mm_per_year": float(np.mean(np.abs(predicted_rate - observed_rate))),
        "normal_change_pearson": safe_corr(predicted_rate, observed_rate, "pearson"),
        "normal_change_spearman": safe_corr(predicted_rate, observed_rate, "spearman"),
        "hotspot_dice": hotspot_dice(predicted_rate, observed_rate),
    }


def unload_local_modules(script_path: Path) -> None:
    for name in ("common", "models", "evaluate", "inr_geometry", "c4_objective"):
        sys.modules.pop(name, None)
    sys.path[:] = [item for item in sys.path if item != str(script_path)]


def load_local_modules(task_path: Path):
    script_path = task_path / "scripts"
    unload_local_modules(script_path)
    sys.path.insert(0, str(script_path))
    common = importlib.import_module("common")
    models = importlib.import_module("models")
    return script_path, common, models


def load_direct_flow(models, config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device):
    flow = models.DirectC4Flow(
        int(config["model"]["latent_dim"]),
        int(config["model"]["width"]),
        int(config["model"]["residual_blocks"]),
        float(config["model"].get("dropout", 0.0)),
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
    return flow.eval()


def require_full_evaluation(run_dir: Path, inr: bool = False) -> int:
    if run_dir.name.startswith("smoke_") or any(part.startswith("smoke_") for part in run_dir.parts):
        raise ValueError(f"Smoke run rejected: {run_dir}")
    report = read_json(run_dir / "evaluation" / "test" / "summary.json")
    if inr:
        rows = int(report["proxy_pair_metrics"]["all_forward"]["groups"]["overall"]["pairs"])
    else:
        rows = int(report["pair_metrics"]["all_forward"]["groups"]["overall"]["rows"])
    if rows < 100:
        raise ValueError(f"Incomplete test evaluation rejected ({rows} pairs): {run_dir}")
    return rows


def age_slope_per_year(archive: dict[str, np.ndarray]) -> float:
    age = archive["visit_age_years"].astype(np.float64)
    norm = archive["visit_age_norm_train"].astype(np.float64)
    slope = float(np.polyfit(age, norm, 1)[0])
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("Invalid normalized-age scale")
    return slope


def observed_velocity(latents: np.ndarray, archive: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    output = np.full_like(latents, np.nan, dtype=np.float64)
    gaps = np.full(len(latents), np.nan, dtype=np.float64)
    methods = ["unavailable"] * len(latents)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    ages = archive["visit_age_years"].astype(np.float64)
    for subject in range(len(offsets) - 1):
        start, stop = int(offsets[subject]), int(offsets[subject + 1])
        for index in range(start, stop):
            if index == start:
                left, right, method = start, start + 1, "forward"
            elif index == stop - 1:
                left, right, method = stop - 2, stop - 1, "backward"
            else:
                left, right, method = index - 1, index + 1, "central"
            gap = float(ages[right] - ages[left])
            output[index] = (latents[right] - latents[left]) / gap
            gaps[index], methods[index] = gap, method
    return output, gaps, methods


@torch.no_grad()
def velocity_rows(
    method: str,
    label: str,
    flow,
    archive: dict[str, np.ndarray],
    latent_key: str,
    device: torch.device,
    eps_years: float,
) -> list[dict[str, Any]]:
    latents = archive[latent_key].astype(np.float32)
    observed, gaps, schemes = observed_velocity(latents, archive)
    age = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    latent = torch.from_numpy(latents).to(device)
    diagnosis = archive["visit_diagnoses"].astype(str)
    condition = torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device)
    scale = age_slope_per_year(archive)
    diagonal = flow.average_velocity(latent, age, age, condition) * scale
    epsilon_norm = float(eps_years * scale)
    finite = (flow.transport(latent, age, age + epsilon_norm, condition) - latent) / float(eps_years)
    cn = flow.average_velocity(latent, age, age, torch.zeros_like(condition)) * scale
    ad = flow.average_velocity(latent, age, age, torch.ones_like(condition)) * scale
    diagonal_np, finite_np = diagonal.cpu().numpy(), finite.cpu().numpy()
    cn_np, ad_np = cn.cpu().numpy(), ad.cpu().numpy()
    rows = []
    for index in range(len(latents)):
        real, model = observed[index], diagonal_np[index]
        real_rms = float(np.sqrt(np.mean(real**2)))
        model_rms = float(np.sqrt(np.mean(model**2)))
        denominator = float(np.linalg.norm(real) * np.linalg.norm(model))
        cosine = float(np.dot(real, model) / denominator) if denominator > 1.0e-12 else 0.0
        rows.append({
            "method": method,
            "method_label": label,
            "split": "test",
            "subject_id": str(archive["visit_subject_ids"][index]),
            "scan_id": str(archive["visit_scan_ids"][index]),
            "diagnosis": str(diagnosis[index]),
            "age_years": float(archive["visit_age_years"][index]),
            "visit_order": int(archive["visit_orders"][index]),
            "real_velocity_scheme": schemes[index],
            "real_velocity_gap_years": float(gaps[index]),
            "real_rms_per_coordinate_per_year": real_rms,
            "model_rms_per_coordinate_per_year": model_rms,
            "velocity_rmse_per_coordinate_per_year": float(np.sqrt(np.mean((model - real) ** 2))),
            "velocity_cosine": cosine,
            "model_to_real_speed_ratio": model_rms / max(real_rms, 1.0e-12),
            "diagonal_fd_rmse_per_coordinate_per_year": float(np.sqrt(np.mean((model - finite_np[index]) ** 2))),
            "ad_minus_cn_condition_velocity_rms_per_coordinate_per_year": float(np.sqrt(np.mean((ad_np[index] - cn_np[index]) ** 2))),
        })
    return rows


def summary_and_bootstrap(endpoint: pd.DataFrame, samples: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, bootstraps = [], []
    rng = np.random.default_rng(seed)
    for (method, label, diagnosis), group in endpoint.groupby(["method", "method_label", "diagnosis"], sort=False):
        groups = [(diagnosis, group)]
        if diagnosis == "CN":
            pass
        for _, current in groups:
            row = {"method": method, "method_label": label, "diagnosis": diagnosis, "subjects": int(current.subject_id.nunique())}
            for metric in METRICS:
                row[metric] = float(current[metric].mean())
                row[f"nochange_{metric}"] = float(current[f"nochange_{metric}"].mean())
                row[f"representation_floor_{metric}"] = float(current[f"representation_floor_{metric}"].mean())
            summaries.append(row)
            indices = rng.integers(0, len(current), size=(samples, len(current)))
            for metric in METRICS:
                values = current[metric].to_numpy(dtype=float)
                draws = values[indices].mean(axis=1)
                bootstraps.append({
                    "method": method, "method_label": label, "diagnosis": diagnosis, "metric": metric,
                    "mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)), "samples": samples,
                })
    for method, group in endpoint.groupby("method", sort=False):
        label = str(group.method_label.iloc[0])
        row = {"method": method, "method_label": label, "diagnosis": "overall", "subjects": int(group.subject_id.nunique())}
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
            row[f"nochange_{metric}"] = float(group[f"nochange_{metric}"].mean())
            row[f"representation_floor_{metric}"] = float(group[f"representation_floor_{metric}"].mean())
        summaries.append(row)
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            draws = values[indices].mean(axis=1)
            bootstraps.append({
                "method": method, "method_label": label, "diagnosis": "overall", "metric": metric,
                "mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)), "samples": samples,
            })
    return summaries, bootstraps


def velocity_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "real_rms_per_coordinate_per_year", "model_rms_per_coordinate_per_year",
        "velocity_rmse_per_coordinate_per_year", "velocity_cosine", "model_to_real_speed_ratio",
        "diagonal_fd_rmse_per_coordinate_per_year", "ad_minus_cn_condition_velocity_rms_per_coordinate_per_year",
    ]
    rows = []
    augmented = pd.concat((frame, frame.assign(diagnosis="overall")), ignore_index=True)
    for (method, label, diagnosis), group in augmented.groupby(["method", "method_label", "diagnosis"], sort=False):
        row = {"method": method, "method_label": label, "diagnosis": diagnosis, "scans": len(group), "subjects": group.subject_id.nunique()}
        for field in fields:
            row[f"{field}_mean"] = float(group[field].mean())
            row[f"{field}_median"] = float(group[field].median())
        rows.append(row)
    return rows


def paired_tests(endpoint: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for metric, higher_is_better in METRICS.items():
        pivot = endpoint.pivot(index="subject_id", columns="method", values=metric)
        current_rows = []
        for first, second in combinations(CURRENT_ORDER, 2):
            values = pivot[[first, second]].dropna()
            raw_difference = values[first].to_numpy() - values[second].to_numpy()
            oriented = raw_difference if higher_is_better else -raw_difference
            p_value = 1.0 if np.allclose(raw_difference, 0.0) else float(wilcoxon(raw_difference, alternative="two-sided", zero_method="wilcox").pvalue)
            current_rows.append({
                "metric": metric, "higher_is_better": higher_is_better,
                "first_method": first, "second_method": second, "subjects": len(values),
                "first_minus_second_mean": float(raw_difference.mean()),
                "oriented_first_better_mean": float(oriented.mean()),
                "first_win_fraction": float(np.mean(oriented > 0.0)),
                "wilcoxon_p": p_value,
            })
        order = np.argsort([row["wilcoxon_p"] for row in current_rows])
        running = 0.0
        adjusted = np.ones(len(current_rows), dtype=float)
        for rank, index in enumerate(order):
            candidate = min(1.0, current_rows[index]["wilcoxon_p"] * (len(current_rows) - rank))
            running = max(running, candidate)
            adjusted[index] = running
        for index, row in enumerate(current_rows):
            row["holm_adjusted_p"] = float(adjusted[index])
            rows.append(row)
    return rows


def consistency_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in CURRENT_ORDER:
        specification = registry["current_models"][method]
        run_dir = resolve(specification["run_dir"])
        evaluation = run_dir / ("evaluation_mc256_exact" if method == "inr" else "evaluation") / "test" / "summary.json"
        report = read_json(evaluation)
        defects = report["consistency_defects"]
        rows.append({
            "method": method, "method_label": specification["label"],
            "semigroup_mean": float(defects["relative_semigroup_defect_mean"]),
            "semigroup_p95": float(defects["relative_semigroup_defect_p95"]),
            "inverse_mean": float(defects["relative_inverse_defect_mean"]),
            "inverse_p95": float(defects["relative_inverse_defect_p95"]),
        })
    return rows


def copy_legacy(registry: dict[str, Any], tables: Path) -> pd.DataFrame:
    source = pd.read_csv(resolve(registry["legacy"]["mesh_metrics"]), low_memory=False)
    source = source[(source.dataset == "qc_large") & (source.split == "test") & source.decode_success.astype(bool)].copy()
    selected = []
    for model_name, (label, transport) in LEGACY_MODELS.items():
        current = source[(source.model == model_name) & (source.transport_method == transport)].copy()
        current["method_label"] = label
        selected.append(current)
    legacy = pd.concat(selected, ignore_index=True)
    columns = [
        "model", "method_label", "family", "subject_id", "diagnosis", "source_scan_id", "target_scan_id",
        "pair_type", "gap_years", "transport_method", "assd", "hd95", "chamfer_l2_squared",
        "volume_relative_error", "source_gt_no_change_assd", "source_gt_no_change_hd95",
        "source_gt_no_change_chamfer_l2_squared", "source_gt_no_change_volume_relative_error",
    ]
    legacy[columns].to_csv(tables / "legacy_endpoint_metrics.csv", index=False)
    velocity = pd.read_csv(resolve(registry["legacy"]["velocity_per_scan"]), low_memory=False)
    velocity = velocity[(velocity.split == "test") & (velocity.condition_eval == "observed")].copy()
    label_map = {
        "qc_brainode_pca150": "BrainODE", "pca150_direct_cocycle_flow": "PCA Cocycle",
        "qc_siren_latent_ode": "Latent ODE", "qc_siren_drop_bad_min2": "INR Cocycle",
    }
    velocity = velocity[velocity.model.isin(label_map)].copy()
    velocity["method_label"] = velocity.model.map(label_map)
    velocity.to_csv(tables / "legacy_velocity_per_scan.csv", index=False)
    return legacy


def legacy_surface_maps(legacy: pd.DataFrame, destination: Path, pca_group_map_root: Path) -> None:
    desired = legacy[legacy.predicted_mesh_path.notna()].copy()
    keys = ["subject_id", "diagnosis", "source_scan_id", "target_scan_id"]
    maps: dict[str, dict[str, list[np.ndarray]]] = {label: {"CN": [], "AD": []} for label, _ in LEGACY_MODELS.values()}
    maps["Observed"] = {"CN": [], "AD": []}
    subject_maps: dict[tuple[str, str, str], list[np.ndarray]] = {}
    template_vertices, template_faces = [], None
    grouped = desired.groupby(keys, sort=False)
    total = len(grouped)
    for number, (key, group) in enumerate(grouped, start=1):
        diagnosis, subject = str(key[1]), str(key[0])
        first = group.iloc[0]
        source_mesh = trimesh.load(str(first.source_mesh_path), force="mesh", process=False)
        target_mesh = trimesh.load(str(first.target_mesh_path), force="mesh", process=False)
        years = float(first.gap_years)
        if template_faces is None:
            template_faces = np.asarray(source_mesh.faces)
        template_vertices.append(np.asarray(source_mesh.vertices))
        observed = normal_rate(np.asarray(source_mesh.vertices), np.asarray(target_mesh.vertices), np.asarray(source_mesh.faces), years)
        subject_maps.setdefault(("Observed", diagnosis, subject), []).append(observed)
        for item in group.itertuples(index=False):
            predicted = trimesh.load(str(item.predicted_mesh_path), force="mesh", process=False)
            if len(predicted.vertices) == len(source_mesh.vertices):
                rate = normal_rate(np.asarray(source_mesh.vertices), np.asarray(predicted.vertices), np.asarray(source_mesh.faces), years)
            else:
                rate = normal_rate(np.asarray(source_mesh.vertices), predicted, np.asarray(source_mesh.faces), years)
            subject_maps.setdefault((str(item.method_label), diagnosis, subject), []).append(rate)
        if number % 20 == 0 or number == total:
            print(f"legacy map {number:03d}/{total:03d}", flush=True)
    for (label, diagnosis, _), values in subject_maps.items():
        maps[label][diagnosis].append(np.mean(np.stack(values), axis=0))
    arrays: dict[str, np.ndarray] = {
        "template_vertices": np.mean(np.stack(template_vertices), axis=0).astype(np.float32),
        "faces": np.asarray(template_faces, dtype=np.int32),
    }
    name_map = {"Observed": "observed", "PCA Cocycle": "pca_cocycle", "INR Cocycle": "inr_cocycle", "Latent ODE": "latent_ode", "BrainODE": "brainode"}
    pca_cn = pd.read_csv(pca_group_map_root / "group_cn_change_maps.csv")["cocycle_flow_normal_change_rate"].to_numpy(dtype=np.float32)
    pca_ad = pd.read_csv(pca_group_map_root / "group_ad_change_maps.csv")["cocycle_flow_normal_change_rate"].to_numpy(dtype=np.float32)
    arrays["pca_cocycle_cn"], arrays["pca_cocycle_ad"] = pca_cn, pca_ad
    arrays["pca_cocycle_gap"] = pca_ad - pca_cn
    for label, key in name_map.items():
        if label == "PCA Cocycle":
            continue
        for diagnosis in ("CN", "AD"):
            arrays[f"{key}_{diagnosis.lower()}"] = np.mean(np.stack(maps[label][diagnosis]), axis=0).astype(np.float32)
        arrays[f"{key}_gap"] = arrays[f"{key}_ad"] - arrays[f"{key}_cn"]
    np.savez_compressed(destination, **arrays)


def main() -> int:
    args = parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in {"", ".", ".."}:
        raise ValueError("output-name must be one safe path component")
    registry = read_json(REGISTRY_PATH)
    output_root = resolve(registry["output_root"])
    destination = output_root / args.output_name
    if destination.exists():
        manifest = destination / "manifest.json"
        if args.reuse and manifest.is_file() and read_json(manifest).get("status") == "complete":
            print(f"REUSING {destination}")
            return 0
        raise FileExistsError(f"Refusing to overwrite {destination}")
    tables, arrays = destination / "tables", destination / "arrays"
    tables.mkdir(parents=True, exist_ok=False)
    arrays.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "manifest.json", {"status": "building", "registry": str(REGISTRY_PATH), "source_meshes_modified": False})
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    task128 = resolve(registry["current_128_task"])
    script128, C128, M128 = load_local_modules(task128)
    archive128 = C128.load_archive("pca128", "test")
    raw_vertices = np.asarray(C128.cached_vertices("test", mmap=False), dtype=np.float32)
    inr_registry = read_json(resolve(registry["current_inr_task"]) / "configs" / "inr256_representation.json")
    with np.load(resolve(inr_registry["output_root"]) / "representations" / "inr256" / "test_subject_sequences_256.npz", allow_pickle=False) as loaded:
        inr_test_probe = {key: loaded[key] for key in ("subject_ids", "visit_scan_ids")}
    shared_subjects = set(inr_test_probe["subject_ids"].astype(str))
    shared_scans = set(inr_test_probe["visit_scan_ids"].astype(str))
    if not shared_subjects.issubset(set(archive128["subject_ids"].astype(str))):
        raise ValueError("INR subjects are not contained in the current test cohort")
    selected_indices = np.asarray([i for i, scan in enumerate(archive128["visit_scan_ids"].astype(str)) if scan in shared_scans], dtype=np.int64)
    if len(selected_indices) != 100:
        raise ValueError(f"Expected 100 shared scans, got {len(selected_indices)}")
    first_last_128 = [row for row in C128.first_last_pairs(archive128) if row.subject in shared_subjects]
    if len(first_last_128) != 20:
        raise ValueError(f"Expected 20 shared first-last pairs, got {len(first_last_128)}")
    faces = np.load(resolve(C128.load_registry()["representations"]["pca128"]["pca_model_root"]) / "faces.npy", allow_pickle=False)
    template_sources = [raw_vertices[row.source] for row in first_last_128]
    map_values: dict[str, dict[str, list[np.ndarray]]] = {name: {"CN": [], "AD": []} for name in ("observed", *CURRENT_ORDER)}
    endpoint_rows: list[dict[str, Any]] = []
    all_velocity_rows: list[dict[str, Any]] = []
    baseline_cache: dict[str, tuple[dict[str, float], np.ndarray]] = {}
    checkpoint_inventory = []

    for method in CURRENT_ORDER[:3]:
        specification = registry["current_models"][method]
        representation, label = specification["representation"], specification["label"]
        run_dir = resolve(specification["run_dir"])
        config = read_json(run_dir / "resolved_config.json")["config"]
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        evaluation_pairs = require_full_evaluation(run_dir)
        checkpoint_inventory.append({"method": method, "method_label": label, "run_dir": str(run_dir), "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]), "test_evaluation_pairs": evaluation_pairs, "checkpoint_sha256": sha256(checkpoint_path), "status": "full", "primary": True})
        train_archive = C128.load_archive(representation, "train")
        archive = C128.load_archive(representation, "test")
        geometry = C128.build_geometry(representation, train_archive, device)
        flow = load_direct_flow(M128, config, checkpoint, device)
        latent = torch.from_numpy(archive["visit_latent_standardized_128"].astype(np.float32)).to(device)
        age = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
        condition = torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device)
        all_velocity_rows.extend(
            item for item in velocity_rows(method, label, flow, archive, "visit_latent_standardized_128", device, args.velocity_eps_years)
            if item["subject_id"] in shared_subjects
        )
        for number, row in enumerate(first_last_128, start=1):
            prediction_z = flow.transport(latent[row.source:row.source + 1], age[row.source:row.source + 1], age[row.target:row.target + 1], condition[row.source:row.source + 1])
            predicted_vertices = geometry.vertices(prediction_z)[0].detach().cpu().numpy()
            floor_vertices = geometry.vertices(latent[row.target:row.target + 1])[0].detach().cpu().numpy()
            source_vertices, target_vertices = raw_vertices[row.source], raw_vertices[row.target]
            source_mesh, target_mesh = mesh(source_vertices, faces), mesh(target_vertices, faces)
            predicted_mesh, floor_mesh = mesh(predicted_vertices, faces), mesh(floor_vertices, faces)
            years = float(row.delta_years)
            observed_map = normal_rate(source_vertices, target_vertices, faces, years)
            predicted_map = normal_rate(source_vertices, predicted_vertices, faces, years)
            floor_map = normal_rate(source_vertices, floor_vertices, faces, years)
            if row.subject not in baseline_cache:
                baseline_cache[row.subject] = (
                    endpoint_metrics(source_mesh, target_mesh, abs(float(source_mesh.volume)), abs(float(target_mesh.volume)), years, np.zeros_like(observed_map), observed_map, args.surface_samples, args.voxel_pitch_mm, 10000 + number),
                    observed_map,
                )
                map_values["observed"][row.diagnosis].append(observed_map)
            predicted_metrics = endpoint_metrics(predicted_mesh, target_mesh, abs(float(source_mesh.volume)), abs(float(target_mesh.volume)), years, predicted_map, observed_map, args.surface_samples, args.voxel_pitch_mm, 20000 + 100 * CURRENT_ORDER.index(method) + number)
            floor_metrics = endpoint_metrics(floor_mesh, target_mesh, abs(float(source_mesh.volume)), abs(float(target_mesh.volume)), years, floor_map, observed_map, args.surface_samples, args.voxel_pitch_mm, 30000 + 100 * CURRENT_ORDER.index(method) + number)
            baseline_metrics = baseline_cache[row.subject][0]
            record = {"method": method, "method_label": label, "subject_id": row.subject, "diagnosis": row.diagnosis, "source_scan_id": str(archive["visit_scan_ids"][row.source]), "target_scan_id": str(archive["visit_scan_ids"][row.target]), "followup_years": years}
            record.update(predicted_metrics)
            for key in METRICS:
                record[f"nochange_{key}"] = baseline_metrics[key]
                record[f"representation_floor_{key}"] = floor_metrics[key]
            endpoint_rows.append(record)
            map_values[method][row.diagnosis].append(predicted_map)
            print(f"current {label} {number:02d}/20 {row.subject} {row.diagnosis}", flush=True)
        del geometry, flow, latent
        if device.type == "cuda":
            torch.cuda.empty_cache()

    unload_local_modules(script128)
    task_inr = resolve(registry["current_inr_task"])
    script_inr, CI, MI = load_local_modules(task_inr)
    GI = importlib.import_module("inr_geometry")
    specification = registry["current_models"]["inr"]
    run_dir = resolve(specification["run_dir"])
    config = read_json(run_dir / "resolved_config.json")["config"]
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    evaluation_pairs = require_full_evaluation(run_dir, inr=True)
    checkpoint_inventory.append({"method": "inr", "method_label": "INR", "run_dir": str(run_dir), "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]), "test_evaluation_pairs": evaluation_pairs, "checkpoint_sha256": sha256(checkpoint_path), "status": "full", "primary": True})
    train_inr, test_inr = CI.load_archive("train"), CI.load_archive("test")
    geometry_inr = GI.build_geometry(train_inr, device)
    flow_inr = load_direct_flow(MI, config, checkpoint, device)
    all_velocity_rows.extend(velocity_rows("inr", "INR", flow_inr, test_inr, "visit_latent_standardized_256", device, args.velocity_eps_years))
    latent = torch.from_numpy(test_inr["visit_latent_standardized_256"].astype(np.float32)).to(device)
    age = torch.from_numpy(test_inr["visit_age_norm_train"].astype(np.float32)).to(device)
    condition = torch.from_numpy(test_inr["visit_label_ad"].astype(np.float32)).to(device)
    raw_index = {scan: i for i, scan in enumerate(archive128["visit_scan_ids"].astype(str))}
    first_last_inr = CI.first_last_pairs(test_inr)
    for number, row in enumerate(first_last_inr, start=1):
        prediction_z = flow_inr.transport(latent[row.source:row.source + 1], age[row.source:row.source + 1], age[row.target:row.target + 1], condition[row.source:row.source + 1])
        predicted_mesh = geometry_inr.mesh(prediction_z, args.surface_resolution)
        floor_mesh = geometry_inr.mesh(latent[row.target:row.target + 1], args.surface_resolution)
        source_scan, target_scan = str(test_inr["visit_scan_ids"][row.source]), str(test_inr["visit_scan_ids"][row.target])
        source_vertices, target_vertices = raw_vertices[raw_index[source_scan]], raw_vertices[raw_index[target_scan]]
        source_mesh, target_mesh = mesh(source_vertices, faces), mesh(target_vertices, faces)
        years = float(row.delta_years)
        observed_map = normal_rate(source_vertices, target_vertices, faces, years)
        predicted_map = normal_rate(source_vertices, predicted_mesh, faces, years)
        floor_map = normal_rate(source_vertices, floor_mesh, faces, years)
        predicted_metrics = endpoint_metrics(predicted_mesh, target_mesh, abs(float(source_mesh.volume)), abs(float(target_mesh.volume)), years, predicted_map, observed_map, args.surface_samples, args.voxel_pitch_mm, 40000 + number)
        floor_metrics = endpoint_metrics(floor_mesh, target_mesh, abs(float(source_mesh.volume)), abs(float(target_mesh.volume)), years, floor_map, observed_map, args.surface_samples, args.voxel_pitch_mm, 50000 + number)
        baseline_metrics = baseline_cache[row.subject][0]
        record = {"method": "inr", "method_label": "INR", "subject_id": row.subject, "diagnosis": row.diagnosis, "source_scan_id": source_scan, "target_scan_id": target_scan, "followup_years": years}
        record.update(predicted_metrics)
        for key in METRICS:
            record[f"nochange_{key}"] = baseline_metrics[key]
            record[f"representation_floor_{key}"] = floor_metrics[key]
        endpoint_rows.append(record)
        map_values["inr"][row.diagnosis].append(predicted_map)
        print(f"current INR {number:02d}/20 {row.subject} {row.diagnosis}", flush=True)

    endpoint = pd.DataFrame(endpoint_rows)
    endpoint.to_csv(tables / "current_endpoint_metrics.csv", index=False)
    velocity = pd.DataFrame(all_velocity_rows)
    velocity.to_csv(tables / "current_velocity_per_scan.csv", index=False)
    summaries, bootstraps = summary_and_bootstrap(endpoint, args.bootstrap_samples, args.seed)
    write_csv(tables / "current_metric_summary.csv", summaries)
    write_csv(tables / "current_metric_bootstrap.csv", bootstraps)
    write_csv(tables / "current_velocity_summary.csv", velocity_summary(velocity))
    write_csv(tables / "current_paired_tests.csv", paired_tests(endpoint))
    write_csv(tables / "current_consistency.csv", consistency_rows(registry))
    write_csv(tables / "checkpoint_inventory.csv", checkpoint_inventory)

    current_arrays: dict[str, np.ndarray] = {
        "template_vertices": np.mean(np.stack(template_sources), axis=0).astype(np.float32),
        "faces": faces.astype(np.int32),
    }
    for method in ("observed", *CURRENT_ORDER):
        for diagnosis in ("CN", "AD"):
            current_arrays[f"{method}_{diagnosis.lower()}"] = np.mean(np.stack(map_values[method][diagnosis]), axis=0).astype(np.float32)
        current_arrays[f"{method}_gap"] = current_arrays[f"{method}_ad"] - current_arrays[f"{method}_cn"]
    np.savez_compressed(arrays / "current_surface_maps.npz", **current_arrays)

    legacy = copy_legacy(registry, tables)
    legacy_surface_maps(legacy, arrays / "legacy_surface_maps.npz", resolve(registry["legacy"]["pca_group_map_root"]))
    manifest = {
        "status": "complete", "schema_version": 1, "seed": args.seed,
        "surface_resolution": args.surface_resolution, "surface_samples": args.surface_samples,
        "voxel_pitch_mm": args.voxel_pitch_mm, "bootstrap_samples": args.bootstrap_samples,
        "velocity_eps_years": args.velocity_eps_years, "shared_subjects": len(shared_subjects),
        "shared_scans": len(shared_scans), "current_endpoint_rows": len(endpoint),
        "current_velocity_rows": len(velocity), "legacy_endpoint_rows": len(legacy),
        "primary_methods": list(CURRENT_ORDER), "legacy_results_separate": True,
        "smoke_runs_excluded": True, "source_meshes_modified": False,
        "surface_distance_definition": "deterministic bidirectional sampled-surface nearest-neighbour distance via cKDTree",
        "files": sorted(str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file() and path.name != "manifest.json"),
    }
    atomic_json(destination / "manifest.json", manifest)
    print(f"WROTE {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
