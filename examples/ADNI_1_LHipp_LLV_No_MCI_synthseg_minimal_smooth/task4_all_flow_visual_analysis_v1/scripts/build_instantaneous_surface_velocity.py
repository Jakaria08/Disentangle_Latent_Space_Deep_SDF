#!/usr/bin/env python3
"""Precompute instantaneous surface-velocity results for the load-only notebook.

The current and legacy stages intentionally use their existing project environments:

* current: ``pytorch_geo`` (registered-mesh decoders require torch_scatter)
* legacy: ``inr_sdf`` (the completed ODE/BrainODE reference loaders)

No checkpoint or source mesh is modified.  Outputs are small CSV/NPZ supplements
stored beside this analysis rather than mixed into the completed base cache.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import trimesh


REPO = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parents[1]
REGISTRY = TASK / "configs" / "model_registry.json"
DEFAULT_OUTPUT = TASK / "derived_cache" / "instantaneous_surface_velocity_v1"
AGE_BINS = (-np.inf, 70.0, 75.0, 80.0, 85.0, np.inf)
AGE_LABELS = ("<70", "70–75", "75–80", "80–85", "85+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("current", "legacy"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epsilon-years", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO / value


def prepare_output(root: Path, stage: str, force: bool) -> tuple[Path, Path, Path]:
    root = root.expanduser().resolve()
    tables, arrays = root / "tables", root / "arrays"
    tables.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    targets = (
        tables / f"{stage}_instantaneous_surface_velocity.csv",
        arrays / f"{stage}_instantaneous_surface_maps.npz",
        root / f"manifest_{stage}.json",
    )
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    return targets


def outward_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    normals = np.array(mesh.vertex_normals, dtype=np.float64, copy=True)
    signed_volume = float(mesh.volume)
    if signed_volume < 0.0:
        normals *= -1.0
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)
    area = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(area, np.asarray(faces).reshape(-1), np.repeat(face_area / 3.0, 3))
    return normals, area, abs(signed_volume)


def safe_corr(left: np.ndarray, right: np.ndarray, method: str = "pearson") -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
        return 0.0
    if method == "spearman":
        return float(pd.Series(left).corr(pd.Series(right), method="spearman"))
    return float(np.corrcoef(left, right)[0, 1])


def weighted_stats(values: np.ndarray, area: np.ndarray, volume: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    total = max(float(area.sum()), 1.0e-12)
    return {
        "normal_mean_mm_per_year": float(np.sum(area * values) / total),
        "normal_abs_mean_mm_per_year": float(np.sum(area * np.abs(values)) / total),
        "normal_rms_mm_per_year": float(np.sqrt(np.sum(area * values**2) / total)),
        "contracting_area_fraction": float(np.sum(area[values < 0.0]) / total),
        "surface_integral_log_volume_rate_per_year": float(np.sum(area * values) / max(volume, 1.0e-12)),
    }


def comparison_stats(model: np.ndarray, observed: np.ndarray, area: np.ndarray) -> dict[str, float]:
    total = max(float(area.sum()), 1.0e-12)
    observed_rms = math.sqrt(float(np.sum(area * observed**2) / total))
    model_rms = math.sqrt(float(np.sum(area * model**2) / total))
    cutoff = float(np.quantile(np.abs(observed), 0.8))
    truth_hot = np.abs(observed) >= cutoff
    model_cutoff = float(np.quantile(np.abs(model), 0.8))
    model_hot = np.abs(model) >= model_cutoff
    intersection = int(np.logical_and(truth_hot, model_hot).sum())
    return {
        "normal_mae_mm_per_year": float(np.sum(area * np.abs(model - observed)) / total),
        "normal_rmse_mm_per_year": float(np.sqrt(np.sum(area * (model - observed) ** 2) / total)),
        "normal_pearson": safe_corr(model, observed),
        "normal_spearman": safe_corr(model, observed, "spearman"),
        "normal_sign_agreement": float(np.sum(area * (np.sign(model) == np.sign(observed))) / total),
        "hotspot_dice": float(2 * intersection / max(int(truth_hot.sum() + model_hot.sum()), 1)),
        "model_to_observed_speed_ratio": float(model_rms / max(observed_rms, 1.0e-12)),
    }


def observed_surface_fields(
    vertices: np.ndarray,
    ages: np.ndarray,
    offsets: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str], np.ndarray]:
    fields: list[np.ndarray] = []
    normals_all: list[np.ndarray] = []
    area_all: list[np.ndarray] = []
    volumes = np.empty(len(vertices), dtype=np.float64)
    gap = np.empty(len(vertices), dtype=np.float64)
    scheme = [""] * len(vertices)
    for index in range(len(vertices)):
        normals, area, volume = outward_geometry(vertices[index], faces)
        normals_all.append(normals)
        area_all.append(area)
        volumes[index] = volume
    for subject in range(len(offsets) - 1):
        start, stop = int(offsets[subject]), int(offsets[subject + 1])
        for index in range(start, stop):
            if index == start:
                left, right, label = start, start + 1, "forward"
            elif index == stop - 1:
                left, right, label = stop - 2, stop - 1, "backward"
            else:
                left, right, label = index - 1, index + 1, "central"
            years = float(ages[right] - ages[left])
            vector = (vertices[right] - vertices[left]) / years
            fields.append(np.sum(vector * normals_all[index], axis=1))
            gap[index], scheme[index] = years, label
    return fields, normals_all, area_all, volumes, scheme, gap


def aggregate_maps(
    template_vertices: np.ndarray,
    faces: np.ndarray,
    metadata: pd.DataFrame,
    observed: list[np.ndarray],
    model_maps: dict[str, dict[str, list[np.ndarray]]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "template_vertices": np.asarray(template_vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
    }

    def subject_then_group(values: list[np.ndarray], diagnosis: str) -> np.ndarray:
        grouped: list[np.ndarray] = []
        frame = metadata.reset_index(drop=True)
        for _, indices in frame[frame.diagnosis.eq(diagnosis)].groupby("subject_id", sort=False).groups.items():
            grouped.append(np.mean(np.stack([values[int(i)] for i in indices]), axis=0))
        return np.mean(np.stack(grouped), axis=0).astype(np.float32)

    arrays["observed_gt_cn"] = subject_then_group(observed, "CN")
    arrays["observed_gt_ad"] = subject_then_group(observed, "AD")
    arrays["observed_gt_group_gap"] = arrays["observed_gt_ad"] - arrays["observed_gt_cn"]
    for method, modes in model_maps.items():
        arrays[f"{method}_model_cn_group"] = subject_then_group(modes["observed"], "CN")
        arrays[f"{method}_model_ad_group"] = subject_then_group(modes["observed"], "AD")
        arrays[f"{method}_model_group_gap"] = arrays[f"{method}_model_ad_group"] - arrays[f"{method}_model_cn_group"]
        per_subject: list[np.ndarray] = []
        for _, indices in metadata.reset_index(drop=True).groupby("subject_id", sort=False).groups.items():
            per_subject.append(
                np.mean(np.stack([modes["ad"][int(i)] - modes["cn"][int(i)] for i in indices]), axis=0)
            )
        arrays[f"{method}_conditional_gap"] = np.mean(np.stack(per_subject), axis=0).astype(np.float32)
    return arrays


def add_surface_row(
    base: dict[str, Any],
    observed: np.ndarray,
    model_observed: np.ndarray,
    cn: np.ndarray,
    ad: np.ndarray,
    area: np.ndarray,
    volume: float,
) -> dict[str, Any]:
    row = dict(base)
    for prefix, values in (("observed_gt", observed), ("model", model_observed), ("cn_condition", cn), ("ad_condition", ad)):
        row.update({f"{prefix}_{key}": value for key, value in weighted_stats(values, area, volume).items()})
    row.update(comparison_stats(model_observed, observed, area))
    gap_stats = weighted_stats(ad - cn, area, volume)
    row.update({f"conditional_ad_minus_cn_{key}": value for key, value in gap_stats.items()})
    return row


def fixed_decoder_fields(
    flow: torch.nn.Module,
    geometry: torch.nn.Module,
    latent: torch.Tensor,
    age: torch.Tensor,
    epsilon_norm: float,
    epsilon_years: float,
    normals: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with torch.no_grad():
        baseline = geometry.vertices(latent).detach().cpu().numpy()
        cn_latent = flow.transport(latent, age, age + epsilon_norm, torch.zeros_like(age))
        ad_latent = flow.transport(latent, age, age + epsilon_norm, torch.ones_like(age))
        cn_vertices = geometry.vertices(cn_latent).detach().cpu().numpy()
        ad_vertices = geometry.vertices(ad_latent).detach().cpu().numpy()
    cn = [np.sum(((cn_vertices[i] - baseline[i]) / epsilon_years) * normals[i], axis=1) for i in range(len(normals))]
    ad = [np.sum(((ad_vertices[i] - baseline[i]) / epsilon_years) * normals[i], axis=1) for i in range(len(normals))]
    return cn, ad


def current_inr_field(
    geometry: torch.nn.Module,
    latent: torch.Tensor,
    future: torch.Tensor,
    vertices_mm: np.ndarray,
    normals_mm: np.ndarray,
    epsilon_years: float,
) -> np.ndarray:
    xyz_np = ((vertices_mm / geometry.distance_unscale_factor) - geometry.range_global_min)
    xyz_np = xyz_np * geometry.range_linear_scale_factor + geometry.target_range_min
    xyz = torch.from_numpy(xyz_np.astype(np.float32)).to(latent.device).requires_grad_(True)
    baseline = geometry.sdf(latent, xyz)[0]
    gradient = torch.autograd.grad(baseline.sum(), xyz, create_graph=False)[0]
    with torch.no_grad():
        projected = xyz - baseline[:, None] * gradient / gradient.square().sum(dim=1, keepdim=True).clamp_min(1.0e-10)
    projected = projected.detach().requires_grad_(True)
    baseline = geometry.sdf(latent, projected)[0]
    gradient = torch.autograd.grad(baseline.sum(), projected, create_graph=False)[0]
    with torch.no_grad():
        future_sdf = geometry.sdf(future, projected)[0]
        gradient_norm = torch.linalg.vector_norm(gradient, dim=1).clamp_min(1.0e-8)
        gradient_unit = gradient / gradient_norm[:, None]
        normals = torch.from_numpy(normals_mm.astype(np.float32)).to(latent.device)
        normal_alignment = torch.sum(gradient_unit * normals, dim=1)
        level_speed = -(future_sdf - baseline) / float(epsilon_years) / gradient_norm
        velocity = level_speed * normal_alignment * float(geometry.linear_normalized_to_mm)
    return velocity.detach().cpu().numpy().astype(np.float64)


def build_current(args: argparse.Namespace, csv_path: Path, npz_path: Path, manifest_path: Path) -> None:
    registry = read_json(REGISTRY)
    sys.path.insert(0, str(TASK / "scripts"))
    import build_analysis_cache as base

    device = torch.device(args.device)
    task128 = resolve(registry["current_128_task"])
    script128, common, models = base.load_local_modules(task128)
    pca_archive = common.load_archive("pca128", "test")
    raw_all = np.asarray(common.cached_vertices("test", mmap=False), dtype=np.float64)
    inr_archive_path = (
        Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1")
        / "representations" / "inr256" / "test_subject_sequences_256.npz"
    )
    with np.load(inr_archive_path, allow_pickle=False) as loaded:
        shared_scans = set(loaded["visit_scan_ids"].astype(str))
    selected = np.asarray([i for i, scan in enumerate(pca_archive["visit_scan_ids"].astype(str)) if scan in shared_scans])
    raw = raw_all[selected]
    ages = pca_archive["visit_age_years"][selected].astype(np.float64)
    subjects = pca_archive["visit_subject_ids"][selected].astype(str)
    diagnoses = pca_archive["visit_diagnoses"][selected].astype(str)
    labels = pca_archive["visit_label_ad"][selected].astype(np.int64)
    scan_ids = pca_archive["visit_scan_ids"][selected].astype(str)
    offsets = np.concatenate(([0], np.cumsum(pd.Series(subjects).value_counts(sort=False).to_numpy(dtype=np.int64))))
    faces = np.load(resolve(common.load_registry()["representations"]["pca128"]["pca_model_root"]) / "faces.npy", allow_pickle=False)
    observed, normals, areas, volumes, schemes, gaps = observed_surface_fields(raw, ages, offsets, faces)
    metadata = pd.DataFrame({"subject_id": subjects, "scan_id": scan_ids, "diagnosis": diagnoses, "label_ad": labels, "age_years": ages})
    rows: list[dict[str, Any]] = []
    maps: dict[str, dict[str, list[np.ndarray]]] = {}

    for method in ("pca", "spiral", "adaptive"):
        spec = registry["current_models"][method]
        run = resolve(spec["run_dir"])
        config = read_json(run / "resolved_config.json")["config"]
        checkpoint = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        archive = common.load_archive(spec["representation"], "test")
        train = common.load_archive(spec["representation"], "train")
        geometry = common.build_geometry(spec["representation"], train, device)
        flow = base.load_direct_flow(models, config, checkpoint, device)
        z = torch.from_numpy(archive["visit_latent_standardized_128"][selected].astype(np.float32)).to(device)
        age = torch.from_numpy(archive["visit_age_norm_train"][selected].astype(np.float32)).to(device)
        slope = base.age_slope_per_year(archive)
        cn, ad = fixed_decoder_fields(flow, geometry, z, age, args.epsilon_years * slope, args.epsilon_years, normals)
        observed_condition = [ad[i] if labels[i] else cn[i] for i in range(len(labels))]
        maps[method] = {"observed": observed_condition, "cn": cn, "ad": ad}
        for i in range(len(metadata)):
            rows.append(add_surface_row({
                "cohort": "current_matched", "method": method, "method_label": spec["label"],
                "generator_kind": "cocycle_diagonal_small_step", "subject_id": subjects[i],
                "scan_id": scan_ids[i], "diagnosis": diagnoses[i], "label_ad": int(labels[i]),
                "age_years": float(ages[i]), "observed_gt_method": schemes[i],
                "observed_gt_gap_years": float(gaps[i]), "epsilon_years": float(args.epsilon_years),
            }, observed[i], observed_condition[i], cn[i], ad[i], areas[i], volumes[i]))
        del geometry, flow, z
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base.unload_local_modules(script128)
    task_inr = resolve(registry["current_inr_task"])
    script_inr, common_inr, models_inr = base.load_local_modules(task_inr)
    inr_geometry = importlib.import_module("inr_geometry")
    spec = registry["current_models"]["inr"]
    run = resolve(spec["run_dir"])
    config = read_json(run / "resolved_config.json")["config"]
    checkpoint = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    train_inr, archive_inr = common_inr.load_archive("train"), common_inr.load_archive("test")
    geometry = inr_geometry.build_geometry(train_inr, device)
    flow = base.load_direct_flow(models_inr, config, checkpoint, device)
    z = torch.from_numpy(archive_inr["visit_latent_standardized_256"].astype(np.float32)).to(device)
    age = torch.from_numpy(archive_inr["visit_age_norm_train"].astype(np.float32)).to(device)
    slope = base.age_slope_per_year(archive_inr)
    epsilon_norm = float(args.epsilon_years * slope)
    scan_to_raw = {scan: i for i, scan in enumerate(scan_ids)}
    cn: list[np.ndarray] = []
    ad: list[np.ndarray] = []
    for i, scan in enumerate(archive_inr["visit_scan_ids"].astype(str)):
        raw_index = scan_to_raw[scan]
        zi, ti = z[i:i + 1], age[i:i + 1]
        with torch.no_grad():
            cn_future = flow.transport(zi, ti, ti + epsilon_norm, torch.zeros_like(ti))
            ad_future = flow.transport(zi, ti, ti + epsilon_norm, torch.ones_like(ti))
        cn.append(current_inr_field(geometry, zi, cn_future, raw[raw_index], normals[raw_index], args.epsilon_years))
        ad.append(current_inr_field(geometry, zi, ad_future, raw[raw_index], normals[raw_index], args.epsilon_years))
        if (i + 1) % 10 == 0:
            print(f"current INR surface velocity {i + 1:03d}/{len(z):03d}", flush=True)
    reorder = [list(archive_inr["visit_scan_ids"].astype(str)).index(scan) for scan in scan_ids]
    cn = [cn[i] for i in reorder]
    ad = [ad[i] for i in reorder]
    observed_condition = [ad[i] if labels[i] else cn[i] for i in range(len(labels))]
    maps["inr"] = {"observed": observed_condition, "cn": cn, "ad": ad}
    for i in range(len(metadata)):
        rows.append(add_surface_row({
            "cohort": "current_matched", "method": "inr", "method_label": "INR",
            "generator_kind": "implicit_level_set_small_step", "subject_id": subjects[i],
            "scan_id": scan_ids[i], "diagnosis": diagnoses[i], "label_ad": int(labels[i]),
            "age_years": float(ages[i]), "observed_gt_method": schemes[i],
            "observed_gt_gap_years": float(gaps[i]), "epsilon_years": float(args.epsilon_years),
        }, observed[i], observed_condition[i], cn[i], ad[i], areas[i], volumes[i]))
    arrays = aggregate_maps(np.mean(raw, axis=0), faces, metadata, observed, maps)
    frame = pd.DataFrame(rows).sort_values(["method", "subject_id", "age_years"]).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **arrays)
    manifest_path.write_text(json.dumps({
        "status": "complete", "stage": "current", "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()), "scans": int(frame.scan_id.nunique()),
        "methods": sorted(frame.method_label.unique().tolist()), "epsilon_years": args.epsilon_years,
        "observed_label_contract": "Observed (GT estimate) is derived from longitudinal scans and is not directly measured instantaneous motion.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {csv_path}")


def decoder_values(decoder: torch.nn.Module, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    expanded = latent.reshape(1, -1).expand(len(points), -1)
    output = decoder(torch.cat((expanded, points), dim=1))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.reshape(len(points), -1)[:, 0]


def legacy_implicit_field(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    velocity: torch.Tensor,
    vertices: np.ndarray,
    normals: np.ndarray,
    distance_scale_mm: float,
    epsilon_years: float,
) -> np.ndarray:
    points = torch.from_numpy(vertices.astype(np.float32)).to(latent.device).requires_grad_(True)
    baseline = decoder_values(decoder, latent, points)
    gradient = torch.autograd.grad(baseline.sum(), points, create_graph=False)[0]
    with torch.no_grad():
        projected = points - baseline[:, None] * gradient / gradient.square().sum(dim=1, keepdim=True).clamp_min(1.0e-10)
    projected = projected.detach().requires_grad_(True)
    baseline = decoder_values(decoder, latent, projected)
    gradient = torch.autograd.grad(baseline.sum(), projected, create_graph=False)[0]
    with torch.no_grad():
        future = decoder_values(decoder, latent + float(epsilon_years) * velocity, projected)
        norm = torch.linalg.vector_norm(gradient, dim=1).clamp_min(1.0e-8)
        unit = gradient / norm[:, None]
        normal_tensor = torch.from_numpy(normals.astype(np.float32)).to(latent.device)
        alignment = torch.sum(unit * normal_tensor, dim=1)
        level_speed = -(future - baseline) / float(epsilon_years) / norm
        result = level_speed * alignment * float(distance_scale_mm)
    return result.detach().cpu().numpy().astype(np.float64)


def legacy_observed_fields(
    frame: pd.DataFrame,
    mesh_paths: dict[str, str],
    distance_scale_mm: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str], np.ndarray]:
    meshes = [trimesh.load(mesh_paths[str(scan)], force="mesh", process=False) for scan in frame.scan_id]
    vertices = np.stack([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes])
    faces = np.asarray(meshes[0].faces, dtype=np.int64)
    ages = frame.age_years.to_numpy(dtype=np.float64)
    counts = frame.groupby("subject_id", sort=False).size().to_numpy(dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    observed, normals, areas, volumes, schemes, gaps = observed_surface_fields(vertices, ages, offsets, faces)
    observed = [field * distance_scale_mm for field in observed]
    areas = [area * distance_scale_mm**2 for area in areas]
    volumes = volumes * distance_scale_mm**3
    return vertices, faces, observed, normals, areas, volumes, schemes, gaps


def build_legacy(args: argparse.Namespace, csv_path: Path, npz_path: Path, manifest_path: Path) -> None:
    legacy_scripts = REPO / "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts"
    sys.path.insert(0, str(legacy_scripts))
    import longitudinal_visual_notebook_support as legacy

    context = legacy.LongitudinalVisualizationContext(device=args.device)
    distance_scale = float(context.distance_scale_mm)
    pair_path = legacy.TABLE_DIR / "mesh_pair_metrics.csv"
    pairs = pd.read_csv(pair_path, low_memory=False)
    pairs = pairs[(pairs.dataset.astype(str) == "qc_large") & (pairs.split.astype(str) == "test")]
    mesh_paths: dict[str, str] = {}
    for scan, path in zip(pairs.source_scan_id.astype(str), pairs.source_mesh_path.astype(str)):
        mesh_paths[scan] = path
    for scan, path in zip(pairs.target_scan_id.astype(str), pairs.target_mesh_path.astype(str)):
        mesh_paths[scan] = path

    model_specs = (
        ("pca_cocycle", "PCA Cocycle", "pca", "cocycle_diagonal_generator"),
        ("inr_cocycle", "INR Cocycle", "qc_siren_drop_bad_min2", "cocycle_diagonal_generator"),
        ("latent_ode", "Latent ODE", "qc_siren_latent_ode", "ode_vector_field"),
        ("brainode", "BrainODE", "brainode", "ode_vector_field"),
    )
    rows: list[dict[str, Any]] = []
    maps: dict[str, dict[str, list[np.ndarray]]] = {}
    reference_metadata: pd.DataFrame | None = None
    reference_observed: list[np.ndarray] | None = None
    reference_vertices: np.ndarray | None = None
    reference_faces: np.ndarray | None = None

    for method, label, support_name, kind in model_specs:
        if support_name == "pca":
            support = context.pca_support
        elif support_name == "brainode":
            support = context.brainode_support
        else:
            support = context.flow_support(support_name)
        scan_frame, latents = support.scan_latent_table()
        mask = scan_frame.split.astype(str).eq("test").to_numpy()
        frame = scan_frame.loc[mask].copy().sort_values(["subject_id", "age_years", "visit_order", "scan_id"]).reset_index()
        latent = latents[frame["index"].to_numpy(dtype=np.int64)]
        frame = frame.drop(columns="index").reset_index(drop=True)
        vertices, faces, observed, normals, areas, volumes, schemes, gaps = legacy_observed_fields(frame, mesh_paths, distance_scale)
        age_norm = frame.age_norm.to_numpy(dtype=np.float32)
        observed_condition = frame.condition_observed.to_numpy(dtype=np.float32)
        cn_velocity = support.latent_velocity_batch(latent, age_norm, np.zeros(len(frame), dtype=np.float32))
        ad_velocity = support.latent_velocity_batch(latent, age_norm, np.ones(len(frame), dtype=np.float32))
        actual_velocity = support.latent_velocity_batch(latent, age_norm, observed_condition)
        fields: dict[str, list[np.ndarray]] = {"observed": [], "cn": [], "ad": []}
        if method in {"pca_cocycle", "brainode"}:
            components = support.components if method == "pca_cocycle" else support.predictor.components
            for mode, velocity in (("observed", actual_velocity), ("cn", cn_velocity), ("ad", ad_velocity)):
                coordinate = (velocity @ np.asarray(components)).reshape(len(frame), len(vertices[0]), 3) * distance_scale
                fields[mode] = [np.sum(coordinate[i] * normals[i], axis=1) for i in range(len(frame))]
        else:
            decoder = support.bundle.decoder
            device = support.bundle.device
            for i in range(len(frame)):
                zi = torch.from_numpy(latent[i:i + 1].astype(np.float32)).to(device)
                for mode, velocity in (("observed", actual_velocity), ("cn", cn_velocity), ("ad", ad_velocity)):
                    vi = torch.from_numpy(velocity[i:i + 1].astype(np.float32)).to(device)
                    fields[mode].append(legacy_implicit_field(
                        decoder, zi, vi, vertices[i], normals[i], distance_scale, args.epsilon_years
                    ))
                if (i + 1) % 20 == 0:
                    print(f"legacy {label} surface velocity {i + 1:03d}/{len(frame):03d}", flush=True)
        maps[method] = fields
        for i, item in frame.iterrows():
            rows.append(add_surface_row({
                "cohort": "legacy_reference", "method": method, "method_label": label,
                "generator_kind": kind, "subject_id": str(item.subject_id), "scan_id": str(item.scan_id),
                "diagnosis": str(item.diagnosis), "label_ad": int(item.label_ad),
                "age_years": float(item.age_years), "observed_gt_method": schemes[i],
                "observed_gt_gap_years": float(gaps[i]), "epsilon_years": float(args.epsilon_years),
            }, observed[i], fields["observed"][i], fields["cn"][i], fields["ad"][i], areas[i], volumes[i]))
        if reference_metadata is None:
            reference_metadata = frame
            reference_observed = observed
            reference_vertices = vertices
            reference_faces = faces
        if support_name == "pca":
            context._pca_support = None
        elif support_name == "brainode":
            context._brainode_support = None
        else:
            context._flow_support.pop(support_name, None)
        del support
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert reference_metadata is not None and reference_observed is not None
    arrays = aggregate_maps(np.mean(reference_vertices, axis=0) * distance_scale, reference_faces, reference_metadata, reference_observed, maps)
    frame = pd.DataFrame(rows).sort_values(["method", "subject_id", "age_years"]).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **arrays)
    manifest_path.write_text(json.dumps({
        "status": "complete", "stage": "legacy", "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()), "scans": int(frame.scan_id.nunique()),
        "methods": sorted(frame.method_label.unique().tolist()), "epsilon_years": args.epsilon_years,
        "distance_scale_mm": distance_scale,
        "observed_label_contract": "Observed (GT estimate) is derived from longitudinal scans and is not directly measured instantaneous motion.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {csv_path}")


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.epsilon_years) or args.epsilon_years <= 0.0:
        raise ValueError("epsilon-years must be positive")
    csv_path, npz_path, manifest_path = prepare_output(args.output_dir, args.stage, args.force)
    if args.stage == "current":
        build_current(args, csv_path, npz_path, manifest_path)
    else:
        build_legacy(args, csv_path, npz_path, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
