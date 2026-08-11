#!/usr/bin/env python3
"""Validation/test subject-macro evaluator for direct SIREN-256 transports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from siren256_common import decode_sdf, ensure_prepared, load_basis, load_cache, load_config, load_pairs, load_sequences, read_json, root_dir, write_json
from siren256_decoder_geometry import C3GeometryLoss
from siren256_transport_models import build_transport


def surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[:, faces]
    return 0.5 * torch.linalg.vector_norm(torch.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0], dim=-1), dim=-1).sum(dim=1)


def nearest_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("SciPy is required for Chamfer/ASSD/HD95 evaluation.") from error
    forward = cKDTree(target).query(predicted, k=1)[0]
    backward = cKDTree(predicted).query(target, k=1)[0]
    distances = np.concatenate((forward, backward))
    return float((np.mean(forward**2) + np.mean(backward**2)) / 2.0), float(distances.mean()), float(np.percentile(distances, 95.0))


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return abs(float(np.einsum("fi,fi->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6.0))


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum())


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    triangles = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area).all() or float(area.sum()) <= 0.0:
        raise ValueError("Cannot sample a zero-area decoded surface.")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=area / area.sum())]
    first, second = rng.random(count), rng.random(count)
    root = np.sqrt(first)
    return (1.0 - root)[:, None] * chosen[:, 0] + (root * (1.0 - second))[:, None] * chosen[:, 1] + (root * second)[:, None] * chosen[:, 2]


def decode_zero_level_mesh(decoder: torch.nn.Module, latent: torch.Tensor, resolution: int, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError("scikit-image is required for decoded SIREN mesh evaluation.") from error
    axis = torch.linspace(lower, upper, resolution, device=latent.device, dtype=latent.dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 3)
    field = decode_sdf(decoder, latent, grid).reshape(resolution, resolution, resolution).cpu().numpy()
    if float(field.min()) > 0.0 or float(field.max()) < 0.0:
        raise RuntimeError("Predicted SIREN field has no zero-level surface inside the evaluation cube.")
    spacing = (upper - lower) / float(resolution - 1)
    vertices, faces, _, _ = marching_cubes(field, level=0.0, spacing=(spacing, spacing, spacing))
    return vertices.astype(np.float32) + lower, faces.astype(np.int64)


def subject_slope_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, pairs in table.groupby("subject_id", sort=True):
        anchored = pairs.loc[pairs.source_visit_order == 0].sort_values("target_time")
        if anchored.empty:
            continue
        base = anchored.iloc[0]
        predicted = pd.concat((pd.DataFrame({"time": [base.source_time], "volume": [base.source_decoded_volume]}), anchored[["target_time", "predicted_decoded_volume"]].rename(columns={"target_time": "time", "predicted_decoded_volume": "volume"})), ignore_index=True).drop_duplicates("time")
        observed = pd.concat((pd.DataFrame({"time": [base.source_time], "volume": [base.source_volume]}), anchored[["target_time", "target_volume"]].rename(columns={"target_time": "time", "target_volume": "volume"})), ignore_index=True).drop_duplicates("time")
        if len(predicted) < 2:
            continue
        predicted_slope = float(np.polyfit(predicted.time, np.log(predicted.volume.clip(lower=1.0e-8)), 1)[0])
        observed_slope = float(np.polyfit(observed.time, np.log(observed.volume.clip(lower=1.0e-8)), 1)[0])
        rows.append({"subject_id": subject_id, "diagnosis": str(base.diagnosis), "predicted_log_volume_slope": predicted_slope, "observed_log_volume_slope": observed_slope, "subject_log_volume_slope_mae": abs(predicted_slope - observed_slope)})
    return pd.DataFrame(rows)


def test_rate_ordering(table: pd.DataFrame) -> dict[str, Any]:
    first_last = table.loc[table.is_first_last]
    subject = first_last.groupby(["subject_id", "diagnosis"])[["predicted_log_volume_rate", "target_log_volume_rate"]].mean().reset_index()
    by_diagnosis = subject.groupby("diagnosis")[["predicted_log_volume_rate", "target_log_volume_rate"]].mean().to_dict(orient="index")
    result: dict[str, Any] = {"subjects": int(len(subject)), "by_diagnosis_subject_macro": by_diagnosis}
    if set(by_diagnosis) == {"CN", "AD"}:
        observed = bool(by_diagnosis["AD"]["target_log_volume_rate"] < by_diagnosis["CN"]["target_log_volume_rate"])
        predicted = bool(by_diagnosis["AD"]["predicted_log_volume_rate"] < by_diagnosis["CN"]["predicted_log_volume_rate"])
        result.update({"observed_ad_rate_is_more_negative_than_cn": observed, "predicted_ad_rate_is_more_negative_than_cn": predicted, "predicted_matches_observed_order": observed == predicted})
    return result


def summarize_subject_macro(table: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    if table.empty:
        return {name: float("nan") for name in metrics}
    subject = table.groupby("subject_id", sort=True)[metrics].mean(numeric_only=True)
    return {name: float(subject[name].mean()) for name in metrics}


def output_panels(table: pd.DataFrame, metrics: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"all_subject_macro": summarize_subject_macro(table, metrics)}
    result["first_last_subject_macro"] = summarize_subject_macro(table.loc[table.is_first_last], metrics)
    result["by_diagnosis_subject_macro"] = {diagnosis: summarize_subject_macro(part, metrics) for diagnosis, part in table.groupby("diagnosis", sort=True)}
    result["by_pair_type_subject_macro"] = {("adjacent" if adjacent else "non_adjacent"): summarize_subject_macro(part, metrics) for adjacent, part in table.groupby("is_adjacent", sort=True)}
    bins = pd.cut(table.gap_years.abs(), bins=[-np.inf, 1.0, 2.0, np.inf], labels=["under_1y", "1_to_2y", "over_2y"])
    result["by_gap_subject_macro"] = {str(label): summarize_subject_macro(table.loc[bins == label], metrics) for label in bins.cat.categories}
    return result


@torch.no_grad()
def sequence_rollout_metrics(model: torch.nn.Module, cache: dict[str, np.ndarray], sequences: list[dict[str, Any]], criterion: C3GeometryLoss, device: torch.device) -> pd.DataFrame:
    rows = []
    for sequence in sequences:
        indices = np.asarray(sequence["cache_indices"], dtype=int)
        if len(indices) < 3:
            continue
        latents = torch.from_numpy(np.array(cache["latents"][indices], copy=True)).to(device)
        times = torch.from_numpy(np.asarray(sequence["times"], dtype=np.float32)).to(device)
        condition = torch.tensor([[float(sequence["label_ad"])]], device=device)
        source, current = latents[:1], latents[:1]
        source_vertices, normals, target_vertices = (torch.from_numpy(np.array(cache[key][index : index + 1], copy=True)).to(device) for key, index in (("vertices", indices[0]), ("normals", indices[0]), ("vertices", indices[-1])))
        for index in range(1, len(indices)):
            current = model.transport(current, times[index - 1 : index], times[index : index + 1], condition)
        direct = model.transport(source, times[:1], times[-1:], condition)
        rollout_displacement, _, _ = criterion.normal_proxy(source, current, source_vertices, normals)
        direct_displacement, _, _ = criterion.normal_proxy(source, direct, source_vertices, normals)
        target_displacement = ((target_vertices - source_vertices) * normals).sum(dim=-1)
        rows.append({"subject_id": sequence["subject_id"], "diagnosis": "AD" if sequence["label_ad"] else "CN", "direct_normal_mae": float((direct_displacement - target_displacement).abs().mean().cpu()), "rollout_normal_mae": float((rollout_displacement - target_displacement).abs().mean().cpu())})
    return pd.DataFrame(rows)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--tag", default=None, help="Optional safe subdirectory tag for comparing validation checkpoints.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    root, run = root_dir(), args.run.resolve()
    ensure_prepared(root)
    config = read_json(run / "config.json")
    if args.split == "test" and bool(config.get("test_data_loaded_during_training", False)):
        raise RuntimeError("Checkpoint violates test-isolation contract.")
    checkpoint = run / "checkpoints" / f"{args.checkpoint}.pt"
    if not checkpoint.exists() and args.checkpoint == "best":
        checkpoint = run / "checkpoints" / "best_candidate.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.tag is not None and Path(args.tag).name != args.tag:
        raise ValueError("Evaluation tag must be one path component.")
    device = torch.device(args.device)
    cache, basis = load_cache(config), load_basis(root)
    from siren256_common import load_frozen_decoder
    decoder = load_frozen_decoder(config, device)
    model = build_transport(config, basis).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["flow_state_dict"], strict=True)
    model.eval()
    faces = torch.from_numpy(np.array(cache["faces"], copy=True)).to(device)
    criterion = C3GeometryLoss(decoder, faces, read_json(root / "metadata" / "loss_scales.json"), config["LossWeights"], float(config["AgeRangeYears"]), options=config).to(device)
    pairs = load_pairs(args.split, root=root)
    if args.max_pairs is not None:
        pairs = pairs.iloc[: args.max_pairs].reset_index(drop=True)
    resolution = int(config["EvaluationGridResolution"])
    lower, upper = (float(value) for value in config["EvaluationBounds"])
    surface_samples = int(config["EvaluationSurfaceSamples"])
    face_array = np.asarray(cache["faces"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    future_rates: list[dict[str, Any]] = []
    decoded_source_cache: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for row in pairs.itertuples(index=False):
        source_index, target_index = int(row.source_cache_index), int(row.target_cache_index)
        source = torch.from_numpy(np.array(cache["latents"][source_index : source_index + 1], copy=True)).to(device)
        source_time = torch.tensor([[float(row.source_time)]], device=device)
        target_time = torch.tensor([[float(row.target_time)]], device=device)
        condition = torch.tensor([[float(row.label_ad)]], device=device)
        source_vertices = torch.from_numpy(np.array(cache["vertices"][source_index : source_index + 1], copy=True)).to(device)
        target_vertices = torch.from_numpy(np.array(cache["vertices"][target_index : target_index + 1], copy=True)).to(device)
        normals = torch.from_numpy(np.array(cache["normals"][source_index : source_index + 1], copy=True)).to(device)
        prediction = model.transport(source, source_time, target_time, condition)
        displacement, proxy, proxy_volume = criterion.normal_proxy(source, prediction, source_vertices, normals)
        target_displacement = ((target_vertices - source_vertices) * normals).sum(dim=-1)
        normal_mae = float((displacement - target_displacement).abs().mean().cpu())
        nochange_normal = float(target_displacement.abs().mean().cpu())
        decoded_vertices, decoded_faces = decode_zero_level_mesh(decoder, prediction, resolution, lower, upper)
        if source_index not in decoded_source_cache:
            source_decoded_vertices, source_decoded_faces = decode_zero_level_mesh(decoder, source, resolution, lower, upper)
            decoded_source_cache[source_index] = (source_decoded_vertices, source_decoded_faces, mesh_volume(source_decoded_vertices, source_decoded_faces), mesh_area(source_decoded_vertices, source_decoded_faces))
        _, _, source_decoded_volume, source_decoded_area = decoded_source_cache[source_index]
        target_vertices_array = target_vertices[0].cpu().numpy()
        seed = source_index * 104729 + target_index
        chamfer, assd, hd95 = nearest_metrics(sample_mesh_surface(decoded_vertices, decoded_faces, surface_samples, seed), sample_mesh_surface(target_vertices_array, face_array, surface_samples, seed + 1))
        target_volume, source_volume = mesh_volume(target_vertices_array, face_array), mesh_volume(source_vertices[0].cpu().numpy(), face_array)
        predicted_volume_value = mesh_volume(decoded_vertices, decoded_faces)
        gap = max(abs(float(row.gap_years)), 0.05)
        target_rate = np.log(max(target_volume, 1e-8) / max(source_volume, 1e-8)) / gap
        predicted_rate = np.log(max(predicted_volume_value, 1e-8) / max(source_decoded_volume, 1e-8)) / gap
        predicted_rate_legacy = np.log(max(predicted_volume_value, 1e-8) / max(source_volume, 1e-8)) / gap
        decoded_area, target_area = mesh_area(decoded_vertices, decoded_faces), mesh_area(target_vertices_array, face_array)
        inverse = model.transport(prediction, target_time, source_time, condition)
        virtual_time = 0.5 * (source_time + target_time)
        virtual = model.transport(model.transport(source, source_time, virtual_time, condition), virtual_time, target_time, condition)
        observed_defect = float("nan")
        if int(row.observed_cache_index) >= 0:
            middle = torch.tensor([[float(row.observed_time)]], device=device)
            observed_defect = float(torch.mean((prediction - model.transport(model.transport(source, source_time, middle, condition), middle, target_time, condition)) ** 2).cpu())
        predicted_hot = set(torch.topk(displacement[0].abs(), max(1, displacement.shape[1] // 10)).indices.cpu().tolist())
        target_hot = set(torch.topk(target_displacement[0].abs(), max(1, target_displacement.shape[1] // 10)).indices.cpu().tolist())
        rows.append({"subject_id": str(row.subject_id), "diagnosis": str(row.diagnosis), "source_scan_id": str(row.source_scan_id), "target_scan_id": str(row.target_scan_id), "source_visit_order": int(row.source_visit_order), "target_visit_order": int(row.target_visit_order), "source_time": float(row.source_time), "target_time": float(row.target_time), "is_adjacent": bool(row.is_adjacent), "is_first_last": bool(row.is_first_last), "gap_years": float(row.gap_years), "source_volume": source_volume, "source_decoded_volume": source_decoded_volume, "target_volume": target_volume, "predicted_decoded_volume": predicted_volume_value, "source_decoder_volume_relative_error": abs(source_decoded_volume - source_volume) / max(source_volume, 1e-8), "registered_normal_mae": normal_mae, "registered_proxy_vertex_mae": float(torch.linalg.vector_norm(proxy - target_vertices, dim=-1).mean().cpu()), "chamfer_l2_squared": chamfer, "assd": assd, "hd95": hd95, "volume_relative_error": abs(predicted_volume_value - target_volume) / max(target_volume, 1e-8), "target_log_volume_rate": target_rate, "predicted_log_volume_rate": predicted_rate, "predicted_log_volume_rate_legacy": predicted_rate_legacy, "annual_log_volume_rate_mae": abs(predicted_rate - target_rate), "surface_area_relative_error": abs(decoded_area - target_area) / max(target_area, 1e-8), "semigroup_defect": float(torch.mean((prediction - virtual) ** 2).cpu()), "observed_semigroup_defect": observed_defect, "inverse_defect": float(torch.mean((inverse - source) ** 2).cpu()), "nochange_improvement_ratio": (nochange_normal - normal_mae) / max(nochange_normal, 1e-8), "hotspot_overlap": len(predicted_hot & target_hot) / len(predicted_hot | target_hot)})
        if bool(row.is_first_last):
            future_time = target_time + (target_time - source_time)
            future_latent = model.transport(source, source_time, future_time, condition)
            future_vertices, future_faces = decode_zero_level_mesh(decoder, future_latent, resolution, lower, upper)
            future_volume = mesh_volume(future_vertices, future_faces)
            future_rates.append({"diagnosis": str(row.diagnosis), "annual_log_volume_rate": float(np.log(max(future_volume, 1.0e-8) / max(source_decoded_volume, 1.0e-8)) / max(2.0 * gap, 0.05)), "volume_estimator": "decoded_SIREN_zero_level_mesh_temporal"})
    table = pd.DataFrame(rows)
    primary = ["registered_normal_mae", "chamfer_l2_squared", "assd", "volume_relative_error", "annual_log_volume_rate_mae"]
    secondary = ["hd95", "surface_area_relative_error", "semigroup_defect", "inverse_defect", "nochange_improvement_ratio", "hotspot_overlap", "registered_proxy_vertex_mae"]
    panels = output_panels(table, primary + secondary)
    sequence = sequence_rollout_metrics(model, cache, load_sequences(args.split, root), criterion, device)
    slope = subject_slope_metrics(table)
    if sequence.empty:
        rollout_summary: dict[str, Any] = {"subjects": 0}
    else:
        subject_sequence = sequence.groupby("subject_id")[["direct_normal_mae", "rollout_normal_mae"]].mean()
        rollout_summary = {"subjects": int(len(subject_sequence)), "direct_normal_mae": float(subject_sequence.direct_normal_mae.mean()), "rollout_normal_mae": float(subject_sequence.rollout_normal_mae.mean()), "rollout_minus_direct": float((subject_sequence.rollout_normal_mae - subject_sequence.direct_normal_mae).mean())}
    future = pd.DataFrame(future_rates)
    trend = {"first_last_subjects": int(len(future)), "by_diagnosis": future.groupby("diagnosis").annual_log_volume_rate.mean().to_dict() if not future.empty else {}}
    if set(trend["by_diagnosis"]) == {"CN", "AD"}:
        trend["ad_rate_is_more_negative_than_cn"] = bool(trend["by_diagnosis"]["AD"] < trend["by_diagnosis"]["CN"])
    slope_summary = {"subjects": int(len(slope)), "subject_log_volume_slope_mae": float(slope.subject_log_volume_slope_mae.mean()) if not slope.empty else float("nan"), "by_diagnosis_subject_macro": slope.groupby("diagnosis").subject_log_volume_slope_mae.mean().to_dict() if not slope.empty else {}}
    ordering = test_rate_ordering(table)
    evaluation = run / "evaluation" / args.split
    if args.tag:
        evaluation = evaluation / args.tag
    evaluation.mkdir(parents=True, exist_ok=True)
    table.to_csv(evaluation / "per_pair_metrics.csv", index=False)
    sequence.to_csv(evaluation / "sequence_rollout_metrics.csv", index=False)
    slope.to_csv(evaluation / "subject_slope_metrics.csv", index=False)
    summary = {"run": str(run), "checkpoint": str(checkpoint), "checkpoint_tag": args.tag, "split": args.split, "pair_count": int(len(table)), "subject_macro": panels, "subject_log_volume_slope": slope_summary, "test_rate_ordering": ordering, "direct_vs_rollout": rollout_summary, "ood_future_trends": trend, "primary_metrics": primary, "volume_rate_definition": "log(decoded predicted volume / decoded source volume) per year; target remains registered target/source volume", "evaluation_surface": {"method": "frozen_SIREN_zero_level_marching_cubes", "grid_resolution": resolution, "bounds": [lower, upper], "samples_per_surface": surface_samples}, "attention_contract": getattr(model, "attention_contract", "not_applicable")}
    write_json(evaluation / "summary.json", summary)
    print(json_dumps_compact({"split": args.split, "pairs": len(table), **panels["all_subject_macro"]}))
    return 0


def json_dumps_compact(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
