#!/usr/bin/env python3
"""Surface-space endpoint, cocycle, anatomy, and velocity objectives."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from conditional_spiral_unet import ConditionalSpiralUNet, mesh_volume, vertex_normals
from data import PreparedSplit


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def scaled_vertex_loss(prediction: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    coordinate = F.smooth_l1_loss(prediction / scale, target / scale)
    euclidean = torch.linalg.vector_norm(prediction - target, dim=-1).mean() / scale
    return 0.5 * (coordinate + euclidean)


def normal_displacement_loss(
    prediction: torch.Tensor, target: torch.Tensor, source: torch.Tensor, faces: torch.Tensor, scale: float
) -> torch.Tensor:
    normals = vertex_normals(source, faces)
    predicted = torch.sum((prediction - source) * normals, dim=-1)
    observed = torch.sum((target - source) * normals, dim=-1)
    return F.smooth_l1_loss(predicted / scale, observed / scale)


def composition_mse(left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.mean(((left - right) / scale) ** 2)


def flipped_face_fraction(
    source: torch.Tensor,
    prediction: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Return a per-mesh fraction of reversed or degenerate predicted faces."""
    source_tri = source[:, faces]
    predicted_tri = prediction[:, faces]
    source_cross = torch.cross(
        source_tri[:, :, 1] - source_tri[:, :, 0],
        source_tri[:, :, 2] - source_tri[:, :, 0],
        dim=-1,
    )
    predicted_cross = torch.cross(
        predicted_tri[:, :, 1] - predicted_tri[:, :, 0],
        predicted_tri[:, :, 2] - predicted_tri[:, :, 0],
        dim=-1,
    )
    orientation = torch.sum(source_cross * predicted_cross, dim=-1)
    predicted_area = torch.linalg.vector_norm(predicted_cross, dim=-1)
    invalid = torch.logical_or(orientation <= 0.0, predicted_area <= 1.0e-12)
    return invalid.float().mean(dim=1)


def mesh_regularizers(
    source: torch.Tensor,
    prediction: torch.Tensor,
    velocity: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_edges = source[:, edges[:, 0]] - source[:, edges[:, 1]]
    predicted_edges = prediction[:, edges[:, 0]] - prediction[:, edges[:, 1]]
    source_length = torch.linalg.vector_norm(source_edges, dim=-1).clamp_min(1.0e-6)
    predicted_length = torch.linalg.vector_norm(predicted_edges, dim=-1)
    edge = F.smooth_l1_loss(predicted_length / source_length, torch.ones_like(predicted_length))
    velocity_edge = velocity[:, edges[:, 0]] - velocity[:, edges[:, 1]]
    smoothness = torch.mean(velocity_edge.square())
    normals = vertex_normals(source, faces)
    normal_component = torch.sum(velocity * normals, dim=-1, keepdim=True) * normals
    tangent = torch.mean((velocity - normal_component).square())
    source_tri = source[:, faces]
    predicted_tri = prediction[:, faces]
    source_cross = torch.cross(source_tri[:, :, 1] - source_tri[:, :, 0], source_tri[:, :, 2] - source_tri[:, :, 0], dim=-1)
    predicted_cross = torch.cross(
        predicted_tri[:, :, 1] - predicted_tri[:, :, 0],
        predicted_tri[:, :, 2] - predicted_tri[:, :, 0],
        dim=-1,
    )
    orientation = torch.sum(F.normalize(source_cross, dim=-1) * F.normalize(predicted_cross, dim=-1), dim=-1)
    flip = F.relu(0.1 - orientation).mean()
    return {"edge": edge, "smoothness": smoothness, "tangent": tangent, "flip": flip}


def pair_objective(
    model: ConditionalSpiralUNet,
    split: PreparedSplit,
    batch: dict[str, torch.Tensor],
    statistics: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
) -> dict[str, torch.Tensor]:
    endpoint_scale = float(statistics["endpoint_scale_mm"])
    velocity_scale = float(statistics["velocity_reference_scale_mm_per_year"])
    faces = model.faces
    edges = unique_edges(faces)
    source, target = batch["source"], batch["target"]
    source_age, target_age, label = batch["source_age"], batch["target_age"], batch["label"]
    forward_velocity = model.average_velocity(source, source_age, target_age, label)
    backward_velocity = model.average_velocity(target, target_age, source_age, label)
    elapsed = (target_age - source_age).reshape(-1, 1, 1)
    forward = source + elapsed * forward_velocity
    backward = target - elapsed * backward_velocity
    endpoint = 0.5 * (
        scaled_vertex_loss(forward, target, endpoint_scale)
        + scaled_vertex_loss(backward, source, endpoint_scale)
    )
    normal_endpoint = 0.5 * (
        normal_displacement_loss(forward, target, source, faces, endpoint_scale)
        + normal_displacement_loss(backward, source, target, faces, endpoint_scale)
    )

    virtual_ratio = torch.empty_like(source_age).uniform_(0.15, 0.85)
    virtual_age = source_age + virtual_ratio * (target_age - source_age)
    direct_forward = forward
    first_leg = model.transport(source, source_age, virtual_age, label)
    composed_forward = model.transport(first_leg, virtual_age, target_age, label)
    first_back = model.transport(target, target_age, virtual_age, label)
    composed_back = model.transport(first_back, virtual_age, source_age, label)
    virtual_cocycle = 0.5 * (
        composition_mse(direct_forward, composed_forward, endpoint_scale)
        + composition_mse(backward, composed_back, endpoint_scale)
    )

    observed_cocycle = source.sum() * 0.0
    valid = batch["intermediate_index"] >= 0
    if bool(valid.any()):
        middle_index = batch["intermediate_index"][valid]
        middle_age = split.ages[middle_index]
        observed_first = model.transport(source[valid], source_age[valid], middle_age, label[valid])
        observed_composed = model.transport(observed_first, middle_age, target_age[valid], label[valid])
        observed_cocycle = composition_mse(forward[valid], observed_composed, endpoint_scale)

    inverse_forward = model.transport(forward, target_age, source_age, label)
    inverse_backward = model.transport(backward, source_age, target_age, label)
    inverse = 0.5 * (
        composition_mse(inverse_forward, source, endpoint_scale)
        + composition_mse(inverse_backward, target, endpoint_scale)
    )

    weights = config["loss"]
    local_cocycle = source.sum() * 0.0
    if float(weights.get("local_cocycle_weight", 0.0)) > 0.0:
        epsilon = torch.minimum(torch.full_like(source_age, 0.25), 0.1 * (target_age - source_age).abs())
        epsilon = epsilon.clamp_min(0.02)
        age_one = source_age + epsilon
        age_two = source_age + 2.0 * epsilon
        local_direct = model.transport(source, source_age, age_two, label)
        local_first = model.transport(source, source_age, age_one, label)
        local_composed = model.transport(local_first, age_one, age_two, label)
        local_cocycle = composition_mse(local_direct, local_composed, endpoint_scale)

    diagonal_source = model.instantaneous_velocity(source, source_age, label)
    diagonal_target = model.instantaneous_velocity(target, target_age, label)
    source_reliability = batch["velocity_reference_weight"].reshape(-1, 1, 1)
    target_reliability = batch["target_velocity_reference_weight"].reshape(-1, 1, 1)
    reliability_sum = source_reliability.mean() + target_reliability.mean()
    if float(reliability_sum.detach()) > 0.0:
        diagonal_velocity = (
            (F.smooth_l1_loss(
                diagonal_source / velocity_scale,
                batch["velocity_reference"] / velocity_scale,
                reduction="none",
            ) * source_reliability).mean()
            + (F.smooth_l1_loss(
                diagonal_target / velocity_scale,
                batch["target_velocity_reference"] / velocity_scale,
                reduction="none",
            ) * target_reliability).mean()
        ) / reliability_sum.clamp_min(1.0e-6)
    else:
        diagonal_velocity = source.sum() * 0.0

    predicted_forward_volume = mesh_volume(forward, faces)
    predicted_backward_volume = mesh_volume(backward, faces)
    volume = 0.5 * (
        F.smooth_l1_loss(torch.log(predicted_forward_volume), torch.log(batch["target_volume"]))
        + F.smooth_l1_loss(torch.log(predicted_backward_volume), torch.log(batch["source_volume"]))
    )
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    observed_rate = (torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])) / years
    predicted_rate = (torch.log(predicted_forward_volume) - torch.log(batch["source_volume"])) / years
    rate = F.smooth_l1_loss(predicted_rate, observed_rate)
    group_rate = source.sum() * 0.0
    disease_gap = source.sum() * 0.0
    is_ad = label >= 0.5
    if bool(is_ad.any()) and bool((~is_ad).any()):
        targets = statistics["group_log_volume_rate_targets"]
        group_rate = 0.5 * (
            F.smooth_l1_loss(predicted_rate[~is_ad].mean(), torch.as_tensor(targets["CN"], device=source.device))
            + F.smooth_l1_loss(predicted_rate[is_ad].mean(), torch.as_tensor(targets["AD"], device=source.device))
        )
        predicted_gap = predicted_rate[is_ad].mean() - predicted_rate[~is_ad].mean()
        disease_gap = F.smooth_l1_loss(
            predicted_gap,
            torch.as_tensor(statistics["ad_minus_cn_log_volume_rate_target"], device=source.device),
        )
    regular = mesh_regularizers(source, forward, forward_velocity, faces, edges)

    consistency_ramp = min(1.0, float(epoch) / max(int(config["training"]["consistency_ramp_epochs"]), 1))
    anatomy_ramp = min(1.0, float(epoch) / max(int(config["training"]["anatomy_ramp_epochs"]), 1))
    total = (
        float(weights["endpoint_weight"]) * endpoint
        + float(weights["normal_endpoint_weight"]) * normal_endpoint
        + consistency_ramp
        * (
            float(weights["observed_cocycle_weight"]) * observed_cocycle
            + float(weights["virtual_cocycle_weight"]) * virtual_cocycle
            + float(weights["local_cocycle_weight"]) * local_cocycle
            + float(weights["inverse_weight"]) * inverse
            + float(weights["diagonal_velocity_weight"]) * diagonal_velocity
        )
        + anatomy_ramp
        * (
            float(weights["volume_weight"]) * volume
            + float(weights["rate_weight"]) * rate
            + float(weights["group_rate_weight"]) * group_rate
            + float(weights["disease_gap_weight"]) * disease_gap
            + float(weights["edge_weight"]) * regular["edge"]
            + float(weights["smoothness_weight"]) * regular["smoothness"]
            + float(weights["tangent_weight"]) * regular["tangent"]
            + float(weights["flip_weight"]) * regular["flip"]
        )
    )
    return {
        "total": total,
        "endpoint": endpoint,
        "normal_endpoint": normal_endpoint,
        "observed_cocycle": observed_cocycle,
        "virtual_cocycle": virtual_cocycle,
        "local_cocycle": local_cocycle,
        "inverse": inverse,
        "diagonal_velocity": diagonal_velocity,
        "volume": volume,
        "rate": rate,
        "group_rate": group_rate,
        "disease_gap": disease_gap,
        **regular,
    }


def first_last_rows(split: PreparedSplit) -> list[C.PairRow]:
    rows = []
    offsets = split.subject_visit_offsets
    for subject_index in range(len(offsets) - 1):
        first, last_exclusive = int(offsets[subject_index]), int(offsets[subject_index + 1])
        last = last_exclusive - 1
        if last <= first:
            continue
        rows.append(
            C.PairRow(
                first,
                last,
                -1,
                str(split.subject_ids[first]),
                str(split.diagnoses[first]),
                "nonadjacent" if last > first + 1 else "adjacent",
                float(split.years_from_baseline[last] - split.years_from_baseline[first]),
            )
        )
    return rows


@torch.no_grad()
def evaluate_rows(
    model: ConditionalSpiralUNet,
    split: PreparedSplit,
    rows: list[C.PairRow],
    batch_size: int,
    max_pairs: int | None = None,
    compute_flips: bool = False,
) -> dict[str, Any]:
    model.eval()
    if max_pairs is None or int(max_pairs) >= len(rows):
        selected = rows
    else:
        grouped = {diagnosis: [row for row in rows if row.diagnosis == diagnosis] for diagnosis in ("CN", "AD")}
        selected = []
        for index in range(int(max_pairs)):
            diagnosis = ("CN", "AD")[index % 2]
            selected.append(grouped[diagnosis][(index // 2) % len(grouped[diagnosis])])
    records: list[dict[str, Any]] = []
    for chunk in C.chunked(selected, batch_size):
        batch = split.pair_batch(chunk)
        prediction = model.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        predicted_volume = mesh_volume(prediction, model.faces)
        error = torch.linalg.vector_norm(prediction - batch["target"], dim=-1)
        baseline = torch.linalg.vector_norm(batch["source"] - batch["target"], dim=-1)
        flipped = flipped_face_fraction(batch["source"], prediction, model.faces) if compute_flips else None
        for index, row in enumerate(chunk):
            elapsed = max(float(row.delta_years), 1.0e-6)
            source_volume = batch["source_volume"][index].clamp_min(1.0e-8)
            target_volume = batch["target_volume"][index].clamp_min(1.0e-8)
            predicted_volume_item = predicted_volume[index].clamp_min(1.0e-8)
            observed_rate = (torch.log(target_volume) - torch.log(source_volume)) / elapsed
            predicted_rate = (torch.log(predicted_volume_item) - torch.log(source_volume)) / elapsed
            record = {
                    "source_index": row.source,
                    "target_index": row.target,
                    "source_scan_id": str(split.scan_ids[row.source]),
                    "target_scan_id": str(split.scan_ids[row.target]),
                    "subject": row.subject,
                    "diagnosis": row.diagnosis,
                    "pair_type": row.pair_type,
                    "delta_years": row.delta_years,
                    "source_age_years": float(batch["source_age"][index].cpu()),
                    "target_age_years": float(batch["target_age"][index].cpu()),
                    "mean_vertex_error_mm": float(error[index].mean().cpu()),
                    "vertex_rmse_mm": float(torch.sqrt(torch.mean(error[index] ** 2)).cpu()),
                    "nochange_mean_vertex_error_mm": float(baseline[index].mean().cpu()),
                    "source_volume_mm3": float(source_volume.cpu()),
                    "observed_target_volume_mm3": float(target_volume.cpu()),
                    "predicted_target_volume_mm3": float(predicted_volume_item.cpu()),
                    "observed_log_volume_rate_per_year": float(observed_rate.cpu()),
                    "predicted_log_volume_rate_per_year": float(predicted_rate.cpu()),
                    "volume_rate_abs_error_per_year": float(torch.abs(predicted_rate - observed_rate).cpu()),
                    "volume_relative_error": float(
                        (torch.abs(predicted_volume_item - target_volume) / target_volume).cpu()
                    ),
                }
            if flipped is not None:
                record["flipped_face_fraction"] = float(flipped[index].cpu())
            records.append(record)
    groups: dict[str, Any] = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = records if diagnosis == "overall" else [row for row in records if row["diagnosis"] == diagnosis]
        if not current:
            continue
        predicted = np.asarray([row["mean_vertex_error_mm"] for row in current])
        baseline = np.asarray([row["nochange_mean_vertex_error_mm"] for row in current])
        summary = {
            "pairs": len(current),
            "mean_vertex_error_mm": float(predicted.mean()),
            "vertex_rmse_mm": float(np.mean([row["vertex_rmse_mm"] for row in current])),
            "nochange_mean_vertex_error_mm": float(baseline.mean()),
            "error_to_nochange_ratio": float(predicted.mean() / max(baseline.mean(), 1.0e-8)),
            "volume_relative_error": float(np.mean([row["volume_relative_error"] for row in current])),
            "observed_log_volume_rate_per_year": float(
                np.mean([row["observed_log_volume_rate_per_year"] for row in current])
            ),
            "predicted_log_volume_rate_per_year": float(
                np.mean([row["predicted_log_volume_rate_per_year"] for row in current])
            ),
            "volume_rate_abs_error_per_year": float(
                np.mean([row["volume_rate_abs_error_per_year"] for row in current])
            ),
        }
        if compute_flips:
            summary["flipped_face_fraction"] = float(np.mean([row["flipped_face_fraction"] for row in current]))
        groups[diagnosis] = summary
    return {"groups": groups, "records": records}


@torch.no_grad()
def evaluate_cocycle(
    model: ConditionalSpiralUNet,
    split: PreparedSplit,
    rows: list[C.PairRow],
    endpoint_scale: float,
    batch_size: int,
    limit: int,
) -> dict[str, float]:
    selected = rows[: int(limit)]
    defects, inverses = [], []
    for chunk in C.chunked(selected, batch_size):
        batch = split.pair_batch(chunk)
        midpoint = 0.5 * (batch["source_age"] + batch["target_age"])
        direct = model.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        first = model.transport(batch["source"], batch["source_age"], midpoint, batch["label"])
        composed = model.transport(first, midpoint, batch["target_age"], batch["label"])
        recovered = model.transport(direct, batch["target_age"], batch["source_age"], batch["label"])
        defects.append(torch.sqrt(torch.mean(((direct - composed) / endpoint_scale) ** 2, dim=(1, 2))).cpu())
        inverses.append(torch.sqrt(torch.mean(((recovered - batch["source"]) / endpoint_scale) ** 2, dim=(1, 2))).cpu())
    defect = torch.cat(defects).numpy()
    inverse = torch.cat(inverses).numpy()
    return {
        "relative_cocycle_defect_mean": float(defect.mean()),
        "relative_cocycle_defect_p95": float(np.quantile(defect, 0.95)),
        "relative_inverse_defect_mean": float(inverse.mean()),
        "relative_inverse_defect_p95": float(np.quantile(inverse, 0.95)),
    }


def _velocity_scan_indices(split: PreparedSplit, max_scans: int | None) -> list[int]:
    candidates = [
        index for index, weight in enumerate(split.velocity_reference_weight.detach().cpu().tolist()) if weight > 0.0
    ]
    if not candidates:
        raise ValueError("Validation has no positive-reliability velocity references")
    if max_scans is None or int(max_scans) <= 0 or int(max_scans) >= len(candidates):
        return candidates

    # A limited run is used only for smoke validation. Alternate diagnosis so
    # that ordering in the cache cannot accidentally produce a one-group check.
    by_diagnosis = {
        diagnosis: [index for index in candidates if str(split.diagnoses[index]) == diagnosis]
        for diagnosis in ("CN", "AD")
    }
    selected: list[int] = []
    cursor = {"CN": 0, "AD": 0}
    while len(selected) < int(max_scans):
        changed = False
        for diagnosis in ("CN", "AD"):
            values = by_diagnosis[diagnosis]
            if cursor[diagnosis] < len(values):
                selected.append(values[cursor[diagnosis]])
                cursor[diagnosis] += 1
                changed = True
                if len(selected) == int(max_scans):
                    break
        if not changed:
            break
    return selected


def _weighted_velocity_group(records: list[dict[str, float | str]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("Cannot summarize an empty velocity group")
    weights = np.asarray([float(row["reference_reliability"]) for row in records], dtype=np.float64)
    weight_sum = float(weights.sum())
    if not weight_sum > 0.0:
        raise ValueError("Velocity group has zero total reliability")

    def weighted(name: str) -> float:
        values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
        return float(np.average(values, weights=weights))

    vector_error = weighted("vector_rmse_mm_per_year")
    vector_zero = weighted("zero_vector_rmse_mm_per_year")
    normal_error = weighted("normal_rmse_mm_per_year")
    normal_zero = weighted("zero_normal_rmse_mm_per_year")
    predicted_speed = weighted("predicted_speed_mm_per_year")
    observed_speed = weighted("observed_speed_mm_per_year")
    vector_ratio = vector_error / max(vector_zero, 1.0e-8)
    normal_ratio = normal_error / max(normal_zero, 1.0e-8)
    return {
        "scans": len(records),
        "reference_weight_sum": weight_sum,
        "reference_reliability_mean": float(weights.mean()),
        "vector_rmse_mm_per_year": vector_error,
        "zero_vector_rmse_mm_per_year": vector_zero,
        "vector_error_to_zero_ratio": float(vector_ratio),
        "normal_rmse_mm_per_year": normal_error,
        "zero_normal_rmse_mm_per_year": normal_zero,
        "normal_error_to_zero_ratio": float(normal_ratio),
        "normalized_error_ratio": float(0.5 * (vector_ratio + normal_ratio)),
        "vector_cosine": weighted("vector_cosine"),
        "normal_pearson": weighted("normal_pearson"),
        "normal_sign_agreement": weighted("normal_sign_agreement"),
        "predicted_speed_mm_per_year": predicted_speed,
        "observed_speed_mm_per_year": observed_speed,
        "speed_ratio": float(predicted_speed / max(observed_speed, 1.0e-8)),
    }


@torch.no_grad()
def evaluate_velocity_selection(
    model: ConditionalSpiralUNet,
    split: PreparedSplit,
    batch_size: int,
    max_scans: int | None = None,
) -> dict[str, Any]:
    """Fast reliability-weighted diagonal-velocity validation for model selection."""
    model.eval()
    selected = _velocity_scan_indices(split, max_scans)
    records: list[dict[str, float | str]] = []
    for chunk in C.chunked(selected, batch_size):
        indices = torch.as_tensor(chunk, dtype=torch.long, device=split.vertices.device)
        vertices = split.vertices[indices]
        observed = split.velocity_reference[indices]
        predicted = model.instantaneous_velocity(vertices, split.ages[indices], split.labels[indices])
        normals = vertex_normals(vertices, model.faces)
        predicted_normal = torch.sum(predicted * normals, dim=-1)
        observed_normal = torch.sum(observed * normals, dim=-1)
        vector_error = torch.sqrt(torch.mean((predicted - observed).square(), dim=(1, 2)))
        vector_zero = torch.sqrt(torch.mean(observed.square(), dim=(1, 2)))
        normal_error = torch.sqrt(torch.mean((predicted_normal - observed_normal).square(), dim=1))
        normal_zero = torch.sqrt(torch.mean(observed_normal.square(), dim=1))
        predicted_flat = predicted.reshape(len(chunk), -1)
        observed_flat = observed.reshape(len(chunk), -1)
        vector_cosine = torch.sum(predicted_flat * observed_flat, dim=1) / (
            torch.linalg.vector_norm(predicted_flat, dim=1)
            * torch.linalg.vector_norm(observed_flat, dim=1)
        ).clamp_min(1.0e-8)
        predicted_centered = predicted_normal - predicted_normal.mean(dim=1, keepdim=True)
        observed_centered = observed_normal - observed_normal.mean(dim=1, keepdim=True)
        normal_pearson = torch.sum(predicted_centered * observed_centered, dim=1) / (
            torch.linalg.vector_norm(predicted_centered, dim=1)
            * torch.linalg.vector_norm(observed_centered, dim=1)
        ).clamp_min(1.0e-8)
        sign_agreement = (torch.sign(predicted_normal) == torch.sign(observed_normal)).float().mean(dim=1)
        predicted_speed = torch.linalg.vector_norm(predicted, dim=-1).mean(dim=1)
        observed_speed = torch.linalg.vector_norm(observed, dim=-1).mean(dim=1)
        for local_index, split_index in enumerate(chunk):
            records.append(
                {
                    "scan_id": str(split.scan_ids[split_index]),
                    "diagnosis": str(split.diagnoses[split_index]),
                    "reference_reliability": float(split.velocity_reference_weight[split_index].cpu()),
                    "vector_rmse_mm_per_year": float(vector_error[local_index].cpu()),
                    "zero_vector_rmse_mm_per_year": float(vector_zero[local_index].cpu()),
                    "normal_rmse_mm_per_year": float(normal_error[local_index].cpu()),
                    "zero_normal_rmse_mm_per_year": float(normal_zero[local_index].cpu()),
                    "vector_cosine": float(vector_cosine[local_index].cpu()),
                    "normal_pearson": float(normal_pearson[local_index].cpu()),
                    "normal_sign_agreement": float(sign_agreement[local_index].cpu()),
                    "predicted_speed_mm_per_year": float(predicted_speed[local_index].cpu()),
                    "observed_speed_mm_per_year": float(observed_speed[local_index].cpu()),
                }
            )
    groups: dict[str, Any] = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = records if diagnosis == "overall" else [row for row in records if row["diagnosis"] == diagnosis]
        if current:
            groups[diagnosis] = _weighted_velocity_group(current)
    return {"weighting": "visit reliability normalized within each reported group", "groups": groups}


def validation_selection(
    first_last: dict[str, Any],
    defects: dict[str, float],
    config: dict[str, Any],
    velocity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = first_last["groups"]
    if "CN" not in groups or "AD" not in groups:
        raise ValueError("Validation requires both CN and AD")
    macro_ratio = 0.5 * (groups["CN"]["error_to_nochange_ratio"] + groups["AD"]["error_to_nochange_ratio"])
    selection = config["selection"]
    score = macro_ratio + float(selection["cocycle_tiebreak_weight"]) * defects["relative_cocycle_defect_mean"]
    gates = {
        "cocycle": defects["relative_cocycle_defect_mean"]
        <= float(selection["max_relative_cocycle_defect"]),
        "inverse": defects["relative_inverse_defect_mean"]
        <= float(selection["max_relative_inverse_defect"]),
    }
    output: dict[str, Any] = {
        "version": str(selection.get("version", "endpoint_v1")),
        "macro_endpoint_ratio": float(macro_ratio),
        "cocycle_tiebreak": float(
            float(selection["cocycle_tiebreak_weight"]) * defects["relative_cocycle_defect_mean"]
        ),
    }

    if bool(selection.get("velocity_aware", False)):
        if velocity is None or "overall" not in velocity.get("groups", {}):
            raise ValueError("Velocity-aware selection requires overall velocity validation metrics")
        overall_velocity = velocity["groups"]["overall"]
        velocity_ratio = float(overall_velocity["normalized_error_ratio"])
        speed_ratio = float(overall_velocity["speed_ratio"])
        velocity_contribution = float(selection["velocity_score_weight"]) * velocity_ratio
        score += velocity_contribution
        predicted_cn_rate = float(groups["CN"]["predicted_log_volume_rate_per_year"])
        predicted_ad_rate = float(groups["AD"]["predicted_log_volume_rate_per_year"])
        flipped = float(groups["overall"].get("flipped_face_fraction", 0.0))
        gates.update(
            {
                "velocity_error": velocity_ratio <= float(selection["max_normalized_velocity_error"]),
                "velocity_speed": float(selection["min_velocity_speed_ratio"])
                <= speed_ratio
                <= float(selection["max_velocity_speed_ratio"]),
                "ad_stronger_volume_decline": (
                    not bool(selection.get("require_ad_stronger_volume_decline", False))
                    or predicted_ad_rate < predicted_cn_rate
                ),
                "mesh_flips": flipped <= float(selection["max_first_last_flipped_face_fraction"]),
            }
        )
        output.update(
            {
                "velocity_normalized_error_ratio": velocity_ratio,
                "velocity_score_contribution": velocity_contribution,
                "velocity_speed_ratio": speed_ratio,
                "predicted_cn_log_volume_rate_per_year": predicted_cn_rate,
                "predicted_ad_log_volume_rate_per_year": predicted_ad_rate,
                "first_last_flipped_face_fraction": flipped,
            }
        )

    output["gates"] = gates
    output["feasible"] = bool(all(gates.values()))
    output["score"] = float(score)
    return output


def validation_score(
    first_last: dict[str, Any],
    defects: dict[str, float],
    config: dict[str, Any],
    velocity: dict[str, Any] | None = None,
) -> tuple[float, bool]:
    selection = validation_selection(first_last, defects, config, velocity)
    return float(selection["score"]), bool(selection["feasible"])
