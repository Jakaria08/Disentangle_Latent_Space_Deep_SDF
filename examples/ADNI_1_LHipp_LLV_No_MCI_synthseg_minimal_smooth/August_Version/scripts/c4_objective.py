#!/usr/bin/env python3
"""Direct C4 losses and validation for arbitrary frozen 128-D decoders."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler

import common as C


@torch.no_grad()
def attach_reference_geometry(
    values: dict[str, torch.Tensor],
    geometry: C.FrozenGeometry,
    batch_size: int,
) -> None:
    vertices: list[torch.Tensor] = []
    volumes: list[torch.Tensor] = []
    for start in range(0, len(values["z"]), int(batch_size)):
        current = geometry.vertices(values["z"][start : start + int(batch_size)])
        vertices.append(current)
        volumes.append(geometry.volume_from_vertices(current))
    values["reference_vertices"] = torch.cat(vertices, dim=0)
    values["reference_volume"] = torch.cat(volumes, dim=0)
    if not torch.isfinite(values["reference_vertices"]).all() or not torch.isfinite(values["reference_volume"]).all():
        raise RuntimeError("Frozen decoder produced a non-finite reference geometry")


def indexed(values: dict[str, torch.Tensor], raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = C.indexed(values, raw)
    source_index = raw["source"].to(values["z"].device)
    target_index = raw["target"].to(values["z"].device)
    output.update({
        "source_index": source_index,
        "target_index": target_index,
        "source_vertices": values["reference_vertices"][source_index],
        "target_vertices": values["reference_vertices"][target_index],
        "source_volume": values["reference_volume"][source_index],
        "target_volume": values["reference_volume"][target_index],
    })
    return output


def balanced_subset(rows: list[C.PairRow], limit: int | None) -> list[C.PairRow]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 4:
        raise ValueError("A balanced subset requires at least four rows")
    buckets: dict[tuple[str, str], list[C.PairRow]] = {}
    for row in rows:
        gap = "adjacent" if row.pair_type == "adjacent" else "nonadjacent"
        buckets.setdefault((row.diagnosis, gap), []).append(row)
    required = (("CN", "adjacent"), ("CN", "nonadjacent"), ("AD", "adjacent"), ("AD", "nonadjacent"))
    if any(not buckets.get(key) for key in required):
        raise ValueError("Cannot form CN/AD adjacent/nonadjacent balanced subset")
    output: list[C.PairRow] = []
    cursor = 0
    while len(output) < limit:
        key = required[cursor % len(required)]
        items = buckets[key]
        output.append(items[(cursor // len(required)) % len(items)])
        cursor += 1
    return output


def balanced_first_last_subset(rows: list[C.PairRow], limit: int | None) -> list[C.PairRow]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 2:
        raise ValueError("First-last validation subset must contain CN and AD")
    grouped = {diagnosis: [row for row in rows if row.diagnosis == diagnosis] for diagnosis in ("CN", "AD")}
    if not grouped["CN"] or not grouped["AD"]:
        raise ValueError("First-last validation requires CN and AD")
    output = []
    for index in range(limit):
        diagnosis = ("CN", "AD")[index % 2]
        output.append(grouped[diagnosis][(index // 2) % len(grouped[diagnosis])])
    return output


@torch.no_grad()
def training_statistics(
    values: dict[str, torch.Tensor],
    rows: list[C.PairRow],
    archive: dict[str, np.ndarray],
    pair_limit: int | None = None,
) -> dict[str, Any]:
    selected = balanced_subset(rows, pair_limit)
    collected: dict[str, list[np.ndarray]] = {
        "latent": [],
        "coordinate": [],
        "euclidean": [],
        "volume_log": [],
        "rate": [],
        "displacement": [],
    }
    rates_by_subject: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}}
    for start in range(0, len(selected), 512):
        chunk = selected[start : start + 512]
        batch = indexed(values, C.collate_pairs(chunk))
        delta_vertex = batch["target_vertices"] - batch["source_vertices"]
        delta_latent = batch["target"] - batch["source"]
        gap = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        log_delta = torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])
        collected["latent"].append(torch.mean(delta_latent.square(), dim=1).cpu().numpy())
        collected["coordinate"].append(torch.mean(torch.abs(delta_vertex), dim=(1, 2)).cpu().numpy())
        collected["euclidean"].append(torch.linalg.vector_norm(delta_vertex, dim=2).mean(dim=1).cpu().numpy())
        collected["volume_log"].append(torch.abs(log_delta).cpu().numpy())
        collected["rate"].append(torch.abs(log_delta / gap).cpu().numpy())
        collected["displacement"].append(torch.sqrt(torch.mean(delta_latent.square(), dim=1)).cpu().numpy())
        for offset, row in enumerate(chunk):
            rates_by_subject[row.diagnosis].setdefault(row.subject, []).append(float((log_delta[offset] / gap[offset]).cpu()))
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    slopes = []
    for index in range(len(offsets) - 1):
        first, last = int(offsets[index]), int(offsets[index + 1])
        slopes.append(np.asarray([
            abs(float(C.line_slope(values["years"][first:last], torch.log(values["reference_volume"][first:last])).cpu()))
        ]))
    group_targets = {}
    for diagnosis in ("CN", "AD"):
        subject_means = [float(np.mean(items)) for items in rates_by_subject[diagnosis].values()]
        if not subject_means:
            raise ValueError(f"No {diagnosis} rate observations for training statistics")
        group_targets[diagnosis] = float(np.mean(subject_means))
    return {
        "normalization_scales": {
            "latent": C.safe_median(collected["latent"]),
            "coordinate": C.safe_median(collected["coordinate"]),
            "euclidean": C.safe_median(collected["euclidean"]),
            "volume_log": C.safe_median(collected["volume_log"]),
            "rate": C.safe_median(collected["rate"]),
            "slope": C.safe_median(slopes),
            "displacement": C.safe_median(collected["displacement"]),
        },
        "group_log_volume_rate_targets": group_targets,
        "ad_minus_cn_log_volume_rate_target": group_targets["AD"] - group_targets["CN"],
        "subjects_by_diagnosis": {diagnosis: len(items) for diagnosis, items in rates_by_subject.items()},
        "statistics_source": "train split only; frozen-decoder geometry; subject-balanced forward pairs",
        "pair_rows_used": len(selected),
    }


def balanced_pair_sampler(rows: list[C.PairRow], seed: int, epoch: int, samples: int) -> WeightedRandomSampler:
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        gap = "adjacent" if row.pair_type == "adjacent" else "nonadjacent"
        buckets.setdefault((row.diagnosis, gap, row.subject), []).append(index)
    strata: dict[tuple[str, str], list[str]] = {}
    for diagnosis, gap, subject in buckets:
        strata.setdefault((diagnosis, gap), []).append(subject)
    required = {("CN", "adjacent"), ("CN", "nonadjacent"), ("AD", "adjacent"), ("AD", "nonadjacent")}
    if set(strata) != required:
        raise ValueError(f"Each diagnosis/gap stratum is required; found {sorted(strata)}")
    weights = np.zeros(len(rows), dtype=np.float64)
    for key, subjects in strata.items():
        for subject in subjects:
            indices = buckets[(key[0], key[1], subject)]
            weights[np.asarray(indices, dtype=np.int64)] = 0.25 / len(subjects) / len(indices)
    generator = torch.Generator().manual_seed(int(seed) + 100_003 * int(epoch))
    return WeightedRandomSampler(torch.from_numpy(weights), int(samples), replacement=True, generator=generator)


def balanced_sequence_starts(archive: dict[str, np.ndarray], seed: int, epoch: int) -> list[int]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    diagnoses = archive["subject_diagnoses"].astype(str)
    grouped = {
        diagnosis: [int(offsets[index]) for index, item in enumerate(diagnoses) if item == diagnosis]
        for diagnosis in ("CN", "AD")
    }
    generator = random.Random(int(seed) + 97_003 * int(epoch))
    for items in grouped.values():
        if not items:
            raise ValueError("Sequence sampling requires CN and AD")
        generator.shuffle(items)
    output = []
    for index in range(max(len(grouped["CN"]), len(grouped["AD"]))):
        output.extend((grouped["CN"][index % len(grouped["CN"])], grouped["AD"][index % len(grouped["AD"])]))
    return output


def shape_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_vertices: torch.Tensor,
    geometry: C.FrozenGeometry,
    scales: dict[str, float],
) -> dict[str, torch.Tensor]:
    latent = torch.mean((prediction - target).square()) / float(scales["latent"])
    predicted_vertices = geometry.vertices(prediction)
    coordinate = torch.mean(torch.abs(predicted_vertices - target_vertices)) / float(scales["coordinate"])
    euclidean = torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=2).mean() / float(scales["euclidean"])
    return {
        "latent": latent,
        "vertex": 0.5 * (coordinate + euclidean),
        "coordinate": coordinate,
        "euclidean": euclidean,
    }


def mean_scaled_mse(left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.mean((left - right).square()) / float(scale)


def pair_terms(
    flow: nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch = indexed(values, raw)
    scales = statistics["normalization_scales"]
    forward_prediction = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
    backward_prediction = flow.transport(batch["target"], batch["target_age"], batch["source_age"], batch["label"])
    forward = shape_terms(forward_prediction, batch["target"], batch["target_vertices"], geometry, scales)
    backward = shape_terms(backward_prediction, batch["source"], batch["source_vertices"], geometry, scales)

    valid_middle = batch["intermediate_index"] >= 0
    observed = forward_prediction.sum() * 0.0
    if bool(valid_middle.any()):
        middle_index = batch["intermediate_index"][valid_middle]
        middle_age = values["age"][middle_index]
        label = batch["label"][valid_middle]
        forward_middle = flow.transport(batch["source"][valid_middle], batch["source_age"][valid_middle], middle_age, label)
        forward_composed = flow.transport(forward_middle, middle_age, batch["target_age"][valid_middle], label)
        backward_middle = flow.transport(batch["target"][valid_middle], batch["target_age"][valid_middle], middle_age, label)
        backward_composed = flow.transport(backward_middle, middle_age, batch["source_age"][valid_middle], label)
        observed = 0.5 * (
            mean_scaled_mse(forward_prediction[valid_middle], forward_composed, scales["latent"])
            + mean_scaled_mse(backward_prediction[valid_middle], backward_composed, scales["latent"])
        )

    ratio = torch.empty_like(batch["source_age"]).uniform_(0.2, 0.8)
    virtual_age = batch["source_age"] + ratio * (batch["target_age"] - batch["source_age"])
    forward_middle = flow.transport(batch["source"], batch["source_age"], virtual_age, batch["label"])
    forward_composed = flow.transport(forward_middle, virtual_age, batch["target_age"], batch["label"])
    backward_middle = flow.transport(batch["target"], batch["target_age"], virtual_age, batch["label"])
    backward_composed = flow.transport(backward_middle, virtual_age, batch["source_age"], batch["label"])
    virtual = 0.5 * (
        mean_scaled_mse(forward_prediction, forward_composed, scales["latent"])
        + mean_scaled_mse(backward_prediction, backward_composed, scales["latent"])
    )
    inverse_forward = flow.transport(forward_prediction, batch["target_age"], batch["source_age"], batch["label"])
    inverse_backward = flow.transport(backward_prediction, batch["source_age"], batch["target_age"], batch["label"])
    inverse = 0.5 * (
        mean_scaled_mse(inverse_forward, batch["source"], scales["latent"])
        + mean_scaled_mse(inverse_backward, batch["target"], scales["latent"])
    )

    predicted_forward_vertices = geometry.vertices(forward_prediction)
    predicted_backward_vertices = geometry.vertices(backward_prediction)
    predicted_forward_volume = geometry.volume_from_vertices(predicted_forward_vertices)
    predicted_backward_volume = geometry.volume_from_vertices(predicted_backward_vertices)
    log_forward = torch.log(predicted_forward_volume) - torch.log(batch["target_volume"])
    log_backward = torch.log(predicted_backward_volume) - torch.log(batch["source_volume"])
    volume = 0.5 * (
        F.smooth_l1_loss(log_forward / float(scales["volume_log"]), torch.zeros_like(log_forward))
        + F.smooth_l1_loss(log_backward / float(scales["volume_log"]), torch.zeros_like(log_backward))
    )
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    observed_rate = (torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])) / years
    rate_forward = (torch.log(predicted_forward_volume) - torch.log(batch["source_volume"])) / years
    rate_backward = (torch.log(predicted_backward_volume) - torch.log(batch["target_volume"])) / (-years)
    rate = 0.5 * (
        F.smooth_l1_loss((rate_forward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_forward))
        + F.smooth_l1_loss((rate_backward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_backward))
    )
    is_ad = batch["label"] >= 0.5
    group_rate = forward_prediction.sum() * 0.0
    disease_gap = forward_prediction.sum() * 0.0
    if bool(is_ad.any()) and bool((~is_ad).any()):
        losses = []
        for diagnosis, mask in (("CN", ~is_ad), ("AD", is_ad)):
            target = torch.tensor(float(statistics["group_log_volume_rate_targets"][diagnosis]), device=rate_forward.device)
            losses.append(F.smooth_l1_loss((rate_forward[mask].mean() - target) / float(scales["rate"]), torch.zeros((), device=rate_forward.device)))
        group_rate = torch.stack(losses).mean()
        target_gap = torch.tensor(float(statistics["ad_minus_cn_log_volume_rate_target"]), device=rate_forward.device)
        disease_gap = F.smooth_l1_loss(
            ((rate_forward[is_ad].mean() - rate_forward[~is_ad].mean()) - target_gap) / float(scales["rate"]),
            torch.zeros((), device=rate_forward.device),
        )
    return {
        "real_latent": 0.5 * (forward["latent"] + backward["latent"]),
        "real_vertex": 0.5 * (forward["vertex"] + backward["vertex"]),
        "real_coordinate": 0.5 * (forward["coordinate"] + backward["coordinate"]),
        "real_euclidean": 0.5 * (forward["euclidean"] + backward["euclidean"]),
        "observed_semigroup": observed,
        "virtual_semigroup": virtual,
        "inverse": inverse,
        "volume": volume,
        "rate": rate,
        "group_rate": group_rate,
        "disease_gap": disease_gap,
    }


def sequence_terms(
    flow: nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    start: int,
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_index = int(np.searchsorted(offsets, int(start), side="right") - 1)
    end = int(offsets[subject_index + 1])
    z = values["z"][start:end]
    ages = values["age"][start:end]
    years = values["years"][start:end]
    label = values["label"][start : start + 1]
    reference_vertices = values["reference_vertices"][start:end]
    reference_volume = values["reference_volume"][start:end]
    if len(z) < 2:
        raise ValueError("Sequence needs at least two visits")
    scales = statistics["normalization_scales"]
    count = len(z) - 1
    direct_forward = flow.transport(z[:1].expand(count, -1), ages[:1].expand(count), ages[1:], label.expand(count))
    rollout_forward = []
    current = z[:1]
    previous_age = ages[:1]
    for index in range(1, len(z)):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
        rollout_forward.append(current)
        previous_age = ages[index : index + 1]
    rollout_forward = torch.cat(rollout_forward)

    reverse_targets = z[:-1].flip(0)
    reverse_vertices = reference_vertices[:-1].flip(0)
    direct_backward = flow.transport(z[-1:].expand(count, -1), ages[-1:].expand(count), ages[:-1].flip(0), label.expand(count))
    rollout_backward = []
    current = z[-1:]
    previous_age = ages[-1:]
    for index in range(len(z) - 2, -1, -1):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
        rollout_backward.append(current)
        previous_age = ages[index : index + 1]
    rollout_backward = torch.cat(rollout_backward)
    shapes = (
        shape_terms(direct_forward, z[1:], reference_vertices[1:], geometry, scales),
        shape_terms(rollout_forward, z[1:], reference_vertices[1:], geometry, scales),
        shape_terms(direct_backward, reverse_targets, reverse_vertices, geometry, scales),
        shape_terms(rollout_backward, reverse_targets, reverse_vertices, geometry, scales),
    )
    sequence_latent = torch.stack([item["latent"] for item in shapes]).mean()
    sequence_vertex = torch.stack([item["vertex"] for item in shapes]).mean()
    semigroup = 0.5 * (
        mean_scaled_mse(direct_forward, rollout_forward, scales["latent"])
        + mean_scaled_mse(direct_backward, rollout_backward, scales["latent"])
    )
    forward_volume = geometry.volume(torch.cat((z[:1], direct_forward)))
    backward_volume = geometry.volume(torch.cat((z[-1:], direct_backward)))
    slope_observed = C.line_slope(years, torch.log(reference_volume))
    slope_forward = C.line_slope(years, torch.log(forward_volume))
    slope_backward = C.line_slope(years.flip(0), torch.log(backward_volume))
    slope = 0.5 * (
        F.smooth_l1_loss((slope_forward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_forward))
        + F.smooth_l1_loss((slope_backward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_backward))
    )
    return {
        "sequence_latent": sequence_latent,
        "sequence_vertex": sequence_vertex,
        "sequence_semigroup": semigroup,
        "slope": slope,
    }


def ramp(epoch: int, epochs: int) -> float:
    return 1.0 if epochs <= 0 else min(1.0, float(epoch) / float(epochs))


def total_loss(
    pair: dict[str, torch.Tensor],
    sequence: dict[str, torch.Tensor],
    config: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = config["loss"]
    consistency = ramp(epoch, int(config["training"]["consistency_ramp_epochs"]))
    anatomy = ramp(epoch, int(config["training"]["anatomy_ramp_epochs"]))
    total = (
        float(weights["real_latent_weight"]) * pair["real_latent"]
        + float(weights["real_vertex_weight"]) * pair["real_vertex"]
        + consistency * (
            float(weights["observed_semigroup_weight"]) * pair["observed_semigroup"]
            + float(weights["virtual_semigroup_weight"]) * pair["virtual_semigroup"]
            + float(weights["inverse_weight"]) * pair["inverse"]
            + float(weights["sequence_latent_weight"]) * sequence["sequence_latent"]
            + float(weights["sequence_vertex_weight"]) * sequence["sequence_vertex"]
            + float(weights["sequence_semigroup_weight"]) * sequence["sequence_semigroup"]
        )
        + anatomy * (
            float(weights["volume_weight"]) * pair["volume"]
            + float(weights["rate_weight"]) * pair["rate"]
            + float(weights["slope_weight"]) * sequence["slope"]
            + float(weights["group_rate_weight"]) * pair["group_rate"]
            + float(weights["disease_gap_weight"]) * pair["disease_gap"]
        )
    )
    terms = pair | sequence | {
        "total": total,
        "consistency_factor": torch.tensor(consistency, device=total.device),
        "anatomy_factor": torch.tensor(anatomy, device=total.device),
    }
    return total, {name: float(value.detach().cpu()) for name, value in terms.items()}


def _empty_eval_group() -> dict[str, list[float]]:
    return {name: [] for name in (
        "latent",
        "coordinate",
        "euclidean",
        "end_to_end_coordinate_rmse",
        "end_to_end_coordinate_mae",
        "end_to_end_euclidean",
        "volume_relative",
        "rate",
        "predicted_signed_rate",
        "observed_signed_rate",
        "nochange_latent",
        "nochange_coordinate",
        "nochange_euclidean",
        "nochange_end_to_end_coordinate_mae",
        "nochange_end_to_end_euclidean",
        "nochange_volume_relative",
        "nochange_rate",
    )}


@torch.no_grad()
def evaluate_pairs(
    flow: nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    rows: list[C.PairRow],
    raw_vertices_mm: np.ndarray,
    batch_size: int,
    include_rows: bool = False,
) -> dict[str, Any]:
    grouped = {diagnosis: _empty_eval_group() for diagnosis in ("CN", "AD", "overall")}
    row_metrics: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = indexed(values, C.collate_pairs(chunk))
        prediction = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        predicted_vertices = geometry.vertices(prediction)
        predicted_volume = geometry.volume_from_vertices(predicted_vertices)
        raw_target = torch.from_numpy(
            np.asarray(raw_vertices_mm[batch["target_index"].cpu().numpy()], dtype=np.float32).copy()
        ).to(predicted_vertices.device)
        years = torch.abs(batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        predicted_signed_rate = (torch.log(predicted_volume) - torch.log(batch["source_volume"])) / years
        observed_signed_rate = (torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])) / years
        transport_delta = predicted_vertices - batch["target_vertices"]
        end_delta = predicted_vertices - raw_target
        nochange_transport = batch["source_vertices"] - batch["target_vertices"]
        nochange_end = batch["source_vertices"] - raw_target
        tensors = {
            "latent": torch.mean((prediction - batch["target"]).square(), dim=1),
            "coordinate": torch.mean(torch.abs(transport_delta), dim=(1, 2)),
            "euclidean": torch.linalg.vector_norm(transport_delta, dim=2).mean(dim=1),
            "end_to_end_coordinate_rmse": torch.sqrt(torch.mean(end_delta.square(), dim=(1, 2))),
            "end_to_end_coordinate_mae": torch.mean(torch.abs(end_delta), dim=(1, 2)),
            "end_to_end_euclidean": torch.linalg.vector_norm(end_delta, dim=2).mean(dim=1),
            "volume_relative": torch.abs(predicted_volume - batch["target_volume"]) / batch["target_volume"],
            "rate": torch.abs((torch.log(predicted_volume) - torch.log(batch["target_volume"])) / years),
            "predicted_signed_rate": predicted_signed_rate,
            "observed_signed_rate": observed_signed_rate,
            "nochange_latent": torch.mean((batch["source"] - batch["target"]).square(), dim=1),
            "nochange_coordinate": torch.mean(torch.abs(nochange_transport), dim=(1, 2)),
            "nochange_euclidean": torch.linalg.vector_norm(nochange_transport, dim=2).mean(dim=1),
            "nochange_end_to_end_coordinate_mae": torch.mean(torch.abs(nochange_end), dim=(1, 2)),
            "nochange_end_to_end_euclidean": torch.linalg.vector_norm(nochange_end, dim=2).mean(dim=1),
            "nochange_volume_relative": torch.abs(batch["source_volume"] - batch["target_volume"]) / batch["target_volume"],
            "nochange_rate": torch.abs((torch.log(batch["source_volume"]) - torch.log(batch["target_volume"])) / years),
        }
        for index, row in enumerate(chunk):
            for bucket in (row.diagnosis, "overall"):
                    for name, tensor in tensors.items():
                        grouped[bucket][name].append(float(tensor[index].cpu()))
            if include_rows:
                row_metrics.append({
                    "subject": row.subject,
                    "diagnosis": row.diagnosis,
                    "pair_type": row.pair_type,
                    "delta_years": float(row.delta_years),
                    **{name: float(tensor[index].cpu()) for name, tensor in tensors.items()},
                })
    output = {
        "groups": {
            diagnosis: {f"{name}_mean": float(np.mean(items)) if items else float("nan") for name, items in values.items()}
            | {"rows": len(values["latent"])}
            for diagnosis, values in grouped.items()
        }
    }
    if include_rows:
        output["row_metrics"] = row_metrics
    return output


@torch.no_grad()
def cocycle_defects(
    flow: nn.Module,
    values: dict[str, torch.Tensor],
    rows: list[C.PairRow],
    statistics: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    semigroup = []
    inverse_values = []
    scale = float(statistics["normalization_scales"]["displacement"])
    for start in range(0, len(rows), batch_size):
        batch = indexed(values, C.collate_pairs(rows[start : start + batch_size]))
        direct = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(batch["source"], batch["source_age"], middle_age, batch["label"])
        composed = flow.transport(middle, middle_age, batch["target_age"], batch["label"])
        inverse = flow.transport(direct, batch["target_age"], batch["source_age"], batch["label"])
        semigroup.extend((torch.sqrt(torch.mean((direct - composed).square(), dim=1)) / scale).cpu().tolist())
        inverse_values.extend((torch.sqrt(torch.mean((inverse - batch["source"]).square(), dim=1)) / scale).cpu().tolist())
    return {
        "relative_semigroup_defect_mean": float(np.mean(semigroup)),
        "relative_semigroup_defect_p95": float(np.quantile(semigroup, 0.95)),
        "relative_inverse_defect_mean": float(np.mean(inverse_values)),
        "relative_inverse_defect_p95": float(np.quantile(inverse_values, 0.95)),
    }


def validation_score(
    all_pairs: dict[str, Any],
    first_last: dict[str, Any],
    defects: dict[str, float],
    selection: dict[str, Any],
) -> tuple[float, bool, dict[str, float]]:
    macro_shape = []
    macro_volume = []
    macro_rate = []
    macro_trend = []
    signed = {}
    feasible = True
    for diagnosis in ("CN", "AD"):
        values = first_last["groups"][diagnosis]
        coordinate_ratio = values["coordinate_mean"] / max(values["nochange_coordinate_mean"], 1.0e-8)
        euclidean_ratio = values["euclidean_mean"] / max(values["nochange_euclidean_mean"], 1.0e-8)
        latent_ratio = values["latent_mean"] / max(values["nochange_latent_mean"], 1.0e-8)
        volume_ratio = values["volume_relative_mean"] / max(values["nochange_volume_relative_mean"], 1.0e-8)
        rate_ratio = values["rate_mean"] / max(values["nochange_rate_mean"], 1.0e-8)
        predicted_rate = float(values["predicted_signed_rate_mean"])
        observed_rate = float(values["observed_signed_rate_mean"])
        trend_ratio = abs(predicted_rate - observed_rate) / max(values["nochange_rate_mean"], 1.0e-8)
        macro_shape.append(0.5 * (coordinate_ratio + euclidean_ratio))
        macro_volume.append(volume_ratio)
        macro_rate.append(rate_ratio)
        macro_trend.append(trend_ratio)
        signed[diagnosis] = (predicted_rate, observed_rate)
        feasible = feasible and latent_ratio <= 1.0 + float(selection["latent_nochange_tolerance"])
        feasible = feasible and coordinate_ratio <= 1.0 + float(selection["coordinate_nochange_tolerance"])
    overall = all_pairs["groups"]["overall"]
    all_shape = 0.5 * (
        overall["coordinate_mean"] / max(overall["nochange_coordinate_mean"], 1.0e-8)
        + overall["euclidean_mean"] / max(overall["nochange_euclidean_mean"], 1.0e-8)
    )
    all_volume = overall["volume_relative_mean"] / max(overall["nochange_volume_relative_mean"], 1.0e-8)
    score = (
        float(np.mean(macro_shape))
        + float(selection["all_pair_shape_weight"]) * all_shape
        + float(selection["volume_tiebreak_weight"]) * all_volume
    )
    feasible = feasible and math.isfinite(score)
    feasible = feasible and defects["relative_semigroup_defect_mean"] <= float(selection["max_relative_semigroup_defect"])
    feasible = feasible and defects["relative_inverse_defect_mean"] <= float(selection["max_relative_inverse_defect"])
    ratios = {
        "macro_first_last_shape": float(np.mean(macro_shape)),
        "all_pair_shape": float(all_shape),
        "all_pair_volume": float(all_volume),
        "macro_first_last_volume": float(np.mean(macro_volume)),
        "macro_first_last_rate": float(np.mean(macro_rate)),
        "macro_group_trend": float(np.mean(macro_trend)),
        "first_last_cn_predicted_signed_rate": signed["CN"][0],
        "first_last_cn_observed_signed_rate": signed["CN"][1],
        "first_last_ad_predicted_signed_rate": signed["AD"][0],
        "first_last_ad_observed_signed_rate": signed["AD"][1],
        "score": float(score),
    }
    C.assert_finite_mapping(ratios)
    return float(score), bool(feasible), ratios
