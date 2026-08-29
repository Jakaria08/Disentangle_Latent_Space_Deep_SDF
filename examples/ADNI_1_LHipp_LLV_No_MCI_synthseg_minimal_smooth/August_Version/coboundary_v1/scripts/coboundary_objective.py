#!/usr/bin/env python3
"""Matched C4 anatomy losses with fixed-context exact-coboundary transport."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


BASE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(BASE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPTS))

import common as C  # noqa: E402
import c4_objective as B  # noqa: E402


attach_reference_geometry = B.attach_reference_geometry
indexed = B.indexed
balanced_subset = B.balanced_subset
balanced_first_last_subset = B.balanced_first_last_subset
training_statistics = B.training_statistics
balanced_pair_sampler = B.balanced_pair_sampler
balanced_sequence_starts = B.balanced_sequence_starts
shape_terms = B.shape_terms
mean_scaled_mse = B.mean_scaled_mse
total_loss = B.total_loss
validation_score = B.validation_score


def _transport(
    flow: nn.Module,
    latent: torch.Tensor,
    source_age: torch.Tensor,
    target_age: torch.Tensor,
    label: torch.Tensor,
    context: torch.Tensor,
    context_age: torch.Tensor,
) -> torch.Tensor:
    return flow.transport(latent, source_age, target_age, label, context, context_age)


def pair_terms(
    flow: nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    batch = indexed(values, raw)
    scales = statistics["normalization_scales"]
    context, context_age = batch["context"], batch["context_age"]
    forward_prediction = _transport(
        flow, batch["source"], batch["source_age"], batch["target_age"], batch["label"], context, context_age
    )
    backward_prediction = _transport(
        flow, batch["target"], batch["target_age"], batch["source_age"], batch["label"], context, context_age
    )
    forward = shape_terms(forward_prediction, batch["target"], batch["target_vertices"], geometry, scales)
    backward = shape_terms(backward_prediction, batch["source"], batch["source_vertices"], geometry, scales)

    valid_middle = batch["intermediate_index"] >= 0
    observed = forward_prediction.sum() * 0.0
    if bool(valid_middle.any()):
        middle_index = batch["intermediate_index"][valid_middle]
        middle_age = values["age"][middle_index]
        label = batch["label"][valid_middle]
        current_context = context[valid_middle]
        current_context_age = context_age[valid_middle]
        forward_middle = _transport(
            flow, batch["source"][valid_middle], batch["source_age"][valid_middle], middle_age,
            label, current_context, current_context_age,
        )
        forward_composed = _transport(
            flow, forward_middle, middle_age, batch["target_age"][valid_middle],
            label, current_context, current_context_age,
        )
        backward_middle = _transport(
            flow, batch["target"][valid_middle], batch["target_age"][valid_middle], middle_age,
            label, current_context, current_context_age,
        )
        backward_composed = _transport(
            flow, backward_middle, middle_age, batch["source_age"][valid_middle],
            label, current_context, current_context_age,
        )
        observed = 0.5 * (
            mean_scaled_mse(forward_prediction[valid_middle], forward_composed, scales["latent"])
            + mean_scaled_mse(backward_prediction[valid_middle], backward_composed, scales["latent"])
        )

    ratio = torch.empty_like(batch["source_age"]).uniform_(0.2, 0.8)
    virtual_age = batch["source_age"] + ratio * (batch["target_age"] - batch["source_age"])
    forward_middle = _transport(flow, batch["source"], batch["source_age"], virtual_age, batch["label"], context, context_age)
    forward_composed = _transport(flow, forward_middle, virtual_age, batch["target_age"], batch["label"], context, context_age)
    backward_middle = _transport(flow, batch["target"], batch["target_age"], virtual_age, batch["label"], context, context_age)
    backward_composed = _transport(flow, backward_middle, virtual_age, batch["source_age"], batch["label"], context, context_age)
    virtual = 0.5 * (
        mean_scaled_mse(forward_prediction, forward_composed, scales["latent"])
        + mean_scaled_mse(backward_prediction, backward_composed, scales["latent"])
    )
    inverse_forward = _transport(flow, forward_prediction, batch["target_age"], batch["source_age"], batch["label"], context, context_age)
    inverse_backward = _transport(flow, backward_prediction, batch["source_age"], batch["target_age"], batch["label"], context, context_age)
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
            losses.append(F.smooth_l1_loss(
                (rate_forward[mask].mean() - target) / float(scales["rate"]),
                torch.zeros((), device=rate_forward.device),
            ))
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
    context = values["context"][start : start + 1]
    context_age = values["context_age"][start : start + 1]
    reference_vertices = values["reference_vertices"][start:end]
    reference_volume = values["reference_volume"][start:end]
    if len(z) < 2:
        raise ValueError("Sequence needs at least two visits")
    scales = statistics["normalization_scales"]
    count = len(z) - 1
    direct_forward = _transport(
        flow, z[:1].expand(count, -1), ages[:1].expand(count), ages[1:], label.expand(count),
        context.expand(count, -1), context_age.expand(count),
    )
    rollout_forward = []
    current, previous_age = z[:1], ages[:1]
    for index in range(1, len(z)):
        current = _transport(flow, current, previous_age, ages[index:index + 1], label, context, context_age)
        rollout_forward.append(current)
        previous_age = ages[index:index + 1]
    rollout_forward = torch.cat(rollout_forward)

    reverse_targets = z[:-1].flip(0)
    reverse_vertices = reference_vertices[:-1].flip(0)
    direct_backward = _transport(
        flow, z[-1:].expand(count, -1), ages[-1:].expand(count), ages[:-1].flip(0), label.expand(count),
        context.expand(count, -1), context_age.expand(count),
    )
    rollout_backward = []
    current, previous_age = z[-1:], ages[-1:]
    for index in range(len(z) - 2, -1, -1):
        current = _transport(flow, current, previous_age, ages[index:index + 1], label, context, context_age)
        rollout_backward.append(current)
        previous_age = ages[index:index + 1]
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


def _empty_eval_group() -> dict[str, list[float]]:
    return {name: [] for name in (
        "latent", "coordinate", "euclidean", "end_to_end_coordinate_rmse", "end_to_end_coordinate_mae",
        "end_to_end_euclidean", "volume_relative", "rate", "predicted_signed_rate", "observed_signed_rate",
        "nochange_latent", "nochange_coordinate", "nochange_euclidean", "nochange_end_to_end_coordinate_mae",
        "nochange_end_to_end_euclidean", "nochange_volume_relative", "nochange_rate",
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
        chunk = rows[start:start + batch_size]
        batch = indexed(values, C.collate_pairs(chunk))
        prediction = _transport(
            flow, batch["source"], batch["source_age"], batch["target_age"], batch["label"],
            batch["context"], batch["context_age"],
        )
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
            diagnosis: {
                **{f"{name}_mean": float(np.mean(items)) if items else float("nan") for name, items in current.items()},
                "rows": len(current["latent"]),
            }
            for diagnosis, current in grouped.items()
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
    semigroup, inverse_values, identity_values = [], [], []
    scale = float(statistics["normalization_scales"]["displacement"])
    for start in range(0, len(rows), batch_size):
        batch = indexed(values, C.collate_pairs(rows[start:start + batch_size]))
        args = (batch["label"], batch["context"], batch["context_age"])
        direct = flow.transport(batch["source"], batch["source_age"], batch["target_age"], *args)
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(batch["source"], batch["source_age"], middle_age, *args)
        composed = flow.transport(middle, middle_age, batch["target_age"], *args)
        inverse = flow.transport(direct, batch["target_age"], batch["source_age"], *args)
        identity = flow.transport(batch["source"], batch["source_age"], batch["source_age"], *args)
        semigroup.extend((torch.sqrt(torch.mean((direct - composed).square(), dim=1)) / scale).cpu().tolist())
        inverse_values.extend((torch.sqrt(torch.mean((inverse - batch["source"]).square(), dim=1)) / scale).cpu().tolist())
        identity_values.extend((torch.sqrt(torch.mean((identity - batch["source"]).square(), dim=1)) / scale).cpu().tolist())
    return {
        "relative_semigroup_defect_mean": float(np.mean(semigroup)),
        "relative_semigroup_defect_p95": float(np.quantile(semigroup, 0.95)),
        "relative_inverse_defect_mean": float(np.mean(inverse_values)),
        "relative_inverse_defect_p95": float(np.quantile(inverse_values, 0.95)),
        "relative_identity_defect_mean": float(np.mean(identity_values)),
        "relative_identity_defect_p95": float(np.quantile(identity_values, 0.95)),
    }

