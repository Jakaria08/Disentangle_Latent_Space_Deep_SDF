#!/usr/bin/env python3
"""Differentiable exact-SDF objectives and proxy evaluation for INR direct C4."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def _safe(value: torch.Tensor, scale: float) -> torch.Tensor:
    return value / max(float(scale), 1.0e-8)


@torch.no_grad()
def compute_scales(flow, geometry, values: dict[str, torch.Tensor], rows: list[C.PairRow], training: dict[str, Any], limit: int = 64) -> dict[str, float]:
    rows = rows[: min(len(rows), int(limit))]
    batch = C.indexed(values, rows)
    points = min(int(training["field_samples_per_pair"]), batch["target_xyz"].shape[1])
    source_on_target = geometry.sdf(batch["source"], batch["target_xyz"][:, :points], int(training["decoder_point_chunk"]))
    latent = torch.mean((batch["source"] - batch["target"]).square(), dim=1)
    sdf = torch.mean(torch.abs(source_on_target - batch["target_sdf"][:, :points]), dim=1)
    volume = torch.abs(batch["source_volume"] - batch["target_volume"]) / batch["target_volume"].clamp_min(1.0)
    delta = (batch["target_years"] - batch["source_years"]).abs().clamp_min(1.0 / 12.0)
    rate = torch.abs(torch.log(batch["target_volume"].clamp_min(1.0) / batch["source_volume"].clamp_min(1.0))) / delta
    result = {
        "latent": float(torch.median(latent).clamp_min(1.0e-6).cpu()),
        "sdf": float(torch.median(sdf).clamp_min(1.0e-5).cpu()),
        "volume": float(torch.median(volume).clamp_min(1.0e-4).cpu()),
        "rate": float(torch.median(rate).clamp_min(1.0e-4).cpu()),
        "displacement": float(torch.median(torch.sqrt(latent)).clamp_min(1.0e-4).cpu()),
    }
    C.assert_finite_mapping(result)
    return result


def pair_loss(flow, geometry, batch: dict[str, torch.Tensor], weights: dict[str, float], scales: dict[str, float], training: dict[str, Any], consistency_ramp: float, anatomy_ramp: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    source, target = batch["source"], batch["target"]
    source_age, target_age, label = batch["source_age"], batch["target_age"], batch["label"]
    forward = flow.transport(source, source_age, target_age, label)
    backward = flow.transport(target, target_age, source_age, label)
    points = min(int(training["field_samples_per_pair"]), batch["target_xyz"].shape[1])
    forward_sdf = geometry.sdf(forward, batch["target_xyz"][:, :points], int(training["decoder_point_chunk"]))
    backward_sdf = geometry.sdf(backward, batch["source_xyz"][:, :points], int(training["decoder_point_chunk"]))
    latent = 0.5 * (_safe(F.mse_loss(forward, target), scales["latent"]) + _safe(F.mse_loss(backward, source), scales["latent"]))
    sdf = 0.5 * (
        _safe(torch.mean(torch.abs(forward_sdf - batch["target_sdf"][:, :points])), scales["sdf"])
        + _safe(torch.mean(torch.abs(backward_sdf - batch["source_sdf"][:, :points])), scales["sdf"])
    )
    virtual_age = 0.5 * (source_age + target_age)
    virtual = flow.transport(source, source_age, virtual_age, label)
    virtual_composed = flow.transport(virtual, virtual_age, target_age, label)
    virtual_semigroup = _safe(F.mse_loss(virtual_composed, forward), scales["latent"])
    inverse = 0.5 * (
        _safe(F.mse_loss(flow.transport(forward, target_age, source_age, label), source), scales["latent"])
        + _safe(F.mse_loss(flow.transport(backward, source_age, target_age, label), target), scales["latent"])
    )
    observed_semigroup = source.new_zeros(())
    sequence_latent = source.new_zeros(())
    sequence_sdf = source.new_zeros(())
    sequence_semigroup = source.new_zeros(())
    valid = batch["valid_middle"]
    if bool(valid.any()):
        direct = forward[valid]
        first = flow.transport(source[valid], source_age[valid], batch["middle_age"][valid], label[valid])
        composed = flow.transport(first, batch["middle_age"][valid], target_age[valid], label[valid])
        observed_semigroup = _safe(F.mse_loss(composed, direct), scales["latent"])
        sequence_latent = _safe(F.mse_loss(composed, target[valid]), scales["latent"])
        composed_sdf = geometry.sdf(composed, batch["target_xyz"][valid, :points], int(training["decoder_point_chunk"]))
        sequence_sdf = _safe(torch.mean(torch.abs(composed_sdf - batch["target_sdf"][valid, :points])), scales["sdf"])
        sequence_semigroup = observed_semigroup
    predicted_volume = geometry.soft_volume(forward, int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
    predicted_source_volume = geometry.soft_volume(backward, int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
    decoded_source_volume = geometry.soft_volume(source, int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
    decoded_target_volume = geometry.soft_volume(target, int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
    volume = 0.5 * (
        _safe(torch.mean(torch.abs(predicted_volume - decoded_target_volume) / decoded_target_volume.clamp_min(1.0)), scales["volume"])
        + _safe(torch.mean(torch.abs(predicted_source_volume - decoded_source_volume) / decoded_source_volume.clamp_min(1.0)), scales["volume"])
    )
    delta_years = (batch["target_years"] - batch["source_years"]).abs().clamp_min(1.0 / 12.0)
    predicted_rate = torch.log(predicted_volume.clamp_min(1.0) / decoded_source_volume.clamp_min(1.0)) / delta_years
    observed_rate = torch.log(batch["target_volume"].clamp_min(1.0) / batch["source_volume"].clamp_min(1.0)) / delta_years
    rate = _safe(F.smooth_l1_loss(predicted_rate, observed_rate), scales["rate"])
    slope = rate
    group_rate = source.new_zeros(())
    disease_gap = source.new_zeros(())
    groups = []
    for diagnosis in (0.0, 1.0):
        mask = label == diagnosis
        if bool(mask.any()):
            group_rate = group_rate + _safe(torch.abs(predicted_rate[mask].mean() - observed_rate[mask].mean()), scales["rate"])
            groups.append((predicted_rate[mask].mean(), observed_rate[mask].mean()))
    if len(groups) == 2:
        disease_gap = _safe(torch.abs((groups[1][0] - groups[0][0]) - (groups[1][1] - groups[0][1])), scales["rate"])
    bound = 0.5 * (geometry.bound_excess(forward).square().mean() + geometry.bound_excess(backward).square().mean())
    terms = {
        "real_latent": latent,
        "real_sdf": sdf,
        "observed_semigroup": observed_semigroup,
        "virtual_semigroup": virtual_semigroup,
        "inverse": inverse,
        "sequence_latent": sequence_latent,
        "sequence_sdf": sequence_sdf,
        "sequence_semigroup": sequence_semigroup,
        "volume": volume,
        "rate": rate,
        "slope": slope,
        "group_rate": group_rate,
        "disease_gap": disease_gap,
        "latent_bound": bound,
    }
    total = (
        float(weights["real_latent_weight"]) * latent
        + anatomy_ramp * float(weights["real_sdf_weight"]) * sdf
        + consistency_ramp * float(weights["observed_semigroup_weight"]) * observed_semigroup
        + consistency_ramp * float(weights["virtual_semigroup_weight"]) * virtual_semigroup
        + consistency_ramp * float(weights["inverse_weight"]) * inverse
        + float(weights["sequence_latent_weight"]) * sequence_latent
        + anatomy_ramp * float(weights["sequence_sdf_weight"]) * sequence_sdf
        + consistency_ramp * float(weights["sequence_semigroup_weight"]) * sequence_semigroup
        + anatomy_ramp * float(weights["volume_weight"]) * volume
        + anatomy_ramp * float(weights["rate_weight"]) * rate
        + anatomy_ramp * float(weights["slope_weight"]) * slope
        + anatomy_ramp * float(weights["group_rate_weight"]) * group_rate
        + anatomy_ramp * float(weights["disease_gap_weight"]) * disease_gap
        + float(weights["latent_bound_weight"]) * bound
    )
    return total, terms | {"total": total}


def _empty_group() -> dict[str, list[float]]:
    return {key: [] for key in ("latent", "sdf", "nochange_latent", "nochange_sdf", "volume_relative", "nochange_volume_relative", "end_to_end_volume_relative", "representation_floor_volume_relative", "bound_excess")}


@torch.no_grad()
def evaluate_pairs(flow, geometry, values: dict[str, torch.Tensor], rows: list[C.PairRow], training: dict[str, Any], batch_size: int, include_rows: bool = False) -> dict[str, Any]:
    groups = {name: _empty_group() for name in ("CN", "AD", "overall")}
    records = []
    for start in range(0, len(rows), int(batch_size)):
        current = rows[start:start + int(batch_size)]
        batch = C.indexed(values, current)
        prediction = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        points = min(int(training["field_samples_per_pair"]), batch["target_xyz"].shape[1])
        predicted_sdf = geometry.sdf(prediction, batch["target_xyz"][:, :points], int(training["decoder_point_chunk"]))
        source_sdf = geometry.sdf(batch["source"], batch["target_xyz"][:, :points], int(training["decoder_point_chunk"]))
        predicted_volume = geometry.soft_volume(prediction, int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
        decoded_source_volume = geometry.soft_volume(batch["source"], int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
        decoded_target_volume = geometry.soft_volume(batch["target"], int(training["volume_samples"]), float(training["volume_temperature"]), int(training["decoder_point_chunk"]))
        metrics = {
            "latent": torch.sqrt(torch.mean((prediction - batch["target"]).square(), dim=1)),
            "sdf": torch.mean(torch.abs(predicted_sdf - batch["target_sdf"][:, :points]), dim=1),
            "nochange_latent": torch.sqrt(torch.mean((batch["source"] - batch["target"]).square(), dim=1)),
            "nochange_sdf": torch.mean(torch.abs(source_sdf - batch["target_sdf"][:, :points]), dim=1),
            "volume_relative": torch.abs(predicted_volume - decoded_target_volume) / decoded_target_volume.clamp_min(1.0),
            "nochange_volume_relative": torch.abs(decoded_source_volume - decoded_target_volume) / decoded_target_volume.clamp_min(1.0),
            "end_to_end_volume_relative": torch.abs(predicted_volume - batch["target_volume"]) / batch["target_volume"].clamp_min(1.0),
            "representation_floor_volume_relative": torch.abs(decoded_target_volume - batch["target_volume"]) / batch["target_volume"].clamp_min(1.0),
            "bound_excess": geometry.bound_excess(prediction),
        }
        for index, row in enumerate(current):
            record = {"subject": row.subject, "diagnosis": row.diagnosis, "source_index": row.source, "target_index": row.target, "pair_type": row.pair_type}
            for name, tensor in metrics.items():
                value = float(tensor[index].cpu())
                record[name] = value
                groups[row.diagnosis][name].append(value)
                groups["overall"][name].append(value)
            if include_rows:
                records.append(record)
    summary = {"pairs": len(rows), "groups": {}}
    for name, metrics in groups.items():
        summary["groups"][name] = {f"{key}_mean": float(np.mean(items)) if items else None for key, items in metrics.items()} | {"pairs": len(metrics["latent"])}
    if include_rows:
        summary["row_metrics"] = records
    return summary


@torch.no_grad()
def cocycle_defects(flow, values: dict[str, torch.Tensor], rows: list[C.PairRow], displacement_scale: float, batch_size: int) -> dict[str, float]:
    semigroup, inverse = [], []
    for start in range(0, len(rows), int(batch_size)):
        batch = C.indexed(values, rows[start:start + int(batch_size)])
        direct = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(batch["source"], batch["source_age"], middle_age, batch["label"])
        composed = flow.transport(middle, middle_age, batch["target_age"], batch["label"])
        restored = flow.transport(direct, batch["target_age"], batch["source_age"], batch["label"])
        semigroup.extend((torch.sqrt(torch.mean((direct - composed).square(), dim=1)) / max(displacement_scale, 1.0e-8)).cpu().tolist())
        inverse.extend((torch.sqrt(torch.mean((restored - batch["source"]).square(), dim=1)) / max(displacement_scale, 1.0e-8)).cpu().tolist())
    return {
        "relative_semigroup_defect_mean": float(np.mean(semigroup)),
        "relative_semigroup_defect_p95": float(np.quantile(semigroup, 0.95)),
        "relative_inverse_defect_mean": float(np.mean(inverse)),
        "relative_inverse_defect_p95": float(np.quantile(inverse, 0.95)),
    }


def validation_score(all_pairs: dict[str, Any], first_last: dict[str, Any], defects: dict[str, float], selection: dict[str, Any]) -> tuple[float, bool, dict[str, float]]:
    macro_sdf, macro_latent, macro_volume = [], [], []
    feasible = True
    for diagnosis in ("CN", "AD"):
        values = first_last["groups"][diagnosis]
        if not values["pairs"]:
            raise ValueError(f"Validation has no {diagnosis} first-last pairs")
        sdf_ratio = values["sdf_mean"] / max(values["nochange_sdf_mean"], 1.0e-8)
        latent_ratio = values["latent_mean"] / max(values["nochange_latent_mean"], 1.0e-8)
        volume_ratio = values["volume_relative_mean"] / max(values["nochange_volume_relative_mean"], 1.0e-8)
        macro_sdf.append(sdf_ratio)
        macro_latent.append(latent_ratio)
        macro_volume.append(volume_ratio)
        feasible = feasible and sdf_ratio <= 1.0 + float(selection["sdf_nochange_tolerance"])
        feasible = feasible and latent_ratio <= 1.0 + float(selection["latent_nochange_tolerance"])
    overall = all_pairs["groups"]["overall"]
    all_sdf = overall["sdf_mean"] / max(overall["nochange_sdf_mean"], 1.0e-8)
    all_volume = overall["volume_relative_mean"] / max(overall["nochange_volume_relative_mean"], 1.0e-8)
    score = float(np.mean(macro_sdf)) + float(selection["all_pair_shape_weight"]) * all_sdf + float(selection["volume_tiebreak_weight"]) * all_volume
    feasible = feasible and defects["relative_semigroup_defect_mean"] <= float(selection["max_relative_semigroup_defect"])
    feasible = feasible and defects["relative_inverse_defect_mean"] <= float(selection["max_relative_inverse_defect"])
    ratios = {"macro_first_last_sdf": float(np.mean(macro_sdf)), "macro_first_last_latent": float(np.mean(macro_latent)), "macro_first_last_volume": float(np.mean(macro_volume)), "all_pair_sdf": float(all_sdf), "all_pair_volume": float(all_volume), "score": float(score)}
    C.assert_finite_mapping(ratios)
    return score, bool(feasible and math.isfinite(score)), ratios
