#!/usr/bin/env python3
"""Volume-aware objective and selector for exact coboundary V2."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


AUGUST_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
V1_SCRIPTS = Path(__file__).resolve().parents[2] / "coboundary_v1" / "scripts"
for path in (V1_SCRIPTS, AUGUST_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common as C  # noqa: E402
import coboundary_objective as V1  # noqa: E402


attach_reference_geometry = V1.attach_reference_geometry
indexed = V1.indexed
balanced_subset = V1.balanced_subset
balanced_first_last_subset = V1.balanced_first_last_subset
training_statistics = V1.training_statistics
balanced_pair_sampler = V1.balanced_pair_sampler
balanced_sequence_starts = V1.balanced_sequence_starts
sequence_terms = V1.sequence_terms
evaluate_pairs = V1.evaluate_pairs
cocycle_defects = V1.cocycle_defects


def ramp(epoch: int, epochs: int) -> float:
    return 1.0 if epochs <= 0 else min(1.0, float(epoch) / float(epochs))


@torch.no_grad()
def fit_train_volume_axis(
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    ridge_alpha: float,
) -> dict[str, Any]:
    """Subject-balanced train-only ridge fit of decoded log-volume on latent."""

    latent = values["z"].detach().cpu().double().numpy()
    target = torch.log(values["reference_volume"]).detach().cpu().double().numpy()
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    weights = np.empty(len(latent), dtype=np.float64)
    for index in range(len(offsets) - 1):
        first, last = int(offsets[index]), int(offsets[index + 1])
        weights[first:last] = 1.0 / float(last - first)
    weights *= len(weights) / weights.sum()
    x_mean = np.average(latent, axis=0, weights=weights)
    y_mean = float(np.average(target, weights=weights))
    centered_x, centered_y = latent - x_mean, target - y_mean
    weighted_x = centered_x * np.sqrt(weights[:, None])
    weighted_y = centered_y * np.sqrt(weights)
    gram = weighted_x.T @ weighted_x + float(ridge_alpha) * np.eye(latent.shape[1], dtype=np.float64)
    coefficient = np.linalg.solve(gram, weighted_x.T @ weighted_y)
    intercept = y_mean - float(x_mean @ coefficient)
    prediction = intercept + latent @ coefficient
    residual = target - prediction
    denominator = float(np.sum(weights * (target - y_mean) ** 2))
    r2 = 1.0 - float(np.sum(weights * residual**2)) / max(denominator, 1.0e-12)
    rmse = math.sqrt(float(np.sum(weights * residual**2) / np.sum(weights)))
    norm = float(np.linalg.norm(coefficient))
    if not np.isfinite(coefficient).all() or not math.isfinite(norm) or norm <= 1.0e-10:
        raise RuntimeError("Invalid train-only decoded-volume direction")
    return {
        "coefficient": coefficient.astype(np.float32).tolist(),
        "coefficient_norm": norm,
        "intercept": intercept,
        "subject_balanced_weighted_r2": r2,
        "subject_balanced_log_volume_rmse": rmse,
        "ridge_alpha": float(ridge_alpha),
        "visits": len(latent),
        "subjects": len(offsets) - 1,
        "source": "train split only; frozen-decoder log-volume; inverse-visit-count subject weights",
    }


def pair_terms(flow, geometry, values, raw, statistics: dict[str, Any]) -> dict[str, torch.Tensor]:
    output = V1.pair_terms(flow, geometry, values, raw, statistics)
    batch = indexed(values, raw)
    scales = statistics["normalization_scales"]
    source_potential = flow.volume_potential(
        batch["context"], batch["context_age"], batch["source_age"], batch["label"]
    ).reshape(-1)
    target_potential = flow.volume_potential(
        batch["context"], batch["context_age"], batch["target_age"], batch["label"]
    ).reshape(-1)
    axis_delta = target_potential - source_potential
    observed_delta = torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    axis_rate = axis_delta / years
    observed_rate = observed_delta / years
    axis_volume = F.smooth_l1_loss(
        (axis_delta - observed_delta) / float(scales["volume_log"]),
        torch.zeros_like(axis_delta),
    )
    axis_rate_loss = F.smooth_l1_loss(
        (axis_rate - observed_rate) / float(scales["rate"]),
        torch.zeros_like(axis_rate),
    )
    is_ad = batch["label"] >= 0.5
    zero = axis_rate.sum() * 0.0
    axis_group_rate, axis_disease_gap, group_order = zero, zero, zero
    if bool(is_ad.any()) and bool((~is_ad).any()):
        group_losses = []
        means = {}
        for diagnosis, mask in (("CN", ~is_ad), ("AD", is_ad)):
            means[diagnosis] = axis_rate[mask].mean()
            target = torch.tensor(
                float(statistics["group_log_volume_rate_targets"][diagnosis]), device=axis_rate.device
            )
            group_losses.append(F.smooth_l1_loss(
                (means[diagnosis] - target) / float(scales["rate"]),
                torch.zeros((), device=axis_rate.device),
            ))
        axis_group_rate = torch.stack(group_losses).mean()
        target_gap = torch.tensor(
            float(statistics["ad_minus_cn_log_volume_rate_target"]), device=axis_rate.device
        )
        axis_disease_gap = F.smooth_l1_loss(
            ((means["AD"] - means["CN"]) - target_gap) / float(scales["rate"]),
            torch.zeros((), device=axis_rate.device),
        )
        group_order = (
            F.smooth_l1_loss(
                torch.relu(means["CN"]) / float(scales["rate"]),
                torch.zeros((), device=axis_rate.device),
            )
            + F.smooth_l1_loss(
                torch.relu(means["AD"] - means["CN"]) / float(scales["rate"]),
                torch.zeros((), device=axis_rate.device),
            )
        )
    output.update({
        "axis_volume": axis_volume,
        "axis_rate": axis_rate_loss,
        "axis_group_rate": axis_group_rate,
        "axis_disease_gap": axis_disease_gap,
        "axis_group_order": group_order,
    })
    return output


def blend_pair_terms(
    ordinary: dict[str, torch.Tensor],
    long_horizon: dict[str, torch.Tensor],
    long_horizon_fraction: float,
) -> dict[str, torch.Tensor]:
    fraction = float(long_horizon_fraction)
    if not 0.0 <= fraction <= 1.0 or set(ordinary) != set(long_horizon):
        raise ValueError("Invalid long-horizon pair blend")
    return {
        name: (1.0 - fraction) * ordinary[name] + fraction * long_horizon[name]
        for name in ordinary
    }


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
            float(weights["sequence_latent_weight"]) * sequence["sequence_latent"]
            + float(weights["sequence_vertex_weight"]) * sequence["sequence_vertex"]
        )
        + anatomy * (
            float(weights["volume_weight"]) * pair["volume"]
            + float(weights["rate_weight"]) * pair["rate"]
            + float(weights["slope_weight"]) * sequence["slope"]
            + float(weights["group_rate_weight"]) * pair["group_rate"]
            + float(weights["disease_gap_weight"]) * pair["disease_gap"]
            + float(weights["axis_volume_weight"]) * pair["axis_volume"]
            + float(weights["axis_rate_weight"]) * pair["axis_rate"]
            + float(weights["axis_group_rate_weight"]) * pair["axis_group_rate"]
            + float(weights["axis_disease_gap_weight"]) * pair["axis_disease_gap"]
            + float(weights["axis_group_order_weight"]) * pair["axis_group_order"]
        )
    )
    terms = pair | sequence | {
        "total": total,
        "consistency_factor": torch.tensor(consistency, device=total.device),
        "anatomy_factor": torch.tensor(anatomy, device=total.device),
    }
    return total, {name: float(value.detach().cpu()) for name, value in terms.items()}


def validation_score(
    all_pairs: dict[str, Any],
    first_last: dict[str, Any],
    defects: dict[str, float],
    selection: dict[str, Any],
) -> tuple[float, bool, dict[str, float]]:
    shape, volume, rate, trend = [], [], [], []
    signed = {}
    feasible = True
    for diagnosis in ("CN", "AD"):
        values = first_last["groups"][diagnosis]
        coordinate_ratio = values["coordinate_mean"] / max(values["nochange_coordinate_mean"], 1.0e-8)
        euclidean_ratio = values["euclidean_mean"] / max(values["nochange_euclidean_mean"], 1.0e-8)
        latent_ratio = values["latent_mean"] / max(values["nochange_latent_mean"], 1.0e-8)
        volume_ratio = values["volume_relative_mean"] / max(values["nochange_volume_relative_mean"], 1.0e-8)
        rate_ratio = values["rate_mean"] / max(values["nochange_rate_mean"], 1.0e-8)
        predicted = float(values["predicted_signed_rate_mean"])
        observed = float(values["observed_signed_rate_mean"])
        shape.append(0.5 * (coordinate_ratio + euclidean_ratio))
        volume.append(volume_ratio)
        rate.append(rate_ratio)
        trend.append(abs(predicted - observed) / max(values["nochange_rate_mean"], 1.0e-8))
        signed[diagnosis] = (predicted, observed)
        feasible = feasible and latent_ratio <= 1.0 + float(selection["latent_nochange_tolerance"])
        feasible = feasible and coordinate_ratio <= 1.0 + float(selection["coordinate_nochange_tolerance"])
    overall = all_pairs["groups"]["overall"]
    all_shape = 0.5 * (
        overall["coordinate_mean"] / max(overall["nochange_coordinate_mean"], 1.0e-8)
        + overall["euclidean_mean"] / max(overall["nochange_euclidean_mean"], 1.0e-8)
    )
    all_volume = overall["volume_relative_mean"] / max(overall["nochange_volume_relative_mean"], 1.0e-8)
    predicted_gap = signed["AD"][0] - signed["CN"][0]
    observed_gap = signed["AD"][1] - signed["CN"][1]
    gap_ratio = abs(predicted_gap - observed_gap) / max(abs(observed_gap), 1.0e-8)
    ratios = {
        "macro_first_last_shape": float(np.mean(shape)),
        "all_pair_shape": float(all_shape),
        "all_pair_volume": float(all_volume),
        "macro_first_last_volume": float(np.mean(volume)),
        "macro_first_last_rate": float(np.mean(rate)),
        "macro_group_trend": float(np.mean(trend)),
        "diagnosis_gap_error_ratio": float(gap_ratio),
        "first_last_cn_predicted_signed_rate": signed["CN"][0],
        "first_last_cn_observed_signed_rate": signed["CN"][1],
        "first_last_ad_predicted_signed_rate": signed["AD"][0],
        "first_last_ad_observed_signed_rate": signed["AD"][1],
    }
    score = (
        ratios["macro_first_last_shape"]
        + float(selection["all_pair_shape_weight"]) * ratios["all_pair_shape"]
        + float(selection["volume_weight"]) * ratios["macro_first_last_volume"]
        + float(selection["rate_weight"]) * ratios["macro_first_last_rate"]
        + float(selection["trend_weight"]) * ratios["macro_group_trend"]
        + float(selection["diagnosis_gap_weight"]) * ratios["diagnosis_gap_error_ratio"]
    )
    ratios["score"] = float(score)
    feasible = feasible and math.isfinite(score)
    feasible = feasible and defects["relative_semigroup_defect_mean"] <= float(selection["max_relative_semigroup_defect"])
    feasible = feasible and defects["relative_inverse_defect_mean"] <= float(selection["max_relative_inverse_defect"])
    C.assert_finite_mapping(ratios)
    return float(score), bool(feasible), ratios
