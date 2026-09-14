#!/usr/bin/env python3
"""Reduced latent-cocycle objective.

The published direct_c4 objective carries thirteen weighted terms. Three groups of them
are algebraically redundant rather than merely small:

  * ``real_latent`` / ``sequence_latent`` duplicate the decoded-vertex terms (for the PCA
    representation the decoder is affine, so they are the same quantity in another metric);
  * ``observed_semigroup``, ``virtual_semigroup``, ``sequence_semigroup`` and ``inverse``
    are one law, Phi(Phi(z,s,r),r,t) = Phi(z,s,t), under four samplers of the split point r;
  * ``volume``, ``rate``, ``slope``, ``group_rate`` and ``disease_gap`` are five functionals
    of a single decoded log-volume trajectory.

This module keeps one representative per group and folds the four composition penalties
into a single ``comp`` term with a configurable mixture of families. Shared machinery
(statistics, samplers, ``shape_terms``, and every evaluation function) is imported from
the baseline ``c4_objective`` so measurement stays identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOSS_SETS: dict[str, tuple[str, ...]] = {
    "full13": ("baseline",),
    "lean5": ("vtx", "lat", "seq_vtx", "comp", "rate"),
    "lean4": ("vtx", "seq_vtx", "comp", "rate"),
    "lean4t": ("vtx", "seq_vtx", "comp", "rate", "trend"),
    "lean3": ("vtx", "comp", "rate"),
}
COMP_FAMILIES = ("virtual", "observed", "inverse", "rollout")


def ramp(step: int, total: int) -> float:
    """Linear warm-up expressed in optimizer steps, not epochs."""
    return 1.0 if total <= 0 else min(1.0, float(step) / float(total))


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def pair_terms(
    flow: nn.Module,
    geometry: Any,
    values: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    statistics: dict[str, Any],
    O: Any,
    families: tuple[str, ...] = COMP_FAMILIES,
    need_trend: bool = False,
) -> dict[str, torch.Tensor]:
    batch = O.indexed(values, raw)
    scales = statistics["normalization_scales"]
    latent_scale = scales["latent"]
    source, target = batch["source"], batch["target"]
    source_age, target_age, label = batch["source_age"], batch["target_age"], batch["label"]

    forward = flow.transport(source, source_age, target_age, label)
    backward = flow.transport(target, target_age, source_age, label)
    shape_f = O.shape_terms(forward, target, batch["target_vertices"], geometry, scales)
    shape_b = O.shape_terms(backward, source, batch["source_vertices"], geometry, scales)

    # ---- one composition law, several samplers of the split point -------------------
    parts: list[torch.Tensor] = []
    if "virtual" in families:
        ratio = torch.empty_like(source_age).uniform_(0.2, 0.8)
        middle_age = source_age + ratio * (target_age - source_age)
        composed_f = flow.transport(
            flow.transport(source, source_age, middle_age, label), middle_age, target_age, label
        )
        composed_b = flow.transport(
            flow.transport(target, target_age, middle_age, label), middle_age, source_age, label
        )
        parts.append(
            0.5
            * (
                O.mean_scaled_mse(forward, composed_f, latent_scale)
                + O.mean_scaled_mse(backward, composed_b, latent_scale)
            )
        )
    if "observed" in families:
        valid = batch["intermediate_index"] >= 0
        if bool(valid.any()):
            middle_age = values["age"][batch["intermediate_index"][valid]]
            sub_label = label[valid]
            composed_f = flow.transport(
                flow.transport(source[valid], source_age[valid], middle_age, sub_label),
                middle_age,
                target_age[valid],
                sub_label,
            )
            composed_b = flow.transport(
                flow.transport(target[valid], target_age[valid], middle_age, sub_label),
                middle_age,
                source_age[valid],
                sub_label,
            )
            parts.append(
                0.5
                * (
                    O.mean_scaled_mse(forward[valid], composed_f, latent_scale)
                    + O.mean_scaled_mse(backward[valid], composed_b, latent_scale)
                )
            )
    if "inverse" in families:
        parts.append(
            0.5
            * (
                O.mean_scaled_mse(
                    flow.transport(forward, target_age, source_age, label), source, latent_scale
                )
                + O.mean_scaled_mse(
                    flow.transport(backward, source_age, target_age, label), target, latent_scale
                )
            )
        )
    comp = torch.stack(parts).mean() if parts else _zero(forward)

    # ---- one decoded log-volume functional ------------------------------------------
    predicted_forward_volume = geometry.volume_from_vertices(geometry.vertices(forward))
    predicted_backward_volume = geometry.volume_from_vertices(geometry.vertices(backward))
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    observed_rate = (torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])) / years
    rate_forward = (torch.log(predicted_forward_volume) - torch.log(batch["source_volume"])) / years
    rate_backward = (torch.log(predicted_backward_volume) - torch.log(batch["target_volume"])) / (-years)
    rate_scale = float(scales["rate"])
    rate = 0.5 * (
        F.smooth_l1_loss((rate_forward - observed_rate) / rate_scale, torch.zeros_like(rate_forward))
        + F.smooth_l1_loss((rate_backward - observed_rate) / rate_scale, torch.zeros_like(rate_backward))
    )

    trend = _zero(forward)
    if need_trend:
        is_ad = label >= 0.5
        if bool(is_ad.any()) and bool((~is_ad).any()):
            device = rate_forward.device
            zero = torch.zeros((), device=device)
            pieces = []
            for diagnosis, mask in (("CN", ~is_ad), ("AD", is_ad)):
                goal = torch.tensor(
                    float(statistics["group_log_volume_rate_targets"][diagnosis]), device=device
                )
                pieces.append(
                    F.smooth_l1_loss((rate_forward[mask].mean() - goal) / rate_scale, zero)
                )
            goal_gap = torch.tensor(
                float(statistics["ad_minus_cn_log_volume_rate_target"]), device=device
            )
            gap = (rate_forward[is_ad].mean() - rate_forward[~is_ad].mean() - goal_gap) / rate_scale
            trend = torch.stack(pieces).mean() + F.smooth_l1_loss(gap, zero)

    return {
        "vtx": 0.5 * (shape_f["vertex"] + shape_b["vertex"]),
        "lat": 0.5 * (shape_f["latent"] + shape_b["latent"]),
        "comp": comp,
        "rate": rate,
        "trend": trend,
    }


def sequence_terms(
    flow: nn.Module,
    geometry: Any,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    start: int,
    statistics: dict[str, Any],
    O: Any,
    with_rollout_comp: bool = True,
) -> dict[str, torch.Tensor]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_index = int(np.searchsorted(offsets, int(start), side="right") - 1)
    end = int(offsets[subject_index + 1])
    z = values["z"][start:end]
    ages = values["age"][start:end]
    reference_vertices = values["reference_vertices"][start:end]
    label = values["label"][start : start + 1]
    if len(z) < 2:
        raise ValueError("Sequence needs at least two visits")
    scales = statistics["normalization_scales"]
    count = len(z) - 1

    direct_forward = flow.transport(
        z[:1].expand(count, -1), ages[:1].expand(count), ages[1:], label.expand(count)
    )
    rollout_forward = []
    current, previous_age = z[:1], ages[:1]
    for index in range(1, len(z)):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
        rollout_forward.append(current)
        previous_age = ages[index : index + 1]
    rollout_forward = torch.cat(rollout_forward)

    reverse_targets = z[:-1].flip(0)
    reverse_vertices = reference_vertices[:-1].flip(0)
    direct_backward = flow.transport(
        z[-1:].expand(count, -1), ages[-1:].expand(count), ages[:-1].flip(0), label.expand(count)
    )
    rollout_backward = []
    current, previous_age = z[-1:], ages[-1:]
    for index in range(len(z) - 2, -1, -1):
        current = flow.transport(current, previous_age, ages[index : index + 1], label)
        rollout_backward.append(current)
        previous_age = ages[index : index + 1]
    rollout_backward = torch.cat(rollout_backward)

    shapes = (
        O.shape_terms(direct_forward, z[1:], reference_vertices[1:], geometry, scales),
        O.shape_terms(rollout_forward, z[1:], reference_vertices[1:], geometry, scales),
        O.shape_terms(direct_backward, reverse_targets, reverse_vertices, geometry, scales),
        O.shape_terms(rollout_backward, reverse_targets, reverse_vertices, geometry, scales),
    )
    output = {
        "seq_vtx": torch.stack([item["vertex"] for item in shapes]).mean(),
        "seq_lat": torch.stack([item["latent"] for item in shapes]).mean(),
    }
    output["seq_comp"] = (
        0.5
        * (
            O.mean_scaled_mse(direct_forward, rollout_forward, scales["latent"])
            + O.mean_scaled_mse(direct_backward, rollout_backward, scales["latent"])
        )
        if with_rollout_comp
        else _zero(direct_forward)
    )
    return output


def total_loss(
    pair: dict[str, torch.Tensor],
    sequence: dict[str, torch.Tensor] | None,
    weights: dict[str, float],
    active: tuple[str, ...],
    comp_ramp: float,
    anatomy_ramp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble the reduced objective.

    ``comp`` merges the pair-level composition families with the sequence direct-versus-
    rollout defect, so the single reported term covers every instance of the cocycle law.
    """
    parts: dict[str, torch.Tensor] = {}
    if "vtx" in active:
        parts["vtx"] = float(weights["vtx"]) * pair["vtx"]
    if "lat" in active:
        parts["lat"] = float(weights["lat"]) * pair["lat"]
    if "seq_vtx" in active and sequence is not None:
        parts["seq_vtx"] = comp_ramp * float(weights["seq_vtx"]) * sequence["seq_vtx"]
    if "comp" in active:
        merged = pair["comp"]
        if sequence is not None and "seq_comp" in sequence:
            merged = 0.5 * (merged + sequence["seq_comp"])
        parts["comp"] = comp_ramp * float(weights["comp"]) * merged
    if "rate" in active:
        parts["rate"] = anatomy_ramp * float(weights["rate"]) * pair["rate"]
    if "trend" in active:
        parts["trend"] = anatomy_ramp * float(weights["trend"]) * pair["trend"]
    total = torch.stack(list(parts.values())).sum()
    report = {f"w_{name}": float(value.detach().cpu()) for name, value in parts.items()}
    report.update({f"raw_{name}": float(value.detach().cpu()) for name, value in pair.items()})
    if sequence is not None:
        report.update({f"raw_{name}": float(value.detach().cpu()) for name, value in sequence.items()})
    report["total"] = float(total.detach().cpu())
    report["comp_ramp"] = comp_ramp
    report["anatomy_ramp"] = anatomy_ramp
    return total, report
