#!/usr/bin/env python3
"""Matched full and reduced objectives built from the original 13 raw losses."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

LEAF_NAMES = (
    "real_latent",
    "real_vertex",
    "observed_semigroup",
    "virtual_semigroup",
    "inverse",
    "sequence_latent",
    "sequence_vertex",
    "sequence_semigroup",
    "volume",
    "rate",
    "slope",
    "group_rate",
    "disease_gap",
)
LOSS_COUNTS = {"full13": 13, "lean6": 6, "lean5": 5, "lean4": 4}


def ramp(step: int, ramp_steps: int) -> float:
    return 1.0 if ramp_steps <= 0 else min(1.0, float(step) / float(ramp_steps))


def _weighted_leaves(
    pair: Mapping[str, torch.Tensor],
    sequence: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    consistency_factor: float,
    anatomy_factor: float,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("real_latent", float(weights["real_latent"]) * pair["real_latent"]),
            ("real_vertex", float(weights["real_vertex"]) * pair["real_vertex"]),
            (
                "observed_semigroup",
                consistency_factor
                * float(weights["observed_semigroup"])
                * pair["observed_semigroup"],
            ),
            (
                "virtual_semigroup",
                consistency_factor
                * float(weights["virtual_semigroup"])
                * pair["virtual_semigroup"],
            ),
            ("inverse", consistency_factor * float(weights["inverse"]) * pair["inverse"]),
            (
                "sequence_latent",
                consistency_factor
                * float(weights["sequence_latent"])
                * sequence["sequence_latent"],
            ),
            (
                "sequence_vertex",
                consistency_factor
                * float(weights["sequence_vertex"])
                * sequence["sequence_vertex"],
            ),
            (
                "sequence_semigroup",
                consistency_factor
                * float(weights["sequence_semigroup"])
                * sequence["sequence_semigroup"],
            ),
            ("volume", anatomy_factor * float(weights["volume"]) * pair["volume"]),
            ("rate", anatomy_factor * float(weights["rate"]) * pair["rate"]),
            ("slope", anatomy_factor * float(weights["slope"]) * sequence["slope"]),
            (
                "group_rate",
                anatomy_factor * float(weights["group_rate"]) * pair["group_rate"],
            ),
            (
                "disease_gap",
                anatomy_factor * float(weights["disease_gap"]) * pair["disease_gap"],
            ),
        )
    )


def loss_groups(
    pair: Mapping[str, torch.Tensor],
    sequence: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    loss_set: str,
    consistency_factor: float = 1.0,
    anatomy_factor: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    """Return the top-level optimization terms for one configured objective.

    All arms compute the same 13 raw values. This deliberately keeps random-number use,
    data sampling, and diagnostics matched. The reduced arms differ only in which weighted
    values are summed and how related values are named at the top level.
    """
    if loss_set not in LOSS_COUNTS:
        raise ValueError(f"Unknown loss set {loss_set!r}")
    leaf = _weighted_leaves(pair, sequence, weights, consistency_factor, anatomy_factor)
    if loss_set == "full13":
        return leaf

    composition = (
        leaf["observed_semigroup"]
        + leaf["virtual_semigroup"]
        + leaf["inverse"]
        + leaf["sequence_semigroup"]
    )
    common = OrderedDict(
        (
            ("vertex", leaf["real_vertex"]),
            ("sequence_vertex", leaf["sequence_vertex"]),
            ("composition", composition),
            ("rate", leaf["rate"]),
        )
    )
    if loss_set == "lean4":
        return common
    common["group_trend"] = leaf["group_rate"] + leaf["disease_gap"]
    if loss_set == "lean5":
        return common
    common["volume"] = leaf["volume"]
    return common


def total_loss(
    pair: Mapping[str, torch.Tensor],
    sequence: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    loss_set: str,
    consistency_factor: float = 1.0,
    anatomy_factor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    groups = loss_groups(
        pair, sequence, weights, loss_set, consistency_factor, anatomy_factor
    )
    total = torch.stack(tuple(groups.values())).sum()
    report = {f"weighted_{key}": float(value.detach().cpu()) for key, value in groups.items()}
    report.update({f"raw_{key}": float(value.detach().cpu()) for key, value in pair.items()})
    report.update({f"raw_{key}": float(value.detach().cpu()) for key, value in sequence.items()})
    report.update(
        {
            "total": float(total.detach().cpu()),
            "consistency_factor": float(consistency_factor),
            "anatomy_factor": float(anatomy_factor),
        }
    )
    return total, report
