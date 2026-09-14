#!/usr/bin/env python3
"""Latent-preserving compact objectives built from the original 13 losses."""

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

OBJECTIVES = {
    "full13": {
        "top_level_count": 13,
        "active_leaf_count": 13,
        "omitted_leaves": (),
    },
    "compact6": {
        "top_level_count": 6,
        "active_leaf_count": 12,
        "omitted_leaves": ("slope",),
    },
    "compact5": {
        "top_level_count": 5,
        "active_leaf_count": 10,
        "omitted_leaves": ("slope", "group_rate", "disease_gap"),
    },
}
LOSS_COUNTS = {name: int(meta["top_level_count"]) for name, meta in OBJECTIVES.items()}
ACTIVE_LEAF_COUNTS = {
    name: int(meta["active_leaf_count"]) for name, meta in OBJECTIVES.items()
}


def ramp(step: int, ramp_steps: int) -> float:
    return 1.0 if ramp_steps <= 0 else min(1.0, float(step) / float(ramp_steps))


def weighted_leaves(
    pair: Mapping[str, torch.Tensor],
    sequence: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    consistency_factor: float,
    anatomy_factor: float,
) -> OrderedDict[str, torch.Tensor]:
    """Apply the unchanged task3/task9 leaf weights."""
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
            (
                "inverse",
                consistency_factor * float(weights["inverse"]) * pair["inverse"],
            ),
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
    objective: str,
    consistency_factor: float = 1.0,
    anatomy_factor: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    """Return the top-level optimization groups for one objective.

    Every arm computes all 13 raw quantities so data sampling and random-number use stay
    matched. ``compact6`` keeps all latent prediction terms and omits only the sequence
    slope. ``compact5`` additionally omits the two population-trend leaves.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}")
    leaf = weighted_leaves(
        pair, sequence, weights, consistency_factor, anatomy_factor
    )
    if objective == "full13":
        return leaf

    groups = OrderedDict(
        (
            (
                "endpoint_prediction",
                leaf["real_vertex"] + leaf["real_latent"],
            ),
            (
                "sequence_prediction",
                leaf["sequence_vertex"] + leaf["sequence_latent"],
            ),
            (
                "composition",
                leaf["observed_semigroup"]
                + leaf["virtual_semigroup"]
                + leaf["inverse"]
                + leaf["sequence_semigroup"],
            ),
            ("volume", leaf["volume"]),
            ("individual_rate", leaf["rate"]),
        )
    )
    if objective == "compact6":
        groups["population_trend"] = leaf["group_rate"] + leaf["disease_gap"]
    return groups


def total_loss(
    pair: Mapping[str, torch.Tensor],
    sequence: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    objective: str,
    consistency_factor: float = 1.0,
    anatomy_factor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    groups = loss_groups(
        pair, sequence, weights, objective, consistency_factor, anatomy_factor
    )
    total = torch.stack(tuple(groups.values())).sum()
    report = {
        f"weighted_group_{key}": float(value.detach().cpu())
        for key, value in groups.items()
    }
    report.update(
        {f"raw_{key}": float(value.detach().cpu()) for key, value in pair.items()}
    )
    report.update(
        {f"raw_{key}": float(value.detach().cpu()) for key, value in sequence.items()}
    )
    report.update(
        {
            "total": float(total.detach().cpu()),
            "top_level_loss_count": LOSS_COUNTS[objective],
            "active_leaf_count": ACTIVE_LEAF_COUNTS[objective],
            "consistency_factor": float(consistency_factor),
            "anatomy_factor": float(anatomy_factor),
        }
    )
    return total, report
