#!/usr/bin/env python3
"""Experiment configuration expansion and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from objective import LOSS_COUNTS

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = TASK_ROOT / "configs" / "experiment.json"


def read_experiment(path: Path = DEFAULT_EXPERIMENT) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    validate_experiment(value)
    return value


def validate_experiment(config: dict[str, Any]) -> None:
    if config.get("loss_sets") != ["full13", "lean6", "lean5", "lean4"]:
        raise ValueError("Expected the preregistered full13/lean6/lean5/lean4 order")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be unique")
    if set(config["loss_sets"]) != set(LOSS_COUNTS):
        raise ValueError("Configured and implemented loss sets differ")
    if int(config["model"]["latent_dim"]) != 128:
        raise ValueError("This experiment is fixed to a 128-D latent-only network")
    training = config["training"]
    if int(training["warmup_steps"]) >= int(training["epochs"]) * int(
        training["steps_per_epoch"]
    ):
        raise ValueError("Warmup must finish before training ends")
    for name in config["representations"]:
        for key in (
            "batch_size_by_representation",
            "decoder_batch_size_by_representation",
            "evaluation_batch_size_by_representation",
        ):
            if int(training[key][name]) <= 0:
                raise ValueError(f"Missing positive {key}.{name}")
    contract = config["scientific_contract"]
    required_true = (
        "latent_only",
        "direct_non_ode",
        "architecture_fixed_across_loss_sets",
        "paired_seed_batches",
        "decoder_frozen",
        "decoder_in_optimizer_loss",
    )
    if any(not bool(contract.get(key)) for key in required_true):
        raise ValueError("Scientific contract was weakened")
    if bool(contract.get("test_loaded_during_training", True)):
        raise ValueError("Training must not load test data")


def make_job(
    experiment: dict[str, Any], representation: str, loss_set: str, seed: int
) -> dict[str, Any]:
    if representation not in experiment["representations"]:
        raise KeyError(representation)
    if loss_set not in experiment["loss_sets"]:
        raise KeyError(loss_set)
    training = dict(experiment["training"])
    training["batch_size"] = int(
        training.pop("batch_size_by_representation")[representation]
    )
    training["decoder_batch_size"] = int(
        training.pop("decoder_batch_size_by_representation")[representation]
    )
    training["evaluation_batch_size"] = int(
        training.pop("evaluation_batch_size_by_representation")[representation]
    )
    training["seed"] = int(seed)
    return {
        "schema_version": 1,
        "experiment": experiment["name"],
        "id": f"{representation}__{loss_set}__s{int(seed)}",
        "representation": representation,
        "loss_set": loss_set,
        "loss_count": LOSS_COUNTS[loss_set],
        "model": dict(experiment["model"]),
        "training": training,
        "loss_weights": dict(experiment["loss_weights"]),
        "selection": dict(experiment["selection"]),
        "scientific_contract": dict(experiment["scientific_contract"]),
    }


def all_jobs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        make_job(experiment, representation, loss_set, seed)
        for representation in experiment["representations"]
        for seed in experiment["seeds"]
        for loss_set in experiment["loss_sets"]
    ]


def validate_job(config: dict[str, Any]) -> None:
    if config.get("loss_set") not in LOSS_COUNTS:
        raise ValueError("Unknown loss set")
    if int(config.get("loss_count", -1)) != LOSS_COUNTS[config["loss_set"]]:
        raise ValueError("Loss count disagrees with implementation")
    model = config["model"]
    if int(model["latent_dim"]) != 128 or int(model["width"]) != 256:
        raise ValueError("Model differs from the fixed task3 control")
    if int(model["residual_blocks"]) != 2 or float(model["dropout"]) != 0.0:
        raise ValueError("Model differs from the fixed task3 control")
    training = config["training"]
    for key in (
        "epochs",
        "steps_per_epoch",
        "batch_size",
        "decoder_batch_size",
        "evaluation_batch_size",
    ):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if bool(config["selection"].get("test_used_for_selection", True)):
        raise ValueError("Test data cannot participate in selection")

