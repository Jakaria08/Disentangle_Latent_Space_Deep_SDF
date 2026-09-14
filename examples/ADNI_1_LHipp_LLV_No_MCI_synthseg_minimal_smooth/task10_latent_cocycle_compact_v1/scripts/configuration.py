#!/usr/bin/env python3
"""Expand and validate the compact latent-objective experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from objective import ACTIVE_LEAF_COUNTS, LOSS_COUNTS, OBJECTIVES

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = TASK_ROOT / "configs" / "experiment.json"


def read_experiment(path: Path = DEFAULT_EXPERIMENT) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    validate_experiment(value)
    return value


def validate_experiment(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 2:
        raise ValueError("Expected task10 experiment schema version 2")
    representations = list(config["representations"])
    if len(representations) != len(set(representations)):
        raise ValueError("Representations must be unique")
    if len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Seeds must be unique")

    arms = list(config["arms"])
    names = [arm["name"] for arm in arms]
    if len(names) != len(set(names)):
        raise ValueError("Arm names must be unique")
    expected = {
        "compact6_standard",
        "compact5_standard",
        "full13_pca_low",
        "compact6_pca_low",
        "compact5_pca_low",
    }
    if set(names) != expected:
        raise ValueError("The preregistered five-arm design changed")

    profiles = config["training"]["learning_rate_profiles"]
    for arm in arms:
        if arm["objective"] not in OBJECTIVES:
            raise ValueError(f"Unknown objective in {arm['name']}")
        if arm["learning_rate_profile"] not in profiles:
            raise ValueError(f"Unknown learning-rate profile in {arm['name']}")
        if not arm["representations"] or not set(arm["representations"]) <= set(
            representations
        ):
            raise ValueError(f"Invalid representations in {arm['name']}")
        if arm["learning_rate_profile"] == "pca_low" and arm[
            "representations"
        ] != ["pca128"]:
            raise ValueError("The lower learning rate is a PCA-only intervention")

    model = config["model"]
    if (
        int(model["latent_dim"]) != 128
        or int(model["width"]) != 256
        or int(model["residual_blocks"]) != 2
        or float(model["dropout"]) != 0.0
    ):
        raise ValueError("Model differs from the fixed task3/task9 control")

    training = config["training"]
    if int(training["warmup_steps"]) >= int(training["epochs"]) * int(
        training["steps_per_epoch"]
    ):
        raise ValueError("Warmup must finish before training ends")
    for name in representations:
        for key in (
            "batch_size_by_representation",
            "decoder_batch_size_by_representation",
            "evaluation_batch_size_by_representation",
        ):
            if int(training[key][name]) <= 0:
                raise ValueError(f"Missing positive {key}.{name}")
    for name, profile in profiles.items():
        initial = float(profile["initial_learning_rate"])
        peak = float(profile["peak_learning_rate"])
        minimum = float(profile["minimum_learning_rate"])
        if not (0.0 < initial <= peak and 0.0 < minimum <= peak):
            raise ValueError(f"Invalid learning-rate profile {name}")

    contract = config["scientific_contract"]
    required_true = (
        "latent_only",
        "direct_non_ode",
        "architecture_fixed_across_arms",
        "paired_seed_batches",
        "decoder_frozen",
        "decoder_in_optimizer_loss",
        "all_raw_terms_logged",
        "task9_controls_read_only",
    )
    if any(not bool(contract.get(key)) for key in required_true):
        raise ValueError("Scientific contract was weakened")
    if bool(contract.get("test_loaded_during_training", True)):
        raise ValueError("Training must not load test data")
    controls = config["external_controls"]
    if not controls.get("task9_experiment_config") or not controls.get(
        "task9_full13_root"
    ):
        raise ValueError("Task9 control provenance is incomplete")
    if set(controls["historical_roots"]) != set(representations):
        raise ValueError("Historical control roots must cover every representation")


def arm_by_name(experiment: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return next(arm for arm in experiment["arms"] if arm["name"] == name)
    except StopIteration as exc:
        raise KeyError(name) from exc


def make_job(
    experiment: dict[str, Any], representation: str, arm_name: str, seed: int
) -> dict[str, Any]:
    if representation not in experiment["representations"]:
        raise KeyError(representation)
    arm = arm_by_name(experiment, arm_name)
    if representation not in arm["representations"]:
        raise ValueError(f"Arm {arm_name} is not defined for {representation}")

    training = dict(experiment["training"])
    profiles = training.pop("learning_rate_profiles")
    profile_name = str(arm["learning_rate_profile"])
    training.update(profiles[profile_name])
    training["learning_rate_profile"] = profile_name
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
    objective = str(arm["objective"])
    return {
        "schema_version": 2,
        "experiment": experiment["name"],
        "id": f"{representation}__{arm_name}__s{int(seed)}",
        "representation": representation,
        "arm": arm_name,
        "objective": objective,
        "loss_set": objective,
        "loss_count": LOSS_COUNTS[objective],
        "active_leaf_count": ACTIVE_LEAF_COUNTS[objective],
        "comparison_control": arm["comparison_control"],
        "model": dict(experiment["model"]),
        "training": training,
        "loss_weights": dict(experiment["loss_weights"]),
        "selection": dict(experiment["selection"]),
        "scientific_contract": dict(experiment["scientific_contract"]),
    }


def all_jobs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        make_job(experiment, representation, arm["name"], seed)
        for representation in experiment["representations"]
        for seed in experiment["seeds"]
        for arm in experiment["arms"]
        if representation in arm["representations"]
    ]


def validate_job(config: dict[str, Any]) -> None:
    objective = config.get("objective")
    if objective not in OBJECTIVES or config.get("loss_set") != objective:
        raise ValueError("Unknown or inconsistent objective")
    if int(config.get("loss_count", -1)) != LOSS_COUNTS[objective]:
        raise ValueError("Top-level loss count disagrees with implementation")
    if int(config.get("active_leaf_count", -1)) != ACTIVE_LEAF_COUNTS[objective]:
        raise ValueError("Active leaf count disagrees with implementation")
    model = config["model"]
    if (
        int(model["latent_dim"]) != 128
        or int(model["width"]) != 256
        or int(model["residual_blocks"]) != 2
        or float(model["dropout"]) != 0.0
    ):
        raise ValueError("Model differs from the fixed task3/task9 control")
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
