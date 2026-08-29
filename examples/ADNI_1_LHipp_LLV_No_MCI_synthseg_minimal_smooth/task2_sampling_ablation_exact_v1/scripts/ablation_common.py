#!/usr/bin/env python3
"""Shared configuration and safety checks for the sampling ablation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
MATRIX_PATH = TASK_DIR / "configs" / "sampling_ablation_matrix.json"
BULK_ROOT = Path("/mnt/bulk10tb").resolve()
BASE_TASK = TASK_DIR.parent / "task2_inr_multires_single_field_v1"
BASE_SCRIPTS = BASE_TASK / "scripts"
TRAIN_SCRIPT = SCRIPT_DIR / "train_sampling_ablation.py"
EVALUATE_SCRIPT = BASE_SCRIPTS / "periodic_evaluate_multires.py"
EXPORT_SCRIPT = BASE_SCRIPTS / "fit_export_multires_latents.py"
BULK_WRAPPER = SCRIPT_DIR / "run_on_bulk.sh"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def require_bulk_path(value: str | Path, description: str = "output") -> Path:
    path = resolve_repo_path(value)
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must be below {BULK_ROOT}; got {path}") from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root.")
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_matrix() -> dict[str, Any]:
    matrix = load_json(MATRIX_PATH)
    require_bulk_path(matrix["bulk_output_root"], "bulk_output_root")
    return matrix


def experiment_definition(matrix: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in matrix["experiments"]:
        raise KeyError(f"Unknown experiment {name!r}; choose from {sorted(matrix['experiments'])}.")
    return matrix["experiments"][name]


def runtime_config_path(matrix: dict[str, Any], name: str) -> Path:
    experiment_definition(matrix, name)
    return require_bulk_path(matrix["bulk_output_root"]) / "runtime_configs" / f"{name}.json"


def run_dir(matrix: dict[str, Any], name: str) -> Path:
    experiment_definition(matrix, name)
    return require_bulk_path(matrix["bulk_output_root"]) / "runs" / name


def initialization_path(matrix: dict[str, Any], name: str) -> Path:
    return run_dir(matrix, name) / "checkpoints" / "population_initialization.pth"


def _fixed_control_sampling(base: dict[str, Any]) -> dict[str, Any]:
    sampling = copy.deepcopy(base["sampling"])
    sampling["ultra_near_band"] = 0.03
    sampling["near_band"] = 0.1
    sampling.pop("mode", None)
    sampling.pop("shells", None)
    return sampling


def materialize_config(matrix: dict[str, Any], name: str) -> dict[str, Any]:
    experiment = experiment_definition(matrix, name)
    base_path = resolve_repo_path(matrix["base_config"])
    actual = sha256_file(base_path)
    if actual != matrix["base_config_sha256"]:
        raise RuntimeError(
            f"Pinned base config changed: expected {matrix['base_config_sha256']}, found {actual}."
        )
    base = load_json(base_path)
    policy = matrix["training_policy"]
    config = copy.deepcopy(base)
    config["name"] = name
    config["description"] = experiment["description"]
    config["run_role"] = "exact_sdf_sampling_ablation_pretrained_population"
    config["output_dir"] = str(run_dir(matrix, name))
    config["require_bulk_output"] = True
    config["latent_size"] = int(policy["latent_size"])
    config["seed"] = int(policy["seed"])
    config["total_epochs"] = int(policy["total_epochs"])
    config["checkpoint_every_epochs"] = int(policy["checkpoint_every_epochs"])
    config["checkpoint_latest_every_epochs"] = int(policy["checkpoint_latest_every_epochs"])
    config["learning_rates"] = copy.deepcopy(policy["learning_rates"])
    config["learning_rate_decay_interval"] = int(policy["learning_rate_decay_interval"])
    config["learning_rate_decay_factor"] = float(policy["learning_rate_decay_factor"])
    config["eikonal"] = copy.deepcopy(policy["eikonal"])
    config["second_order"] = {"enabled": False, "weight": 0.0}
    config["level_schedule"] = [
        {"resolution": int(resolution), "start_epoch": 1, "end_epoch": 1}
        for resolution in config["network_specs"]["grid_resolutions"]
    ]
    config.pop("decoder_warm_start", None)
    config["population_initialization"] = {
        "enabled": True,
        "mode": "model_and_training_latents_fresh_optimizer",
        "checkpoint": matrix["source_checkpoint"],
        "checkpoint_sha256": matrix["source_checkpoint_sha256"],
        "expected_source_epoch": int(matrix["source_checkpoint_epoch"]),
        "expected_architecture": "single_field_dense_multiresolution_sdf",
        "expected_latent_size": int(policy["latent_size"]),
        "require_training_scan_order_match": True,
        "load_model": True,
        "load_train_latents": True,
        "load_optimizer": False,
        "load_rng": False,
        "reset_epoch": True
    }
    sampling = copy.deepcopy(config["sampling"])
    sampling["mode"] = experiment["sampling_mode"]
    sampling["ultra_near_band"] = float(experiment["ultra_near_band"])
    sampling["near_band"] = float(experiment["near_band"])
    if experiment["sampling_mode"] == "shell_stratified":
        sampling["shells"] = copy.deepcopy(experiment["shells"])
    else:
        sampling.pop("shells", None)
    config["sampling"] = sampling
    # Evaluation/inference sampling is deliberately identical for every arm.
    config["latent_fit"]["sampling"] = _fixed_control_sampling(base)
    config["validation"]["latent_fit"]["sampling"] = copy.deepcopy(
        base["validation"]["latent_fit"]["sampling"]
    )
    config["validation"]["latent_fit"]["sampling"]["ultra_near_band"] = 0.03
    config["validation"]["latent_fit"]["sampling"]["near_band"] = 0.1
    config["validation"]["every_epochs"] = int(policy["validation_every_epochs"])
    config["validation"]["scans_per_validation"] = int(policy["validation_scans"])
    periodic = config["periodic_evaluation"]
    periodic["enabled"] = True
    periodic["every_epochs"] = int(policy["periodic_evaluation_every_epochs"])
    periodic["also_final_epoch"] = True
    periodic["splits"] = list(policy["periodic_evaluation_splits"])
    periodic["per_split"] = int(policy["periodic_evaluation_per_split"])
    periodic["fixed_selection_json"] = matrix["fixed_validation_selection"]
    periodic["source_unseen_validation_only"] = False
    periodic["fscore_thresholds_mm"] = list(policy["fscore_thresholds_mm"])
    periodic["script"] = str(EVALUATE_SCRIPT.resolve())
    config["latent_export"] = {"enabled": False}
    config["sampling_ablation"] = {
        "experiment": name,
        "training_sampling_only_variable": True,
        "validation_sampling_fixed_to_control": True,
        "test_locked_during_selection": True,
        "implementation": str((SCRIPT_DIR / "sampling_dataset.py").resolve())
    }
    return config


def atomic_write_json(path: str | Path, value: Any) -> Path:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def write_runtime_config(matrix: dict[str, Any], name: str, config: dict[str, Any]) -> Path:
    path = runtime_config_path(matrix, name)
    if path.is_file():
        existing = load_json(path)
        if existing != config:
            raise FileExistsError(
                f"Existing runtime config differs from the current definition: {path}. "
                "Refusing a silent overwrite. Use a new output root or reconcile it explicitly."
            )
        return path
    return atomic_write_json(path, config)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def all_persistent_paths(config: dict[str, Any]) -> list[Path]:
    paths = [require_bulk_path(config["output_dir"], "run output")]
    periodic = config["periodic_evaluation"]
    if "fixed_selection_json" in periodic:
        paths.append(require_bulk_path(periodic["fixed_selection_json"], "validation selection"))
    paths.append(require_bulk_path(config["population_initialization"]["checkpoint"], "source checkpoint"))
    return paths
