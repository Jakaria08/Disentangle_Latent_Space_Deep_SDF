#!/usr/bin/env python3
"""Shared utilities for the isolated dense-multiresolution SDF experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BULK_ROOT = Path("/mnt/bulk10tb").resolve()
REFERENCE_TASK = (
    TASK_DIR.parent / "task2_inr_representations_v1" / "scripts" / "shared_grid_common.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_reference_common():
    spec = importlib.util.spec_from_file_location("task2_compact_common_reference", REFERENCE_TASK)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference utilities from {REFERENCE_TASK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reference = _load_reference_common()
ContinuousSDFDataset = _reference.ContinuousSDFDataset
load_manifest = _reference.load_manifest
load_sdf_arrays = _reference.load_sdf_arrays
load_checkpoint_training_scan_ids = _reference.load_checkpoint_training_scan_ids
sample_continuous_sdf_pair = _reference.sample_continuous_sdf_pair
select_stratified_rows = _reference.select_stratified_rows
stable_seed = _reference.stable_seed
_reference_validate_manifest_contract = _reference.validate_manifest_contract
seed_dataloader_worker = _reference.seed_dataloader_worker
set_global_seed = _reference.set_global_seed
sha256_file = _reference.sha256_file


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_manifest_contract(
    rows: list[dict[str, str]], grid_aabb: list[list[float]]
) -> dict[str, Any]:
    """Validate inputs, optionally applying manifest-defined mesh centring in memory."""
    centered = any(row.get("mesh_center_x", "") not in {None, ""} for row in rows)
    if not centered:
        return _reference_validate_manifest_contract(rows, grid_aabb)
    scan_ids = [row["scan_id"] for row in rows]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Manifest scan_id values are not unique.")
    subject_splits: dict[str, set[str]] = {}
    population_min = np.full(3, np.inf, dtype=np.float64)
    population_max = np.full(3, -np.inf, dtype=np.float64)
    center_fields = ("mesh_center_x", "mesh_center_y", "mesh_center_z")
    for row in rows:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
        for key in ("mesh_path", "sdf_npz_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"Manifest {key} does not exist: {row[key]}")
        present = [row.get(field, "") not in {None, ""} for field in center_fields]
        if not all(present):
            raise ValueError(f"Incomplete mesh_center_x/y/z for {row['scan_id']}.")
        vertices, _faces = _reference.load_obj_arrays(row["mesh_path"])
        center = np.asarray([float(row[field]) for field in center_fields], dtype=np.float64)
        if not len(vertices) or not np.isfinite(vertices).all() or not np.isfinite(center).all():
            raise ValueError(f"Invalid centred mesh inputs for {row['scan_id']}.")
        vertices = np.asarray(vertices, dtype=np.float64) - center[None, :]
        population_min = np.minimum(population_min, vertices.min(axis=0))
        population_max = np.maximum(population_max, vertices.max(axis=0))
    leakage = {subject: sorted(splits) for subject, splits in subject_splits.items() if len(splits) != 1}
    if leakage:
        raise ValueError(f"Subjects occur in multiple splits: {dict(list(leakage.items())[:5])}")
    aabb = np.asarray(grid_aabb, dtype=np.float64)
    lower_margin = population_min - aabb[0]
    upper_margin = aabb[1] - population_max
    if np.any(lower_margin < 0.0) or np.any(upper_margin < 0.0):
        raise ValueError(
            "At least one centred mesh lies outside grid_aabb: "
            f"population={[population_min.tolist(), population_max.tolist()]}, aabb={aabb.tolist()}"
        )
    return {
        "scan_count": len(rows),
        "subject_count": len(subject_splits),
        "split_scan_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "split_subject_counts": {split: len({row["subject_id"] for row in rows if row["split"] == split}) for split in ("train", "val", "test")},
        "population_mesh_min": population_min.tolist(),
        "population_mesh_max": population_max.tolist(),
        "grid_aabb": aabb.tolist(),
        "lower_margin": lower_margin.tolist(),
        "upper_margin": upper_margin.tolist(),
        "subject_split_disjoint": True,
        "all_mesh_and_sdf_paths_exist": True,
        "optional_manifest_mesh_center_applied": True,
    }


def require_bulk_path(value: str | Path, description: str = "output") -> Path:
    """Reject every persistent output path outside the 10-TB mount."""
    path = resolve_repo_path(value)
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(
            f"{description} must be below {BULK_ROOT}; refusing SSD path {path}"
        ) from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root itself.")
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    config["_output_dir"] = str(require_bulk_path(config["output_dir"], "config output_dir"))
    expected_sampling_source = config.get("reference_sampling_utility")
    expected_sampling_hash = config.get("reference_sampling_sha256")
    if expected_sampling_source:
        configured_source = resolve_repo_path(expected_sampling_source)
        if configured_source != REFERENCE_TASK.resolve():
            raise ValueError(
                f"Configured sampling utility {configured_source} does not match imported {REFERENCE_TASK}."
            )
    if expected_sampling_hash:
        digest = hashlib.sha256(REFERENCE_TASK.read_bytes()).hexdigest()
        if digest != expected_sampling_hash:
            raise RuntimeError(
                "The pinned Compact-style sampling utility changed: "
                f"expected {expected_sampling_hash}, found {digest}. Audit the change before training."
            )
    return config


def write_json(path: str | Path, value: Any) -> None:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.is_file()
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def choose_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid CUDA device {index}; visible device count is {torch.cuda.device_count()}."
            )
        return torch.device("cuda", index)
    return device


def build_decoder(config: dict[str, Any], device: torch.device):
    module = importlib.import_module(f"networks.{config['network_arch']}")
    specs = dict(config["network_specs"])
    # These two values control dataset cell balancing, not decoder construction.
    specs.pop("grid_resolution", None)
    specs.pop("sampling_balance_resolution", None)
    decoder = module.Decoder(int(config["latent_size"]), **specs)
    return decoder.to(device)


def level_weights_for_epoch(epoch: int, config: dict[str, Any]) -> list[float]:
    resolutions = tuple(int(value) for value in config["network_specs"]["grid_resolutions"])
    entries = config["level_schedule"]
    by_resolution = {int(item["resolution"]): item for item in entries}
    if set(by_resolution) != set(resolutions):
        raise ValueError("level_schedule must define every grid resolution exactly once.")
    result = []
    for resolution in resolutions:
        item = by_resolution[resolution]
        start = int(item["start_epoch"])
        end = int(item["end_epoch"])
        if start < 1 or end < start:
            raise ValueError(f"Invalid level schedule for resolution {resolution}.")
        if epoch < start:
            weight = 0.0
        elif end == start or epoch >= end:
            weight = 1.0
        else:
            unit = (epoch - start) / (end - start)
            weight = unit * unit * (3.0 - 2.0 * unit)
        result.append(float(weight))
    return result


def effective_eikonal_weight(epoch: int, config: dict[str, Any]) -> float:
    settings = config.get("eikonal", {})
    if not bool(settings.get("enabled", False)):
        return 0.0
    start = int(settings.get("start_epoch", 1))
    if epoch < start:
        return 0.0
    warmup = max(1, int(settings.get("warmup_epochs", 1)))
    unit = min(1.0, (epoch - start + 1) / warmup)
    return float(settings.get("weight", 0.0)) * unit


def finite_difference_epsilon(
    epoch: int, config: dict[str, Any], level_weights: list[float]
) -> np.ndarray:
    settings = config.get("eikonal", {})
    resolutions = np.asarray(config["network_specs"]["grid_resolutions"], dtype=np.int64)
    active = np.flatnonzero(np.asarray(level_weights) >= 0.5)
    resolution = int(resolutions[active[-1] if len(active) else 0])
    aabb = np.asarray(config["network_specs"]["grid_aabb"], dtype=np.float64)
    epsilon = (aabb[1] - aabb[0]) / max(1, resolution - 1)
    scale = float(settings.get("epsilon_cell_scale", 1.0))
    final_scale = float(settings.get("final_epsilon_cell_scale", scale))
    final_start = int(settings.get("final_epsilon_start_epoch", config["total_epochs"] + 1))
    if epoch >= final_start:
        scale = final_scale
    return (epsilon * scale).astype(np.float32)


def expand_codes(codes: torch.Tensor, count: int) -> torch.Tensor:
    return codes[:, None, :].expand(-1, count, -1).reshape(-1, codes.shape[1])


def clamped_l1(prediction: torch.Tensor, target: torch.Tensor, distance: float) -> torch.Tensor:
    return torch.abs(
        prediction.clamp(-distance, distance) - target.clamp(-distance, distance)
    ).mean()


def select_near_surface_points(
    latent_codes: torch.Tensor,
    samples: torch.Tensor,
    target_band: float,
    total_points: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select an approximately equal number of already-shuffled points per scene."""
    scene_count = samples.shape[0]
    per_scene = max(1, math.ceil(total_points / scene_count))
    xyz_parts = []
    code_parts = []
    replacement_shortfall = 0
    for scene in range(scene_count):
        eligible = torch.nonzero(
            samples[scene, :, 3].abs() <= target_band, as_tuple=False
        )[:, 0]
        if not len(eligible):
            continue
        if len(eligible) < per_scene:
            repeat = math.ceil(per_scene / len(eligible))
            chosen = eligible.repeat(repeat)[:per_scene]
            replacement_shortfall += per_scene - len(eligible)
        else:
            chosen = eligible[:per_scene]
        xyz_parts.append(samples[scene, chosen, :3])
        code_parts.append(latent_codes[scene : scene + 1].expand(len(chosen), -1))
    if not xyz_parts:
        raise RuntimeError(f"No Eikonal points satisfy |target SDF| <= {target_band}.")
    xyz = torch.cat(xyz_parts, dim=0)[:total_points]
    codes = torch.cat(code_parts, dim=0)[:total_points]
    return xyz, codes, replacement_shortfall


def numerical_spatial_gradient(
    decoder,
    codes: torch.Tensor,
    xyz: torch.Tensor,
    epsilon_xyz: torch.Tensor,
    level_weights: list[float],
) -> torch.Tensor:
    """Central differences with a single batched six-offset decoder evaluation."""
    if epsilon_xyz.shape != (3,) or torch.any(epsilon_xyz <= 0.0):
        raise ValueError("epsilon_xyz must be a positive three-vector.")
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    plus = xyz[None, :, :] + identity[:, None, :] * epsilon_xyz[None, None, :]
    minus = xyz[None, :, :] - identity[:, None, :] * epsilon_xyz[None, None, :]
    queries = torch.cat((plus, minus), dim=0).reshape(-1, 3)
    repeated_codes = codes[None, :, :].expand(6, -1, -1).reshape(-1, codes.shape[1])
    values = decoder(
        torch.cat((repeated_codes, queries), dim=1), level_weights=level_weights
    ).reshape(6, len(xyz), 1)
    return ((values[:3] - values[3:]) / (2.0 * epsilon_xyz[:, None, None])).squeeze(-1).transpose(0, 1)


def checkpoint_path(config: dict[str, Any], checkpoint: str | Path) -> Path:
    value = Path(checkpoint)
    if value.is_file():
        return value.resolve()
    name = str(checkpoint)
    name = name if name.endswith(".pth") else f"{name}.pth"
    return require_bulk_path(config["output_dir"]) / "checkpoints" / name


def load_decoder_checkpoint(config: dict[str, Any], checkpoint: str | Path, device: torch.device):
    path = checkpoint_path(config, checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    decoder = build_decoder(config, device)
    state = payload.get("model_state_dict", payload)
    decoder.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    decoder.eval()
    return decoder, payload, path


def warm_start_full_decoder(
    decoder,
    checkpoint: str | Path,
    device: torch.device,
    expected_network_arch: str | None = None,
) -> dict[str, Any]:
    """Load only a compatible population decoder/grid state from a source run.

    This deliberately does *not* restore source latent codes, optimizer state,
    or RNG state.  Those objects belong to the source run's 2,037-shape
    training table and must not be carried into a new pilot manifest.
    """
    path = resolve_repo_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Decoder warm-start checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    source = payload.get("model_state_dict", payload)
    if not isinstance(source, dict):
        raise ValueError(f"Decoder warm-start checkpoint has no state dictionary: {path}")
    source = {key.removeprefix("module."): value for key, value in source.items()}
    target = decoder.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    mismatched = sorted(
        key
        for key in set(target).intersection(source)
        if tuple(target[key].shape) != tuple(source[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Full decoder warm-start is incompatible with the requested architecture: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={mismatched[:5]}."
        )
    source_config = payload.get("config", {})
    source_arch = source_config.get("network_arch")
    if expected_network_arch and source_arch and source_arch != expected_network_arch:
        raise ValueError(
            f"Decoder warm-start architecture is {source_arch!r}, expected "
            f"{expected_network_arch!r}."
        )
    source_latent_size = source_config.get("latent_size", decoder.latent_size)
    if int(source_latent_size) != decoder.latent_size:
        raise ValueError(
            f"Decoder warm-start latent size is {source_latent_size}, expected "
            f"{decoder.latent_size}."
        )
    decoder.load_state_dict(source, strict=True)
    source_ids = payload.get("training_scan_ids", payload.get("train_scan_ids", []))
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "source_epoch": payload.get("epoch"),
        "source_best_validation_l1": payload.get("best_validation_l1"),
        "source_best_mesh_assd": payload.get("best_mesh_assd"),
        "source_network_arch": source_arch,
        "latent_size": decoder.latent_size,
        "source_training_scan_count": len(source_ids),
        "loaded_tensor_count": len(source),
        "target_tensor_count": len(target),
        "state_dict_key_and_shape_match": True,
        "transfer": "strict_full_decoder",
        "latent_transfer": "disabled_new_pilot_codes",
        "optimizer_state_transfer": "disabled_new_optimizer",
        "rng_state_transfer": "disabled_new_run_seed",
    }


def balance_resolution(network_specs: dict[str, Any]) -> int:
    """Cell-balancing resolution for the sampler.

    Written as a function because ``dict.get(key, default)`` evaluates the
    default eagerly: the previous inline form called ``max(grid_resolutions)``
    even when the key was present, which raises for a grid-free architecture
    whose ladder is legitimately empty.
    """
    configured = network_specs.get("sampling_balance_resolution")
    if configured:
        return int(configured)
    resolutions = network_specs.get("grid_resolutions") or []
    if not resolutions:
        raise ValueError(
            "sampling_balance_resolution must be set when there is no grid ladder."
        )
    return int(max(resolutions))


def latent_objective(
    decoder,
    latent: torch.Tensor,
    broad: np.ndarray,
    near: np.ndarray,
    clamp_distance: float,
    level_weights: list[float],
    broad_weight: float,
    near_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    broad_tensor = torch.from_numpy(broad).to(latent.device)
    near_tensor = torch.from_numpy(near).to(latent.device)
    broad_prediction = decoder(
        torch.cat((latent.expand(len(broad_tensor), -1), broad_tensor[:, :3]), dim=1),
        level_weights=level_weights,
    )
    near_prediction = decoder(
        torch.cat((latent.expand(len(near_tensor), -1), near_tensor[:, :3]), dim=1),
        level_weights=level_weights,
    )
    broad_l1 = clamped_l1(broad_prediction, broad_tensor[:, 3:4], clamp_distance)
    near_l1 = clamped_l1(near_prediction, near_tensor[:, 3:4], clamp_distance)
    return broad_weight * broad_l1 + near_weight * near_l1, broad_l1, near_l1


def fit_single_latent(
    decoder,
    sdf_path: str | Path,
    latent_size: int,
    fit_config: dict[str, Any],
    clamp_distance: float,
    device: torch.device,
    seed: int,
    network_specs: dict[str, Any],
    steps_override: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize only a new scan's 256-D code with a frozen single-field decoder."""
    rng = np.random.default_rng(seed)
    pos, neg = load_sdf_arrays(sdf_path)
    fraction = float(fit_config.get("holdout_fraction", 0.1))

    def split(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(len(array))
        count = min(len(array) - 1, max(1, int(round(len(array) * fraction))))
        return array[order[count:]], array[order[:count]]

    pos_fit, pos_holdout = split(pos)
    neg_fit, neg_holdout = split(neg)
    initial = rng.normal(
        0.0, float(fit_config.get("initial_std", 0.01)), size=(1, latent_size)
    ).astype(np.float32)
    latent = torch.from_numpy(initial).to(device)
    latent.requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=float(fit_config.get("learning_rate", 0.005)))
    steps = int(fit_config["steps"] if steps_override is None else steps_override)
    if steps < 1:
        raise ValueError("Latent fitting requires at least one optimization step.")
    level_weights = [1.0] * len(network_specs["grid_resolutions"])
    broad_weight = float(fit_config.get("broad_weight", 1.0))
    near_weight = float(fit_config.get("near_weight", 1.0))
    regularization = float(fit_config.get("code_regularization_lambda", 0.0))
    bound = float(fit_config.get("code_bound", 0.0))
    best = math.inf
    best_latent = None
    no_improvement = 0
    patience = int(fit_config.get("early_stop_patience", steps))
    minimum_delta = float(fit_config.get("early_stop_min_delta", 0.0))
    decoder.eval()
    parameters = list(decoder.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        for step in range(steps):
            broad, near = sample_continuous_sdf_pair(
                pos_fit,
                neg_fit,
                fit_config["sampling"],
                rng,
                network_specs["grid_aabb"],
                balance_resolution(network_specs),
            )
            optimizer.zero_grad(set_to_none=True)
            objective, _broad_l1, _near_l1 = latent_objective(
                decoder,
                latent,
                broad,
                near,
                clamp_distance,
                level_weights,
                broad_weight,
                near_weight,
            )
            loss = objective + regularization * latent.square().mean()
            loss.backward()
            optimizer.step()
            if bound > 0.0:
                with torch.no_grad():
                    norm = latent.norm(dim=1, keepdim=True)
                    latent.mul_(torch.clamp(bound / (norm + 1.0e-12), max=1.0))
            value = float(loss.detach().cpu())
            if value < best - minimum_delta:
                best = value
                best_latent = latent.detach().clone()
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= patience:
                break
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)
    if best_latent is None:
        best_latent = latent.detach().clone()
    evaluation_broad, evaluation_near = sample_continuous_sdf_pair(
        pos_holdout,
        neg_holdout,
        fit_config["sampling"],
        rng,
        network_specs["grid_aabb"],
        balance_resolution(network_specs),
    )
    with torch.no_grad():
        heldout_objective, broad_l1, near_l1 = latent_objective(
            decoder,
            best_latent,
            evaluation_broad,
            evaluation_near,
            clamp_distance,
            level_weights,
            broad_weight,
            near_weight,
        )
    data_weight = broad_weight + near_weight
    if data_weight <= 0.0:
        raise ValueError("At least one latent-fit data weight must be positive.")
    heldout_l1 = (broad_weight * broad_l1 + near_weight * near_l1) / data_weight
    return best_latent[0].cpu().numpy(), {
        "steps_requested": steps,
        "steps_completed": step + 1,
        "best_fit_objective": best,
        "heldout_objective": float(heldout_objective.cpu()),
        "heldout_sdf_l1": float(heldout_l1.cpu()),
        "heldout_broad_l1": float(broad_l1.cpu()),
        "heldout_near_l1": float(near_l1.cpu()),
        "latent_l2_mean": float(best_latent.square().mean().cpu()),
        "latent_norm": float(best_latent.norm().cpu()),
        "finite": bool(torch.isfinite(best_latent).all().cpu()),
    }


def decode_latent_to_mesh(
    decoder,
    latent: np.ndarray,
    output_path: str | Path,
    resolution: int,
    max_batch: int,
    device: torch.device,
    scaling: dict[str, float] | None = None,
) -> dict[str, Any]:
    import trimesh
    from skimage.measure import marching_cubes

    output = require_bulk_path(output_path, "mesh output")
    output.parent.mkdir(parents=True, exist_ok=True)
    latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).reshape(1, -1).to(device)
    total = int(resolution) ** 3
    values = np.empty(total, dtype=np.float32)
    step = 2.0 / (int(resolution) - 1)
    weights = [1.0] * len(decoder.grid_resolutions)
    decoder.eval()
    with torch.no_grad():
        for start in range(0, total, int(max_batch)):
            stop = min(start + int(max_batch), total)
            index = torch.arange(start, stop, device=device)
            x = torch.div(index, resolution * resolution, rounding_mode="floor")
            y = torch.div(index, resolution, rounding_mode="floor") % resolution
            z = index % resolution
            xyz = torch.stack((x, y, z), dim=1).float() * step - 1.0
            prediction = decoder(
                torch.cat((latent_tensor.expand(len(index), -1), xyz), dim=1),
                level_weights=weights,
            )
            values[start:stop] = prediction[:, 0].detach().cpu().numpy()
    volume = values.reshape(resolution, resolution, resolution)
    value_min, value_max = float(volume.min()), float(volume.max())
    if not value_min <= 0.0 <= value_max:
        raise RuntimeError(f"No zero level set: [{value_min}, {value_max}]")
    vertices, faces, _normals, _values = marching_cubes(
        volume, level=0.0, spacing=(step, step, step), method="lewiner"
    )
    vertices += np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    coordinate_space = "normalized"
    if scaling is not None:
        convert_normalized_mesh_to_mm(mesh, scaling)
        coordinate_space = "physical_mm"
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    mesh.export(temporary)
    os.replace(temporary, output)
    return {
        "mesh_path": str(output),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "coordinate_space": coordinate_space,
        "sdf_grid_min": value_min,
        "sdf_grid_max": value_max,
    }


def read_scaling(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    keys = ("target_range_min", "range_global_min", "range_linear_scale_factor", "distance_unscale_factor")
    values = {key: float(row[key]) for key in keys}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite mesh scaling in {path}")
    return values


def convert_normalized_mesh_to_mm(mesh, scaling: dict[str, float]) -> None:
    mesh.vertices = (
        (mesh.vertices - scaling["target_range_min"])
        / scaling["range_linear_scale_factor"]
        + scaling["range_global_min"]
    ) * scaling["distance_unscale_factor"]


def file_signature(path: str | Path) -> dict[str, Any]:
    value = resolve_repo_path(path)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256_file(value)}


def config_public(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
