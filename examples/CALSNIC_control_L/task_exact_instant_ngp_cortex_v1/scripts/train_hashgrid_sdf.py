#!/usr/bin/env python3
"""Train a 256-D auto-decoder over Instant-NGP hash grids on CALSNIC cortex.

Forked from ``task2_inr_multires_single_field_v1/scripts/train_multires_sdf.py``.
The optimizer groups, gradient accumulation, latent norm-ball projection,
checkpoint payload, provenance files and selection policy are deliberately
unchanged so that a run here is comparable with MR64/MR128.  Three things differ:

* the loss understands the two-branch Compact-SDF decoder (generalization,
  overfitting, fused, and an off-surface agreement term);
* Eikonal gradients are taken with a millimetre-pinned finite difference and may
  constrain more than one branch;
* the exact-SDF audit gate runs before anything is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hashgrid_common import (
    CALSNIC_TASK,
    ContinuousSDFDataset,
    REFERENCE_TASK,
    REPO_ROOT,
    append_csv,
    architecture_name,
    atomic_torch_save,
    build_decoder,
    choose_device,
    clamped_l1,
    config_public,
    effective_eikonal_weight,
    expand_codes,
    far_field_agreement,
    field_callable,
    file_signature,
    finite_difference_epsilon,
    fit_single_latent,
    hash_capacity_report,
    is_deformation_field,
    is_two_branch,
    level_weights_for_epoch,
    load_config,
    load_manifest,
    numerical_gradient_and_laplacian,
    numerical_jacobian_squared,
    require_bulk_path,
    seed_dataloader_worker,
    select_near_surface_points,
    select_stratified_rows,
    set_global_seed,
    stable_seed,
    validate_manifest_contract,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path, latest, or best_sdf.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--benchmark-step",
        action="store_true",
        help="Time one full-sampling chunk without writing experiment files.",
    )
    parser.add_argument("--benchmark-scenes", type=int, default=2)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--skip-periodic-evaluation", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Exact-SDF audit gate (same contract as the CALSNIC multires task)
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enforce_audit(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    audit_value = config.get("required_exact_audit")
    if not audit_value:
        raise ValueError("Every CALSNIC run config must pin required_exact_audit.")
    audit_path = Path(audit_value)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Required full exact-SDF audit does not exist: {audit_path}")
    report = json.loads(audit_path.read_text())
    if not (
        report.get("passed")
        and report.get("full_cohort_audit")
        and report.get("training_allowed")
    ):
        raise RuntimeError(f"Exact-SDF audit does not authorize training: {audit_path}")
    manifest = Path(config["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if report.get("exact_manifest_sha256") != sha256(manifest):
        raise RuntimeError("Exact manifest changed after the successful audit; rerun the audit.")


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------
def decoder_parameters(model) -> list[torch.nn.Parameter]:
    grid_ids = {id(parameter) for parameter in model.grids}
    return [parameter for parameter in model.parameters() if id(parameter) not in grid_ids]


def make_optimizer(model, embedding: torch.nn.Embedding, config: dict[str, Any]):
    rates = config["learning_rates"]
    groups = [
        {"name": "decoder", "params": decoder_parameters(model), "lr": float(rates["decoder"])},
        {"name": "latent", "params": embedding.parameters(), "lr": float(rates["latent"])},
    ]
    grids = list(model.grids)
    if grids:  # a grid-free architecture contributes no group at all
        groups.insert(1, {"name": "grid", "params": grids, "lr": float(rates["grid"])})
    return torch.optim.Adam(groups, betas=(0.9, 0.999))


def configure_learning_rates(optimizer, config: dict[str, Any], epoch: int) -> None:
    interval = max(1, int(config["learning_rate_decay_interval"]))
    decay = float(config["learning_rate_decay_factor"]) ** ((epoch - 1) // interval)
    rates = config["learning_rates"]
    for group in optimizer.param_groups:
        group["lr"] = float(rates[group["name"]]) * decay


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def eikonal_terms(
    model,
    latent_codes: torch.Tensor,
    near_samples: torch.Tensor,
    epoch: int,
    config: dict[str, Any],
    level_weights: list[float],
) -> tuple[torch.Tensor, dict[str, float], np.ndarray]:
    """Squared unit-gradient penalty over every configured branch."""
    epsilon_np = finite_difference_epsilon(epoch, config, level_weights)
    weight = effective_eikonal_weight(epoch, config)
    zero = latent_codes.new_zeros(())
    settings = config["eikonal"]
    branches = list(settings.get("branches", ["fused"]))
    # The key set must not depend on whether Eikonal is active yet: append_csv
    # writes the header from the first epoch's row, so a key appearing later
    # would silently shift every column in training_history.csv.
    metrics = {
        "eikonal_loss": 0.0,
        "eikonal_weight": float(weight),
        "gradient_norm_mean": 0.0,
        "gradient_norm_median": 0.0,
        "gradient_norm_p95": 0.0,
        "gradient_norm_absolute_error": 0.0,
        "eikonal_points": 0.0,
        "eikonal_replacement_shortfall": 0.0,
        "second_order_loss": 0.0,
        "second_order_weight": 0.0,
        "curvature_excess_fraction": 0.0,
        "laplacian_abs_mean_per_mm": 0.0,
        "laplacian_abs_p95_per_mm": 0.0,
        **{f"eikonal_{branch}_loss": 0.0 for branch in branches},
    }
    if weight <= 0.0:
        return zero, metrics, epsilon_np

    selected_xyz, selected_codes, shortfall = select_near_surface_points(
        latent_codes,
        near_samples,
        float(settings["target_band"]),
        int(settings["points_per_chunk"]),
    )
    epsilon = torch.as_tensor(
        epsilon_np, device=selected_xyz.device, dtype=selected_xyz.dtype
    )
    losses = []
    norms_all = []
    laplacians = []
    for branch in branches:
        gradient, laplacian = numerical_gradient_and_laplacian(
            field_callable(model, branch),
            selected_codes,
            selected_xyz,
            epsilon,
            level_weights,
        )
        norm = torch.linalg.vector_norm(gradient, dim=1)
        losses.append(torch.square(norm - 1.0).mean())
        norms_all.append(norm)
        laplacians.append(laplacian)
        metrics[f"eikonal_{branch}_loss"] = float(losses[-1].detach().cpu())
    loss = torch.stack(losses).mean()
    norm = torch.cat(norms_all)
    laplacian = torch.cat(laplacians)
    metrics.update(
        {
            "eikonal_loss": float(loss.detach().cpu()),
            "gradient_norm_mean": float(norm.mean().detach().cpu()),
            "gradient_norm_median": float(norm.median().detach().cpu()),
            "gradient_norm_p95": float(torch.quantile(norm, 0.95).detach().cpu()),
            "gradient_norm_absolute_error": float(
                torch.abs(norm - 1.0).mean().detach().cpu()
            ),
            "eikonal_points": float(len(norm)),
            "eikonal_replacement_shortfall": float(shortfall),
        }
    )
    # Second-order term, band selected by the same step size (see
    # numerical_gradient_and_laplacian).  Reported in 1/mm so the magnitude is
    # interpretable as a curvature rather than a unitless model number.
    per_mm = 1.0 / float(config["mm_per_normalized_unit"])
    second_weight = effective_second_order_weight(epoch, config)
    total = weight * loss
    if second_weight > 0.0:
        # Hinge, not L2. Measured on the trained two-branch model at h=0.5 mm,
        # the grid-free branch sits at |lap| ~ 0.39 /mm (a 5 mm curvature radius,
        # i.e. real cortex) while the fused branch reaches 3.5 /mm and collapses
        # ~9x when h widens to 2 mm -- the signature of sub-millimetre noise
        # rather than folding. Penalising |lap| toward zero would flatten genuine
        # folds, so only curvature above a physical threshold is charged for.
        threshold = float(config["second_order"].get("curvature_threshold_per_mm", 1.0))
        excess = torch.clamp(laplacian.abs() * per_mm - threshold, min=0.0)
        second_loss = excess.square().mean()
        total = total + second_weight * second_loss
        metrics["second_order_loss"] = float(second_loss.detach().cpu())
        metrics["curvature_excess_fraction"] = float((excess > 0).float().mean().cpu())
    metrics["second_order_weight"] = float(second_weight)
    absolute = laplacian.abs().detach() * per_mm
    metrics["laplacian_abs_mean_per_mm"] = float(absolute.mean().cpu())
    metrics["laplacian_abs_p95_per_mm"] = float(torch.quantile(absolute, 0.95).cpu())
    return total, metrics, epsilon_np


def deformation_terms(
    model,
    latent_codes: torch.Tensor,
    near_samples: torch.Tensor,
    parts: dict[str, torch.Tensor],
    epoch: int,
    config: dict[str, Any],
    level_weights: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warp magnitude, warp smoothness, and residual penalties.

    Only a deformation architecture contributes these; every other architecture
    returns an empty metric dict so its ``training_history.csv`` schema is
    unchanged.  Within a deformation run every key is emitted on every epoch --
    ``append_csv`` writes the header from the first row, so a key that appeared
    only once a term became active would silently shift every later column.

    ``deformation_jacobian`` is the term that decides whether this architecture
    works.  Too weak and the warp re-creates the high-frequency noise that
    fragmented the grid arms; too strong and it cannot move folds far enough to
    reach subject position, which is the entire point of the factorization.
    """
    zero = latent_codes.new_zeros(())
    if not is_deformation_field(config):
        return zero, {}

    weights = config["loss_weights"]
    settings = config.get("deformation", {})
    mm = float(config["mm_per_normalized_unit"])
    warp = parts["warp"]
    residual = parts["residual"]

    magnitude = warp.square().sum(dim=1).mean()
    residual_l2 = residual.square().mean()
    # Two different things, reported separately because confusing them makes the
    # log unreadable: the displacement is a 3-vector whose NORM is what moves a
    # fold, but warp_scale_mm bounds each COMPONENT, so the norm may legitimately
    # reach warp_scale * sqrt(3) without anything being saturated.
    norm_mm = warp.detach().norm(dim=1) * mm
    component_mm = warp.detach().abs() * mm
    unit = parts["warp_unit"].detach().abs()
    metrics = {
        "deformation_l2": float(magnitude.detach().cpu()),
        "deformation_jacobian": 0.0,
        "residual_l2": float(residual_l2.detach().cpu()),
        "warp_norm_mean_mm": float(norm_mm.mean().cpu()),
        "warp_norm_p95_mm": float(torch.quantile(norm_mm, 0.95).cpu()),
        # Directly comparable to warp_scale_mm.
        "warp_component_p50_mm": float(component_mm.median().cpu()),
        "warp_component_max_mm": float(component_mm.max().cpu()),
        # The early warning that matters: |tanh| -> 1 kills the warp's gradient.
        "warp_saturation_fraction": float((unit > 0.9).float().mean().cpu()),
        "warp_gradient_scale_mean": float((1.0 - unit.square()).mean().cpu()),
    }
    total = float(weights.get("deformation_l2", 0.0)) * magnitude + float(
        weights.get("residual_l2", 0.0)
    ) * residual_l2

    jacobian_weight = float(weights.get("deformation_jacobian", 0.0))
    if jacobian_weight > 0.0:
        selected_xyz, selected_codes, _shortfall = select_near_surface_points(
            latent_codes,
            near_samples,
            float(settings.get("target_band", config["eikonal"]["target_band"])),
            int(settings.get("points_per_chunk", 1024)),
        )
        epsilon = torch.as_tensor(
            finite_difference_epsilon(epoch, config, level_weights),
            device=selected_xyz.device,
            dtype=selected_xyz.dtype,
        )
        jacobian = numerical_jacobian_squared(
            model.warp, selected_codes, selected_xyz, epsilon
        ).mean()
        total = total + jacobian_weight * jacobian
        metrics["deformation_jacobian"] = float(jacobian.detach().cpu())
    return total, metrics


def effective_second_order_weight(epoch: int, config: dict[str, Any]) -> float:
    """Warm-in schedule for the finite-difference Laplacian penalty."""
    settings = config.get("second_order", {})
    if not bool(settings.get("enabled", False)):
        return 0.0
    start = int(settings.get("start_epoch", 1))
    if epoch < start:
        return 0.0
    warmup = max(1, int(settings.get("warmup_epochs", 1)))
    unit = min(1.0, (epoch - start + 1) / warmup)
    return float(settings.get("weight", 0.0)) * unit


def compute_loss(
    model,
    latent_codes: torch.Tensor,
    broad_samples: torch.Tensor,
    near_samples: torch.Tensor,
    epoch: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = config["loss_weights"]
    clamp_distance = float(config["clamp_distance"])
    level_weights = level_weights_for_epoch(epoch, config)
    two_branch = is_two_branch(config)

    broad_xyz = broad_samples[:, :, :3].reshape(-1, 3)
    broad_target = broad_samples[:, :, 3:4].reshape(-1, 1)
    broad_codes = expand_codes(latent_codes, broad_samples.shape[1])
    broad_parts = model(
        torch.cat((broad_codes, broad_xyz), dim=1),
        level_weights=level_weights,
        return_parts=True,
    )

    near_xyz = near_samples[:, :, :3].reshape(-1, 3)
    near_target = near_samples[:, :, 3:4].reshape(-1, 1)
    near_codes = expand_codes(latent_codes, near_samples.shape[1])
    near_parts = model(
        torch.cat((near_codes, near_xyz), dim=1),
        level_weights=level_weights,
        return_parts=True,
    )

    code_l2 = latent_codes.square().mean()
    grid_l2, grid_tv = model.grid_regularization(level_weights)
    data_metrics: dict[str, float] = {}

    if two_branch:
        terms = {
            "global_broad_sdf_l1": clamped_l1(
                broad_parts["sdf_global"], broad_target, clamp_distance
            ),
            "global_near_sdf_l1": clamped_l1(
                near_parts["sdf_global"], near_target, clamp_distance
            ),
            "local_broad_sdf_l1": clamped_l1(
                broad_parts["sdf_local"], broad_target, clamp_distance
            ),
            "local_near_sdf_l1": clamped_l1(
                near_parts["sdf_local"], near_target, clamp_distance
            ),
            "fused_near_sdf_l1": clamped_l1(
                near_parts["sdf_smooth_gate"], near_target, clamp_distance
            ),
            "far_field_agreement": far_field_agreement(
                broad_parts["sdf_global"],
                broad_parts["sdf_local"],
                broad_target,
                float(config["sampling"]["near_band"]),
                float(config.get("far_field_margin", 0.0)),
            ),
        }
        total = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
        data_metrics = {name: float(value.detach().cpu()) for name, value in terms.items()}
        # Reported under the shared names so logs line up with MR64/MR128.
        data_metrics["broad_sdf_l1"] = data_metrics["global_broad_sdf_l1"]
        data_metrics["near_sdf_l1"] = data_metrics["fused_near_sdf_l1"]
    else:
        broad_l1 = clamped_l1(broad_parts["sdf"], broad_target, clamp_distance)
        near_l1 = clamped_l1(near_parts["sdf"], near_target, clamp_distance)
        total = float(weights.get("broad_sdf_l1", 1.0)) * broad_l1 + float(
            weights.get("near_sdf_l1", 1.0)
        ) * near_l1
        data_metrics = {
            "broad_sdf_l1": float(broad_l1.detach().cpu()),
            "near_sdf_l1": float(near_l1.detach().cpu()),
        }

    total = (
        total
        + float(weights.get("latent_l2", 0.0)) * code_l2
        + float(weights.get("grid_l2", 0.0)) * grid_l2
        + float(weights.get("grid_tv", 0.0)) * grid_tv
    )

    # Measured on the near samples, where the warp has to place folds precisely.
    deformation_loss, deformation_metrics = deformation_terms(
        model, latent_codes, near_samples, near_parts, epoch, config, level_weights
    )
    total = total + deformation_loss

    eikonal_loss, eikonal_metrics, epsilon_np = eikonal_terms(
        model, latent_codes, near_samples, epoch, config, level_weights
    )
    total = total + eikonal_loss

    metrics = {
        **data_metrics,
        **deformation_metrics,
        "latent_l2": float(code_l2.detach().cpu()),
        "grid_l2": float(grid_l2.detach().cpu()),
        "grid_tv": float(grid_tv.detach().cpu()),
        **eikonal_metrics,
        "epsilon_x": float(epsilon_np[0]),
        "epsilon_y": float(epsilon_np[1]),
        "epsilon_z": float(epsilon_np[2]),
        **{
            f"level_{resolution}_weight": float(weight)
            for resolution, weight in zip(model.grid_resolutions, level_weights)
        },
        "total": float(total.detach().cpu()),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def checkpoint_payload(
    epoch: int,
    model,
    embedding,
    optimizer,
    best_validation_l1: float,
    best_mesh_assd: float,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": architecture_name(config),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "latent_codes": embedding.weight.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_l1": float(best_validation_l1),
        "best_mesh_assd": float(best_mesh_assd),
        "config": config_public(config),
        "training_scan_ids": [row["scan_id"] for row in train_rows],
        "level_weights": level_weights_for_epoch(epoch, config),
        "warm_start_report": {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        payload["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    output_dir: Path,
    label: str,
    epoch: int,
    model,
    embedding,
    optimizer,
    best_validation_l1: float,
    best_mesh_assd: float,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    compatibility: bool,
) -> None:
    payload = checkpoint_payload(
        epoch,
        model,
        embedding,
        optimizer,
        best_validation_l1,
        best_mesh_assd,
        config,
        train_rows,
    )
    atomic_torch_save(payload, output_dir / "checkpoints" / f"{label}.pth")
    if compatibility:
        atomic_torch_save(
            {"epoch": epoch, "model_state_dict": payload["model_state_dict"]},
            output_dir / "ModelParameters" / f"{label}.pth",
        )
        atomic_torch_save(
            {"epoch": epoch, "optimizer_state_dict": payload["optimizer_state_dict"]},
            output_dir / "OptimizerParameters" / f"{label}.pth",
        )
        atomic_torch_save(
            {
                "epoch": epoch,
                "latent_codes": {"weight": payload["latent_codes"]},
                "train_scan_ids": payload["training_scan_ids"],
            },
            output_dir / "LatentCodes" / f"{label}.pth",
        )


def restore_rng(payload: dict[str, Any]) -> None:
    """Restore RNG state saved in a checkpoint.

    ``torch.load(..., map_location=device)`` moves every tensor in the payload to
    that device, including the RNG ByteTensors, but ``set_rng_state`` requires
    them on the CPU as uint8. Without this they arrive as CUDA tensors and every
    ``--resume`` onto a GPU fails.
    """
    def as_cpu_byte(value):
        return value.detach().cpu().to(torch.uint8) if torch.is_tensor(value) else value

    state = payload.get("rng_state", {})
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(as_cpu_byte(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([as_cpu_byte(v) for v in state["cuda"]])


def validate_codes(
    model, rows: list[dict[str, str]], config: dict[str, Any], device, epoch: int
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    per_scan = []
    for row in rows:
        _latent, metrics = fit_single_latent(
            model,
            row["sdf_npz_path"],
            int(config["latent_size"]),
            copy.deepcopy(config["validation"]["latent_fit"]),
            float(config["clamp_distance"]),
            device,
            stable_seed(row["scan_id"], int(config["seed"])),
            config["network_specs"],
        )
        per_scan.append({"epoch": epoch, "scan_id": row["scan_id"], **metrics})
    return float(np.mean([row["heldout_sdf_l1"] for row in per_scan])), per_scan


def run_periodic_evaluation(
    config: dict[str, Any], checkpoint: Path, device, epoch: int
) -> dict[str, Any]:
    configured = config.get("periodic_evaluation", {}).get("script")
    script = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent / "evaluate_hashgrid_variants.py"
    )
    if not script.is_absolute():
        script = REPO_ROOT / script
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Periodic evaluation script does not exist: {script}")
    output = require_bulk_path(config["output_dir"]) / "periodic_evaluation" / f"epoch_{epoch:04d}"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config["_config_path"]),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--device",
            str(device),
        ],
        check=True,
    )
    return json.loads((output / "summary.json").read_text())


def smoke_overrides(config: dict[str, Any]) -> None:
    config["scenes_per_batch"] = 2
    config["scenes_per_chunk"] = 2
    config["num_workers"] = 0
    config["load_dataset_into_ram"] = False
    config["sampling"].update(
        {
            "global_near_samples_per_scene": 16,
            "global_positive_samples_per_scene": 8,
            "global_negative_samples_per_scene": 8,
            "local_ultra_near_samples_per_scene": 16,
            "local_positive_samples_per_scene": 8,
            "local_negative_samples_per_scene": 8,
        }
    )
    config["eikonal"]["points_per_chunk"] = 8


def main() -> None:
    args = parse_args()
    enforce_audit(Path(args.config).resolve())
    config = load_config(args.config)
    if args.smoke_test and args.benchmark_step:
        raise ValueError("Choose either --smoke-test or --benchmark-step.")
    if args.smoke_test:
        config = copy.deepcopy(config)
        smoke_overrides(config)
    elif args.benchmark_step:
        if args.benchmark_scenes < 1:
            raise ValueError("--benchmark-scenes must be positive.")
        config = copy.deepcopy(config)
        # Keep the real batching geometry so the epoch projection stays honest.
        benchmark_geometry = {
            "scenes_per_batch": int(config["scenes_per_batch"]),
            "scenes_per_chunk": int(config["scenes_per_chunk"]),
        }
        config["scenes_per_batch"] = int(args.benchmark_scenes)
        config["scenes_per_chunk"] = int(args.benchmark_scenes)
        config["num_workers"] = 0
        config["load_dataset_into_ram"] = False

    set_global_seed(int(config["seed"]))
    device = choose_device(args.device)
    rows = load_manifest(config["manifest"])
    input_report = validate_manifest_contract(rows, config["network_specs"]["grid_aabb"])
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_pool = [row for row in rows if row["split"] == "val"]
    validation_rows = select_stratified_rows(
        validation_pool,
        min(int(config["validation"]["scans_per_validation"]), len(validation_pool)),
        int(config["seed"]) + 7001,
    )
    output_dir = require_bulk_path(config["output_dir"])
    capacity = hash_capacity_report(config)

    if args.validate_only:
        print(
            f"Input validation passed: scans={len(rows)} train={len(train_rows)} "
            f"val={len(validation_pool)} grid_params={capacity['grid_parameters']} "
            f"finest_cell={capacity['finest_geometric_mean_cell_mm']:.3f}mm output={output_dir}",
            flush=True,
        )
        return

    model = build_decoder(config, device)
    embedding = torch.nn.Embedding(len(train_rows), int(config["latent_size"]), device=device)
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=float(config["code_initial_std"]))
    optimizer = make_optimizer(model, embedding, config)
    start_epoch = 1
    best_validation_l1 = math.inf
    best_mesh_assd = math.inf

    if args.resume:
        resume = (
            Path(args.resume).resolve()
            if Path(args.resume).is_file()
            else output_dir / "checkpoints" / f"{args.resume}.pth"
        )
        if not resume.is_file():
            raise FileNotFoundError(resume)
        payload = torch.load(resume, map_location=device)
        if payload.get("training_scan_ids") != [row["scan_id"] for row in train_rows]:
            raise ValueError("Resume checkpoint training scan order does not match the manifest.")
        model.load_state_dict(payload["model_state_dict"])
        embedding.weight.data.copy_(payload["latent_codes"].to(device))
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation_l1 = float(payload.get("best_validation_l1", math.inf))
        best_mesh_assd = float(payload.get("best_mesh_assd", math.inf))
        restore_rng(payload)

    if args.smoke_test:
        dataset_rows = train_rows[:2]
    elif args.benchmark_step:
        dataset_rows = train_rows[: int(args.benchmark_scenes)]
    else:
        dataset_rows = train_rows
    dataset = ContinuousSDFDataset(dataset_rows, config)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["scenes_per_batch"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_dataloader_worker,
    )

    if args.smoke_test:
        broad, near, indices = next(iter(loader))
        broad, near, indices = broad.to(device), near.to(device), indices.to(device)
        for epoch in [1, 650, max(int(config["eikonal"]["start_epoch"]) + 200, 900)]:
            configure_learning_rates(optimizer, config, epoch)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, embedding(indices), broad, near, epoch, config)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(embedding.parameters()),
                float(config["gradient_clip_norm"]),
            )
            if not math.isfinite(metrics["total"]) or not torch.isfinite(gradient_norm):
                raise RuntimeError("Smoke test produced a non-finite loss or gradient.")
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.grids
            ):
                raise RuntimeError("At least one hash table received no finite gradient.")
            optimizer.step()
            print(
                f"smoke epoch={epoch} total={metrics['total']:.6g} "
                f"near={metrics['near_sdf_l1']:.6g} eik={metrics['eikonal_loss']:.6g} "
                f"eps_mm={metrics['epsilon_x'] * float(config['mm_per_normalized_unit']):.3f}",
                flush=True,
            )
        print("Smoke test passed; no experiment files were written.", flush=True)
        return

    if args.benchmark_step:
        sampling_started = time.perf_counter()
        broad, near, indices = next(iter(loader))
        sampling_seconds = time.perf_counter() - sampling_started
        broad, near, indices = broad.to(device), near.to(device), indices.to(device)

        def one_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, embedding(indices), broad, near, 1500, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(embedding.parameters()),
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
            return metrics["total"]

        # The first steps pay for kernel selection across sixteen table shapes;
        # timing them would overstate the steady-state cost several-fold.
        for _ in range(max(0, int(args.benchmark_warmup))):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(max(1, int(args.benchmark_iters))):
            total = one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = (time.perf_counter() - started) / max(1, int(args.benchmark_iters))
        peak_gib = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else float("nan")
        )
        batches = math.ceil(len(train_rows) / benchmark_geometry["scenes_per_batch"])
        chunks_per_batch = math.ceil(
            benchmark_geometry["scenes_per_batch"] / benchmark_geometry["scenes_per_chunk"]
        )
        scale = benchmark_geometry["scenes_per_chunk"] / max(1, int(args.benchmark_scenes))
        projected = batches * chunks_per_batch * step_seconds * scale
        print(
            f"benchmark passed: scenes={len(indices)} broad_per_scene={broad.shape[1]} "
            f"near_per_scene={near.shape[1]} sampling_seconds={sampling_seconds:.3f} "
            f"warmup={args.benchmark_warmup} iters={args.benchmark_iters} "
            f"forward_backward_step_seconds={step_seconds:.4f} peak_cuda_gib={peak_gib:.3f} "
            f"projected_epoch_seconds={projected:.1f} "
            f"projected_total_hours={projected * int(config['total_epochs']) / 3600:.1f} "
            f"loss={total:.6g}; no experiment files were written.",
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "architecture": architecture_name(config),
        "manifest": file_signature(config["manifest"]),
        "encoding_source": file_signature(
            REPO_ROOT / "networks" / "multires_hashgrid_encoding.py"
        ),
        "network_source": file_signature(
            REPO_ROOT / "networks" / f"{config['network_arch']}.py"
        ),
        "sampling_utility_source": file_signature(REFERENCE_TASK),
        "trainer_source": file_signature(Path(__file__).resolve()),
        "config_source": file_signature(config["_config_path"]),
        "baseline_task": str(CALSNIC_TASK),
        "input_report": input_report,
        "hash_capacity": capacity,
        "persistent_output_policy": f"all paths below {output_dir}",
    }
    generator_source = config.get("sdf_supervision", {}).get("source_file")
    if generator_source:
        provenance["sdf_generator_source"] = file_signature(generator_source)
    write_json(output_dir / "run_config.json", config_public(config))
    write_json(output_dir / "input_contract_report.json", input_report)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "hash_capacity_report.json", capacity)
    write_json(output_dir / "training_scan_ids.json", [row["scan_id"] for row in train_rows])
    write_json(
        output_dir / "validation_selection.json", [row["scan_id"] for row in validation_rows]
    )

    total_epochs = int(args.epochs or config["total_epochs"])
    history_fields: list[str] | None = None
    for epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        dataset.set_epoch(epoch)
        generator.manual_seed(stable_seed(f"train-order:{epoch}", int(config["seed"])))
        configure_learning_rates(optimizer, config, epoch)
        model.train()
        running_sum: defaultdict[str, float] = defaultdict(float)
        running_weight: defaultdict[str, float] = defaultdict(float)
        for broad, near, indices in loader:
            broad = broad.to(device, non_blocking=True)
            near = near.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_count = len(indices)
            chunk_size = int(config["scenes_per_chunk"])
            for start in range(0, batch_count, chunk_size):
                stop = min(start + chunk_size, batch_count)
                scene_weight = (stop - start) / batch_count
                loss, metrics = compute_loss(
                    model,
                    embedding(indices[start:stop]),
                    broad[start:stop],
                    near[start:stop],
                    epoch,
                    config,
                )
                (loss * scene_weight).backward()
                for key, value in metrics.items():
                    running_sum[key] += value * (stop - start)
                    running_weight[key] += stop - start
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(embedding.parameters()),
                float(config["gradient_clip_norm"]),
            )
            optimizer.step()
            code_bound = float(config.get("code_bound", 0.0))
            if code_bound > 0.0:
                with torch.no_grad():
                    norms = embedding.weight.norm(dim=1, keepdim=True)
                    embedding.weight.mul_(torch.clamp(code_bound / (norms + 1.0e-12), max=1.0))

        row: dict[str, Any] = {
            "epoch": epoch,
            "stage": "hashgrid_c2f",
            "seconds": round(time.time() - started, 3),
            **{f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups},
            **{key: running_sum[key] / running_weight[key] for key in sorted(running_sum)},
        }
        # append_csv writes the header once, from the first row, then appends
        # each later row in its own key order.  A schema change would therefore
        # shift every column silently instead of failing.
        if history_fields is None:
            history_fields = list(row)
        elif list(row) != history_fields:
            raise RuntimeError(
                "training_history schema changed at epoch "
                f"{epoch}: added={sorted(set(row) - set(history_fields))}, "
                f"removed={sorted(set(history_fields) - set(row))}."
            )
        print(
            f"epoch={epoch:04d} total={row['total']:.6g} near={row['near_sdf_l1']:.6g} "
            f"eik={row['eikonal_loss']:.6g} seconds={row['seconds']:.1f}",
            flush=True,
        )

        if epoch % int(config["validation"]["every_epochs"]) == 0 or epoch == total_epochs:
            value, per_scan = validate_codes(model, validation_rows, config, device, epoch)
            for item in per_scan:
                append_csv(output_dir / "logs" / "validation_per_scan.csv", item)
            append_csv(
                output_dir / "logs" / "validation_history.csv",
                {"epoch": epoch, "selection_metric": "heldout_sdf_l1", "value": value},
            )
            if value < best_validation_l1:
                best_validation_l1 = value
                save_checkpoint(
                    output_dir, "best_sdf", epoch, model, embedding, optimizer,
                    best_validation_l1, best_mesh_assd, config, train_rows, compatibility=True,
                )
        append_csv(output_dir / "logs" / "training_history.csv", row)

        latest_every = max(1, int(config.get("checkpoint_latest_every_epochs", 1)))
        if epoch % latest_every == 0 or epoch == total_epochs:
            save_checkpoint(
                output_dir, "latest", epoch, model, embedding, optimizer,
                best_validation_l1, best_mesh_assd, config, train_rows, compatibility=False,
            )
        if epoch % int(config["checkpoint_every_epochs"]) == 0 or epoch == total_epochs:
            label = f"epoch_{epoch:04d}"
            save_checkpoint(
                output_dir, label, epoch, model, embedding, optimizer,
                best_validation_l1, best_mesh_assd, config, train_rows, compatibility=True,
            )
            periodic = config.get("periodic_evaluation", {})
            should_evaluate = bool(periodic.get("enabled", False)) and (
                epoch % int(periodic.get("every_epochs", 500)) == 0
                or (epoch == total_epochs and bool(periodic.get("also_final_epoch", True)))
            )
            if should_evaluate and not args.skip_periodic_evaluation:
                report = run_periodic_evaluation(
                    config, output_dir / "checkpoints" / f"{label}.pth", device, epoch
                )
                validation_assd = float(report["selection_metric"]["validation_inr_assd_mm"])
                if validation_assd < best_mesh_assd:
                    best_mesh_assd = validation_assd
                    save_checkpoint(
                        output_dir, "best_mesh", epoch, model, embedding, optimizer,
                        best_validation_l1, best_mesh_assd, config, train_rows,
                        compatibility=True,
                    )
    print(
        f"Training complete: best held-out SDF L1={best_validation_l1:.8f}; "
        f"best validation ASSD={best_mesh_assd:.6f} mm; output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
