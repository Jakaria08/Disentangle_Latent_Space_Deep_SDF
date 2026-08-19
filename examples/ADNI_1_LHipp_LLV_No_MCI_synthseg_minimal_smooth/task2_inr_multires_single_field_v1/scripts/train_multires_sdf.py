#!/usr/bin/env python3
"""Train a 256-D auto-decoder with one dense-multiresolution SDF field."""

from __future__ import annotations

import argparse
import copy
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

from multires_common import (
    ContinuousSDFDataset,
    REFERENCE_TASK,
    append_csv,
    atomic_torch_save,
    build_decoder,
    choose_device,
    clamped_l1,
    config_public,
    effective_eikonal_weight,
    expand_codes,
    file_signature,
    finite_difference_epsilon,
    fit_single_latent,
    level_weights_for_epoch,
    load_config,
    load_manifest,
    numerical_spatial_gradient,
    require_bulk_path,
    seed_dataloader_worker,
    select_near_surface_points,
    select_stratified_rows,
    set_global_seed,
    stable_seed,
    validate_manifest_contract,
    warm_start_full_decoder,
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
        help="Time one full-sampling two-scan chunk without writing experiment files.",
    )
    parser.add_argument(
        "--benchmark-scenes",
        type=int,
        default=2,
        help="Scenes in the no-write benchmark chunk (default: 2).",
    )
    parser.add_argument("--skip-periodic-evaluation", action="store_true")
    return parser.parse_args()


def decoder_parameters(model) -> list[torch.nn.Parameter]:
    grid_ids = {id(parameter) for parameter in model.grids}
    return [parameter for parameter in model.parameters() if id(parameter) not in grid_ids]


def make_optimizer(model, embedding: torch.nn.Embedding, config: dict[str, Any]):
    rates = config["learning_rates"]
    return torch.optim.Adam(
        [
            {"name": "decoder", "params": decoder_parameters(model), "lr": float(rates["decoder"])},
            {"name": "grid", "params": list(model.grids), "lr": float(rates["grid"])},
            {"name": "latent", "params": embedding.parameters(), "lr": float(rates["latent"])},
        ],
        betas=(0.9, 0.999),
    )


def warm_start_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    settings = config.get("decoder_warm_start", {})
    if not bool(settings.get("enabled", False)):
        return None
    if str(settings.get("mode", "")) != "decoder_only":
        raise ValueError("Full decoder warm-start mode must be 'decoder_only'.")
    if not settings.get("checkpoint"):
        raise ValueError("Full decoder warm-start requires a checkpoint path.")
    if any(
        bool(settings.get(key, True))
        for key in ("copy_source_latents", "load_source_optimizer", "load_source_rng")
    ):
        raise ValueError(
            "Full decoder warm-start must not import source latents, optimizer, or RNG state."
        )
    if int(settings.get("latent_adapt_epochs", 0)) < 1:
        raise ValueError("Full decoder warm-start requires at least one latent-adaptation epoch.")
    rates = settings.get("latent_adapt_learning_rates", {})
    required = {"decoder", "grid", "latent"}
    if set(rates) != required:
        raise ValueError(
            "latent_adapt_learning_rates must provide decoder, grid, and latent rates."
        )
    if any(float(rates[name]) < 0.0 for name in required) or float(rates["latent"]) <= 0.0:
        raise ValueError("Latent-adaptation learning rates must be non-negative with latent > 0.")
    return settings


def stage_for_epoch(epoch: int, config: dict[str, Any]) -> str:
    settings = warm_start_settings(config)
    if settings is not None and epoch <= int(settings["latent_adapt_epochs"]):
        return "latent_adapt"
    return "joint"


def epoch_in_stage(epoch: int, stage: str, config: dict[str, Any]) -> int:
    if stage == "latent_adapt":
        return epoch - 1
    settings = warm_start_settings(config)
    return epoch - (int(settings["latent_adapt_epochs"]) if settings else 0) - 1


def configure_trainability(model, embedding: torch.nn.Embedding, stage: str) -> None:
    if stage not in {"latent_adapt", "joint"}:
        raise ValueError(f"Unknown multires training stage {stage!r}.")
    decoder_trainable = stage == "joint"
    for parameter in model.parameters():
        parameter.requires_grad_(decoder_trainable)
    embedding.weight.requires_grad_(True)


def configure_learning_rates(
    optimizer, epoch: int, config: dict[str, Any], stage: str, stage_epoch: int
) -> None:
    interval = max(1, int(config["learning_rate_decay_interval"]))
    decay = float(config["learning_rate_decay_factor"]) ** (stage_epoch // interval)
    settings = warm_start_settings(config)
    rates = (
        settings["latent_adapt_learning_rates"]
        if stage == "latent_adapt" and settings is not None
        else config["learning_rates"]
    )
    for group in optimizer.param_groups:
        group["lr"] = float(rates[group["name"]]) * decay


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

    broad_count = broad_samples.shape[1]
    broad_xyz = broad_samples[:, :, :3].reshape(-1, 3)
    broad_target = broad_samples[:, :, 3:4].reshape(-1, 1)
    broad_codes = expand_codes(latent_codes, broad_count)
    broad_prediction = model(
        torch.cat((broad_codes, broad_xyz), dim=1), level_weights=level_weights
    )
    broad_l1 = clamped_l1(broad_prediction, broad_target, clamp_distance)

    near_count = near_samples.shape[1]
    near_xyz = near_samples[:, :, :3].reshape(-1, 3)
    near_target = near_samples[:, :, 3:4].reshape(-1, 1)
    near_codes = expand_codes(latent_codes, near_count)
    near_parts = model(
        torch.cat((near_codes, near_xyz), dim=1),
        level_weights=level_weights,
        return_parts=True,
    )
    near_l1 = clamped_l1(near_parts["sdf"], near_target, clamp_distance)

    code_l2 = latent_codes.square().mean()
    grid_l2, grid_tv = model.grid_regularization(level_weights)
    total = (
        float(weights.get("broad_sdf_l1", 1.0)) * broad_l1
        + float(weights.get("near_sdf_l1", 1.0)) * near_l1
        + float(weights.get("latent_l2", 0.0)) * code_l2
        + float(weights.get("grid_l2", 0.0)) * grid_l2
        + float(weights.get("grid_tv", 0.0)) * grid_tv
    )

    epsilon_np = finite_difference_epsilon(epoch, config, level_weights)
    eikonal_weight = effective_eikonal_weight(epoch, config)
    eikonal_loss = total.new_zeros(())
    gradient_mean = total.new_zeros(())
    gradient_median = total.new_zeros(())
    gradient_p95 = total.new_zeros(())
    gradient_absolute_error = total.new_zeros(())
    eikonal_points = 0
    replacement_shortfall = 0
    if eikonal_weight > 0.0:
        settings = config["eikonal"]
        selected_xyz, selected_codes, replacement_shortfall = select_near_surface_points(
            latent_codes,
            near_samples,
            float(settings["target_band"]),
            int(settings["points_per_chunk"]),
        )
        epsilon = torch.as_tensor(epsilon_np, device=selected_xyz.device, dtype=selected_xyz.dtype)
        gradient = numerical_spatial_gradient(
            model, selected_codes, selected_xyz, epsilon, level_weights
        )
        norm = torch.linalg.vector_norm(gradient, dim=1)
        eikonal_loss = torch.square(norm - 1.0).mean()
        total = total + eikonal_weight * eikonal_loss
        gradient_mean = norm.mean()
        gradient_median = norm.median()
        gradient_p95 = torch.quantile(norm, 0.95)
        gradient_absolute_error = torch.abs(norm - 1.0).mean()
        eikonal_points = len(norm)

    metrics = {
        "broad_sdf_l1": float(broad_l1.detach().cpu()),
        "near_sdf_l1": float(near_l1.detach().cpu()),
        "latent_l2": float(code_l2.detach().cpu()),
        "grid_l2": float(grid_l2.detach().cpu()),
        "grid_tv": float(grid_tv.detach().cpu()),
        "eikonal_loss": float(eikonal_loss.detach().cpu()),
        "eikonal_weight": float(eikonal_weight),
        "gradient_norm_mean": float(gradient_mean.detach().cpu()),
        "gradient_norm_median": float(gradient_median.detach().cpu()),
        "gradient_norm_p95": float(gradient_p95.detach().cpu()),
        "gradient_norm_absolute_error": float(gradient_absolute_error.detach().cpu()),
        "eikonal_points": float(eikonal_points),
        "eikonal_replacement_shortfall": float(replacement_shortfall),
        "epsilon_x": float(epsilon_np[0]),
        "epsilon_y": float(epsilon_np[1]),
        "epsilon_z": float(epsilon_np[2]),
        **{f"level_{resolution}_weight": float(weight) for resolution, weight in zip(model.grid_resolutions, level_weights)},
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def checkpoint_payload(
    epoch: int,
    model,
    embedding,
    optimizer,
    best_validation_l1: float,
    best_mesh_assd: float,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    warm_start_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "single_field_dense_multiresolution_sdf",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "latent_codes": embedding.weight.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_l1": float(best_validation_l1),
        "best_mesh_assd": float(best_mesh_assd),
        "config": config_public(config),
        "training_scan_ids": [row["scan_id"] for row in train_rows],
        "level_weights": level_weights_for_epoch(epoch, config),
        "warm_start_report": warm_start_report or {},
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
    warm_start_report: dict[str, Any] | None = None,
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
        warm_start_report,
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
    state = payload.get("rng_state", {})
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_codes(
    model,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
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
    value = float(np.mean([row["heldout_sdf_l1"] for row in per_scan]))
    return value, per_scan


def run_periodic_evaluation(
    config: dict[str, Any], checkpoint: Path, device: torch.device, epoch: int
) -> dict[str, Any]:
    script = Path(__file__).resolve().parent / "periodic_evaluate_multires.py"
    output = require_bulk_path(config["output_dir"]) / "periodic_evaluation" / f"epoch_{epoch:04d}"
    command = [
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
    ]
    subprocess.run(command, check=True)
    with (output / "summary.json").open("r", encoding="utf-8") as handle:
        import json

        return json.load(handle)


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

    if args.validate_only:
        print(
            f"Input validation passed: scans={len(rows)} train={len(train_rows)} "
            f"val={len(validation_pool)} output={output_dir}",
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
    decoder_warm_start = warm_start_settings(config)
    warm_start_report: dict[str, Any] = {}
    if args.resume:
        resume = (
            output_dir / "checkpoints" / f"{args.resume}.pth"
            if not Path(args.resume).is_file()
            else Path(args.resume).resolve()
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
        warm_start_report = payload.get("warm_start_report", {})
        restore_rng(payload)
    elif decoder_warm_start is not None:
        warm_start_report = warm_start_full_decoder(
            model,
            decoder_warm_start["checkpoint"],
            device,
            expected_network_arch=config["network_arch"],
        )
        expected_hash = decoder_warm_start.get("checkpoint_sha256")
        if expected_hash and warm_start_report["checkpoint_sha256"] != expected_hash:
            raise RuntimeError(
                "Full decoder warm-start checksum differs from the pinned runtime config."
            )
        warm_start_report.update(
            {
                "mode": "decoder_only",
                "target_training_scan_count": len(train_rows),
                "random_latent_count": len(train_rows),
                "source_latent_codes_loaded": False,
                "source_optimizer_loaded": False,
                "source_rng_loaded": False,
            }
        )

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
        latent_epochs = int(decoder_warm_start["latent_adapt_epochs"]) if decoder_warm_start else 0
        epochs = (
            [1, latent_epochs + 1, max(int(config["eikonal"]["start_epoch"]), latent_epochs + 2)]
            if decoder_warm_start
            else [1, 350, max(int(config["eikonal"]["start_epoch"]), 650)]
        )
        for epoch in epochs:
            stage = stage_for_epoch(epoch, config)
            configure_trainability(model, embedding, stage)
            configure_learning_rates(optimizer, epoch, config, stage, epoch_in_stage(epoch, stage, config))
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(
                model, embedding(indices), broad, near, epoch, config
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(embedding.parameters()),
                float(config["gradient_clip_norm"]),
            )
            if not math.isfinite(metrics["total"]) or not torch.isfinite(gradient_norm):
                raise RuntimeError("Smoke test produced a non-finite loss or gradient.")
            optimizer.step()
            print(
                f"smoke epoch={epoch} stage={stage} total={metrics['total']:.6g} "
                f"eik={metrics['eikonal_loss']:.6g} levels="
                f"{level_weights_for_epoch(epoch, config)}",
                flush=True,
            )
        print("Smoke test passed; no experiment files were written.", flush=True)
        return

    if args.benchmark_step:
        sampling_started = time.perf_counter()
        broad, near, indices = next(iter(loader))
        sampling_seconds = time.perf_counter() - sampling_started
        broad, near, indices = broad.to(device), near.to(device), indices.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_loss(
            model, embedding(indices), broad, near, 650, config
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embedding.parameters()),
            float(config["gradient_clip_norm"]),
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started
        peak_gib = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else float("nan")
        )
        print(
            f"benchmark passed: scenes={len(indices)} broad_per_scene={broad.shape[1]} "
            f"near_per_scene={near.shape[1]} sampling_seconds={sampling_seconds:.3f} "
            f"forward_backward_step_seconds={step_seconds:.3f} peak_cuda_gib={peak_gib:.3f} "
            f"loss={metrics['total']:.6g}; no experiment files were written.",
            flush=True,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    network_source = Path(__file__).resolve().parents[4] / "networks" / "shared_multires_grid_sdf.py"
    provenance = {
        "architecture": "single_field_dense_multiresolution_sdf",
        "manifest": file_signature(config["manifest"]),
        "network_source": file_signature(network_source),
        "sampling_utility_source": file_signature(REFERENCE_TASK),
        "config_source": file_signature(config["_config_path"]),
        "input_report": input_report,
        "persistent_output_policy": f"all paths below {require_bulk_path(config['output_dir'])}",
        "decoder_warm_start": warm_start_report,
    }
    generator_source = config.get("sdf_supervision", {}).get("source_file")
    if generator_source:
        provenance["sdf_generator_source"] = file_signature(generator_source)
    write_json(output_dir / "run_config.json", config_public(config))
    write_json(output_dir / "input_contract_report.json", input_report)
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "warm_start_report.json", warm_start_report)
    write_json(output_dir / "training_scan_ids.json", [row["scan_id"] for row in train_rows])
    write_json(output_dir / "validation_selection.json", [row["scan_id"] for row in validation_rows])

    total_epochs = int(args.epochs or config["total_epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        dataset.set_epoch(epoch)
        generator.manual_seed(stable_seed(f"train-order:{epoch}", int(config["seed"])))
        stage = stage_for_epoch(epoch, config)
        configure_trainability(model, embedding, stage)
        configure_learning_rates(
            optimizer, epoch, config, stage, epoch_in_stage(epoch, stage, config)
        )
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
                    embedding.weight.mul_(
                        torch.clamp(code_bound / (norms + 1.0e-12), max=1.0)
                    )

        row: dict[str, Any] = {
            "epoch": epoch,
            "stage": "multires_c2f" if decoder_warm_start is None else stage,
            "seconds": round(time.time() - started, 3),
            **{f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups},
            **{key: running_sum[key] / running_weight[key] for key in sorted(running_sum)},
        }
        print(
            f"epoch={epoch:04d} total={row['total']:.6g} near={row['near_sdf_l1']:.6g} "
            f"eik={row['eikonal_loss']:.6g} seconds={row['seconds']:.1f}",
            flush=True,
        )

        validation_every = int(config["validation"]["every_epochs"])
        if epoch % validation_every == 0 or epoch == total_epochs:
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
                    output_dir,
                    "best_sdf",
                    epoch,
                    model,
                    embedding,
                    optimizer,
                    best_validation_l1,
                    best_mesh_assd,
                    config,
                    train_rows,
                    compatibility=True,
                    warm_start_report=warm_start_report,
                )
        append_csv(output_dir / "logs" / "training_history.csv", row)
        latest_every = max(1, int(config.get("checkpoint_latest_every_epochs", 1)))
        if epoch % latest_every == 0 or epoch == total_epochs:
            save_checkpoint(
                output_dir,
                "latest",
                epoch,
                model,
                embedding,
                optimizer,
                best_validation_l1,
                best_mesh_assd,
                config,
                train_rows,
                compatibility=False,
                warm_start_report=warm_start_report,
            )
        save_every = int(config["checkpoint_every_epochs"])
        if epoch % save_every == 0 or epoch == total_epochs:
            label = f"epoch_{epoch:04d}"
            save_checkpoint(
                output_dir,
                label,
                epoch,
                model,
                embedding,
                optimizer,
                best_validation_l1,
                best_mesh_assd,
                config,
                train_rows,
                compatibility=True,
                warm_start_report=warm_start_report,
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
                        output_dir,
                        "best_mesh",
                        epoch,
                        model,
                        embedding,
                        optimizer,
                        best_validation_l1,
                        best_mesh_assd,
                        config,
                        train_rows,
                        compatibility=True,
                        warm_start_report=warm_start_report,
                    )
    print(
        f"Training complete: best held-out SDF L1={best_validation_l1:.8f}; "
        f"best validation ASSD={best_mesh_assd:.6f} mm; output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
