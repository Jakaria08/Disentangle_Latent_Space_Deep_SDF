#!/usr/bin/env python3
"""Train the staged global/local shared-grid SDF representation."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared_grid_common import (
    ContinuousSDFDataset,
    build_decoder,
    choose_device,
    fit_single_latent,
    load_checkpoint_training_scan_ids,
    load_config,
    load_manifest,
    resolve_repo_path,
    rows_for_split,
    seed_dataloader_worker,
    select_stratified_rows,
    set_global_seed,
    stable_seed,
    validate_manifest_contract,
    warm_start_full_decoder,
    warm_start_global_decoder,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment JSON config.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument("--resume", default=None, help="Checkpoint path, or latest/best_sdf.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument(
        "--skip-periodic-evaluation",
        action="store_true",
        help="Do not run the expensive 100-per-split geometry evaluation.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run tiny updates for all stages without writing experiment files.",
    )
    return parser.parse_args()


def stage_for_epoch(epoch: int, schedule: dict[str, Any]) -> str:
    latent_end = int(schedule.get("latent_adapt_epochs", 0))
    if epoch <= latent_end:
        return "latent_adapt"
    global_end = latent_end + int(schedule["global_adapt_epochs"])
    local_end = global_end + int(schedule["local_warmup_epochs"])
    if epoch <= global_end:
        return "global_adapt"
    if epoch <= local_end:
        return "local_warmup"
    return "joint"


def epoch_in_stage(epoch: int, schedule: dict[str, Any], stage: str) -> int:
    latent_epochs = int(schedule.get("latent_adapt_epochs", 0))
    if stage == "latent_adapt":
        return epoch - 1
    if stage == "global_adapt":
        return epoch - latent_epochs - 1
    if stage == "local_warmup":
        return epoch - latent_epochs - int(schedule["global_adapt_epochs"]) - 1
    return (
        epoch
        - latent_epochs
        - int(schedule["global_adapt_epochs"])
        - int(schedule["local_warmup_epochs"])
        - 1
    )


def fusion_alpha(epoch: int, schedule: dict[str, Any]) -> float:
    stage = stage_for_epoch(epoch, schedule)
    if stage == "latent_adapt":
        alpha = float(schedule.get("latent_adapt_fusion_alpha", 1.0))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("latent_adapt_fusion_alpha must lie in [0, 1].")
        return alpha
    if stage != "joint":
        return 0.0
    ramp = max(1, int(schedule.get("joint_fusion_ramp_epochs", 1)))
    return min(1.0, (epoch_in_stage(epoch, schedule, "joint") + 1) / ramp)


def make_optimizer(model, embedding: torch.nn.Embedding, config: dict[str, Any]):
    base_lrs = config["stage_learning_rates"]["global_adapt"]
    groups = [
        {"name": "global", "params": model.global_decoder.parameters(), "lr": float(base_lrs["global"])},
        {"name": "local", "params": model.local_decoder.parameters(), "lr": float(base_lrs["local"])},
        {"name": "grid", "params": [model.shared_grid], "lr": float(base_lrs["grid"])},
        {"name": "latent", "params": embedding.parameters(), "lr": float(base_lrs["latent"])},
    ]
    return torch.optim.Adam(groups, betas=(0.9, 0.999))


def configure_stage(
    model,
    embedding: torch.nn.Embedding,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    stage: str,
    stage_epoch: int,
) -> None:
    model.set_train_stage(stage)
    embedding.weight.requires_grad_(stage != "local_warmup")
    rates = config["stage_learning_rates"][stage]
    interval = int(config["learning_rate_decay_interval"])
    decay = float(config["learning_rate_decay_factor"]) ** (stage_epoch // interval)
    for group in optimizer.param_groups:
        group["lr"] = float(rates[group["name"]]) * decay


def _expand_codes(codes: torch.Tensor, count: int) -> torch.Tensor:
    return codes[:, None, :].expand(-1, count, -1).reshape(-1, codes.shape[1])


def _clamped_l1(prediction: torch.Tensor, target: torch.Tensor, distance: float) -> torch.Tensor:
    return torch.abs(
        prediction.clamp(-distance, distance) - target.clamp(-distance, distance)
    ).mean()


def compute_loss(
    model,
    latent_codes: torch.Tensor,
    global_samples: torch.Tensor,
    local_samples: torch.Tensor,
    stage: str,
    epoch: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compact-SDF-style direct branch losses plus latent/grid regularization."""
    weights = config["loss_weights"]
    clamp_distance = float(config["clamp_distance"])
    alpha = fusion_alpha(epoch, config["schedule"])
    metrics: dict[str, torch.Tensor] = {}
    total = latent_codes.sum() * 0.0

    if stage in {"latent_adapt", "global_adapt", "joint"}:
        global_count = global_samples.shape[1]
        global_xyz = global_samples[:, :, :3].reshape(-1, 3)
        global_target = global_samples[:, :, 3:4].reshape(-1, 1)
        global_codes = _expand_codes(latent_codes, global_count)
        global_parts = model(
            torch.cat((global_codes, global_xyz), dim=1),
            return_parts=True,
            global_only=stage == "global_adapt",
            fusion_alpha=alpha,
        )
        global_l1 = _clamped_l1(global_parts["global_sdf"], global_target, clamp_distance)
        total = total + float(weights.get("global_l1", 1.0)) * global_l1
        metrics["global_l1"] = global_l1
    else:
        global_parts = None

    if stage in {"latent_adapt", "local_warmup", "joint"}:
        local_count = local_samples.shape[1]
        local_xyz = local_samples[:, :, :3].reshape(-1, 3)
        local_target = local_samples[:, :, 3:4].reshape(-1, 1)
        local_codes = _expand_codes(latent_codes, local_count)
        if stage == "local_warmup":
            local_sdf, features, _inside = model.local_sdf(local_codes, local_xyz)
            local_parts = {"local_sdf": local_sdf, "features": features}
        else:
            local_parts = model(
                torch.cat((local_codes, local_xyz), dim=1),
                return_parts=True,
                fusion_alpha=alpha,
            )
        local_l1 = _clamped_l1(local_parts["local_sdf"], local_target, clamp_distance)
        total = total + float(weights.get("local_l1", 1.0)) * local_l1
        metrics["local_l1"] = local_l1

        if stage in {"latent_adapt", "joint"}:
            fused_global_l1 = _clamped_l1(global_parts["sdf"], global_target, clamp_distance)
            fused_local_l1 = _clamped_l1(local_parts["sdf"], local_target, clamp_distance)
            total = total + float(weights.get("fused_global_l1", 0.0)) * fused_global_l1
            total = total + float(weights.get("fused_local_l1", 0.0)) * fused_local_l1
            metrics["fused_global_l1"] = fused_global_l1
            metrics["fused_local_l1"] = fused_local_l1
            metrics["fusion_alpha"] = total.new_tensor(alpha)

        regularization_scale = min(
            1.0, epoch / max(1, int(config.get("regularization_warmup_epochs", 100)))
        )
        feature_l2 = local_parts["features"].square().mean()
        total = total + regularization_scale * float(
            weights.get("sampled_grid_l2", 0.0)
        ) * feature_l2
        metrics["sampled_grid_l2"] = feature_l2
        if float(weights.get("grid_tv", 0.0)):
            _grid_l2, grid_tv = model.grid_regularization()
            total = total + regularization_scale * float(weights["grid_tv"]) * grid_tv
            metrics["grid_tv"] = grid_tv

    code_l2 = latent_codes.square().mean()
    if stage != "local_warmup":
        regularization_scale = min(
            1.0, epoch / max(1, int(config.get("regularization_warmup_epochs", 100)))
        )
        total = total + regularization_scale * float(weights.get("code_l2", 0.0)) * code_l2
    metrics["code_l2"] = code_l2
    metrics["total"] = total
    return total, {name: float(value.detach().cpu()) for name, value in metrics.items()}


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    epoch: int,
    model,
    embedding,
    optimizer,
    best_validation_l1: float,
    best_mesh_assd: float,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    warm_start_report: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "latent_codes": embedding.weight.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_l1": best_validation_l1,
        "best_mesh_assd": best_mesh_assd,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "training_scan_ids": [row["scan_id"] for row in train_rows],
        "warm_start_report": warm_start_report,
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
    path: Path,
    epoch: int,
    model,
    embedding,
    optimizer,
    best_validation_l1: float,
    best_mesh_assd: float,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    warm_start_report: dict[str, Any],
    compatibility_label: str | None = None,
) -> None:
    payload = checkpoint_payload(
        epoch, model, embedding, optimizer, best_validation_l1, best_mesh_assd,
        config, train_rows, warm_start_report,
    )
    _atomic_torch_save(payload, path)
    if compatibility_label:
        output_dir = resolve_repo_path(config["output_dir"])
        _atomic_torch_save(
            {"epoch": epoch, "model_state_dict": payload["model_state_dict"]},
            output_dir / "ModelParameters" / f"{compatibility_label}.pth",
        )
        _atomic_torch_save(
            {"epoch": epoch, "optimizer_state_dict": payload["optimizer_state_dict"]},
            output_dir / "OptimizerParameters" / f"{compatibility_label}.pth",
        )
        _atomic_torch_save(
            {
                "epoch": epoch,
                "latent_codes": {"weight": payload["latent_codes"]},
                "train_scan_ids": payload["training_scan_ids"],
            },
            output_dir / "LatentCodes" / f"{compatibility_label}.pth",
        )


def restore_rng_state(payload: dict[str, Any]) -> None:
    state = payload.get("rng_state", {})
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def validation_l1(
    model,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> tuple[float, str, list[dict[str, Any]]]:
    stage = stage_for_epoch(epoch, config["schedule"])
    fit_config = copy.deepcopy(config["validation"]["latent_fit"])
    if stage == "global_adapt":
        fit_config["branch_weights"] = {"global": 1.0, "local": 0.0, "fused": 0.0}
        selection_metric = "heldout_global_l1"
    elif stage == "local_warmup":
        fit_config["branch_weights"] = {"global": 1.0, "local": 1.0, "fused": 0.0}
        selection_metric = "heldout_local_l1"
    else:
        selection_metric = "heldout_fused_l1"
    per_scan = []
    for row in rows:
        _latent, metrics = fit_single_latent(
            model,
            row["sdf_npz_path"],
            int(config["latent_size"]),
            fit_config,
            float(config["clamp_distance"]),
            device,
            stable_seed(row["scan_id"], int(config["seed"])),
            mesh_path=row["mesh_path"],
            network_specs=config["network_specs"],
        )
        per_scan.append({"epoch": epoch, "scan_id": row["scan_id"], **metrics})
    return float(np.mean([row[selection_metric] for row in per_scan])), selection_metric, per_scan


def smoke_overrides(config: dict[str, Any]) -> None:
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
    config["scenes_per_batch"] = 2
    config["scenes_per_chunk"] = 1
    config["num_workers"] = 0
    config["load_dataset_into_ram"] = False


def should_run_periodic(epoch: int, total_epochs: int, config: dict[str, Any]) -> bool:
    periodic = config.get("periodic_evaluation", {})
    if not bool(periodic.get("enabled", False)):
        return False
    every = int(periodic.get("every_epochs", 500))
    return epoch % every == 0 or (
        epoch == total_epochs and bool(periodic.get("also_final_epoch", True))
    )


def run_periodic_evaluation(
    config: dict[str, Any], checkpoint: Path, device: torch.device, epoch: int
) -> dict[str, Any]:
    script = Path(__file__).resolve().parent / "periodic_evaluate_shared_grid.py"
    output = resolve_repo_path(config["output_dir"]) / "periodic_evaluation" / f"epoch_{epoch:04d}"
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
        "--use-saved-selection",
    ]
    subprocess.run(command, check=True)
    import json

    with (output / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    device = choose_device(args.device)
    if args.smoke_test:
        smoke_overrides(config)

    manifest_rows = load_manifest(config["manifest"])
    input_report = (
        {"skipped_for_non_writing_smoke_test": True}
        if args.smoke_test
        else validate_manifest_contract(
            manifest_rows, config["network_specs"]["grid_aabb"]
        )
    )
    train_rows = rows_for_split(manifest_rows, "train")
    val_rows = rows_for_split(manifest_rows, "val")
    if args.smoke_test:
        train_rows = train_rows[:2]
        val_rows = val_rows[:1]
    if not train_rows:
        raise RuntimeError("The manifest has no training rows.")

    decoder_warm_start = config.get("decoder_warm_start", {})
    use_full_decoder_warm_start = bool(decoder_warm_start.get("enabled", False))
    if use_full_decoder_warm_start:
        if str(decoder_warm_start.get("mode", "")) != "decoder_only":
            raise ValueError("Full decoder warm-start mode must be 'decoder_only'.")
        if any(
            bool(decoder_warm_start.get(key, True))
            for key in ("copy_source_latents", "load_source_optimizer", "load_source_rng")
        ):
            raise ValueError(
                "Full decoder warm-start must not import source latents, optimizer, or RNG state."
            )
        if bool(config.get("reuse_matching_source_latents", False)):
            raise ValueError("Full decoder warm-start cannot reuse source latent codes.")
        source_checkpoint = decoder_warm_start.get("checkpoint")
        if not source_checkpoint:
            raise ValueError("Full decoder warm-start requires a checkpoint path.")
        source_training_scan_ids = load_checkpoint_training_scan_ids(source_checkpoint)
    else:
        source_checkpoint = config["global_checkpoint"]
        source_training_scan_ids = load_checkpoint_training_scan_ids(
            source_checkpoint, str(config.get("global_checkpoint_scan_id_suffix", ""))
        )
    source_seen_subjects = {
        row["subject_id"] for row in manifest_rows if row["scan_id"] in source_training_scan_ids
    }
    source_overlap = {}
    for split in ("train", "val", "test"):
        split_rows = rows_for_split(manifest_rows, split)
        split_subjects = {row["subject_id"] for row in split_rows}
        source_overlap[split] = {
            "scan_count": len(split_rows),
            "source_seen_scan_count": sum(row["scan_id"] in source_training_scan_ids for row in split_rows),
            "subject_count": len(split_subjects),
            "source_seen_subject_count": len(split_subjects & source_seen_subjects),
            "source_unseen_subject_count": len(split_subjects - source_seen_subjects),
        }
    validation_pool = val_rows
    if bool(config["validation"].get("source_unseen_subjects_only", False)) and not args.smoke_test:
        validation_pool = [row for row in val_rows if row["subject_id"] not in source_seen_subjects]
    if not validation_pool:
        raise RuntimeError("No eligible validation scans are available.")
    validation_rows = select_stratified_rows(
        validation_pool,
        min(len(validation_pool), int(config["validation"]["scans_per_validation"])),
        int(config["seed"]),
    )

    model = build_decoder(config, device)
    embedding = torch.nn.Embedding(
        len(train_rows), int(config["latent_size"]), max_norm=float(config["code_bound"])
    ).to(device)
    torch.nn.init.normal_(embedding.weight, mean=0.0, std=float(config["code_initial_std"]))
    optimizer = make_optimizer(model, embedding, config)
    output_dir = resolve_repo_path(config["output_dir"])
    if bool(config.get("require_bulk_output", False)):
        bulk_root = Path("/mnt/bulk10tb").resolve()
        try:
            output_dir.resolve().relative_to(bulk_root)
        except ValueError as error:
            raise ValueError(
                f"Configured persistent output must be below {bulk_root}, got {output_dir}."
            ) from error
    start_epoch = 1
    best_validation_l1 = math.inf
    best_mesh_assd = math.inf
    warm_report: dict[str, Any]

    if args.resume:
        resume_value = Path(args.resume)
        if not resume_value.is_file():
            name = str(args.resume) if str(args.resume).endswith(".pth") else f"{args.resume}.pth"
            resume_value = output_dir / "checkpoints" / name
        payload = torch.load(resume_value, map_location=device)
        if payload.get("training_scan_ids") != [row["scan_id"] for row in train_rows]:
            raise ValueError("Resume checkpoint training scan order does not match the manifest.")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        embedding.weight.data.copy_(payload["latent_codes"].to(device))
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation_l1 = float(payload.get("best_validation_l1", math.inf))
        best_mesh_assd = float(payload.get("best_mesh_assd", math.inf))
        warm_report = payload.get("warm_start_report", {})
        restore_rng_state(payload)
    else:
        reuse_latents = bool(config.get("reuse_matching_source_latents", False))
        if use_full_decoder_warm_start:
            warm_report = warm_start_full_decoder(
                model,
                decoder_warm_start["checkpoint"],
                device,
                expected_network_arch=config["network_arch"],
            )
            expected_hash = decoder_warm_start.get("checkpoint_sha256")
            if expected_hash and warm_report["checkpoint_sha256"] != expected_hash:
                raise RuntimeError(
                    "Full decoder warm-start checksum differs from the pinned runtime config."
                )
            warm_report.update(
                {
                    "mode": "decoder_only",
                    "target_training_scan_count": len(train_rows),
                    "random_latent_count": len(train_rows),
                    "source_latent_codes_loaded": False,
                    "source_optimizer_loaded": False,
                    "source_rng_loaded": False,
                }
            )
        else:
            warm_report = warm_start_global_decoder(
                model,
                config["global_checkpoint"],
                device,
                latent_embedding=embedding if reuse_latents else None,
                target_scan_ids=[row["scan_id"] for row in train_rows] if reuse_latents else None,
                source_scan_id_suffix=str(config.get("global_checkpoint_scan_id_suffix", "")),
            )
            warm_report["latent_transfer"] = (
                warm_report.get("latent_transfer", "matched_scan_id") if reuse_latents else "disabled_new_qc_mesh_codes"
            )
            warm_report["random_latent_count"] = len(train_rows) if not reuse_latents else warm_report["random_latent_count"]
    warm_report["source_pretraining_overlap"] = source_overlap
    warm_report["validation_selection"] = {
        "source_unseen_subjects_only": bool(config["validation"].get("source_unseen_subjects_only", False)),
        "scan_ids": [row["scan_id"] for row in validation_rows],
    }

    dataset = ContinuousSDFDataset(train_rows, config)
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
    total_epochs = int(args.epochs or config["schedule"]["total_epochs"])

    if args.smoke_test:
        global_samples, local_samples, indices = next(iter(loader))
        global_samples = global_samples.to(device)
        local_samples = local_samples.to(device)
        indices = indices.to(device)
        schedule = config["schedule"]
        boundaries = [
            ("latent_adapt", 1, int(schedule.get("latent_adapt_epochs", 0))),
            (
                "global_adapt",
                int(schedule.get("latent_adapt_epochs", 0)) + 1,
                int(schedule["global_adapt_epochs"]),
            ),
            (
                "local_warmup",
                int(schedule.get("latent_adapt_epochs", 0))
                + int(schedule["global_adapt_epochs"])
                + 1,
                int(schedule["local_warmup_epochs"]),
            ),
            (
                "joint",
                int(schedule.get("latent_adapt_epochs", 0))
                + int(schedule["global_adapt_epochs"])
                + int(schedule["local_warmup_epochs"])
                + 1,
                1,
            ),
        ]
        for stage, epoch, duration in boundaries:
            if duration < 1:
                continue
            configure_stage(model, embedding, optimizer, config, stage, 0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(
                model, embedding(indices), global_samples, local_samples, stage, epoch, config
            )
            loss.backward()
            optimizer.step()
            if not math.isfinite(metrics["total"]):
                raise RuntimeError(f"Smoke loss is non-finite in {stage}.")
            print(f"smoke stage={stage} total={metrics['total']:.6g}", flush=True)
        print("Smoke test passed for all stages; no experiment files were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {key: value for key, value in config.items() if not key.startswith("_")})
    write_json(output_dir / "input_contract_report.json", input_report)
    write_json(output_dir / "warm_start_report.json", warm_report)
    write_json(output_dir / "training_scan_ids.json", [row["scan_id"] for row in train_rows])
    write_json(output_dir / "validation_selection.json", [row["scan_id"] for row in validation_rows])

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        dataset.set_epoch(epoch)
        generator.manual_seed(stable_seed(f"train-order:{epoch}", int(config["seed"])))
        stage = stage_for_epoch(epoch, config["schedule"])
        configure_stage(
            model, embedding, optimizer, config, stage,
            epoch_in_stage(epoch, config["schedule"], stage),
        )
        model.train()
        running_sum: defaultdict[str, float] = defaultdict(float)
        running_weight: defaultdict[str, float] = defaultdict(float)
        for global_samples, local_samples, indices in loader:
            global_samples = global_samples.to(device, non_blocking=True)
            local_samples = local_samples.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            batch_count = len(indices)
            optimizer.zero_grad(set_to_none=True)
            chunk_size = int(config["scenes_per_chunk"])
            for start in range(0, batch_count, chunk_size):
                stop = min(start + chunk_size, batch_count)
                scene_weight = (stop - start) / batch_count
                loss, metrics = compute_loss(
                    model,
                    embedding(indices[start:stop]),
                    global_samples[start:stop],
                    local_samples[start:stop],
                    stage,
                    epoch,
                    config,
                )
                (loss * scene_weight).backward()
                for key, value in metrics.items():
                    running_sum[key] += value * (stop - start)
                    running_weight[key] += stop - start
            trainable_parameters = [
                parameter
                for parameter in list(model.parameters()) + list(embedding.parameters())
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(config["gradient_clip_norm"]))
            optimizer.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "stage": stage,
            "seconds": round(time.time() - epoch_start, 3),
            **{f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups},
            **{key: running_sum[key] / running_weight[key] for key in sorted(running_sum)},
        }
        print(
            f"epoch={epoch:04d} stage={stage} total={row.get('total', math.nan):.6g} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
        validation_every = int(config["validation"]["every_epochs"])
        if epoch % validation_every == 0 or epoch == total_epochs:
            value, metric_name, per_scan = validation_l1(model, validation_rows, config, device, epoch)
            for item in per_scan:
                append_csv(output_dir / "logs" / "validation_per_scan.csv", item)
            append_csv(
                output_dir / "logs" / "validation_history.csv",
                {"epoch": epoch, "stage": stage, "selection_metric": metric_name, "value": value},
            )
            if stage == "joint" and value < best_validation_l1:
                best_validation_l1 = value
                save_checkpoint(
                    output_dir / "checkpoints" / "best_sdf.pth", epoch, model, embedding,
                    optimizer, best_validation_l1, best_mesh_assd, config, train_rows,
                    warm_report, compatibility_label="best_sdf",
                )

        append_csv(output_dir / "logs" / "training_history.csv", row)
        latest_every = int(config.get("checkpoint_latest_every_epochs", 1))
        if epoch % latest_every == 0 or epoch == total_epochs:
            save_checkpoint(
                output_dir / "checkpoints" / "latest.pth", epoch, model, embedding,
                optimizer, best_validation_l1, best_mesh_assd, config, train_rows, warm_report,
            )
        schedule = config["schedule"]
        latent_end = int(schedule.get("latent_adapt_epochs", 0))
        global_end = latent_end + int(schedule["global_adapt_epochs"])
        local_end = global_end + int(schedule["local_warmup_epochs"])
        stage_boundaries = {value for value in (latent_end, global_end, local_end) if value > 0}
        save_every = int(config["checkpoint_every_epochs"])
        if epoch in stage_boundaries or epoch % save_every == 0 or epoch == total_epochs:
            label = f"epoch_{epoch:04d}"
            save_checkpoint(
                output_dir / "checkpoints" / f"{label}.pth", epoch, model, embedding,
                optimizer, best_validation_l1, best_mesh_assd, config, train_rows,
                warm_report, compatibility_label=label,
            )

        if should_run_periodic(epoch, total_epochs, config) and not args.skip_periodic_evaluation:
            checkpoint = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
            report = run_periodic_evaluation(config, checkpoint, device, epoch)
            validation_assd = float(report["selection_metric"]["validation_inr_assd_mm"])
            if validation_assd < best_mesh_assd:
                best_mesh_assd = validation_assd
                save_checkpoint(
                    output_dir / "checkpoints" / "best_mesh.pth", epoch, model, embedding,
                    optimizer, best_validation_l1, best_mesh_assd, config, train_rows,
                    warm_report, compatibility_label="best_mesh",
                )
                save_checkpoint(
                    output_dir / "checkpoints" / "latest.pth", epoch, model, embedding,
                    optimizer, best_validation_l1, best_mesh_assd, config, train_rows,
                    warm_report,
                )

    print(
        f"Training complete. Best validation fused L1={best_validation_l1:.8f}; "
        f"best validation mesh ASSD={best_mesh_assd:.6f} mm"
    )


if __name__ == "__main__":
    main()
