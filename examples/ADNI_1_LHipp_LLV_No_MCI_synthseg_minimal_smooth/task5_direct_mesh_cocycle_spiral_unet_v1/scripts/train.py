#!/usr/bin/env python3
"""Train a direct, non-ODE conditional mesh cocycle."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

import common as C
import data as D
import objectives as O
from conditional_spiral_unet import ConditionalSpiralUNet
from mesh_hierarchy import load_hierarchy
from mesh_layers import adaptive_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("method") != "direct_surface_cocycle":
        raise ValueError("method must be direct_surface_cocycle")
    if bool(config.get("scientific_contract", {}).get("ode_used", True)):
        raise ValueError("This task prohibits ODE use")
    if bool(config.get("scientific_contract", {}).get("global_latent_bottleneck", True)):
        raise ValueError("This task prohibits a global latent bottleneck")
    model = config.get("model", {})
    if model.get("operator") not in {"spiral", "adaptive_spiral"}:
        raise ValueError("model.operator must be spiral or adaptive_spiral")
    if len(model.get("channels", [])) != 4:
        raise ValueError("model.channels must have four entries")
    required_loss = {
        "endpoint_weight",
        "normal_endpoint_weight",
        "observed_cocycle_weight",
        "virtual_cocycle_weight",
        "local_cocycle_weight",
        "inverse_weight",
        "diagonal_velocity_weight",
        "volume_weight",
        "rate_weight",
        "group_rate_weight",
        "disease_gap_weight",
        "edge_weight",
        "smoothness_weight",
        "tangent_weight",
        "flip_weight",
    }
    missing = required_loss.difference(config.get("loss", {}))
    if missing:
        raise KeyError(f"Missing loss weights: {sorted(missing)}")
    selection = config.get("selection", {})
    if bool(selection.get("velocity_aware", False)):
        required_selection = {
            "velocity_score_weight",
            "velocity_validation_batch_size",
            "max_normalized_velocity_error",
            "min_velocity_speed_ratio",
            "max_velocity_speed_ratio",
            "max_first_last_flipped_face_fraction",
        }
        missing_selection = required_selection.difference(selection)
        if missing_selection:
            raise KeyError(f"Missing velocity-aware selection settings: {sorted(missing_selection)}")
        if not 0.0 < float(selection["max_normalized_velocity_error"]):
            raise ValueError("max_normalized_velocity_error must be positive")
        if not 0.0 <= float(selection["min_velocity_speed_ratio"]) < float(
            selection["max_velocity_speed_ratio"]
        ):
            raise ValueError("Velocity speed-ratio bounds are invalid")


def build_model(
    config: dict[str, Any], root: Path, device: torch.device
) -> tuple[ConditionalSpiralUNet, dict[str, Any]]:
    statistics = D.load_statistics(root)
    hierarchy = load_hierarchy(root, device)
    model = ConditionalSpiralUNet(hierarchy, config, statistics).to(device)
    return model, statistics


def validate_epoch(
    model: ConditionalSpiralUNet,
    validation_split: D.PreparedSplit,
    validation_rows: list[C.PairRow],
    statistics: dict[str, Any],
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    batch_size = int(config["training"]["evaluation_batch_size"])
    pair_limit = int(config["training"]["smoke_pair_limit"]) if smoke else None
    velocity_aware = bool(config["selection"].get("velocity_aware", False))
    first_last = O.evaluate_rows(
        model,
        validation_split,
        O.first_last_rows(validation_split),
        batch_size,
        pair_limit,
        compute_flips=velocity_aware,
    )
    all_pairs = O.evaluate_rows(model, validation_split, validation_rows, batch_size, pair_limit)
    defects = O.evaluate_cocycle(
        model,
        validation_split,
        validation_rows,
        float(statistics["endpoint_scale_mm"]),
        batch_size,
        int(config["selection"]["cocycle_validation_pairs"] if not smoke else pair_limit),
    )
    velocity = None
    if velocity_aware:
        configured_limit = int(config["selection"].get("velocity_validation_max_scans", 0))
        velocity = O.evaluate_velocity_selection(
            model,
            validation_split,
            int(config["selection"]["velocity_validation_batch_size"]),
            max_scans=(2 * int(pair_limit)) if smoke else configured_limit,
        )
    selection_metrics = O.validation_selection(first_last, defects, config, velocity)
    return {
        "score": float(selection_metrics["score"]),
        "feasible": bool(selection_metrics["feasible"]),
        "selection_metrics": selection_metrics,
        "first_last": first_last,
        "all_pairs": all_pairs,
        "defects": defects,
        "velocity_selection": velocity,
        "adaptive_support": model.adaptive_support_report(),
    }


def checkpoint_payload(
    epoch: int,
    model: ConditionalSpiralUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    validation: dict[str, Any],
    best_score: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "validation": validation,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "model_class": "ConditionalSpiralUNet",
        "operator": config["model"]["operator"],
        "ode_used": False,
        "global_latent_bottleneck": False,
        "test_data_loaded": False,
    }


def train_experiment(
    config: dict[str, Any],
    device_name: str,
    root: Path,
    run_name: str | None = None,
    seed_override: int | None = None,
    epochs_override: int | None = None,
    resume: bool = False,
    smoke: bool = False,
    trial: Any | None = None,
) -> dict[str, Any]:
    validate_config(config)
    device = C.choose_device(device_name)
    training = config["training"]
    seed = int(training["seed"] if seed_override is None else seed_override)
    epochs = int(training["epochs"] if epochs_override is None else epochs_override)
    if smoke:
        epochs = int(training.get("smoke_epochs", 1))
    C.set_seed(seed)
    name = str(run_name or training["run_name"])
    output = root / "runs" / name
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    # Deliberately load only train and validation. Test is owned by evaluate.py.
    train_split = D.load_split("train", root, device)
    val_split = D.load_split("val", root, device)
    if set(train_split.subject_ids.astype(str)) & set(val_split.subject_ids.astype(str)):
        raise ValueError("Train/validation subject leakage")
    train_rows = C.load_pairs("train", train_split.scan_ids, train_split.subject_ids, train_split.labels.cpu().numpy())
    val_rows = C.load_pairs("val", val_split.scan_ids, val_split.subject_ids, val_split.labels.cpu().numpy())
    model, statistics = build_model(config, root, device)
    adaptive_initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "prefix_pool.predictor" in name
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training["minimum_learning_rate"]),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    stale = 0
    latest = checkpoints / "latest.pt"
    if resume:
        if not latest.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {latest}")
        payload = torch.load(latest, map_location=device, weights_only=False)
        if payload["config"]["model"] != config["model"]:
            raise ValueError("Resume model config mismatch")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        best_epoch = int(payload["best_epoch"])

    baseline_validation = validate_epoch(model, val_split, val_rows, statistics, config, smoke)
    baseline_payload = checkpoint_payload(
        0, model, optimizer, scheduler, config, baseline_validation, baseline_validation["score"], 0
    )
    if not resume:
        C.atomic_torch_save(checkpoints / "epoch_0000_nochange.pt", baseline_payload)
        C.atomic_torch_save(checkpoints / "best.pt", baseline_payload)
        best_score = float(baseline_validation["score"])

    physical_batch = int(training["batch_size"])
    accumulation = 1 if smoke else int(training.get("gradient_accumulation", 1))
    samples = int(training["samples_per_epoch"])
    if smoke:
        samples = physical_batch * int(training.get("smoke_train_batches", 2))
    history_path = output / "history.jsonl"
    started = time.time()
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        indices = C.balanced_pair_indices(train_rows, samples, seed + 100_003 * epoch)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        for step, index_chunk in enumerate(C.chunked(indices, physical_batch), start=1):
            rows = [train_rows[index] for index in index_chunk]
            batch = train_split.pair_batch(rows)
            with torch.cuda.amp.autocast(enabled=use_amp):
                terms = O.pair_objective(model, train_split, batch, statistics, config, epoch)
                loss = terms["total"] / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step * physical_batch >= samples:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batches += 1
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        scheduler.step()
        validation = validate_epoch(model, val_split, val_rows, statistics, config, smoke)
        score = float(validation["score"])
        feasible = bool(validation["feasible"])
        improved = feasible and score < best_score - float(training["early_stopping_min_delta"])
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        payload = checkpoint_payload(epoch, model, optimizer, scheduler, config, validation, best_score, best_epoch)
        C.atomic_torch_save(latest, payload)
        if improved:
            C.atomic_torch_save(checkpoints / "best.pt", payload)
        elapsed = time.time() - started
        history = {
            "epoch": epoch,
            "elapsed_minutes": elapsed / 60.0,
            "epoch_seconds_mean": elapsed / max(epoch - start_epoch + 1, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "validation": validation,
        }
        output.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history, sort_keys=True) + "\n")
        C.atomic_json(
            output / "training_status.json",
            {
                "status": "running",
                "epoch": epoch,
                "epochs_requested": epochs,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale_epochs": stale,
                "operator": config["model"]["operator"],
                "parameters": C.parameter_count(model),
                "test_data_loaded": False,
            },
        )
        print(
            f"epoch {epoch:03d}/{epochs} loss={totals['total']/max(batches,1):.6f} "
            f"val={score:.6f} feasible={feasible} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
        completed_epoch = epoch
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned(f"Pruned at epoch {epoch}")
        if not smoke and stale >= int(training["early_stopping_patience"]):
            break
    adaptive_change = 0.0
    adaptive_change_by_module: dict[str, float] = {}
    if adaptive_initial:
        current_parameters = dict(model.named_parameters())
        for name, initial in adaptive_initial.items():
            module_name = name.split(".prefix_pool.predictor", 1)[0]
            adaptive_change_by_module[module_name] = adaptive_change_by_module.get(module_name, 0.0) + float(
                (current_parameters[name].detach().cpu() - initial).abs().sum()
            )
        adaptive_change = float(sum(adaptive_change_by_module.values()))
        unchanged = [name for name, value in adaptive_change_by_module.items() if value <= 0.0]
        if smoke and unchanged:
            raise RuntimeError(f"Adaptive smoke gate failed; unchanged support predictors: {unchanged}")
    status = {
        "status": "complete",
        "epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selected_checkpoint": str(checkpoints / "best.pt"),
        "operator": config["model"]["operator"],
        "parameters": C.parameter_count(model),
        "adaptive_support_parameter_change_l1": adaptive_change,
        "adaptive_support_parameter_change_by_module": adaptive_change_by_module,
        "test_data_loaded": False,
    }
    C.atomic_json(output / "training_status.json", status)
    return status


def main() -> int:
    args = parse_args()
    config = C.read_json(args.config)
    root = C.output_root(args.output_root)
    status = train_experiment(
        copy.deepcopy(config),
        args.device,
        root,
        run_name=args.run_name,
        seed_override=args.seed,
        epochs_override=args.epochs,
        resume=args.resume,
        smoke=args.smoke,
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
