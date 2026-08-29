#!/usr/bin/env python3
"""Train one end-to-end, direct, non-ODE conditional LAMM surface cocycle."""

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
from _shared import activate_shared
from conditional_lamm_flow import ConditionalLAMMFlow
from region_layout import load_region_layout

activate_shared()
import objectives as O  # noqa: E402  (validated direct-Spiral objective, model-agnostic)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("method") != "direct_surface_cocycle":
        raise ValueError("method must be direct_surface_cocycle")
    contract = config.get("scientific_contract", {})
    if bool(contract.get("ode_used", True)):
        raise ValueError("This task prohibits ODE use")
    if not bool(contract.get("global_latent_bottleneck", False)):
        raise ValueError("LAMM must declare its internal global latent bottleneck")
    if not bool(contract.get("end_to_end", False)):
        raise ValueError("LAMM flow must be trained end to end")
    if not bool(contract.get("identity_by_construction", False)):
        raise ValueError("The direct surface residual must provide exact identity")
    model = config.get("model", {})
    if model.get("operator") != "lamm_mlpmixer":
        raise ValueError("model.operator must be lamm_mlpmixer")
    if [int(value) for value in model.get("region_scales", [])] != [43, 86]:
        raise ValueError("The latest-LAMM layout must use region_scales=[43,86]")
    latent_dim = int(model.get("latent_dim", 0))
    split = [int(value) for value in model.get("latent_split", [])]
    if latent_dim not in {128, 256}:
        raise ValueError("This controlled experiment supports latent_dim 128 or 256")
    if len(split) != 2 or any(value <= 0 for value in split) or sum(split) != latent_dim:
        raise ValueError("latent_split must contain two positive entries summing to latent_dim")
    for name in (
        "token_dim",
        "encoder_depth",
        "decoder_depth",
        "condition_dim",
        "time_frequencies",
        "latent_width",
        "latent_residual_blocks",
    ):
        if int(model.get(name, 0)) <= 0:
            raise ValueError(f"model.{name} must be positive")
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
    required_selection = {
        "velocity_score_weight",
        "velocity_validation_batch_size",
        "max_normalized_velocity_error",
        "min_velocity_speed_ratio",
        "max_velocity_speed_ratio",
        "max_first_last_flipped_face_fraction",
        "cocycle_validation_pairs",
    }
    missing_selection = required_selection.difference(selection)
    if missing_selection:
        raise KeyError(f"Missing selection settings: {sorted(missing_selection)}")
    if not bool(selection.get("velocity_aware", False)):
        raise ValueError("Selection must remain velocity-aware")
    if not 0.0 <= float(selection["min_velocity_speed_ratio"]) < float(
        selection["max_velocity_speed_ratio"]
    ):
        raise ValueError("Velocity speed-ratio bounds are invalid")


def build_model(
    config: dict[str, Any], root: Path, device: torch.device
) -> tuple[ConditionalLAMMFlow, dict[str, Any]]:
    del root  # geometry comes from the separately configured immutable data root
    statistics = D.load_statistics()
    layout = load_region_layout(
        config["model"].get("region_layout_checkpoint"),
        expected_regions=config["model"]["region_scales"],
    )
    model = ConditionalLAMMFlow(layout, statistics["faces"], config, statistics).to(device)
    return model, statistics


def validate_epoch(
    model: ConditionalLAMMFlow,
    validation_split: D.PreparedSplit,
    validation_rows: list[C.PairRow],
    statistics: dict[str, Any],
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    batch_size = int(config["training"]["evaluation_batch_size"])
    pair_limit = int(config["training"]["smoke_pair_limit"]) if smoke else None
    first_last = O.evaluate_rows(
        model,
        validation_split,
        O.first_last_rows(validation_split),
        batch_size,
        pair_limit,
        compute_flips=True,
    )
    all_pairs = O.evaluate_rows(
        model, validation_split, validation_rows, batch_size, pair_limit
    )
    defects = O.evaluate_cocycle(
        model,
        validation_split,
        validation_rows,
        float(statistics["endpoint_scale_mm"]),
        batch_size,
        int(config["selection"]["cocycle_validation_pairs"] if not smoke else pair_limit),
    )
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
        "adaptive_support": [],
    }


def checkpoint_payload(
    epoch: int,
    model: ConditionalLAMMFlow,
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
        "model_class": "ConditionalLAMMFlow",
        "operator": model.operator,
        "latent_dim": model.latent_dim,
        "latent_split": model.latent_split,
        "region_layout_fingerprint": model.layout_fingerprint,
        "ode_used": False,
        "global_latent_bottleneck": True,
        "end_to_end": True,
        "test_data_loaded": False,
    }


def _gradient_group(name: str) -> str | None:
    prefixes = {
        "condition.": "condition",
        "tokenizers.": "tokenizers",
        "encoder.": "encoder",
        "w_down.": "down_projection",
        "latent_input.": "latent_flow",
        "latent_blocks.": "latent_flow",
        "latent_output.": "latent_flow",
        "w_up.": "up_projection",
        "region_tokens.": "decoder_tokens",
        "decoder.": "decoder",
        "velocity_heads.": "velocity_heads",
    }
    return next((group for prefix, group in prefixes.items() if name.startswith(prefix)), None)


def train_experiment(
    config: dict[str, Any],
    device_name: str,
    root: Path,
    data_root: Path | None = None,
    run_name: str | None = None,
    seed_override: int | None = None,
    epochs_override: int | None = None,
    resume: bool = False,
    smoke: bool = False,
    trial: Any | None = None,
) -> dict[str, Any]:
    validate_config(config)
    C.configure_data_root(data_root)
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

    # Training is deliberately blind to test data.
    train_split = D.load_split("train", device=device)
    val_split = D.load_split("val", device=device)
    if set(train_split.subject_ids.astype(str)) & set(val_split.subject_ids.astype(str)):
        raise ValueError("Train/validation subject leakage")
    train_rows = C.load_pairs(
        "train", train_split.scan_ids, train_split.subject_ids, train_split.labels.cpu().numpy()
    )
    val_rows = C.load_pairs(
        "val", val_split.scan_ids, val_split.subject_ids, val_split.labels.cpu().numpy()
    )
    model, statistics = build_model(config, root, device)
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
        0,
        model,
        optimizer,
        scheduler,
        config,
        baseline_validation,
        baseline_validation["score"],
        0,
    )
    if not resume:
        C.atomic_torch_save(checkpoints / "epoch_0000_nochange.pt", baseline_payload)
        C.atomic_torch_save(checkpoints / "best.pt", baseline_payload)
        best_score = float(baseline_validation["score"])

    physical_batch = int(training["batch_size"])
    accumulation = 1 if smoke else int(training.get("gradient_accumulation", 1))
    samples = int(training["samples_per_epoch"])
    if smoke:
        samples = physical_batch * int(training.get("smoke_train_batches", 4))
    history_path = output / "history.jsonl"
    started = time.time()
    completed_epoch = start_epoch - 1
    gradient_reach = {
        name: 0.0
        for name in (
            "condition",
            "tokenizers",
            "encoder",
            "down_projection",
            "latent_flow",
            "up_projection",
            "decoder_tokens",
            "decoder",
            "velocity_heads",
        )
    }
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
            for parameter_name, parameter in model.named_parameters():
                group = _gradient_group(parameter_name)
                if group is not None and parameter.grad is not None:
                    gradient_reach[group] += float(parameter.grad.detach().abs().sum().cpu())
            if step % accumulation == 0 or step * physical_batch >= samples:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
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
        payload = checkpoint_payload(
            epoch, model, optimizer, scheduler, config, validation, best_score, best_epoch
        )
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
            "end_to_end_gradient_l1": gradient_reach,
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
                "operator": model.operator,
                "latent_dim": model.latent_dim,
                "parameters": C.parameter_count(model),
                "parameter_breakdown": model.parameter_breakdown(),
                "end_to_end_gradient_l1": gradient_reach,
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
    if smoke:
        missing_gradients = [name for name, value in gradient_reach.items() if value <= 0.0]
        if missing_gradients:
            raise RuntimeError(
                "End-to-end smoke gate failed; zero accumulated gradient for "
                f"{missing_gradients}"
            )
    status = {
        "status": "complete",
        "epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selected_checkpoint": str(checkpoints / "best.pt"),
        "operator": model.operator,
        "latent_dim": model.latent_dim,
        "latent_split": model.latent_split,
        "parameters": C.parameter_count(model),
        "parameter_breakdown": model.parameter_breakdown(),
        "end_to_end_gradient_l1": gradient_reach,
        "all_components_received_gradient": all(value > 0.0 for value in gradient_reach.values()),
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
        data_root=args.data_root,
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

