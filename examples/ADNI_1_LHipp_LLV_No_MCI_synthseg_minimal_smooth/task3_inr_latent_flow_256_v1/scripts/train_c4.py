#!/usr/bin/env python3
"""Train direct, non-ODE INR-256 C4 with a frozen differentiable SDF decoder."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import c4_objective as O
import common as C
from inr_geometry import build_geometry
from models import DirectC4Flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="One differentiability/validation audit; no files written")
    parser.add_argument("--smoke", action="store_true", help="One bounded epoch written to a smoke run")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("representation") != "inr256" or config.get("method") != "direct_c4":
        raise ValueError("Expected the inr256 direct_c4 contract")
    if int(config["model"].get("latent_dim", -1)) != C.LATENT_DIM:
        raise ValueError("C4 latent dimension must be 256")
    if config["model"].get("variant") != "direct" or bool(config["model"].get("ode_used", True)):
        raise ValueError("C4 must be direct and non-ODE")
    if float(config["loss"].get("coboundary_weight", -1.0)) != 0.0:
        raise ValueError("Coboundary is prohibited")
    if not bool(config["scientific_contract"].get("exact_sdf_targets")):
        raise ValueError("Direct C4 must declare exact-SDF targets")
    training = config["training"]
    if int(training["batch_size"]) * int(training["gradient_accumulation_steps"]) != int(training["effective_batch_size"]):
        raise ValueError("Effective batch-size contract is inconsistent")


def balanced(rows: list[C.PairRow], limit: int | None) -> list[C.PairRow]:
    if limit is None or len(rows) <= limit:
        return list(rows)
    cn, ad = [row for row in rows if row.diagnosis == "CN"], [row for row in rows if row.diagnosis == "AD"]
    each = max(1, limit // 2)
    return cn[:each] + ad[:each]


def validation(flow, geometry, values, pairs, first_last, training, scales, selection, limit=None):
    current_pairs = balanced(pairs, limit)
    current_first_last = balanced(first_last, limit)
    batch_size = int(training["evaluation_batch_size"])
    all_metrics = O.evaluate_pairs(flow, geometry, values, current_pairs, training, batch_size)
    first_metrics = O.evaluate_pairs(flow, geometry, values, current_first_last, training, batch_size)
    defects = O.cocycle_defects(flow, values, current_pairs, scales["displacement"], batch_size)
    score, feasible, ratios = O.validation_score(all_metrics, first_metrics, defects, selection)
    return {"score": score, "feasible": feasible, "ratios": ratios, "defects": defects, "all_pairs": all_metrics, "first_last": first_metrics}


def sampled_epoch_rows(rows: list[C.PairRow], samples: int, generator: np.random.Generator) -> list[C.PairRow]:
    cn = [row for row in rows if row.diagnosis == "CN"]
    ad = [row for row in rows if row.diagnosis == "AD"]
    half = samples // 2
    chosen = [cn[int(index)] for index in generator.integers(0, len(cn), size=half)]
    chosen += [ad[int(index)] for index in generator.integers(0, len(ad), size=samples - half)]
    generator.shuffle(chosen)
    return chosen


def train_epoch(flow, geometry, values, rows, optimizer, training, weights, scales, epoch, max_batches=None):
    flow.train()
    geometry.eval()
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    consistency_ramp = min(1.0, epoch / max(float(training["consistency_ramp_epochs"]), 1.0))
    anatomy_ramp = min(1.0, epoch / max(float(training["anatomy_ramp_epochs"]), 1.0))
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    observed = 0
    batches = 0
    optimizer_steps = 0
    for start in range(0, len(rows), batch_size):
        current = rows[start:start + batch_size]
        batch = C.indexed(values, current)
        loss, terms = O.pair_loss(flow, geometry, batch, weights, scales, training, consistency_ramp, anatomy_ramp)
        (loss / accumulation).backward()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * len(current)
        observed += len(current)
        batches += 1
        if batches % accumulation == 0 or start + batch_size >= len(rows):
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if max_batches is not None and batches >= int(max_batches):
            break
    if any(parameter.grad is not None for parameter in geometry.decoder.parameters()):
        raise RuntimeError("Frozen decoder accumulated parameter gradients")
    return {key: value / max(observed, 1) for key, value in totals.items()} | {"pairs": float(observed), "batches": float(batches), "optimizer_steps": float(optimizer_steps)}


def main() -> int:
    args = parse_args()
    config_path = C.resolve_path(args.config)
    config = C.read_json(config_path)
    validate_config(config)
    registry = C.load_registry()
    # Deliberately do not load or validate test here.
    train_archive = C.load_archive("train", registry)
    val_archive = C.load_archive("val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Train/validation subject leakage")
    device = C.choose_device(args.device)
    training, weights, selection = config["training"], config["loss"], config["selection"]
    seed = int(args.seed if args.seed is not None else training["seed"])
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    if args.smoke:
        epochs = 1
    C.set_seed(seed)
    train_values, val_values = C.values_on_device(train_archive, device), C.values_on_device(val_archive, device)
    train_pairs = C.load_pairs("train", train_archive, registry)
    val_pairs = C.load_pairs("val", val_archive, registry)
    val_first_last = C.first_last_pairs(val_archive)
    geometry = build_geometry(train_archive, device, registry)
    flow = DirectC4Flow(C.LATENT_DIM, int(config["model"]["width"]), int(config["model"]["residual_blocks"]), float(config["model"].get("dropout", 0.0))).to(device)
    scales = O.compute_scales(flow, geometry, train_values, train_pairs, training)
    print(f"direct_c4 | inr256 | device={device} | flow_params={C.parameter_count(flow):,}")
    print(f"decoder_epoch={geometry.checkpoint_epoch} decoder_frozen={all(not p.requires_grad for p in geometry.decoder.parameters())}")
    print(f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} val_first_last={len(val_first_last)} test_loaded=no")
    print(f"scales={json.dumps(scales, sort_keys=True)}")
    if args.dry_run:
        optimizer = torch.optim.AdamW(flow.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        rows = balanced(train_pairs, int(training["smoke_pair_limit"]))
        metrics = train_epoch(flow, geometry, train_values, rows, optimizer, training, weights, scales, 1, max_batches=1)
        report = validation(flow, geometry, val_values, val_pairs, val_first_last, training, scales, selection, int(training["smoke_pair_limit"]))
        gradient_signal = sum(float(parameter.detach().abs().sum().cpu()) for name, parameter in flow.named_parameters() if "head" in name)
        if not math.isfinite(gradient_signal) or gradient_signal <= 0.0:
            raise RuntimeError("C4 differentiability audit failed: zero transport-head update")
        print("DRY RUN PASSED — exact-SDF gradients reach C4; decoder weights remain frozen; no files written.")
        print(json.dumps({"train": metrics, "validation": report, "head_parameter_signal": gradient_signal}, indent=2, sort_keys=True))
        return 0
    run_name = str(args.run_name or training["run_name"])
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    C.validate_run_name(run_name)
    run_dir = C.output_root(registry) / "training" / "inr256" / "direct_c4" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(checkpoint_dir / "latest.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_json(run_dir / "resolved_config.json", {
        "config_path": str(config_path), "config": config,
        "representation_manifest": C.read_json(C.output_root(registry) / "representations" / "inr256" / "manifest.json"),
        "device": str(device), "seed": seed, "epochs": epochs, "smoke": bool(args.smoke),
        "decoder_in_optimizer_loss": True, "decoder_frozen": True, "test_data_loaded": False,
        "loss_scales": scales,
    })
    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(training["minimum_learning_rate"]))
    start_epoch, best_epoch, best_score, stale = 1, 0, float("inf"), 0
    history = []
    baseline = validation(flow, geometry, val_values, val_pairs, val_first_last, training, scales, selection, int(training["smoke_pair_limit"]) if args.smoke else None)
    baseline_payload = {"epoch": 0, "flow_state_dict": flow.state_dict(), "config": config, "statistics": {"normalization_scales": scales}, "validation": baseline, "best_validation_selection_score": baseline["score"], "best_epoch": 0, "decoder_frozen": True, "test_data_loaded": False}
    best_score = float(baseline["score"])
    C.atomic_torch_save(checkpoint_dir / "best.pt", baseline_payload)
    if args.resume:
        payload = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        flow.load_state_dict(payload["flow_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch, best_epoch = int(payload["epoch"]) + 1, int(payload["best_epoch"])
        best_score, stale, history = float(payload["best_validation_selection_score"]), int(payload.get("stale_epochs", 0)), list(payload.get("history", []))
    started = time.time()
    completed_epoch = start_epoch - 1
    generator = np.random.default_rng(seed + start_epoch)
    for epoch in range(start_epoch, epochs + 1):
        rows = sampled_epoch_rows(train_pairs, int(training["samples_per_epoch"]), generator)
        train_metrics = train_epoch(flow, geometry, train_values, rows, optimizer, training, weights, scales, epoch, int(training["smoke_train_batches"]) if args.smoke else None)
        val = validation(flow, geometry, val_values, val_pairs, val_first_last, training, scales, selection, int(training["smoke_pair_limit"]) if args.smoke else None)
        scheduler.step()
        improved = bool(val["feasible"] and val["score"] < best_score - float(training["early_stopping_min_delta"]))
        if improved:
            best_score, best_epoch, stale = float(val["score"]), epoch, 0
        else:
            stale += 1
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0, "learning_rate": float(optimizer.param_groups[0]["lr"]), "train": train_metrics, "validation": val, "improved": improved}
        history.append(row)
        payload = {"epoch": epoch, "flow_state_dict": flow.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "statistics": {"normalization_scales": scales}, "validation": val, "history": history, "best_validation_selection_score": best_score, "best_epoch": best_epoch, "stale_epochs": stale, "decoder_frozen": True, "test_data_loaded": False}
        C.atomic_torch_save(checkpoint_dir / "latest.pt", payload)
        if improved:
            C.atomic_torch_save(checkpoint_dir / "best.pt", payload)
        C.atomic_json(run_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_selection_score": best_score, "stale_epochs": stale, "coboundary_used": False, "decoder_frozen": True, "test_data_loaded": False})
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"epoch {epoch:03d}/{epochs} loss={train_metrics['total']:.5f} sdf={train_metrics['real_sdf']:.5f} val={val['score']:.6f} feasible={val['feasible']} best={best_score:.6f}@{best_epoch}", flush=True)
        completed_epoch = epoch
        if not args.smoke and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping after {stale} stale epochs", flush=True)
            break
    C.atomic_json(run_dir / "training_status.json", {"status": "complete", "epoch": completed_epoch, "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_selection_score": best_score, "selected_checkpoint": str(checkpoint_dir / "best.pt"), "coboundary_used": False, "decoder_frozen": True, "test_data_loaded": False})
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
