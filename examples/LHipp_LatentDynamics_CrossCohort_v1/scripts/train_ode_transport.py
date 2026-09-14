#!/usr/bin/env python3
"""Train a plain ODE or BrainODE transport, dz/dt = f(z, t, d), on a protocol view.

Training logic is task3/August ``train_latent_ode.py`` (the trainer behind the August plain_ode
and brainode anchors; the file name predates the Latent ODE and does not mean one): the only
optimizer objective is the standardized latent-trajectory MSE over every forward suffix and
backward prefix of each training sequence, integrated with fixed-step RK4. All epochs run;
best.pt is the lowest validation first-to-last selection score. Differences:

* data come from a stage-1 view, any of the four R1 representations is accepted;
* a test-leakage guard checks the loaded archives against the view's test split.

Its trajectory builder, loaders, epoch loop and validation are imported from task3, not copied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import benchmark_common as bc
import dynamics_core as D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--representation", required=True, choices=bc.ALL_REPRESENTATIONS)
    parser.add_argument("--method", required=True, choices=("plain_ode", "brainode"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def task3_ode_trainer():
    """task3's train_latent_ode module (plain ODE / BrainODE trainer), imported read-only."""
    D.core()
    import train_latent_ode as T

    if Path(T.__file__).resolve().parent != D.T3_SCRIPTS.resolve():
        raise ImportError(f"train_latent_ode resolved to {T.__file__}")
    return T


def main() -> int:
    args = parse_args()
    overrides = {key: json.loads(raw) for key, _, raw in (item.partition("=") for item in args.override)}
    config = D.load_recipe(args.method, args.representation, overrides)
    if set(config["loss"]) != {"latent_trajectory_mse_weight"} or float(config["loss"]["latent_trajectory_mse_weight"]) != 1.0:
        raise ValueError("ODE loss contract permits only unit-weight latent trajectory MSE")
    parts = D.core()
    C, M = parts["C"], parts["M"]
    T = task3_ode_trainer()
    registry = D.view_registry(args.view)
    representation = args.representation
    train_archive = C.load_archive(representation, "train", registry)
    val_archive = C.load_archive(representation, "val", registry)
    D.assert_no_test_leakage(args.view, [train_archive, val_archive])
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("train/validation subject leakage")
    device = D.device(args.device)
    training = config["training"]
    seed = int(args.seed)
    epochs = 1 if args.smoke else int(args.epochs if args.epochs is not None else training["epochs"])
    C.set_seed(seed)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    model = M.build_ode(config).to(device)
    train_rows = T.build_trajectories(train_archive, include_backward=True, first_visit_only=False)
    val_forward_rows = T.build_trajectories(val_archive, include_backward=False, first_visit_only=False)
    val_first_last_rows = T.build_trajectories(val_archive, include_backward=False, first_visit_only=True)
    batch_size = int(training["batch_size"])
    train_loaders = T.loaders(train_rows, batch_size, True, seed)
    val_forward_loaders = T.loaders(val_forward_rows, batch_size, False, seed)
    val_first_last_loaders = T.loaders(val_first_last_rows, batch_size, False, seed)
    raw_val = D.view_vertices(val_archive)
    substeps = int(training["integration_substeps"])
    max_train_batches = int(training.get("smoke_train_batches", 2)) if args.smoke else None
    max_val_records = int(training.get("smoke_val_records", 8)) if args.smoke else None

    print("=" * 96)
    print(f"{args.method} | view={args.view} | {representation} | seed={seed} | device={device} | parameters={C.parameter_count(model)}")
    print(f"train trajectories={len(train_rows)} val first-last={len(val_first_last_rows)} | latent-only objective | test loaded: no")

    if args.dry_run:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        metrics = T.train_epoch(model, train_loaders, optimizer, device, training, max_batches=1)
        validation = T.evaluate(model, geometry, val_first_last_loaders, raw_val, device, substeps, int(training.get("smoke_val_records", 8)))
        print("DRY RUN PASSED - nothing written.")
        print(json.dumps({"train": metrics, "validation_score": validation["selection_score"]}, indent=2))
        return 0

    run_name = args.run_name or f"{representation}_{args.method}_s{seed}"
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    output_dir = D.run_directory(args.view, representation, args.method, run_name, args.runs_root)
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"cannot resume without {checkpoint_dir / 'latest.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_json(output_dir / "resolved_config.json", D.provenance(args.view, representation, config, seed, {
        "trainer": "train_ode_transport.py (task3 train_latent_ode semantics)", "epochs": epochs, "device": str(device),
        "smoke": bool(args.smoke), "decoder_in_optimizer_loss": False,
    }))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(training["scheduler_step_size"]), gamma=float(training["scheduler_gamma"]))
    history: list[dict[str, Any]] = []
    start_epoch, best_score, best_epoch = 1, float("inf"), 0
    if args.resume:
        saved = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        history = list(saved.get("history", []))
        start_epoch = int(saved["epoch"]) + 1
        best_score, best_epoch = float(saved["best_selection_score"]), int(saved["best_epoch"])
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = T.train_epoch(model, train_loaders, optimizer, device, training, max_train_batches)
        val_forward = T.evaluate(model, geometry, val_forward_loaders, raw_val, device, substeps, max_val_records)
        val_first_last = T.evaluate(model, geometry, val_first_last_loaders, raw_val, device, substeps, max_val_records)
        scheduler.step()
        score = float(val_first_last["selection_score"])
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               **{f"train_{k}": v for k, v in train_metrics.items()}, "val_forward": val_forward, "val_first_last": val_first_last,
               "val_selection_score": score}
        history.append(row)
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "config": config, "representation": representation,
                 "method": args.method, "history": history, "best_selection_score": best_score, "best_epoch": best_epoch,
                 "view": args.view, "seed": seed, "decoder_in_optimizer_loss": False, "test_data_loaded": False}
        C.atomic_torch_save(checkpoint_dir / "latest.pt", state)
        if improved:
            C.atomic_torch_save(checkpoint_dir / "best.pt", state)
        bc.atomic_json(output_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs,
                       "best_epoch": best_epoch, "best_validation_selection_score": best_score, "test_data_loaded": False})
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=bc._json_default) + "\n")
        print(f"epoch {epoch:03d}/{epochs} latent_mse={train_metrics['latent_trajectory_mse']:.6f} val_score={score:.6f} best={best_score:.6f}@{best_epoch}", flush=True)
    bc.atomic_json(output_dir / "training_status.json", {
        "status": "complete", "epoch": epochs, "epochs_requested": epochs, "best_epoch": best_epoch,
        "best_validation_selection_score": best_score, "selected_checkpoint": str(checkpoint_dir / "best.pt"),
        "elapsed_minutes": (time.time() - started) / 60.0, "decoder_in_optimizer_loss": False, "test_data_loaded": False,
    })
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
