#!/usr/bin/env python3
"""Train the direct C4 cocycle, Phi(z,s,t,d) = z + (t-s)[v_CN + d v_AD], on a protocol view.

Training logic is August_Version/scripts/train_c4.py, the trainer behind the August anchors:
the 13-leaf objective from task3's c4_objective, a subject-balanced pair sampler, cosine LR,
early stopping whose patience counts genuine (unconstrained) score improvement, and best.pt =
best feasible validation epoch. Differences from that file:

* data come from a stage-1 view (any cohort mix), not the ADNI-only registry, and any of the
  four R1 representations is accepted;
* a test-leakage guard checks the loaded archives against the view's test split;
* best_min_epoch.pt additionally keeps the best feasible epoch at or after
  ``selection.min_selection_epoch_sensitivity`` (sensitivity analysis only);
* ``--method`` also trains two stage-5 ablations with exactly this objective and loop:
  ``direct_c4_no_disease`` (A3, the condition never reaches the velocity) and ``brainode_v``
  (A2, BrainODE's RK4 field in place of the direct cocycle).

Only train and val are opened. Test is evaluator-only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

import benchmark_common as bc
import dynamics_core as D

COCYCLE_OBJECTIVE_METHODS = ("direct_c4", "direct_c4_no_disease", "brainode_v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--representation", required=True, choices=bc.ALL_REPRESENTATIONS)
    parser.add_argument("--method", default="direct_c4", choices=COCYCLE_OBJECTIVE_METHODS)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[], help="dotted.key=json_value, e.g. training.learning_rate=1e-4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_overrides(items: list[str]) -> dict[str, Any]:
    output = {}
    for item in items:
        key, _, raw = item.partition("=")
        output[key] = json.loads(raw)
    return output


class PairDataset(Dataset):
    def __init__(self, rows) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def validate_config(config: dict[str, Any]) -> None:
    method = config.get("method")
    if method not in COCYCLE_OBJECTIVE_METHODS or config.get("representation") not in bc.ALL_REPRESENTATIONS:
        raise ValueError("config must be a cocycle-objective method on a known representation")
    model = config["model"]
    if int(model["latent_dim"]) != 128:
        raise ValueError("the transport must act on 128-D codes")
    if method == "brainode_v":
        if model.get("variant") != "brainode_v" or not model.get("ode_used"):
            raise ValueError("BrainODE-V must declare the brainode_v ODE variant")
    elif model.get("variant") != "direct" or model.get("ode_used") or model.get("attention_used"):
        raise ValueError("C4 must be the direct 128-D variant without ODE or attention")
    if method == "direct_c4_no_disease" and model.get("disease_head", True):
        raise ValueError("ablation A3 must declare model.disease_head = false")
    loss = config["loss"]
    if float(loss.get("coboundary_weight", -1.0)) != 0.0:
        raise ValueError("coboundary weight must be present and exactly zero")


def validation(flow, geometry, values, archive, pair_rows, raw_vertices, statistics, config, limit):
    parts = D.core()
    C, O = parts["C"], parts["O"]
    rows = O.balanced_subset(pair_rows, limit)
    first_last = O.balanced_first_last_subset(C.first_last_pairs(archive), limit)
    batch_size = int(config["training"]["evaluation_batch_size"])
    all_metrics = O.evaluate_pairs(flow, geometry, values, rows, raw_vertices, batch_size)
    first_last_metrics = O.evaluate_pairs(flow, geometry, values, first_last, raw_vertices, batch_size)
    defects = O.cocycle_defects(flow, values, rows, statistics, batch_size)
    score, feasible, ratios = O.validation_score(all_metrics, first_last_metrics, defects, config["selection"])
    return {"score": score, "feasible": feasible, "ratios": ratios, "all_pairs": all_metrics, "first_last": first_last_metrics, "defects": defects}


def gradient_norm(module) -> float:
    return math.sqrt(sum(float(torch.sum(p.grad.detach().square()).cpu()) for p in module.parameters() if p.grad is not None))


def main() -> int:
    args = parse_args()
    config = D.load_recipe(args.method, args.representation, parse_overrides(args.override))
    validate_config(config)
    method = config["method"]
    parts = D.core()
    C, O = parts["C"], parts["O"]
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
    train_values = C.values_on_device(train_archive, device)
    val_values = C.values_on_device(val_archive, device)
    O.attach_reference_geometry(train_values, geometry, int(training["decoder_batch_size"]))
    O.attach_reference_geometry(val_values, geometry, int(training["decoder_batch_size"]))
    train_pairs = C.load_pairs("train", train_archive, registry)
    val_pairs = C.load_pairs("val", val_archive, registry)
    smoke_limit = int(training.get("smoke_pair_limit", 16)) if (args.smoke or args.dry_run) else None
    statistics = O.training_statistics(train_values, train_pairs, train_archive, smoke_limit)
    flow = D.build_transport(config, device)
    raw_val = D.view_vertices(val_archive)
    batch_size = int(training["batch_size"])

    print("=" * 96)
    print(f"{method} | view={args.view} | {representation} | seed={seed} | device={device} | parameters={C.parameter_count(flow)}")
    print(f"train scans={len(train_archive['visit_scan_ids'])} pairs={len(train_pairs)} | val scans={len(val_archive['visit_scan_ids'])} | test loaded: no")

    # Gradient routing audit: the frozen-decoder reconstruction term must reach the flow and only the flow.
    loader = DataLoader(PairDataset(train_pairs), batch_size=batch_size, collate_fn=C.collate_pairs,
                        sampler=O.balanced_pair_sampler(train_pairs, seed, 0, max(batch_size, 4)), num_workers=0)
    pair = O.pair_terms(flow, geometry, train_values, next(iter(loader)), statistics)
    sequence = O.sequence_terms(flow, geometry, train_values, train_archive, O.balanced_sequence_starts(train_archive, seed, 1)[0], statistics)
    flow.zero_grad(set_to_none=True)
    pair["real_vertex"].backward(retain_graph=True)
    reconstruction_gradient = gradient_norm(flow)
    decoder_gradient = any(p.grad is not None for p in geometry.parameters())
    flow.zero_grad(set_to_none=True)
    total, terms = O.total_loss(pair, sequence, config, 1)
    total.backward()
    total_gradient = gradient_norm(flow)
    if decoder_gradient or not (math.isfinite(reconstruction_gradient) and reconstruction_gradient > 0 and math.isfinite(total_gradient) and total_gradient > 0):
        raise RuntimeError("C4 gradient audit failed")
    flow.zero_grad(set_to_none=True)
    flow.eval()
    baseline = validation(flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics, config, smoke_limit)
    if args.dry_run:
        print("DRY RUN PASSED - finite C4 gradients, decoder routing and validation; nothing written.")
        print(json.dumps({"loss": float(total.detach().cpu()), "total_gradient": total_gradient, "validation_score": baseline["score"]}, indent=2))
        return 0

    run_name = args.run_name or f"{representation}_{method}_s{seed}"
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    output_dir = D.run_directory(args.view, representation, method, run_name, args.runs_root)
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"cannot resume without {checkpoint_dir / 'latest.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_json(output_dir / "resolved_config.json", D.provenance(args.view, representation, config, seed, {
        "trainer": "train_cocycle.py (August train_c4 semantics)", "epochs": epochs, "device": str(device), "smoke": bool(args.smoke),
    }))
    bc.atomic_json(output_dir / "training_statistics.json", statistics)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training["minimum_learning_rate"]))
    min_epoch = int(config["selection"].get("min_selection_epoch_sensitivity", 0))
    best_score, best_epoch = float(baseline["score"]), 0
    best_any_score, best_any_epoch = float(baseline["score"]), 0
    best_min_score, best_min_epoch = float("inf"), -1
    stale, start_epoch = 0, 1

    def payload(epoch: int, current: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": epoch, "flow_state_dict": flow.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "config": config, "statistics": statistics, "validation": current,
            "best_selection_score": best_score, "best_epoch": best_epoch,
            "best_unconstrained_score": best_any_score, "best_unconstrained_epoch": best_any_epoch,
            "best_min_epoch_score": best_min_score, "best_min_epoch_epoch": best_min_epoch,
            "view": args.view, "seed": seed, "decoder_in_optimizer_loss": True, "test_data_loaded": False,
        }

    if args.resume:
        saved = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        flow.load_state_dict(saved["flow_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_score, best_epoch = float(saved["best_selection_score"]), int(saved["best_epoch"])
        best_any_score, best_any_epoch = float(saved["best_unconstrained_score"]), int(saved["best_unconstrained_epoch"])
        best_min_score, best_min_epoch = float(saved["best_min_epoch_score"]), int(saved["best_min_epoch_epoch"])
        stale = int(bc.read_json(output_dir / "training_status.json").get("stale_epochs", 0))
    else:
        initial = payload(0, baseline)
        bc_save = parts["C"].atomic_torch_save
        bc_save(checkpoint_dir / "epoch_0000_nochange.pt", initial)
        bc_save(checkpoint_dir / "best.pt", initial)

    samples_per_epoch = batch_size * int(training.get("smoke_train_batches", 2)) if args.smoke else int(training["samples_per_epoch"])
    started = time.time()
    completed = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        flow.train()
        loader = DataLoader(PairDataset(train_pairs), batch_size=batch_size, collate_fn=C.collate_pairs, num_workers=0,
                            sampler=O.balanced_pair_sampler(train_pairs, seed, epoch, samples_per_epoch))
        starts = O.balanced_sequence_starts(train_archive, seed, epoch)
        totals: dict[str, float] = {}
        batches = 0
        for step, raw in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            pair = O.pair_terms(flow, geometry, train_values, raw, statistics)
            sequence = O.sequence_terms(flow, geometry, train_values, train_archive, starts[(step - 1) % len(starts)], statistics)
            loss, current_terms = O.total_loss(pair, sequence, config, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C4 loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in current_terms.items():
                totals[name] = totals.get(name, 0.0) + value
        scheduler.step()
        flow.eval()
        current = validation(flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics, config, smoke_limit)
        score, feasible = float(current["score"]), bool(current["feasible"])
        delta = float(training["early_stopping_min_delta"])
        improved_any = score < best_any_score - delta
        if improved_any:
            best_any_score, best_any_epoch = score, epoch
        if feasible and score < best_score - delta:
            best_score, best_epoch = score, epoch
        if feasible and epoch >= min_epoch and score < best_min_score - delta:
            best_min_score, best_min_epoch = score, epoch
        stale = 0 if improved_any else stale + 1
        state = payload(epoch, current)
        C.atomic_torch_save(checkpoint_dir / "latest.pt", state)
        if feasible and best_epoch == epoch:
            C.atomic_torch_save(checkpoint_dir / "best.pt", state)
        if best_any_epoch == epoch:
            C.atomic_torch_save(checkpoint_dir / "best_unconstrained.pt", state)
        if best_min_epoch == epoch:
            C.atomic_torch_save(checkpoint_dir / "best_min_epoch.pt", state)
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "batches": batches, **{f"train_{k}": v / max(batches, 1) for k, v in totals.items()}, "validation": current}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=bc._json_default) + "\n")
        bc.atomic_json(output_dir / "training_status.json", {
            "status": "running", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch,
            "best_validation_selection_score": best_score, "best_unconstrained_epoch": best_any_epoch,
            "best_min_epoch_epoch": best_min_epoch, "stale_epochs": stale, "test_data_loaded": False,
        })
        completed = epoch
        print(f"epoch {epoch:03d}/{epochs} loss={totals['total'] / max(batches, 1):.6f} val={score:.6f} feasible={feasible} "
              f"best={best_score:.6f}@{best_epoch} min_epoch_best@{best_min_epoch}", flush=True)
        if not args.smoke and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch}")
            break
    bc.atomic_json(output_dir / "training_status.json", {
        "status": "complete", "epoch": completed, "epochs_requested": epochs, "best_epoch": best_epoch,
        "best_validation_selection_score": best_score, "best_unconstrained_epoch": best_any_epoch,
        "best_min_epoch_epoch": best_min_epoch, "best_min_epoch_score": best_min_score,
        "selected_checkpoint": str(checkpoint_dir / "best.pt"), "stale_epochs": stale,
        "elapsed_minutes": (time.time() - started) / 60.0, "test_data_loaded": False,
    })
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
