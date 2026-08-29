#!/usr/bin/env python3
"""Train exact coupling-coboundary C4 with a frozen representation decoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


BASE_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(BASE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPTS))

import common as C  # noqa: E402
import coboundary_objective as O  # noqa: E402
from coboundary_model import ExactCouplingCoboundaryFlow, build_flow  # noqa: E402


class PairDataset(Dataset[C.PairRow]):
    def __init__(self, rows: list[C.PairRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> C.PairRow:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("method") != "exact_coboundary_c4":
        raise ValueError("Method must be exact_coboundary_c4")
    if config.get("representation") not in {"pca128", "spiralnet128", "adaptive128"}:
        raise ValueError("Unknown representation")
    model = config.get("model", {})
    if int(model.get("latent_dim", -1)) != C.LATENT_DIM:
        raise ValueError("Coboundary model must use 128-D latent coordinates")
    if model.get("variant") != "exact_coupling_group_coboundary":
        raise ValueError("Wrong exact-coboundary model variant")
    if float(model.get("dropout", 0.0)) != 0.0:
        raise ValueError("Dropout is forbidden inside an analytically inverted coupling map")
    if bool(model.get("ode_used", True)) or bool(model.get("attention_used", True)):
        raise ValueError("Exact coboundary C4 uses neither ODE integration nor attention")
    loss = config.get("loss", {})
    required = {
        "real_latent_weight", "real_vertex_weight", "observed_semigroup_weight",
        "virtual_semigroup_weight", "inverse_weight", "sequence_latent_weight",
        "sequence_vertex_weight", "sequence_semigroup_weight", "volume_weight",
        "rate_weight", "slope_weight", "group_rate_weight", "disease_gap_weight",
    }
    if not required.issubset(loss):
        raise KeyError(f"Loss is missing {sorted(required.difference(loss))}")
    structural = ("observed_semigroup_weight", "virtual_semigroup_weight", "inverse_weight", "sequence_semigroup_weight")
    if any(float(loss[name]) != 0.0 for name in structural):
        raise ValueError("Exact structural identities must be audited, not approximated with loss penalties")
    contract = config.get("scientific_contract", {})
    if not bool(contract.get("coboundary_used")) or not bool(contract.get("structural_exactness")):
        raise ValueError("Scientific contract must explicitly declare exact coboundary use")


def validation(
    flow: ExactCouplingCoboundaryFlow,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    archive: dict,
    pair_rows: list[C.PairRow],
    raw_vertices,
    statistics: dict[str, Any],
    config: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    rows = O.balanced_subset(pair_rows, limit)
    first_last = O.balanced_first_last_subset(C.first_last_pairs(archive), limit)
    batch_size = int(config["training"]["evaluation_batch_size"])
    all_metrics = O.evaluate_pairs(flow, geometry, values, rows, raw_vertices, batch_size)
    first_last_metrics = O.evaluate_pairs(flow, geometry, values, first_last, raw_vertices, batch_size)
    defects = O.cocycle_defects(flow, values, rows, statistics, batch_size)
    score, feasible, ratios = O.validation_score(all_metrics, first_last_metrics, defects, config["selection"])
    return {
        "score": score,
        "feasible": feasible,
        "ratios": ratios,
        "all_pairs": all_metrics,
        "first_last": first_last_metrics,
        "defects": defects,
    }


def checkpoint_payload(
    epoch: int,
    flow: ExactCouplingCoboundaryFlow,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    statistics: dict[str, Any],
    current_validation: dict[str, Any],
    best_score: float,
    best_epoch: int,
    best_any_score: float,
    best_any_epoch: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "flow_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "statistics": statistics,
        "validation": current_validation,
        "best_selection_score": best_score,
        "best_epoch": best_epoch,
        "best_unconstrained_score": best_any_score,
        "best_unconstrained_epoch": best_any_epoch,
        "model_variant": flow.variant,
        "coboundary_used": True,
        "structural_exactness": True,
        "decoder_frozen": True,
        "decoder_in_optimizer_loss": True,
        "test_data_loaded": False,
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(torch.sum(parameter.grad.detach().square()).cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    ))


def main() -> int:
    args = parse_args()
    config_path = C.resolve_path(args.config)
    config = C.read_json(config_path)
    validate_config(config)
    registry = C.load_registry()
    representation = str(config["representation"])
    # Deliberately load train and validation only. Test remains sealed during training.
    train_archive = C.load_archive(representation, "train", registry)
    val_archive = C.load_archive(representation, "val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Train/validation subject leakage")
    device = C.choose_device(args.device)
    training = dict(config["training"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    if args.smoke:
        epochs = 1
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
    flow = build_flow(config).to(device)
    raw_val = C.cached_vertices("val", registry)

    print("=" * 96)
    print(f"exact coupling coboundary C4 | {representation} | device={device} | parameters={C.parameter_count(flow)}")
    print("Phi(z,s,t;c,d)=F(t;c,d)(F(s;c,d)^-1(z)); ODE=no; attention=no; structural coboundary=yes")
    print("context=subject first visit (fixed for all legs); decoder=frozen+differentiable; test loaded=no")
    print(json.dumps(statistics, sort_keys=True))

    initial_sampler = O.balanced_pair_sampler(train_pairs, seed, 0, max(int(training["batch_size"]), 4))
    initial_loader = DataLoader(
        PairDataset(train_pairs), batch_size=int(training["batch_size"]), sampler=initial_sampler,
        collate_fn=C.collate_pairs, num_workers=0,
    )
    raw = next(iter(initial_loader))
    pair = O.pair_terms(flow, geometry, train_values, raw, statistics)
    sequence_start = O.balanced_sequence_starts(train_archive, seed, 1)[0]
    sequence = O.sequence_terms(flow, geometry, train_values, train_archive, sequence_start, statistics)

    flow.zero_grad(set_to_none=True)
    pair["real_vertex"].backward(retain_graph=True)
    reconstruction_gradient = _gradient_norm(flow)
    decoder_has_gradient = any(parameter.grad is not None for parameter in geometry.parameters())
    if not math.isfinite(reconstruction_gradient) or reconstruction_gradient <= 0.0 or decoder_has_gradient:
        raise RuntimeError("Frozen-decoder gradient-routing audit failed")
    flow.zero_grad(set_to_none=True)
    total, terms = O.total_loss(pair, sequence, config, 1)
    total.backward()
    total_gradient = _gradient_norm(flow)
    if not torch.isfinite(total) or not math.isfinite(total_gradient) or total_gradient <= 0.0:
        raise RuntimeError("Total-loss gradient audit failed")
    flow.zero_grad(set_to_none=True)
    flow.eval()
    baseline_validation = validation(
        flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics, config, smoke_limit
    )
    maximum_defect = max(
        baseline_validation["defects"]["relative_semigroup_defect_mean"],
        baseline_validation["defects"]["relative_inverse_defect_mean"],
        baseline_validation["defects"]["relative_identity_defect_mean"],
    )
    if maximum_defect > float(config["selection"]["implementation_audit_max_relative_defect"]):
        raise RuntimeError(f"Exact-coboundary algebra audit failed: defect={maximum_defect:.3e}")
    if args.dry_run:
        print("DRY RUN PASSED — exact algebra, finite gradients, frozen-decoder routing, and validation; no files written.")
        print(json.dumps({
            "loss": float(total.detach().cpu()), "terms": terms,
            "total_gradient_l2_norm": total_gradient,
            "reconstruction_to_flow_gradient_l2_norm": reconstruction_gradient,
            "decoder_parameter_gradient_present": decoder_has_gradient,
            "validation": baseline_validation,
        }, indent=2, sort_keys=True))
        return 0

    run_name = str(args.run_name or training["run_name"])
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    C.validate_run_name(run_name)
    output_dir = C.output_root(registry) / "training" / representation / "exact_coboundary_c4" / run_name
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {checkpoint_dir / 'latest.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_json(output_dir / "resolved_config.json", {
        "config_path": str(config_path), "config": config,
        "representation_manifest": C.read_json(C.output_root(registry) / "representations" / representation / "manifest.json"),
        "device": str(device), "physical_gpu_contract": os.environ.get("PHYSICAL_GPU_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": seed, "epochs": epochs,
        "smoke": bool(args.smoke), "test_data_loaded": False,
    })
    C.atomic_json(output_dir / "training_statistics.json", statistics)
    C.atomic_json(output_dir / "run_contract.json", {
        "transport": "Phi(z,s,t;c,d)=F(t;c,d)(F(s;c,d)^-1(z))",
        "coboundary_group": "invertible affine-coupling diffeomorphisms",
        "fixed_context": "subject first visit latent and age",
        "model_variant": flow.variant, "ode_used": False, "attention_used": False,
        "coboundary_used": True, "identity_exact": True, "inverse_exact": True,
        "composition_exact": True, "decoder_frozen": True,
        "decoder_in_optimizer_loss": True, "test_data_loaded": False,
    })
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training["minimum_learning_rate"])
    )
    start_epoch = 1
    best_score = float(baseline_validation["score"])
    best_epoch = 0
    best_any_score, best_any_epoch = best_score, 0
    stale = 0
    baseline_payload = checkpoint_payload(
        0, flow, optimizer, scheduler, config, statistics, baseline_validation,
        best_score, best_epoch, best_any_score, best_any_epoch,
    )
    if args.resume:
        payload = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        flow.load_state_dict(payload["flow_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_score, best_epoch = float(payload["best_selection_score"]), int(payload["best_epoch"])
        best_any_score = float(payload.get("best_unconstrained_score", best_score))
        best_any_epoch = int(payload.get("best_unconstrained_epoch", best_epoch))
        stale = int(C.read_json(output_dir / "training_status.json").get("stale_epochs", 0))
    else:
        C.atomic_torch_save(checkpoint_dir / "epoch_0000_nochange.pt", baseline_payload)
        C.atomic_torch_save(checkpoint_dir / "best.pt", baseline_payload)
        C.atomic_torch_save(checkpoint_dir / "best_unconstrained.pt", baseline_payload)

    samples_per_epoch = int(training["samples_per_epoch"])
    if args.smoke:
        samples_per_epoch = int(training["batch_size"]) * int(training.get("smoke_train_batches", 2))
    started = time.time()
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        flow.train()
        loader = DataLoader(
            PairDataset(train_pairs), batch_size=int(training["batch_size"]),
            sampler=O.balanced_pair_sampler(train_pairs, seed, epoch, samples_per_epoch),
            collate_fn=C.collate_pairs, num_workers=0,
        )
        sequence_starts = O.balanced_sequence_starts(train_archive, seed, epoch)
        totals: dict[str, float] = {}
        batches = 0
        for step, current_raw in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            pair = O.pair_terms(flow, geometry, train_values, current_raw, statistics)
            sequence = O.sequence_terms(
                flow, geometry, train_values, train_archive,
                sequence_starts[(step - 1) % len(sequence_starts)], statistics,
            )
            loss, current_terms = O.total_loss(pair, sequence, config, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in current_terms.items():
                totals[name] = totals.get(name, 0.0) + value
        scheduler.step()
        flow.eval()
        current_validation = validation(
            flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics, config, smoke_limit
        )
        score, feasible = float(current_validation["score"]), bool(current_validation["feasible"])
        delta = float(training["early_stopping_min_delta"])
        improved_any = score < best_any_score - delta
        if improved_any:
            best_any_score, best_any_epoch = score, epoch
        improved_feasible = feasible and score < best_score - delta
        if improved_feasible:
            best_score, best_epoch = score, epoch
        stale = 0 if improved_any else stale + 1
        payload = checkpoint_payload(
            epoch, flow, optimizer, scheduler, config, statistics, current_validation,
            best_score, best_epoch, best_any_score, best_any_epoch,
        )
        C.atomic_torch_save(checkpoint_dir / "latest.pt", payload)
        if improved_feasible:
            C.atomic_torch_save(checkpoint_dir / "best.pt", payload)
        if improved_any:
            C.atomic_torch_save(checkpoint_dir / "best_unconstrained.pt", payload)
        elapsed_minutes = (time.time() - started) / 60.0
        row = {
            "epoch": epoch, "elapsed_minutes": elapsed_minutes,
            "estimated_total_minutes_at_current_rate": elapsed_minutes * epochs / max(epoch - start_epoch + 1, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]), "batches": batches,
            **{f"train_{name}": value / max(batches, 1) for name, value in totals.items()},
            "validation": current_validation,
        }
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        C.atomic_json(output_dir / "training_status.json", {
            "status": "running", "epoch": epoch, "epochs_requested": epochs,
            "best_epoch": best_epoch, "best_validation_selection_score": best_score,
            "best_unconstrained_epoch": best_any_epoch, "best_unconstrained_score": best_any_score,
            "stale_epochs": stale, "elapsed_minutes": elapsed_minutes,
            "estimated_total_minutes_at_current_rate": row["estimated_total_minutes_at_current_rate"],
            "coboundary_used": True, "structural_exactness": True, "test_data_loaded": False,
        })
        completed_epoch = epoch
        print(
            f"epoch {epoch:03d}/{epochs} loss={totals['total']/max(batches,1):.6f} "
            f"val={score:.6f} feasible={feasible} best={best_score:.6f}@{best_epoch} "
            f"elapsed={elapsed_minutes:.1f}m", flush=True,
        )
        if not args.smoke and stale >= int(training["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    C.atomic_json(output_dir / "training_status.json", {
        "status": "complete", "epoch": completed_epoch, "epochs_requested": epochs,
        "best_epoch": best_epoch, "best_validation_selection_score": best_score,
        "best_unconstrained_epoch": best_any_epoch, "best_unconstrained_score": best_any_score,
        "selected_checkpoint": str(checkpoint_dir / "best.pt"),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "coboundary_used": True, "structural_exactness": True, "test_data_loaded": False,
    })
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
