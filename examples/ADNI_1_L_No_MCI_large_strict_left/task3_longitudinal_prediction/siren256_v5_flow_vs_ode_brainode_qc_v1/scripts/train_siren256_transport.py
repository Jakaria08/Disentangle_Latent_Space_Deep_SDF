#!/usr/bin/env python3
"""Train one of the three matched SIREN-256 C3 transports without loading test pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from siren256_common import (BalancedPairBatchSampler, RegisteredPairDataset, balanced_sequence, ensure_prepared, load_basis, load_config, load_frozen_decoder, load_pairs, load_sequences, load_split_cache, read_json, resolve, root_dir, set_seed, to_device, write_json)
from siren256_decoder_geometry import C3GeometryLoss
from siren256_transport_models import build_transport


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sequence_tensor(sequence: dict[str, Any], cache: dict[str, np.ndarray], device: torch.device, max_targets: int) -> dict[str, torch.Tensor]:
    global_indices = np.asarray(sequence["cache_indices"], dtype=int)
    index_map = cache.get("cache_index_to_local", {})
    indices = np.asarray([index_map.get(int(index), int(index)) for index in global_indices], dtype=int)
    if index_map and any(int(index) not in index_map for index in global_indices):
        raise KeyError(f"Sequence {sequence['subject_id']} is outside the allowed split cache.")
    if max_targets > 0 and len(indices) > max_targets + 1:
        indices = np.concatenate((indices[:1], indices[-max_targets:]))
    times = np.asarray(sequence["times"], dtype=np.float32)
    if len(times) != len(sequence["cache_indices"]):
        raise ValueError("Prepared sequence indices/times are misaligned.")
    if len(indices) != len(times):
        times = np.concatenate((times[:1], times[-max_targets:]))
    return {
        "latents": torch.from_numpy(np.array(cache["latents"][indices], copy=True)).to(device),
        "times": torch.from_numpy(np.array(times, copy=True)).to(device),
        "condition": torch.tensor([[float(sequence["label_ad"])]], device=device),
        "vertices": torch.from_numpy(np.array(cache["vertices"][indices], copy=True)).to(device),
        "normals": torch.from_numpy(np.array(cache["normals"][indices], copy=True)).to(device),
        "volumes": torch.from_numpy(np.array(cache["volumes"][indices], copy=True)).to(device),
    }


def combine_terms(pair_terms: dict[str, torch.Tensor], sequence_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = set(pair_terms) | set(sequence_terms)
    zero = next(iter(pair_terms.values())).sum() * 0.0
    return {name: pair_terms.get(name, zero) + sequence_terms.get(name, zero) for name in names}


@torch.no_grad()
def validate(model: torch.nn.Module, criterion: C3GeometryLoss, loader: DataLoader, device: torch.device, max_batches: int | None, selection_volume_weight: float = 0.01) -> dict[str, float]:
    model.eval()
    by_subject: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    raw_totals: dict[str, float] = defaultdict(float)
    rows, volume_prediction, volume_nochange = 0, 0.0, 0.0
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(raw_batch, device)
        prediction, terms = criterion.pair_terms(model, batch, vertex_count=None, training=False)
        displacement, _, proxy_volume = criterion.normal_proxy(batch["source_latent"], prediction, batch["source_vertices"], batch["source_normals"])
        predicted_volume, _ = criterion.volume_from_prediction(batch["source_latent"], prediction, batch["source_volume"], proxy_volume)
        target_displacement = ((batch["target_vertices"] - batch["source_vertices"]) * batch["source_normals"]).sum(dim=-1)
        predicted_error = torch.nn.functional.smooth_l1_loss(displacement, target_displacement, reduction="none").mean(dim=1)
        nochange_error = torch.nn.functional.smooth_l1_loss(torch.zeros_like(target_displacement), target_displacement, reduction="none").mean(dim=1)
        target_volume = batch["target_volume"].clamp_min(1.0e-8)
        predicted_relative = (predicted_volume - target_volume).abs() / target_volume
        nochange_relative = (batch["source_volume"] - target_volume).abs() / target_volume
        for name, value in terms.items():
            raw_totals[name] += float(value.detach().cpu()) * len(batch["subject_id"])
        for index, subject in enumerate(batch["subject_id"]):
            bucket = by_subject[str(subject)]
            bucket["all_prediction"].append(float(predicted_error[index].cpu()))
            bucket["all_nochange"].append(float(nochange_error[index].cpu()))
            if bool(batch["is_first_last"][index]):
                bucket["first_prediction"].append(float(predicted_error[index].cpu()))
                bucket["first_nochange"].append(float(nochange_error[index].cpu()))
        volume_prediction += float(predicted_relative.sum().cpu())
        volume_nochange += float(nochange_relative.sum().cpu())
        rows += len(batch["subject_id"])
    if not rows:
        raise RuntimeError("Validation loader was empty.")
    def macro(key: str) -> float:
        values = [np.mean(values) for bucket in by_subject.values() if (values := bucket.get(key))]
        return float(np.mean(values))
    first_prediction, first_nochange = macro("first_prediction"), macro("first_nochange")
    all_prediction, all_nochange = macro("all_prediction"), macro("all_nochange")
    result = {name: value / rows for name, value in raw_totals.items()}
    result.update({"rows": float(rows), "subjects": float(len(by_subject)), "macro_first_last_geometry": first_prediction, "macro_first_last_nochange_geometry": first_nochange, "macro_all_pair_geometry": all_prediction, "macro_all_pair_nochange_geometry": all_nochange, "geometry_ratio_first_last": first_prediction / max(first_nochange, 1.0e-8), "geometry_ratio_all_pairs": all_prediction / max(all_nochange, 1.0e-8), "volume_ratio": (volume_prediction / rows) / max(volume_nochange / rows, 1.0e-8)})
    result["selection_score"] = result["geometry_ratio_first_last"] + 0.2 * result["geometry_ratio_all_pairs"] + float(selection_volume_weight) * result["volume_ratio"]
    result["semigroup_scaled"] = (result.get("observed_semigroup", 0.0) + result.get("virtual_semigroup", 0.0)) / criterion.scales["latent"]
    result["inverse_scaled"] = result.get("inverse", 0.0) / criterion.scales["latent"]
    return result


def checkpoint_payload(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, config: dict[str, Any], best_score: float, validation: dict[str, float]) -> dict[str, Any]:
    return {"epoch": epoch, "flow_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "best_validation_score": best_score, "validation": validation, "decoder_frozen": True, "latent_representation": "exact frozen 256-D no-skip SIREN", "attention_contract": getattr(model, "attention_contract", "not_applicable"), "test_data_loaded_during_training": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    root, config = root_dir(), load_config(args.config)
    ensure_prepared(root)
    device, seed = torch.device(args.device), int(config["Seed"])
    set_seed(seed)
    run_name = str(args.run_name or config["RunName"])
    if Path(run_name).name != run_name:
        raise ValueError("Run name must be one path component.")
    run = root / "runs" / run_name
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}")
    # Deliberately load only train and validation tables.  Test evaluation is a
    # separate script after checkpoint choice is complete.
    train_pairs = load_pairs("train", include_backward=bool(config.get("IncludeBackwardPairs", True)), root=root)
    val_pairs = load_pairs("val", root=root)
    cache = load_split_cache(config, ("train", "val"), root)
    train_dataset = RegisteredPairDataset(config, cache, train_pairs, int(config["SamplesPerTarget"]), deterministic=False, seed=seed)
    val_dataset = RegisteredPairDataset(config, cache, val_pairs, int(config["SamplesPerTarget"]), deterministic=True, seed=seed + 991)
    train_sampler: BalancedPairBatchSampler | None = None
    if bool(config.get("BalancedPairSampling", True)):
        train_sampler = BalancedPairBatchSampler(train_pairs, int(config["PairsPerBatch"]), int(config["StepsPerEpoch"]), seed)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=device.type == "cuda")
    else:
        train_loader = DataLoader(train_dataset, batch_size=int(config["PairsPerBatch"]), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=int(config["PairsPerBatch"]), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    train_sequences, val_sequences = load_sequences("train", root), load_sequences("val", root)
    if not train_sequences or not val_sequences:
        raise RuntimeError("Prepared subject sequence tables are empty.")
    basis = load_basis(root)
    if str(config["ModelType"]).lower() == "pca_parity_direct_flow":
        with np.load(resolve(config["LatentArchives"]["train"]), allow_pickle=False) as archive:
            train_latents = np.asarray(archive["latents"], dtype=np.float32)
        latent_mean = train_latents.mean(axis=0, keepdims=True)
        latent_scale = np.maximum(train_latents.std(axis=0, keepdims=True), 1.0e-6)
        basis = {**basis, "latent_mean": latent_mean, "latent_scale": latent_scale}
    decoder = load_frozen_decoder(config, device)
    model = build_transport(config, basis).to(device)
    if "ContinuationOf" in config:
        source = root / "runs" / str(config["ContinuationOf"]) / "checkpoints" / "best.pt"
        model.load_state_dict(torch.load(source, map_location="cpu")["flow_state_dict"], strict=True)
    faces = torch.from_numpy(np.array(cache["faces"], copy=True)).to(device)
    criterion = C3GeometryLoss(decoder, faces, read_json(root / "metadata" / "loss_scales.json"), config["LossWeights"], float(config["AgeRangeYears"]), options=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["LearningRate"]), weight_decay=float(config["WeightDecay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["Epochs"]), eta_min=float(config["MinimumLearningRate"]))
    run.mkdir(parents=True)
    write_json(run / "config.json", config)
    if str(config["ModelType"]).lower() == "pca_parity_direct_flow":
        write_json(run / "train_only_latent_standardization.json", {
            "fit_split": "train",
            "scan_count": int(train_latents.shape[0]),
            "latent_size": int(train_latents.shape[1]),
            "minimum_scale": float(latent_scale.min()),
            "maximum_scale": float(latent_scale.max()),
            "mean_scale": float(latent_scale.mean()),
        })
    history: list[dict[str, float]] = []
    best_score, candidate_score, best_epoch, validated = float("inf"), float("inf"), 0, False
    rng = np.random.default_rng(seed + 17)
    selection_volume_weight = float(config.get("SelectionVolumeWeight", 0.01))
    baseline_score = float("inf")
    if bool(config.get("SaveEpochZero", False)):
        initial = validate(model, criterion, val_loader, device, 2 if args.smoke_steps is not None else None, selection_volume_weight)
        baseline_score = best_score = candidate_score = float(initial["selection_score"])
        validated = True
        payload = checkpoint_payload(0, model, optimizer, scheduler, config, best_score, initial)
        save_checkpoint(run / "checkpoints" / "epoch_0000_nochange.pt", payload)
        save_checkpoint(run / "checkpoints" / "best_candidate.pt", payload)
        save_checkpoint(run / "checkpoints" / "best.pt", payload)
        initial_report = {"epoch": 0.0, "train_steps": 0.0, "train_optimizer_steps": 0.0, "learning_rate": float(optimizer.param_groups[0]["lr"]), **{f"val_{name}": value for name, value in initial.items()}}
        history.append(initial_report)
        write_json(run / "history.json", history)
        print(f"epoch=000 no-change val_score={initial['selection_score']:.6g}", flush=True)
    max_epochs = 1 if args.smoke_steps is not None else int(config["Epochs"])
    accumulation_steps = max(1, int(config.get("GradientAccumulationSteps", 1)))
    for epoch in range(1, max_epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        totals: dict[str, float] = defaultdict(float)
        steps, optimizer_steps, pending_gradients = 0, 0, 0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for raw_batch in train_loader:
            if args.smoke_steps is not None and steps >= args.smoke_steps:
                break
            batch = to_device(raw_batch, device)
            _, pair_terms = criterion.pair_terms(model, batch, int(config["RegisteredVertices"]), training=True)
            if bool(config.get("BalancedSequenceSampling", True)):
                selected_sequence = balanced_sequence(rng, train_sequences)
            else:
                selected_sequence = train_sequences[int(rng.integers(len(train_sequences)))]
            sequence = sequence_tensor(selected_sequence, cache, device, int(config.get("SequenceTargets", 0)))
            sequence_terms = criterion.sequence_terms(model, sequence, int(config["RegisteredVertices"]))
            terms = combine_terms(pair_terms, sequence_terms)
            total = criterion.total(terms)
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {steps}.")
            total.backward()
            pending_gradients += 1
            if pending_gradients >= accumulation_steps:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(pending_gradients))
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["GradientClipNorm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_gradients = 0
            totals["total"] += float(total.detach().cpu())
            for name, value in terms.items():
                totals[name] += float(value.detach().cpu())
            steps += 1
        if pending_gradients:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(pending_gradients))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["GradientClipNorm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        scheduler.step()
        report = {"epoch": float(epoch), "train_steps": float(steps), "train_optimizer_steps": float(optimizer_steps), "seconds": float(time.time() - started), "learning_rate": float(optimizer.param_groups[0]["lr"]), **{f"train_{name}": value / max(steps, 1) for name, value in totals.items()}}
        need_validation = args.smoke_steps is not None or epoch % int(config["ValidationEvery"]) == 0 or epoch == max_epochs
        if need_validation:
            validated = True
            validation = validate(model, criterion, val_loader, device, 2 if args.smoke_steps is not None else None, selection_volume_weight)
            report.update({f"val_{name}": value for name, value in validation.items()})
            feasibility = config["Feasibility"]
            acceptable = validation["semigroup_scaled"] <= float(feasibility["semigroup_max_scaled"]) and validation["inverse_scaled"] <= float(feasibility["inverse_max_scaled"])
            if bool(config.get("RequireBeatNoChangeForBest", False)):
                geometry_tolerance = float(config.get("NoChangeGeometryTolerance", 1.0))
                volume_tolerance = float(config.get("NoChangeVolumeTolerance", 1.0))
                acceptable = acceptable and validation["geometry_ratio_first_last"] <= geometry_tolerance and validation["geometry_ratio_all_pairs"] <= geometry_tolerance and validation["volume_ratio"] <= volume_tolerance and validation["selection_score"] < baseline_score
            payload = checkpoint_payload(epoch, model, optimizer, scheduler, config, best_score, validation)
            save_checkpoint(run / "checkpoints" / "latest.pt", payload)
            if validation["selection_score"] < candidate_score:
                candidate_score = validation["selection_score"]
                save_checkpoint(run / "checkpoints" / "best_candidate.pt", payload)
            if acceptable and validation["selection_score"] < best_score:
                best_score, best_epoch = validation["selection_score"], epoch
                payload["best_validation_score"] = best_score
                save_checkpoint(run / "checkpoints" / "best.pt", payload)
        history.append(report)
        write_json(run / "history.json", history)
        print(f"epoch={epoch:03d} train_total={report['train_total']:.6g}" + (f" val_score={report['val_selection_score']:.6g}" if "val_selection_score" in report else ""), flush=True)
        if args.smoke_steps is None and validated and epoch - best_epoch >= int(config["EarlyStopPatience"]):
            print(f"Early stop after {epoch - best_epoch} epochs without an eligible improvement.", flush=True)
            break
    report_path = root / "metadata" / "validation_report.json"
    validation_report = read_json(report_path) if report_path.exists() else {"runs": {}}
    validation_report.setdefault("runs", {})[run_name] = history[-1] if history else {}
    validation_report.update({"criterion": f"macro first-last geometry ratio + 0.2 all-pair geometry ratio + {selection_volume_weight:g} volume ratio", "feasibility": config["Feasibility"]})
    write_json(report_path, validation_report)
    if not (run / "checkpoints" / "best.pt").exists() and (run / "checkpoints" / "best_candidate.pt").exists():
        print("No checkpoint met feasibility; use best_candidate.pt only after reviewing the recorded defects.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
