#!/usr/bin/env python3
"""Train a matched or PCA-parity SIREN-256 transport without loading test pairs."""

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

from siren256_common import (BalancedPairBatchSampler, RegisteredPairDataset, balanced_sequence, decode_sdf, ensure_prepared, load_basis, load_config, load_frozen_decoder, load_pairs, load_sequences, load_split_cache, read_json, resolve, root_dir, set_seed, to_device, write_json)
from siren256_decoder_geometry import C3GeometryLoss
from siren256_transport_models import build_transport


PCA_PARITY_MODEL_TYPES = {"pca_parity_direct_flow", "pca_parity_geometry_whitened_direct_flow"}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def decoder_geometry_velocity_scale(
    decoder: torch.nn.Module,
    cache: dict[str, Any],
    train_pairs: Any,
    latent_scale: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    smoke: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate a train-only diagonal decoder-Jacobian output preconditioner.

    Hutchinson probes estimate the mean squared SDF sensitivity of each latent
    coordinate at registered training surfaces.  Dividing the ordinary latent
    output scale by this normalized sensitivity makes one model output unit
    less likely to move a decoder-sensitive coordinate excessively.
    """
    indices = np.unique(np.asarray(train_pairs["source_cache_index"], dtype=int))
    index_map = cache["cache_index_to_local"]
    local_indices = np.asarray([index_map[int(index)] for index in indices if int(index) in index_map], dtype=int)
    if not len(local_indices):
        raise RuntimeError("No training anchors available for decoder geometry whitening.")
    scan_count = int(config.get("GeometryWhiteningScanCount", 48))
    point_count = int(config.get("GeometryWhiteningPoints", 128))
    probe_count = int(config.get("GeometryWhiteningHutchinsonProbes", 4))
    if smoke:
        scan_count, point_count, probe_count = min(scan_count, 2), min(point_count, 32), min(probe_count, 1)
    if scan_count <= 0 or point_count <= 0 or probe_count <= 0:
        raise ValueError("Geometry-whitening scan, point, and probe counts must be positive.")
    rng = np.random.default_rng(int(config["Seed"]) + 701)
    chosen = local_indices if len(local_indices) <= scan_count else np.sort(rng.choice(local_indices, size=scan_count, replace=False))
    sensitivity = torch.zeros(int(config["LatentSize"]), device=device)
    used = 0
    for local_index in chosen:
        vertex_count = cache["vertices"].shape[1]
        selected_points = np.sort(rng.choice(vertex_count, size=min(point_count, vertex_count), replace=False))
        points = torch.from_numpy(np.array(cache["vertices"][local_index, selected_points], copy=True)).to(device)[None, :, :]
        source = torch.from_numpy(np.array(cache["latents"][local_index : local_index + 1], copy=True)).to(device).requires_grad_(True)
        for _ in range(probe_count):
            sdf = decode_sdf(decoder, source, points)
            signs = torch.randint(0, 2, sdf.shape, device=device, dtype=torch.int64).to(sdf.dtype).mul_(2.0).sub_(1.0)
            projection = (sdf * signs).sum() / float(np.sqrt(sdf.shape[1]))
            gradient = torch.autograd.grad(projection, source, retain_graph=False, create_graph=False)[0]
            sensitivity += gradient.square().sum(dim=0)
            used += 1
    sensitivity = torch.sqrt(sensitivity / max(used, 1)).detach().cpu().numpy()[None, :]
    median = float(np.median(sensitivity[sensitivity > 0.0])) if np.any(sensitivity > 0.0) else 1.0
    normalized = sensitivity / max(median, 1.0e-8)
    lower, upper = float(config.get("GeometryWhiteningMinSensitivity", 0.25)), float(config.get("GeometryWhiteningMaxSensitivity", 4.0))
    if lower <= 0.0 or upper < lower:
        raise ValueError("Invalid geometry-whitening sensitivity clamp.")
    normalized = np.clip(normalized, lower, upper)
    velocity_scale = np.asarray(latent_scale, dtype=np.float32) / normalized.astype(np.float32)
    report = {
        "fit_split": "train",
        "anchor_scan_count": int(len(chosen)),
        "points_per_anchor": int(min(point_count, cache["vertices"].shape[1])),
        "hutchinson_probes_per_anchor": int(probe_count),
        "sensitivity_min": float(sensitivity.min()),
        "sensitivity_median": float(np.median(sensitivity)),
        "sensitivity_max": float(sensitivity.max()),
        "normalized_sensitivity_min": float(normalized.min()),
        "normalized_sensitivity_max": float(normalized.max()),
        "velocity_scale_min": float(velocity_scale.min()),
        "velocity_scale_max": float(velocity_scale.max()),
    }
    return velocity_scale, report


def pareto_candidate(epoch: int, validation: dict[str, float], config: dict[str, Any], nochange: bool = False) -> dict[str, Any]:
    feasibility = config["Feasibility"]
    constraint_feasible = validation["semigroup_scaled"] <= float(feasibility["semigroup_max_scaled"]) and validation["inverse_scaled"] <= float(feasibility["inverse_max_scaled"])
    geometry_gate = validation["geometry_ratio_first_last"] <= float(config.get("ParetoMaxFirstLastGeometryRatio", float("inf"))) and validation["geometry_ratio_all_pairs"] <= float(config.get("ParetoMaxAllPairGeometryRatio", float("inf")))
    volume_improved = validation["volume_ratio"] < float(config.get("ParetoMaxVolumeRatio", 1.0))
    return {
        "epoch": int(epoch),
        "checkpoint": "epoch_0000_nochange" if nochange else f"epoch_{epoch:04d}",
        "tag": "epoch_0000_nochange" if nochange else f"epoch_{epoch:04d}",
        "nochange": bool(nochange),
        "constraint_feasible": bool(constraint_feasible),
        "geometry_gate": bool(geometry_gate),
        "volume_improved": bool(volume_improved),
        "proxy_feasible": bool(constraint_feasible and geometry_gate and (nochange or volume_improved)),
        **{name: float(value) for name, value in validation.items()},
    }


def pareto_frontier(candidates: list[dict[str, Any]]) -> set[int]:
    eligible = [candidate for candidate in candidates if candidate["constraint_feasible"]]
    metrics = ("geometry_ratio_first_last", "geometry_ratio_all_pairs", "volume_ratio")
    frontier: set[int] = set()
    for candidate in eligible:
        dominated = any(
            other["epoch"] != candidate["epoch"]
            and all(other[metric] <= candidate[metric] for metric in metrics)
            and any(other[metric] < candidate[metric] for metric in metrics)
            for other in eligible
        )
        if not dominated:
            frontier.add(int(candidate["epoch"]))
    return frontier


def finalize_pareto_candidates(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    frontier = pareto_frontier(candidates)
    for candidate in candidates:
        candidate["pareto_frontier"] = int(candidate["epoch"]) in frontier
        candidate["shortlisted"] = False
    maximum = max(1, int(config.get("ParetoMaxCandidates", 5)))
    baseline = next((candidate for candidate in candidates if candidate["nochange"]), None)
    chosen: list[dict[str, Any]] = [baseline] if baseline is not None else []
    pool = [candidate for candidate in candidates if not candidate["nochange"] and candidate["constraint_feasible"]]
    pool.sort(key=lambda candidate: (not candidate["proxy_feasible"], candidate["volume_ratio"], candidate["geometry_ratio_first_last"], candidate["epoch"]))
    for candidate in pool:
        if len(chosen) >= maximum:
            break
        chosen.append(candidate)
    if len(chosen) < maximum:
        for candidate in sorted((candidate for candidate in pool if candidate["pareto_frontier"]), key=lambda candidate: (candidate["selection_score"], candidate["epoch"])):
            if len(chosen) >= maximum:
                break
            if candidate not in chosen:
                chosen.append(candidate)
    for candidate in chosen:
        candidate["shortlisted"] = True
    return candidates


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
    model_type = str(config["ModelType"]).lower()
    basis = load_basis(root)
    train_latents: np.ndarray | None = None
    whitening_report: dict[str, float] | None = None
    if model_type in PCA_PARITY_MODEL_TYPES:
        with np.load(resolve(config["LatentArchives"]["train"]), allow_pickle=False) as archive:
            train_latents = np.asarray(archive["latents"], dtype=np.float32)
        latent_mean = train_latents.mean(axis=0, keepdims=True)
        latent_scale = np.maximum(train_latents.std(axis=0, keepdims=True), 1.0e-6)
        basis = {**basis, "latent_mean": latent_mean, "latent_scale": latent_scale}
    decoder = load_frozen_decoder(config, device)
    if model_type == "pca_parity_geometry_whitened_direct_flow":
        if train_latents is None:
            raise RuntimeError("Geometry-whitened flow requires training latent statistics.")
        velocity_scale, whitening_report = decoder_geometry_velocity_scale(
            decoder, cache, train_pairs, latent_scale, config, device, args.smoke_steps is not None
        )
        basis = {**basis, "velocity_scale": velocity_scale}
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
    if model_type in PCA_PARITY_MODEL_TYPES:
        if train_latents is None:
            raise RuntimeError("PCA-parity flow requires training latent statistics.")
        write_json(run / "train_only_latent_standardization.json", {
            "fit_split": "train",
            "scan_count": int(train_latents.shape[0]),
            "latent_size": int(train_latents.shape[1]),
            "minimum_scale": float(latent_scale.min()),
            "maximum_scale": float(latent_scale.max()),
            "mean_scale": float(latent_scale.mean()),
        })
    if whitening_report is not None:
        write_json(run / "train_only_decoder_geometry_whitening.json", whitening_report)
    history: list[dict[str, float]] = []
    best_score, candidate_score, best_epoch, candidate_epoch, validated = float("inf"), float("inf"), 0, 0, False
    best_validation: dict[str, float] = {}
    candidate_validation: dict[str, float] = {}
    use_pareto_selection = bool(config.get("UseParetoValidationSelection", False))
    if use_pareto_selection and not bool(config.get("SaveEpochZero", False)):
        raise ValueError("Pareto validation selection requires SaveEpochZero=true.")
    pareto_candidates: list[dict[str, Any]] = []
    best_geometry_key = (float("inf"), float("inf"), float("inf"))
    best_volume_key = (float("inf"), float("inf"), float("inf"))
    rng = np.random.default_rng(seed + 17)
    selection_volume_weight = float(config.get("SelectionVolumeWeight", 0.01))
    baseline_score = float("inf")
    if bool(config.get("SaveEpochZero", False)):
        initial = validate(model, criterion, val_loader, device, 2 if args.smoke_steps is not None else None, selection_volume_weight)
        baseline_score = best_score = candidate_score = float(initial["selection_score"])
        best_validation = candidate_validation = dict(initial)
        validated = True
        payload = checkpoint_payload(0, model, optimizer, scheduler, config, best_score, initial)
        save_checkpoint(run / "checkpoints" / "epoch_0000_nochange.pt", payload)
        save_checkpoint(run / "checkpoints" / "best_candidate.pt", payload)
        save_checkpoint(run / "checkpoints" / "best.pt", payload)
        if use_pareto_selection:
            baseline_candidate = pareto_candidate(0, initial, config, nochange=True)
            pareto_candidates.append(baseline_candidate)
            best_geometry_key = (baseline_candidate["geometry_ratio_first_last"], baseline_candidate["geometry_ratio_all_pairs"], baseline_candidate["volume_ratio"])
            save_checkpoint(run / "checkpoints" / "best_geometry.pt", payload)
        initial_report = {"epoch": 0.0, "train_steps": 0.0, "train_optimizer_steps": 0.0, "learning_rate": float(optimizer.param_groups[0]["lr"]), **{f"val_{name}": value for name, value in initial.items()}}
        history.append(initial_report)
        write_json(run / "history.json", history)
        print(f"epoch=000 no-change val_score={initial['selection_score']:.6g}", flush=True)
    max_epochs = 1 if args.smoke_steps is not None else int(config["Epochs"])
    accumulation_steps = max(1, int(config.get("GradientAccumulationSteps", 1)))
    for epoch in range(1, max_epochs + 1):
        model.train()
        criterion.set_epoch(epoch)
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
        report = {"epoch": float(epoch), "train_steps": float(steps), "train_optimizer_steps": float(optimizer_steps), "seconds": float(time.time() - started), "learning_rate": float(optimizer.param_groups[0]["lr"]), **{f"weight_{name}": value for name, value in criterion.effective_weights().items()}, **{f"train_{name}": value / max(steps, 1) for name, value in totals.items()}}
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
            if bool(config.get("SaveValidationSnapshots", False)) and epoch % max(1, int(config.get("CheckpointEvery", 1))) == 0:
                save_checkpoint(run / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
            if use_pareto_selection:
                record = pareto_candidate(epoch, validation, config)
                pareto_candidates.append(record)
                geometry_key = (record["geometry_ratio_first_last"], record["geometry_ratio_all_pairs"], record["volume_ratio"])
                if record["constraint_feasible"] and geometry_key < best_geometry_key:
                    best_geometry_key = geometry_key
                    save_checkpoint(run / "checkpoints" / "best_geometry.pt", payload)
                volume_key = (record["volume_ratio"], record["geometry_ratio_first_last"], record["geometry_ratio_all_pairs"])
                if record["proxy_feasible"] and volume_key < best_volume_key:
                    best_volume_key = volume_key
                    save_checkpoint(run / "checkpoints" / "best_volume_under_geometry_gate.pt", payload)
            if validation["selection_score"] < candidate_score:
                candidate_score, candidate_epoch = validation["selection_score"], epoch
                candidate_validation = dict(validation)
                save_checkpoint(run / "checkpoints" / "best_candidate.pt", payload)
            if not use_pareto_selection and acceptable and validation["selection_score"] < best_score:
                best_score, best_epoch = validation["selection_score"], epoch
                best_validation = dict(validation)
                payload["best_validation_score"] = best_score
                save_checkpoint(run / "checkpoints" / "best.pt", payload)
        history.append(report)
        write_json(run / "history.json", history)
        if use_pareto_selection:
            write_json(run / "validation_candidates.json", {"criteria": {"max_first_last_geometry_ratio": float(config.get("ParetoMaxFirstLastGeometryRatio", float("inf"))), "max_all_pair_geometry_ratio": float(config.get("ParetoMaxAllPairGeometryRatio", float("inf"))), "max_volume_ratio": float(config.get("ParetoMaxVolumeRatio", 1.0)), "feasibility": config["Feasibility"]}, "candidates": pareto_candidates})
        print(f"epoch={epoch:03d} train_total={report['train_total']:.6g}" + (f" val_score={report['val_selection_score']:.6g}" if "val_selection_score" in report else ""), flush=True)
        if args.smoke_steps is None and validated and epoch - best_epoch >= int(config["EarlyStopPatience"]):
            print(f"Early stop after {epoch - best_epoch} epochs without an eligible improvement.", flush=True)
            break
    if use_pareto_selection:
        pareto_candidates = finalize_pareto_candidates(pareto_candidates, config)
        write_json(run / "validation_candidates.json", {"criteria": {"max_first_last_geometry_ratio": float(config.get("ParetoMaxFirstLastGeometryRatio", float("inf"))), "max_all_pair_geometry_ratio": float(config.get("ParetoMaxAllPairGeometryRatio", float("inf"))), "max_volume_ratio": float(config.get("ParetoMaxVolumeRatio", 1.0)), "feasibility": config["Feasibility"], "selection": "validation_surface_required"}, "candidates": pareto_candidates})
    best_checkpoint_exists = (run / "checkpoints" / "best.pt").exists()
    selected_checkpoint = "best.pt" if best_checkpoint_exists else "best_candidate.pt"
    selected_epoch = best_epoch if best_checkpoint_exists else candidate_epoch
    selected_validation = best_validation if best_checkpoint_exists else candidate_validation
    final_validation_report = {
        "run_name": run_name,
        "criterion": f"macro first-last geometry ratio + 0.2 all-pair geometry ratio + {selection_volume_weight:g} volume ratio",
        "feasibility": config["Feasibility"],
        "selected_checkpoint": selected_checkpoint,
        "selected_epoch": int(selected_epoch),
        "result": selected_validation,
        "last_epoch_result": history[-1] if history else {},
        "selection_status": "validation_surface_required" if use_pareto_selection else "complete",
        "proxy_shortlist": [candidate["checkpoint"] for candidate in pareto_candidates if candidate.get("shortlisted")] if use_pareto_selection else [],
    }
    write_json(run / "validation_report.json", final_validation_report)
    if bool(config.get("WriteSharedValidationReport", True)):
        report_path = root / "metadata" / "validation_report.json"
        validation_report = read_json(report_path) if report_path.exists() else {"runs": {}}
        validation_report.setdefault("runs", {})[run_name] = history[-1] if history else {}
        validation_report.update({"criterion": final_validation_report["criterion"], "feasibility": config["Feasibility"]})
        write_json(report_path, validation_report)
    if not best_checkpoint_exists and (run / "checkpoints" / "best_candidate.pt").exists():
        print("No checkpoint met feasibility; use best_candidate.pt only after reviewing the recorded defects.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
