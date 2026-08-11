#!/usr/bin/env python3
"""Train the frozen-v1, AD-only speed-calibrated direct cocycle flow."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from v1_anchored_speed_model import V1AnchoredAdSpeedCalibrator, model_config_from_instance
from v1_speed_utils import (
    EXPERIMENT_DIR,
    PairRecord,
    build_pair_records,
    decode_pca_torch,
    first_visit_sequence_starts,
    load_config,
    load_json,
    load_pca_model,
    load_split_archive,
    load_v1_flow,
    limit_records_stratified,
    mesh_volume_torch,
    pca_latents,
    resolve_device,
    resolve_repo_path,
    subject_end,
    write_json,
)


class PairDataset(Dataset):
    def __init__(self, archive: dict[str, np.ndarray], records: list[PairRecord], components: int) -> None:
        self.archive = archive
        self.records = records
        self.latents = pca_latents(archive, components)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "source": torch.from_numpy(self.latents[record.source_index].copy()).float(),
            "target": torch.from_numpy(self.latents[record.target_index].copy()).float(),
            "source_norm": torch.tensor(record.source_age_norm, dtype=torch.float32),
            "target_norm": torch.tensor(record.target_age_norm, dtype=torch.float32),
            "source_years": torch.tensor(record.source_age_years, dtype=torch.float32),
            "target_years": torch.tensor(record.target_age_years, dtype=torch.float32),
            "condition": torch.tensor(float(record.label_ad), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--run-name", default="v1_anchor_ad_speed_seed42")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def first_last_records(archive: dict[str, np.ndarray]) -> list[PairRecord]:
    records = build_pair_records(archive)
    offsets = archive["subject_visit_offsets"]
    last_by_first = {int(offsets[index]): int(offsets[index + 1] - 1) for index in range(len(offsets) - 1)}
    return [record for record in records if record.source_index in last_by_first and record.target_index == last_by_first[record.source_index]]


def make_model(
    *,
    config: dict[str, Any],
    metadata: dict[str, np.ndarray],
    device: torch.device,
) -> V1AnchoredAdSpeedCalibrator:
    components_count = int(config["components"])
    _, mean_flat, pca_components, faces = load_pca_model(components_count)
    base, _ = load_v1_flow(resolve_repo_path(config["base_v1_checkpoint"]), device)
    model_config = config["model"]
    return V1AnchoredAdSpeedCalibrator(
        base_flow=base,
        feature_pcs=int(model_config["feature_pcs"]),
        hidden_dims=[int(value) for value in model_config["hidden_dims"]],
        dropout=float(model_config["dropout"]),
        minimum_speed=float(model_config["minimum_speed"]),
        maximum_speed=float(model_config["maximum_speed"]),
        initial_ad_speed=float(model_config.get("initial_ad_speed", 1.0)),
        individual_log_span=float(model_config["individual_log_span"]),
        coefficient_mean=torch.from_numpy(metadata["coefficient_mean"]),
        coefficient_std=torch.from_numpy(metadata["coefficient_std"]),
        feature_scalar_mean=torch.from_numpy(metadata["feature_scalar_mean"]),
        feature_scalar_std=torch.from_numpy(metadata["feature_scalar_std"]),
        mean_flat=torch.from_numpy(mean_flat),
        pca_components=torch.from_numpy(pca_components),
        faces=torch.from_numpy(faces),
    ).to(device)


def base_parameter_snapshot(model: V1AnchoredAdSpeedCalibrator) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.base_flow.state_dict().items()}


def assert_base_unchanged(model: V1AnchoredAdSpeedCalibrator, before: dict[str, torch.Tensor]) -> None:
    after = model.base_flow.state_dict()
    for key, value in before.items():
        if not torch.equal(value, after[key].detach().cpu()):
            raise RuntimeError(f"Frozen v1 parameter changed: {key}")


def set_global_speed(model: V1AnchoredAdSpeedCalibrator, value: float) -> None:
    lower, upper = float(model.minimum_speed), float(model.maximum_speed)
    if not lower < float(value) < upper:
        raise ValueError(f"AD speed must be inside ({lower}, {upper})")
    fraction = (float(value) - lower) / (upper - lower)
    with torch.no_grad():
        model.global_speed_logit.copy_(torch.tensor(math.log(fraction / (1.0 - fraction)), device=model.global_speed_logit.device))


def make_pair_loader(
    archive: dict[str, np.ndarray],
    records: list[PairRecord],
    *,
    components: int,
    batch_size: int,
    samples_per_epoch: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    dataset = PairDataset(archive, records, components)
    subject_counts: dict[str, int] = {}
    for record in records:
        subject_counts[record.subject_id] = subject_counts.get(record.subject_id, 0) + 1
    weights = torch.tensor([1.0 / subject_counts[record.subject_id] for record in records], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 100_003 * int(epoch))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(records),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler, num_workers=0)


def sequence_tensors(
    *,
    archive: dict[str, np.ndarray],
    latents: np.ndarray,
    start: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    end = subject_end(archive, start)
    return {
        "source": torch.from_numpy(latents[start : start + 1].copy()).float().to(device),
        "targets": torch.from_numpy(latents[start + 1 : end].copy()).float().to(device),
        "source_norm": torch.tensor([float(archive["visit_continuous_age_norm"][start])], dtype=torch.float32, device=device),
        "target_norm": torch.from_numpy(archive["visit_continuous_age_norm"][start + 1 : end].copy()).float().to(device),
        "source_years": torch.tensor([float(archive["visit_continuous_age_years"][start])], dtype=torch.float32, device=device),
        "target_years": torch.from_numpy(archive["visit_continuous_age_years"][start + 1 : end].copy()).float().to(device),
        "condition": torch.tensor([float(archive["visit_label_ad"][start])], dtype=torch.float32, device=device),
    }


def line_slope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    centered_x = x - x.mean()
    return torch.sum(centered_x * (y - y.mean())) / torch.clamp(torch.sum(centered_x**2), min=1.0e-8)


def sequence_slope_loss(
    *,
    model: V1AnchoredAdSpeedCalibrator,
    sequence: dict[str, torch.Tensor],
    slope_scale: float,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = sequence["targets"]
    count = targets.shape[0]
    source = sequence["source"].expand(count, -1)
    source_norm = sequence["source_norm"].expand(count)
    source_years = sequence["source_years"].expand(count)
    condition = sequence["condition"].expand(count)
    predicted, _ = model.transport(
        source,
        source_norm,
        sequence["target_norm"],
        source_years,
        sequence["target_years"],
        condition,
    )
    predicted_volume = model.volume_from_latent(predicted)
    with torch.no_grad():
        target_volume = model.volume_from_latent(targets)
    times = sequence["target_years"] - sequence["source_years"]
    pred_slope = line_slope(times, torch.log(predicted_volume))
    target_slope = line_slope(times, torch.log(target_volume))
    normalized = (pred_slope - target_slope) / max(float(slope_scale), 1.0e-6)
    loss = F.huber_loss(normalized, torch.zeros_like(normalized), delta=float(huber_delta))
    return loss, pred_slope.detach(), target_slope.detach()


def evaluate_records(
    *,
    model: V1AnchoredAdSpeedCalibrator,
    archive: dict[str, np.ndarray],
    records: list[PairRecord],
    components: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(PairDataset(archive, records, components), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    groups: dict[str, dict[str, list[float]]] = {"CN": {}, "AD": {}, "overall": {}}
    cn_deltas: list[float] = []
    for raw in loader:
        batch = to_device(raw, device)
        predicted, diagnostics = model.transport(
            batch["source"], batch["source_norm"], batch["target_norm"], batch["source_years"], batch["target_years"], batch["condition"]
        )
        base = diagnostics["base_prediction"]
        pred_vertices = model.vertices_from_latent(predicted)
        target_vertices = model.vertices_from_latent(batch["target"])
        source_volume = model.volume_from_latent(batch["source"])
        target_volume = model.volume_from_latent(batch["target"])
        pred_volume = model.volume_from_latent(predicted)
        delta_years = batch["target_years"] - batch["source_years"]
        safe_delta = torch.where(delta_years.abs() < 1.0e-6, torch.full_like(delta_years, 1.0e-6), delta_years)
        values = {
            "pca": torch.mean((predicted - batch["target"]) ** 2, dim=1),
            "vertex": torch.mean(torch.abs(pred_vertices - target_vertices), dim=(1, 2)),
            "volume_relative": torch.abs(pred_volume - target_volume) / target_volume.clamp_min(1.0e-8),
            "rate": torch.abs((torch.log(pred_volume) - torch.log(source_volume)) / safe_delta - (torch.log(target_volume) - torch.log(source_volume)) / safe_delta),
            "speed": diagnostics["speed"],
        }
        labels = batch["condition"] >= 0.5
        for name, mask in (("AD", labels), ("CN", ~labels), ("overall", torch.ones_like(labels, dtype=torch.bool))):
            if not bool(mask.any()):
                continue
            bucket = groups[name]
            for metric, tensor in values.items():
                bucket.setdefault(metric, []).extend(tensor[mask].detach().cpu().tolist())
        if bool((~labels).any()):
            cn_deltas.extend(torch.max(torch.abs(predicted[~labels] - base[~labels]), dim=1).values.detach().cpu().tolist())
    result: dict[str, Any] = {"groups": {}, "cn_max_abs_delta_from_v1": max(cn_deltas, default=0.0)}
    for name, values in groups.items():
        result["groups"][name] = {f"{metric}_mean": float(np.mean(items)) if items else float("nan") for metric, items in values.items()}
        result["groups"][name]["rows"] = len(values.get("pca", []))
    return result


@torch.no_grad()
def evaluate_subject_slopes(
    *,
    model: V1AnchoredAdSpeedCalibrator,
    archive: dict[str, np.ndarray],
    latents: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors: list[float] = []
    observed: list[float] = []
    predicted: list[float] = []
    for start in first_visit_sequence_starts(archive, diagnosis="AD"):
        sequence = sequence_tensors(archive=archive, latents=latents, start=start, device=device)
        count = sequence["targets"].shape[0]
        source = sequence["source"].expand(count, -1)
        model_prediction, _ = model.transport(
            source,
            sequence["source_norm"].expand(count),
            sequence["target_norm"],
            sequence["source_years"].expand(count),
            sequence["target_years"],
            sequence["condition"].expand(count),
        )
        target_volume = model.volume_from_latent(sequence["targets"])
        pred_volume = model.volume_from_latent(model_prediction)
        x = sequence["target_years"] - sequence["source_years"]
        observed_slope = line_slope(x, torch.log(target_volume))
        pred_slope = line_slope(x, torch.log(pred_volume))
        observed.append(float(observed_slope.item()))
        predicted.append(float(pred_slope.item()))
        errors.append(float(torch.abs(pred_slope - observed_slope).item()))
    correlation = float(np.corrcoef(predicted, observed)[0, 1]) if len(observed) > 1 and np.std(predicted) > 1.0e-12 else float("nan")
    return {
        "subjects": float(len(errors)),
        "ad_subject_slope_abs_error": float(np.mean(errors)) if errors else float("nan"),
        "ad_observed_slope_mean": float(np.mean(observed)) if observed else float("nan"),
        "ad_predicted_slope_mean": float(np.mean(predicted)) if predicted else float("nan"),
        "ad_slope_pearson": correlation,
    }


def checkpoint_payload(
    *,
    model: V1AnchoredAdSpeedCalibrator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    metadata_dir: Path,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config_from_instance(model),
        "metrics": metrics,
        "config": config,
        "metadata_dir": str(metadata_dir),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def feasible(
    metrics: dict[str, Any],
    references: dict[str, Any],
    tolerance: float,
    first_last_tolerance: float,
) -> bool:
    if float(metrics["cn_max_abs_delta_from_v1"]) > 2.0e-6:
        return False
    checks = (
        ("val_all_pairs", "groups", float(tolerance)),
        ("val_first_last", "first_last_groups", float(first_last_tolerance)),
    )
    for reference_name, metric_name, allowed_increase in checks:
        for name in ("AD", "overall"):
            current = metrics[metric_name][name]
            reference = references[reference_name][name]
            if int(reference.get("rows", 0)) <= 0:
                continue
            if not math.isfinite(float(current.get("pca_mean", float("nan")))) or not math.isfinite(float(current.get("vertex_mean", float("nan")))):
                return False
            if float(current["pca_mean"]) > float(reference["endpoint_pca_mse"]) * (1.0 + allowed_increase):
                return False
            if float(current["vertex_mean"]) > float(reference["endpoint_vertex_mae"]) * (1.0 + allowed_increase):
                return False
    return True


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_config = dict(config["training"])
    loss_config = dict(config["loss"])
    selection_config = dict(config["selection"])
    device = resolve_device(args.device)
    seed = int(args.seed if args.seed is not None else train_config["seed"])
    set_seed(seed)
    epochs = int(args.epochs if args.epochs is not None else train_config["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else train_config["batch_size"])
    samples_per_epoch = int(args.samples_per_epoch if args.samples_per_epoch is not None else train_config["samples_per_epoch"])
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else EXPERIMENT_DIR / "metadata"
    stats_path = metadata_dir / "speed_feature_stats.npz"
    refs_path = metadata_dir / "frozen_v1_reference_metrics.json"
    if not stats_path.is_file() or not refs_path.is_file():
        raise FileNotFoundError("Run prepare_v1_speed_metadata.py before training")
    with np.load(stats_path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    references = load_json(refs_path)
    output_root = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / "runs"
    run_dir = output_root / str(args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", {"config": config, "args": vars(args), "device": str(device), "seed": seed})

    train_archive = load_split_archive("train")
    val_archive = load_split_archive("val")
    components_count = int(config["components"])
    train_records = build_pair_records(train_archive, diagnosis="AD")
    val_records = build_pair_records(val_archive)
    val_first_last = first_last_records(val_archive)
    if args.max_train_pairs > 0:
        train_records = train_records[: int(args.max_train_pairs)]
    if args.max_val_pairs > 0:
        val_records = limit_records_stratified(val_records, int(args.max_val_pairs))
        val_first_last = limit_records_stratified(val_first_last, int(args.max_val_pairs))
    if not train_records:
        raise RuntimeError("No AD training pairs")

    model = make_model(config=config, metadata=metadata, device=device)
    base_before = base_parameter_snapshot(model)
    head_parameters = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.global_speed_logit], "lr": float(train_config["learning_rate_global"]), "weight_decay": 0.0},
            {"params": head_parameters, "lr": float(train_config["learning_rate_head"]), "weight_decay": float(train_config["weight_decay"])},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=0.1 * float(train_config["learning_rate_head"]))
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_endpoint = float("inf")
    best_volume = float("inf")
    best_feasible = float("inf")
    stale_epochs = 0
    resumed = False

    if args.resume:
        latest = checkpoint_dir / "latest.pth"
        if latest.is_file():
            payload = torch.load(latest, map_location=device)
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload["epoch"]) + 1
            resumed = True
            history_path = run_dir / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text())

    val_latents = pca_latents(val_archive, components_count)
    if resumed:
        def metrics_from(name: str) -> dict[str, Any]:
            path = checkpoint_dir / name
            if not path.is_file():
                return {}
            return dict(torch.load(path, map_location="cpu").get("metrics", {}))

        endpoint_metrics = metrics_from("best_endpoint.pth")
        if endpoint_metrics:
            best_endpoint = float(endpoint_metrics["groups"]["overall"]["pca_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_pca_mse"]) + float(endpoint_metrics["groups"]["overall"]["vertex_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_vertex_mae"])
        volume_metrics = metrics_from("best_ad_volume.pth")
        if volume_metrics:
            best_volume = float(volume_metrics.get("volume_score", float("inf")))
        feasible_metrics = metrics_from("best_feasible_volume.pth")
        if feasible_metrics and bool(feasible_metrics.get("feasible", False)):
            best_feasible = float(feasible_metrics.get("volume_score", float("inf")))
    else:
        # Keep a true v1 state even when the primary run starts at the validated
        # AD speed. This is a safe, directly evaluable fallback checkpoint.
        identity_model = copy.deepcopy(model)
        set_global_speed(identity_model, 1.0)
        identity_val = evaluate_records(model=identity_model, archive=val_archive, records=val_records, components=components_count, device=device, batch_size=batch_size)
        identity_first_last = evaluate_records(model=identity_model, archive=val_archive, records=val_first_last, components=components_count, device=device, batch_size=batch_size)
        identity_slopes = evaluate_subject_slopes(model=identity_model, archive=val_archive, latents=val_latents, device=device)
        identity_metrics = {**identity_val, **identity_slopes, "first_last_groups": identity_first_last["groups"], "epoch": 0, "global_speed": 1.0, "checkpoint_role": "exact_v1_identity"}
        identity_metrics["feasible"] = feasible(identity_metrics, references, float(selection_config["endpoint_tolerance"]), float(selection_config["first_last_tolerance"]))
        identity_metrics["volume_score"] = float(identity_val["groups"]["AD"]["rate_mean"]) + float(identity_slopes["ad_subject_slope_abs_error"]) + float(selection_config["volume_relative_error_weight"]) * float(identity_val["groups"]["AD"]["volume_relative_mean"])
        save_checkpoint(checkpoint_dir / "epoch_0000_v1_identity.pth", checkpoint_payload(model=identity_model, optimizer=optimizer, scheduler=scheduler, epoch=0, metrics=identity_metrics, config=config, metadata_dir=metadata_dir))

        initial_val = evaluate_records(model=model, archive=val_archive, records=val_records, components=components_count, device=device, batch_size=batch_size)
        initial_first_last = evaluate_records(model=model, archive=val_archive, records=val_first_last, components=components_count, device=device, batch_size=batch_size)
        initial_slopes = evaluate_subject_slopes(model=model, archive=val_archive, latents=val_latents, device=device)
        initial_metrics = {**initial_val, **initial_slopes, "first_last_groups": initial_first_last["groups"], "epoch": 0, "global_speed": float(model.global_speed.detach().cpu().item()), "checkpoint_role": "configured_fixed_ad_speed"}
        initial_metrics["feasible"] = feasible(initial_metrics, references, float(selection_config["endpoint_tolerance"]), float(selection_config["first_last_tolerance"]))
        initial_metrics["volume_score"] = float(initial_val["groups"]["AD"]["rate_mean"]) + float(initial_slopes["ad_subject_slope_abs_error"]) + float(selection_config["volume_relative_error_weight"]) * float(initial_val["groups"]["AD"]["volume_relative_mean"])
        initial_endpoint_score = float(initial_val["groups"]["overall"]["pca_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_pca_mse"]) + float(initial_val["groups"]["overall"]["vertex_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_vertex_mae"])
        initial_payload = checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=0, metrics=initial_metrics, config=config, metadata_dir=metadata_dir)
        save_checkpoint(checkpoint_dir / "epoch_0000_configured_speed.pth", initial_payload)
        save_checkpoint(checkpoint_dir / "best_endpoint.pth", initial_payload)
        save_checkpoint(checkpoint_dir / "best_ad_volume.pth", initial_payload)
        save_checkpoint(checkpoint_dir / "best_feasible_volume.pth", initial_payload)
        best_endpoint = initial_endpoint_score
        best_volume = initial_metrics["volume_score"]
        best_feasible = initial_metrics["volume_score"] if initial_metrics["feasible"] else float("inf")

    train_latents = pca_latents(train_archive, components_count)
    sequence_starts = first_visit_sequence_starts(train_archive, diagnosis="AD")
    loss_scales = references["loss_scales"]
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for parameter in head_parameters:
            parameter.requires_grad_(epoch > int(train_config["warmup_global_epochs"]))
        loader = make_pair_loader(
            train_archive,
            train_records,
            components=components_count,
            batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            epoch=epoch,
        )
        sequence_order = list(sequence_starts)
        random.Random(seed + epoch * 200_003).shuffle(sequence_order)
        if int(train_config["sequence_subjects_per_epoch"]) > 0:
            sequence_order = sequence_order[: int(train_config["sequence_subjects_per_epoch"])]
        if not sequence_order:
            raise RuntimeError("No AD sequences with at least three visits")
        totals: dict[str, float] = {key: 0.0 for key in ("total", "pca", "vertex", "pair_rate", "subject_slope", "group_slope", "cocycle", "inverse", "no_degradation", "speed_regularization")}
        rows = 0
        ramp_start = int(train_config["warmup_global_epochs"])
        ramp = min(1.0, max(0.0, float(epoch - ramp_start) / max(float(train_config["consistency_ramp_epochs"]), 1.0)))
        for step, raw in enumerate(loader):
            batch = to_device(raw, device)
            predicted, diagnostics = model.transport(
                batch["source"], batch["source_norm"], batch["target_norm"], batch["source_years"], batch["target_years"], batch["condition"]
            )
            base = diagnostics["base_prediction"]
            pred_vertices = model.vertices_from_latent(predicted)
            target_vertices = model.vertices_from_latent(batch["target"])
            base_vertices = model.vertices_from_latent(base)
            pca_per_row = torch.mean((predicted - batch["target"]) ** 2, dim=1)
            base_pca_per_row = torch.mean((base - batch["target"]) ** 2, dim=1)
            vertex_per_row = torch.mean(torch.abs(pred_vertices - target_vertices), dim=(1, 2))
            base_vertex_per_row = torch.mean(torch.abs(base_vertices - target_vertices), dim=(1, 2))
            pca_loss = pca_per_row.mean() / float(loss_scales["pca"])
            vertex_loss = vertex_per_row.mean() / float(loss_scales["vertex"])
            source_volume = model.volume_from_latent(batch["source"])
            target_volume = model.volume_from_latent(batch["target"])
            pred_volume = model.volume_from_latent(predicted)
            delta_years = batch["target_years"] - batch["source_years"]
            safe_delta = torch.where(delta_years.abs() < 1.0e-6, torch.full_like(delta_years, 1.0e-6), delta_years)
            true_rate = (torch.log(target_volume) - torch.log(source_volume)) / safe_delta
            pred_rate = (torch.log(pred_volume) - torch.log(source_volume)) / safe_delta
            pair_rate_loss = F.huber_loss((pred_rate - true_rate) / float(loss_scales["rate"]), torch.zeros_like(pred_rate), delta=float(loss_config["huber_delta"]))
            group_slope_loss = F.mse_loss(pred_rate.mean() / float(loss_scales["rate"]), true_rate.mean() / float(loss_scales["rate"]))
            sequence = sequence_tensors(archive=train_archive, latents=train_latents, start=sequence_order[step % len(sequence_order)], device=device)
            subject_slope_loss, _, _ = sequence_slope_loss(model=model, sequence=sequence, slope_scale=float(loss_scales["slope"]), huber_delta=float(loss_config["huber_delta"]))
            midpoint_ratio = torch.rand_like(safe_delta)
            midpoint_years = batch["source_years"] + midpoint_ratio * safe_delta
            midpoint_norm = batch["source_norm"] + midpoint_ratio * (batch["target_norm"] - batch["source_norm"])
            midpoint, _ = model.transport(batch["source"], batch["source_norm"], midpoint_norm, batch["source_years"], midpoint_years, batch["condition"])
            composed, _ = model.transport(midpoint, midpoint_norm, batch["target_norm"], midpoint_years, batch["target_years"], batch["condition"])
            cocycle_loss = torch.mean((composed - predicted.detach()) ** 2) / float(loss_scales["consistency"])
            inverse, _ = model.transport(predicted, batch["target_norm"], batch["source_norm"], batch["target_years"], batch["source_years"], batch["condition"])
            inverse_loss = torch.mean((inverse - batch["source"]) ** 2) / float(loss_scales["consistency"])
            tolerance = float(loss_config["no_degradation_tolerance"])
            no_degradation = torch.mean(F.relu((pca_per_row - base_pca_per_row * (1.0 + tolerance)) / float(loss_scales["pca"]))) + torch.mean(F.relu((vertex_per_row - base_vertex_per_row * (1.0 + tolerance)) / float(loss_scales["vertex"])))
            speed_regularization = torch.mean(diagnostics["individual_log_speed"] ** 2)
            loss = (
                float(loss_config["pca_weight"]) * pca_loss
                + float(loss_config["vertex_weight"]) * vertex_loss
                + float(loss_config["pair_rate_weight"]) * pair_rate_loss
                + float(loss_config["subject_slope_weight"]) * subject_slope_loss
                + float(loss_config["group_slope_weight"]) * group_slope_loss
                + ramp * float(loss_config["cocycle_weight"]) * cocycle_loss
                + ramp * float(loss_config["inverse_weight"]) * inverse_loss
                + float(loss_config["no_degradation_weight"]) * no_degradation
                + float(loss_config["speed_regularization_weight"]) * speed_regularization
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_config["gradient_clip_norm"]))
            optimizer.step()
            batch_rows = int(batch["source"].shape[0])
            rows += batch_rows
            for key, value in (("total", loss), ("pca", pca_loss), ("vertex", vertex_loss), ("pair_rate", pair_rate_loss), ("subject_slope", subject_slope_loss), ("group_slope", group_slope_loss), ("cocycle", cocycle_loss), ("inverse", inverse_loss), ("no_degradation", no_degradation), ("speed_regularization", speed_regularization)):
                totals[key] += float(value.detach().cpu().item()) * batch_rows
        scheduler.step()
        assert_base_unchanged(model, base_before)
        row: dict[str, Any] = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "global_speed": float(model.global_speed.detach().cpu().item()),
            "consistency_ramp": ramp,
            "learning_rate_global": float(optimizer.param_groups[0]["lr"]),
            "learning_rate_head": float(optimizer.param_groups[1]["lr"]),
            **{f"train_{key}": value / max(rows, 1) for key, value in totals.items()},
        }
        validation = evaluate_records(model=model, archive=val_archive, records=val_records, components=components_count, device=device, batch_size=batch_size)
        first_last_validation = evaluate_records(model=model, archive=val_archive, records=val_first_last, components=components_count, device=device, batch_size=batch_size)
        slope_validation = evaluate_subject_slopes(model=model, archive=val_archive, latents=val_latents, device=device)
        metrics = {**validation, **slope_validation, "first_last_groups": first_last_validation["groups"], "epoch": epoch, "global_speed": row["global_speed"]}
        metrics["feasible"] = feasible(metrics, references, float(selection_config["endpoint_tolerance"]), float(selection_config["first_last_tolerance"]))
        metrics["volume_score"] = float(validation["groups"]["AD"]["rate_mean"]) + float(slope_validation["ad_subject_slope_abs_error"]) + float(selection_config["volume_relative_error_weight"]) * float(validation["groups"]["AD"]["volume_relative_mean"])
        endpoint_score = float(validation["groups"]["overall"]["pca_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_pca_mse"]) + float(validation["groups"]["overall"]["vertex_mean"]) / float(references["val_all_pairs"]["overall"]["endpoint_vertex_mae"])
        row.update(
            {
                "val_endpoint_score": endpoint_score,
                "val_volume_score": metrics["volume_score"],
                "val_feasible": bool(metrics["feasible"]),
                "val_ad_pca": validation["groups"]["AD"]["pca_mean"],
                "val_ad_vertex": validation["groups"]["AD"]["vertex_mean"],
                "val_ad_volume_relative": validation["groups"]["AD"]["volume_relative_mean"],
                "val_ad_rate": validation["groups"]["AD"]["rate_mean"],
                "val_ad_subject_slope": slope_validation["ad_subject_slope_abs_error"],
                "val_first_last_ad_pca": first_last_validation["groups"]["AD"].get("pca_mean", float("nan")),
                "val_first_last_ad_vertex": first_last_validation["groups"]["AD"].get("vertex_mean", float("nan")),
                "val_cn_max_abs_delta_from_v1": validation["cn_max_abs_delta_from_v1"],
            }
        )
        payload = checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, metrics=metrics, config=config, metadata_dir=metadata_dir)
        save_checkpoint(checkpoint_dir / "latest.pth", payload)
        if epoch % int(train_config["snapshot_frequency"]) == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pth", payload)
        if endpoint_score < best_endpoint - float(train_config["early_stopping_min_delta"]):
            best_endpoint = endpoint_score
            save_checkpoint(checkpoint_dir / "best_endpoint.pth", payload)
        if metrics["volume_score"] < best_volume - float(train_config["early_stopping_min_delta"]):
            best_volume = metrics["volume_score"]
            save_checkpoint(checkpoint_dir / "best_ad_volume.pth", payload)
        if metrics["feasible"] and metrics["volume_score"] < best_feasible - float(train_config["early_stopping_min_delta"]):
            best_feasible = metrics["volume_score"]
            save_checkpoint(checkpoint_dir / "best_feasible_volume.pth", payload)
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(row)
        write_json(run_dir / "history.json", history)
        write_json(run_dir / "training_status.json", {"status": "running", "epoch": epoch, "best_feasible_volume": best_feasible, "run_dir": str(run_dir)})
        print(json.dumps(row, sort_keys=True))
        if stale_epochs >= int(train_config["early_stopping_patience"]):
            print(f"Early stopping after {stale_epochs} stale epochs")
            break
    write_json(run_dir / "training_status.json", {"status": "completed", "epochs_completed": len(history), "best_feasible_volume": best_feasible, "run_dir": str(run_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
