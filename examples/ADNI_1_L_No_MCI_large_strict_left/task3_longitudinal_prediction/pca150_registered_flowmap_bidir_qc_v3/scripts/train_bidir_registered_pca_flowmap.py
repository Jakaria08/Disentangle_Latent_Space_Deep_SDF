#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bidir_registered_pca_model import RegisteredPCAFlowMap
from bidir_registered_pca_utils import (
    DEFAULT_EXPERIMENT_NAME,
    TASK_DIR,
    PairRecord,
    build_pair_records,
    coefficient_stats,
    decode_pca_torch,
    experiment_root,
    finite_mean,
    load_pca_model,
    load_split_archive,
    mesh_volume_torch,
    pca_latents,
    summarize_pair_records,
    vertex_normals_np,
    write_json,
)


DEFAULT_RUN_NAME = "bidir_e04_k16_full_volume_local_seed42"


class PairDataset(Dataset):
    def __init__(
        self,
        archive: dict[str, np.ndarray],
        records: list[PairRecord],
        components: int,
    ) -> None:
        self.archive = archive
        self.records = records
        self.latents = pca_latents(archive, int(components))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "source_latent": torch.from_numpy(self.latents[record.source_index].copy()).float(),
            "target_latent": torch.from_numpy(self.latents[record.target_index].copy()).float(),
            "source_age_norm": torch.tensor(record.source_age_norm, dtype=torch.float32),
            "target_age_norm": torch.tensor(record.target_age_norm, dtype=torch.float32),
            "source_age_years": torch.tensor(record.source_age_years, dtype=torch.float32),
            "target_age_years": torch.tensor(record.target_age_years, dtype=torch.float32),
            "delta_years": torch.tensor(record.delta_years, dtype=torch.float32),
            "abs_gap_years": torch.tensor(record.abs_gap_years, dtype=torch.float32),
            "condition": torch.tensor(float(record.label_ad), dtype=torch.float32),
            "label_ad": torch.tensor(record.label_ad, dtype=torch.long),
            "is_backward": torch.tensor(record.direction == "backward", dtype=torch.bool),
            "has_intermediate": torch.tensor(record.intermediate_index >= 0, dtype=torch.bool),
            "intermediate_latent": torch.from_numpy(
                (
                    self.latents[record.intermediate_index]
                    if record.intermediate_index >= 0
                    else np.zeros(self.latents.shape[1], dtype=np.float32)
                ).copy()
            ).float(),
            "intermediate_age_norm": torch.tensor(
                float(self.archive["visit_continuous_age_norm"][record.intermediate_index])
                if record.intermediate_index >= 0
                else 0.0,
                dtype=torch.float32,
            ),
            "intermediate_age_years": torch.tensor(
                float(self.archive["visit_continuous_age_years"][record.intermediate_index])
                if record.intermediate_index >= 0
                else 0.0,
                dtype=torch.float32,
            ),
        }


class SequenceDataset(Dataset):
    def __init__(self, archive: dict[str, np.ndarray], starts: list[int], components: int) -> None:
        self.archive = archive
        self.starts = starts
        self.latents = pca_latents(archive, int(components))
        self.offsets = archive["subject_visit_offsets"]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        subject_end = int(self.offsets[np.searchsorted(self.offsets, start, side="right")])
        latents = self.latents[start:subject_end]
        return {
            "latents": torch.from_numpy(latents.copy()).float(),
            "age_norm": torch.from_numpy(
                self.archive["visit_continuous_age_norm"][start:subject_end].copy()
            ).float(),
            "age_years": torch.from_numpy(
                self.archive["visit_continuous_age_years"][start:subject_end].copy()
            ).float(),
            "condition": torch.tensor(
                float(self.archive["visit_label_ad"][start]),
                dtype=torch.float32,
            ),
        }


def build_sequence_starts(archive: dict[str, np.ndarray], min_length: int = 3) -> list[int]:
    starts: list[int] = []
    offsets = archive["subject_visit_offsets"]
    for subject_index in range(len(offsets) - 1):
        subject_start = int(offsets[subject_index])
        subject_end = int(offsets[subject_index + 1])
        for start in range(subject_start, subject_end - int(min_length) + 1):
            starts.append(start)
    return starts


def grouped_sequence_loaders(
    archive: dict[str, np.ndarray],
    starts: list[int],
    components: int,
    batch_size: int,
    num_workers: int,
) -> list[DataLoader]:
    by_length: dict[int, list[int]] = {}
    offsets = archive["subject_visit_offsets"]
    for start in starts:
        subject_end = int(offsets[np.searchsorted(offsets, start, side="right")])
        by_length.setdefault(subject_end - int(start), []).append(int(start))
    return [
        DataLoader(
            SequenceDataset(archive, group_starts, int(components)),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
        )
        for _, group_starts in sorted(by_length.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a bidirectional registered PCA flow-map with dynamic-PCA, "
            "volume, local, EMA cocycle, and multi-visit sequence losses."
        )
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--dynamic-dim", type=int, default=16)
    parser.add_argument("--dynamic-basis", default=None)
    parser.add_argument("--rate-targets", default=None)
    parser.add_argument("--local-weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--max-sequences-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-frequency", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-6)
    parser.add_argument("--snapshot-frequency", type=int, default=25)
    parser.add_argument("--latest-frequency", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--latent-condition-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--train-pair-type", choices=("all", "adjacent", "nonadjacent"), default="all")
    parser.add_argument("--train-max-gap-years", type=float, default=0.0)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--include-backward-pairs", action="store_true")
    parser.add_argument("--include-sequence-loss", action="store_true")
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.set_defaults(balanced_sampling=True)
    parser.add_argument("--backward-loss-weight", type=float, default=0.5)
    parser.add_argument("--backward-volume-rate-weight", type=float, default=0.2)
    parser.add_argument("--vertex-weight", type=float, default=1.0)
    parser.add_argument("--pca-weight", type=float, default=0.25)
    parser.add_argument("--dynamic-rate-weight", type=float, default=0.2)
    parser.add_argument("--volume-rate-weight", type=float, default=1.0)
    parser.add_argument("--group-volume-weight", type=float, default=0.25)
    parser.add_argument("--local-normal-weight", type=float, default=0.25)
    parser.add_argument("--ema-cocycle-weight", type=float, default=0.2)
    parser.add_argument("--sequence-rollout-weight", type=float, default=0.25)
    parser.add_argument("--sequence-cocycle-weight", type=float, default=0.1)
    parser.add_argument("--reverse-sequence-weight", type=float, default=0.5)
    parser.add_argument("--inverse-weight", type=float, default=0.1)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--latent-guard-weight", type=float, default=0.02)
    parser.add_argument("--latent-guard-threshold", type=float, default=3.5)
    parser.add_argument("--cocycle-ramp-epochs", type=int, default=15)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--pareto-volume-weight", type=float, default=1.0)
    parser.add_argument("--pareto-local-weight", type=float, default=0.2)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def run_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    return experiment_root(args.experiment_name) / "runs" / str(args.run_name)


def basis_path_for(args: argparse.Namespace) -> Path:
    if args.dynamic_basis:
        return Path(args.dynamic_basis).expanduser()
    return (
        experiment_root(args.experiment_name)
        / "metadata"
        / f"dynamic_pca_velocity_k{int(args.dynamic_dim)}.npz"
    )


def rate_targets_path_for(args: argparse.Namespace) -> Path:
    if args.rate_targets:
        return Path(args.rate_targets).expanduser()
    return experiment_root(args.experiment_name) / "metadata" / "empirical_rate_targets.json"


def local_weights_path_for(args: argparse.Namespace) -> Path:
    if args.local_weights:
        return Path(args.local_weights).expanduser()
    return experiment_root(args.experiment_name) / "metadata" / "local_ad_cn_change_weights.npy"


def load_rate_targets(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    values = data.get("log_volume_rate_per_year", {})
    result: dict[int, float] = {}
    if "CN" in values:
        result[0] = float(values["CN"].get("mean", 0.0))
    if "AD" in values:
        result[1] = float(values["AD"].get("mean", 0.0))
    return result


def make_balanced_epoch_records(
    records: list[PairRecord],
    *,
    samples_per_epoch: int,
    seed: int,
    epoch: int,
    balanced: bool,
) -> list[PairRecord]:
    if not records:
        return []
    count = int(samples_per_epoch) if int(samples_per_epoch) > 0 else len(records)
    rng = random.Random(int(seed) + int(epoch) * 100_003)
    if not balanced:
        selected = list(records)
        rng.shuffle(selected)
        if count <= len(selected):
            return selected[:count]
        return [rng.choice(selected) for _ in range(count)]

    buckets: dict[tuple[str, str, str], dict[str, list[PairRecord]]] = {}
    for record in records:
        key = (record.diagnosis, record.gap_bin, record.direction)
        buckets.setdefault(key, {}).setdefault(record.subject_id, []).append(record)
    keys = sorted(buckets)
    selected: list[PairRecord] = []
    for index in range(count):
        key = keys[index % len(keys)]
        subject_map = buckets[key]
        subject_id = rng.choice(list(subject_map.keys()))
        selected.append(rng.choice(subject_map[subject_id]))
    rng.shuffle(selected)
    return selected


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def weighted_local_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).view(1, -1)
    return torch.sum(values * weights) / torch.clamp(weights.sum() * values.shape[0], min=1.0e-8)


def weighted_row_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).view(-1)
    values = values.view(-1)
    return torch.sum(values * weights) / torch.clamp(weights.sum(), min=1.0e-8)


def smooth_l1_per_row(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(left, right, reduction="none")
    return loss.reshape(loss.shape[0], -1).mean(dim=1)


def local_weighted_per_row(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype).view(1, -1)
    return torch.sum(values * weights, dim=1) / torch.clamp(weights.sum(), min=1.0e-8)


def z_guard_loss(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, threshold: float) -> torch.Tensor:
    z_score = torch.abs((latent - mean) / std)
    excess = torch.relu(z_score - float(threshold))
    return torch.mean(excess**2)


def update_ema_model(
    *,
    ema_model: RegisteredPCAFlowMap,
    model: RegisteredPCAFlowMap,
    decay: float,
) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(float(decay)).add_(param, alpha=1.0 - float(decay))
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_one_epoch(
    *,
    model: RegisteredPCAFlowMap,
    ema_model: RegisteredPCAFlowMap,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    faces: torch.Tensor,
    template_normals: torch.Tensor,
    local_weights: torch.Tensor,
    rate_targets: dict[int, float],
    coefficient_mean: torch.Tensor,
    coefficient_std: torch.Tensor,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "total_loss": 0.0,
        "vertex_loss": 0.0,
        "pca_loss": 0.0,
        "dynamic_rate_loss": 0.0,
        "volume_rate_loss": 0.0,
        "group_volume_loss": 0.0,
        "local_normal_loss": 0.0,
        "ema_cocycle_loss": 0.0,
        "inverse_loss": 0.0,
        "identity_loss": 0.0,
        "latent_guard_loss": 0.0,
    }
    total_rows = 0
    ramp = min(1.0, max(0.0, float(epoch) / max(float(args.cocycle_ramp_epochs), 1.0)))
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        source = batch["source_latent"]
        target = batch["target_latent"]
        source_age_norm = batch["source_age_norm"]
        target_age_norm = batch["target_age_norm"]
        source_age_years = batch["source_age_years"]
        target_age_years = batch["target_age_years"]
        delta_years = batch["delta_years"].view(-1, 1)
        signed_delta_years = torch.where(
            torch.abs(delta_years) < 1.0e-6,
            torch.full_like(delta_years, 1.0e-6),
            delta_years,
        )
        condition = batch["condition"]
        is_backward = batch["is_backward"].view(-1)
        direction_weights = torch.where(
            is_backward,
            torch.full_like(condition, float(args.backward_loss_weight)),
            torch.ones_like(condition),
        )
        volume_direction_weights = torch.where(
            is_backward,
            torch.full_like(condition, float(args.backward_volume_rate_weight)),
            torch.ones_like(condition),
        )

        pred = model.transport(
            source,
            source_age_norm,
            target_age_norm,
            source_age_years,
            target_age_years,
            condition,
        )
        source_vertices = decode_pca_torch(source, mean_flat, components)
        target_vertices = decode_pca_torch(target, mean_flat, components)
        pred_vertices = decode_pca_torch(pred, mean_flat, components)
        vertex_loss = weighted_row_mean(
            smooth_l1_per_row(pred_vertices, target_vertices),
            direction_weights,
        )
        pca_loss = weighted_row_mean(smooth_l1_per_row(pred, target), direction_weights)

        true_rate = (target - source) / signed_delta_years
        pred_rate = (pred - source) / signed_delta_years
        basis = model.dynamic_basis.to(device=device, dtype=source.dtype)
        dynamic_rate_loss = weighted_row_mean(
            smooth_l1_per_row(pred_rate @ basis.T, true_rate @ basis.T),
            direction_weights,
        )

        source_volume = mesh_volume_torch(source_vertices, faces)
        target_volume = mesh_volume_torch(target_vertices, faces)
        pred_volume = mesh_volume_torch(pred_vertices, faces)
        true_log_rate = (torch.log(target_volume) - torch.log(source_volume)) / signed_delta_years.view(-1)
        pred_log_rate = (torch.log(pred_volume) - torch.log(source_volume)) / signed_delta_years.view(-1)
        volume_rate_loss = weighted_row_mean(
            F.smooth_l1_loss(pred_log_rate, true_log_rate, reduction="none"),
            volume_direction_weights,
        )

        group_losses: list[torch.Tensor] = []
        for label, target_rate in rate_targets.items():
            mask = (batch["label_ad"] == int(label)) & (~is_backward)
            if bool(mask.any().item()):
                target_tensor = torch.tensor(float(target_rate), device=device, dtype=source.dtype)
                group_losses.append(F.smooth_l1_loss(pred_log_rate[mask].mean(), target_tensor))
        group_volume_loss = torch.stack(group_losses).mean() if group_losses else source.sum() * 0.0

        normal_true = torch.sum(
            ((target_vertices - source_vertices) / signed_delta_years.view(-1, 1, 1)) * template_normals.view(1, -1, 3),
            dim=2,
        )
        normal_pred = torch.sum(
            ((pred_vertices - source_vertices) / signed_delta_years.view(-1, 1, 1)) * template_normals.view(1, -1, 3),
            dim=2,
        )
        local_values = F.smooth_l1_loss(normal_pred, normal_true, reduction="none")
        local_normal_loss = weighted_row_mean(
            local_weighted_per_row(local_values, local_weights),
            direction_weights,
        )

        ema_cocycle_loss = source.sum() * 0.0
        if float(args.ema_cocycle_weight) > 0.0:
            ratio = torch.rand_like(signed_delta_years)
            mid_age_years = source_age_years.view(-1, 1) + ratio * signed_delta_years
            mid_age_norm = source_age_norm.view(-1, 1) + ratio * (
                target_age_norm.view(-1, 1) - source_age_norm.view(-1, 1)
            )
            mid = model.transport(
                source,
                source_age_norm,
                mid_age_norm.view(-1),
                source_age_years,
                mid_age_years.view(-1),
                condition,
            )
            composed = model.transport(
                mid,
                mid_age_norm.view(-1),
                target_age_norm,
                mid_age_years.view(-1),
                target_age_years,
                condition,
            )
            with torch.no_grad():
                teacher = ema_model.transport(
                    source,
                    source_age_norm,
                    target_age_norm,
                    source_age_years,
                    target_age_years,
                    condition,
                )
            ema_cocycle_loss = weighted_row_mean(
                smooth_l1_per_row(composed, teacher.detach()),
                direction_weights,
            )

        inverse = model.transport(
            pred,
            target_age_norm,
            source_age_norm,
            target_age_years,
            source_age_years,
            condition,
        )
        inverse_loss = weighted_row_mean(smooth_l1_per_row(inverse, source), direction_weights)
        identity = model.transport(
            source,
            source_age_norm,
            source_age_norm,
            source_age_years,
            source_age_years,
            condition,
        )
        identity_loss = F.smooth_l1_loss(identity, source)
        latent_guard = z_guard_loss(pred, coefficient_mean, coefficient_std, float(args.latent_guard_threshold))

        loss = (
            float(args.vertex_weight) * vertex_loss
            + float(args.pca_weight) * pca_loss
            + float(args.dynamic_rate_weight) * dynamic_rate_loss
            + float(args.volume_rate_weight) * volume_rate_loss
            + float(args.group_volume_weight) * group_volume_loss
            + float(args.local_normal_weight) * local_normal_loss
            + float(args.ema_cocycle_weight) * ramp * ema_cocycle_loss
            + float(args.inverse_weight) * inverse_loss
            + float(args.identity_weight) * identity_loss
            + float(args.latent_guard_weight) * latent_guard
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.gradient_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
        optimizer.step()
        update_ema_model(ema_model=ema_model, model=model, decay=float(args.ema_decay))

        batch_rows = int(source.shape[0])
        total_rows += batch_rows
        for key, value in (
            ("total_loss", loss),
            ("vertex_loss", vertex_loss),
            ("pca_loss", pca_loss),
            ("dynamic_rate_loss", dynamic_rate_loss),
            ("volume_rate_loss", volume_rate_loss),
            ("group_volume_loss", group_volume_loss),
            ("local_normal_loss", local_normal_loss),
            ("ema_cocycle_loss", ema_cocycle_loss),
            ("inverse_loss", inverse_loss),
            ("identity_loss", identity_loss),
            ("latent_guard_loss", latent_guard),
        ):
            totals[key] += float(value.detach().cpu().item()) * batch_rows
    return {key: value / max(total_rows, 1) for key, value in totals.items()}


def sequence_loss_for_order(
    *,
    model: RegisteredPCAFlowMap,
    latents: torch.Tensor,
    age_norm: torch.Tensor,
    age_years: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = latents[:, 0, :]
    source_age_norm = age_norm[:, 0]
    source_age_years = age_years[:, 0]
    direct_steps: list[torch.Tensor] = []
    composed_steps: list[torch.Tensor] = []
    current = source
    for step in range(1, latents.shape[1]):
        direct_steps.append(
            model.transport(
                source,
                source_age_norm,
                age_norm[:, step],
                source_age_years,
                age_years[:, step],
                condition,
            )
        )
        current = model.transport(
            current,
            age_norm[:, step - 1],
            age_norm[:, step],
            age_years[:, step - 1],
            age_years[:, step],
            condition,
        )
        composed_steps.append(current)
    direct = torch.stack(direct_steps, dim=1)
    composed = torch.stack(composed_steps, dim=1)
    target = latents[:, 1:, :]
    rollout_loss = 0.5 * (
        F.smooth_l1_loss(direct, target) + F.smooth_l1_loss(composed, target)
    )
    cocycle_loss = F.smooth_l1_loss(composed, direct.detach())
    return rollout_loss, cocycle_loss


def train_sequence_epoch(
    *,
    model: RegisteredPCAFlowMap,
    ema_model: RegisteredPCAFlowMap,
    loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    if not bool(args.include_sequence_loss):
        return {
            "sequence_total_loss": 0.0,
            "sequence_rollout_loss": 0.0,
            "sequence_cocycle_loss": 0.0,
            "sequence_rows": 0.0,
        }
    model.train()
    total_rows = 0
    total_loss = 0.0
    total_rollout = 0.0
    total_cocycle = 0.0
    max_rows = int(args.max_sequences_per_epoch)
    for loader in loaders:
        for raw_batch in loader:
            latents = raw_batch["latents"].to(device)
            age_norm = raw_batch["age_norm"].to(device)
            age_years = raw_batch["age_years"].to(device)
            condition = raw_batch["condition"].to(device)
            rollout_loss, cocycle_loss = sequence_loss_for_order(
                model=model,
                latents=latents,
                age_norm=age_norm,
                age_years=age_years,
                condition=condition,
            )
            loss = (
                float(args.sequence_rollout_weight) * rollout_loss
                + float(args.sequence_cocycle_weight) * cocycle_loss
            )
            if bool(args.include_backward_pairs) and float(args.reverse_sequence_weight) > 0.0:
                reverse_rollout, reverse_cocycle = sequence_loss_for_order(
                    model=model,
                    latents=torch.flip(latents, dims=[1]),
                    age_norm=torch.flip(age_norm, dims=[1]),
                    age_years=torch.flip(age_years, dims=[1]),
                    condition=condition,
                )
                loss = loss + float(args.reverse_sequence_weight) * (
                    float(args.sequence_rollout_weight) * reverse_rollout
                    + float(args.sequence_cocycle_weight) * reverse_cocycle
                )
                rollout_loss = 0.5 * (rollout_loss + reverse_rollout)
                cocycle_loss = 0.5 * (cocycle_loss + reverse_cocycle)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.gradient_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
            optimizer.step()
            update_ema_model(ema_model=ema_model, model=model, decay=float(args.ema_decay))

            batch_rows = int(latents.shape[0])
            total_rows += batch_rows
            total_loss += float(loss.detach().cpu().item()) * batch_rows
            total_rollout += float(rollout_loss.detach().cpu().item()) * batch_rows
            total_cocycle += float(cocycle_loss.detach().cpu().item()) * batch_rows
            if max_rows > 0 and total_rows >= max_rows:
                return {
                    "sequence_total_loss": total_loss / max(total_rows, 1),
                    "sequence_rollout_loss": total_rollout / max(total_rows, 1),
                    "sequence_cocycle_loss": total_cocycle / max(total_rows, 1),
                    "sequence_rows": float(total_rows),
                }
    return {
        "sequence_total_loss": total_loss / max(total_rows, 1),
        "sequence_rollout_loss": total_rollout / max(total_rows, 1),
        "sequence_cocycle_loss": total_cocycle / max(total_rows, 1),
        "sequence_rows": float(total_rows),
    }


@torch.no_grad()
def evaluate_records(
    *,
    model: RegisteredPCAFlowMap,
    dataset: PairDataset,
    batch_size: int,
    device: torch.device,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    faces: torch.Tensor,
    template_normals: torch.Tensor,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    vertex_errors: list[float] = []
    no_change_vertex_errors: list[float] = []
    pca_errors: list[float] = []
    volume_rate_abs_errors: list[float] = []
    ad_volume_rate_abs_errors: list[float] = []
    cn_volume_rate_abs_errors: list[float] = []
    local_normal_maes: list[float] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        source = batch["source_latent"]
        target = batch["target_latent"]
        delta_years = batch["delta_years"].view(-1, 1)
        signed_delta_years = torch.where(
            torch.abs(delta_years) < 1.0e-6,
            torch.full_like(delta_years, 1.0e-6),
            delta_years,
        )
        pred = model.transport(
            source,
            batch["source_age_norm"],
            batch["target_age_norm"],
            batch["source_age_years"],
            batch["target_age_years"],
            batch["condition"],
        )
        source_vertices = decode_pca_torch(source, mean_flat, components)
        target_vertices = decode_pca_torch(target, mean_flat, components)
        pred_vertices = decode_pca_torch(pred, mean_flat, components)
        vertex_error = torch.linalg.norm(pred_vertices - target_vertices, dim=2).mean(dim=1)
        no_change_vertex_error = torch.linalg.norm(source_vertices - target_vertices, dim=2).mean(dim=1)
        pca_error = torch.mean((pred - target) ** 2, dim=1)

        source_volume = mesh_volume_torch(source_vertices, faces)
        target_volume = mesh_volume_torch(target_vertices, faces)
        pred_volume = mesh_volume_torch(pred_vertices, faces)
        true_log_rate = (torch.log(target_volume) - torch.log(source_volume)) / signed_delta_years.view(-1)
        pred_log_rate = (torch.log(pred_volume) - torch.log(source_volume)) / signed_delta_years.view(-1)
        volume_abs = torch.abs(pred_log_rate - true_log_rate)

        normal_true = torch.sum(
            ((target_vertices - source_vertices) / signed_delta_years.view(-1, 1, 1)) * template_normals.view(1, -1, 3),
            dim=2,
        )
        normal_pred = torch.sum(
            ((pred_vertices - source_vertices) / signed_delta_years.view(-1, 1, 1)) * template_normals.view(1, -1, 3),
            dim=2,
        )
        local_mae = torch.mean(torch.abs(normal_pred - normal_true), dim=1)

        vertex_errors.extend(float(value) for value in vertex_error.detach().cpu().tolist())
        no_change_vertex_errors.extend(
            float(value) for value in no_change_vertex_error.detach().cpu().tolist()
        )
        pca_errors.extend(float(value) for value in pca_error.detach().cpu().tolist())
        volume_rate_abs_errors.extend(float(value) for value in volume_abs.detach().cpu().tolist())
        label_ad = batch["label_ad"].detach().cpu().numpy()
        volume_np = volume_abs.detach().cpu().numpy()
        ad_volume_rate_abs_errors.extend(float(value) for value in volume_np[label_ad == 1])
        cn_volume_rate_abs_errors.extend(float(value) for value in volume_np[label_ad == 0])
        local_normal_maes.extend(float(value) for value in local_mae.detach().cpu().tolist())

    vertex_mean = finite_mean(vertex_errors)
    no_change_vertex = finite_mean(no_change_vertex_errors)
    ad_volume = finite_mean(ad_volume_rate_abs_errors)
    local_mean = finite_mean(local_normal_maes)
    return {
        "rows": float(len(vertex_errors)),
        "vertex_euclidean_mean": vertex_mean,
        "no_change_vertex_euclidean_mean": no_change_vertex,
        "vertex_euclidean_improvement": no_change_vertex - vertex_mean,
        "pca_mse_mean": finite_mean(pca_errors),
        "log_volume_rate_mae": finite_mean(volume_rate_abs_errors),
        "ad_log_volume_rate_mae": ad_volume,
        "cn_log_volume_rate_mae": finite_mean(cn_volume_rate_abs_errors),
        "local_normal_rate_mae": local_mean,
    }


def save_checkpoint(
    *,
    path: Path,
    model: RegisteredPCAFlowMap,
    ema_model: RegisteredPCAFlowMap,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    basis_path: Path,
    rate_targets_path: Path,
    local_weights_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "basis_path": str(basis_path),
        "rate_targets_path": str(rate_targets_path),
        "local_weights_path": str(local_weights_path),
        "model_config": {
            "latent_dim": int(args.components),
            "dynamic_dim": int(args.dynamic_dim),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "activation": str(args.activation),
            "dropout": float(args.dropout),
            "latent_condition_dim": int(args.latent_condition_dim),
            "residual_scale": float(args.residual_scale),
        },
        "dynamic_basis": model.dynamic_basis.detach().cpu(),
        "coefficient_mean": model.coefficient_mean.detach().cpu().view(-1),
        "coefficient_std": model.coefficient_std.detach().cpu().view(-1),
        "cn_velocity_mean": model.cn_velocity_mean.detach().cpu().view(-1),
        "ad_velocity_mean": model.ad_velocity_mean.detach().cpu().view(-1),
    }
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    if int(args.components) <= 0 or int(args.components) > 256:
        raise ValueError("--components must be in [1, 256].")
    set_random_seed(int(args.seed))
    device = resolve_device(str(args.device))
    run_dir = run_dir_for(args)
    checkpoint_dir = run_dir / "checkpoints"
    status_path = run_dir / "training_status.json"
    history_path = run_dir / "history.json"
    history_jsonl_path = run_dir / "history.jsonl"
    start_wall = time.time()
    write_json(
        status_path,
        {
            "status": "running",
            "run_name": str(args.run_name),
            "started_at_unix": start_wall,
            "device": str(device),
        },
    )

    try:
        _, pca_model_dir, mean_flat_np, components_np, faces_np = load_pca_model(
            args.config,
            int(args.components),
        )
        train_archive = load_split_archive("train")
        val_archive = load_split_archive("val")
        train_records = build_pair_records(
            train_archive,
            max_gap_years=float(args.train_max_gap_years),
            pair_type=str(args.train_pair_type),
            include_backward=bool(args.include_backward_pairs),
        )
        if int(args.max_train_pairs) > 0:
            train_records = train_records[: int(args.max_train_pairs)]
        val_records = build_pair_records(val_archive, include_backward=False)
        if int(args.max_val_pairs) > 0:
            val_records = val_records[: int(args.max_val_pairs)]
        if not train_records or not val_records:
            raise RuntimeError("Training and validation records must be non-empty.")
        sequence_starts = build_sequence_starts(train_archive) if bool(args.include_sequence_loss) else []
        if int(args.max_sequences_per_epoch) > 0 and sequence_starts:
            random.Random(int(args.seed)).shuffle(sequence_starts)
        sequence_loaders = grouped_sequence_loaders(
            train_archive,
            sequence_starts,
            int(args.components),
            int(args.sequence_batch_size),
            int(args.num_workers),
        )

        basis_path = basis_path_for(args)
        if not basis_path.is_file():
            raise FileNotFoundError(
                f"Missing dynamic basis: {basis_path}. Run build_dynamic_pca_basis.py first."
            )
        basis_archive = np.load(basis_path, allow_pickle=False)
        dynamic_basis = torch.from_numpy(basis_archive["basis"].astype(np.float32))
        cn_velocity_mean = torch.from_numpy(basis_archive["cn_velocity_mean"].astype(np.float32))
        ad_velocity_mean = torch.from_numpy(basis_archive["ad_velocity_mean"].astype(np.float32))
        coefficient_mean_np, coefficient_std_np = coefficient_stats(train_archive, int(args.components))
        coefficient_mean = torch.from_numpy(coefficient_mean_np)
        coefficient_std = torch.from_numpy(coefficient_std_np)
        rate_targets_path = rate_targets_path_for(args)
        rate_targets = load_rate_targets(rate_targets_path)
        local_weights_path = local_weights_path_for(args)
        if local_weights_path.is_file():
            local_weights_np = np.load(local_weights_path).astype(np.float32)
        else:
            local_weights_np = np.ones(mean_flat_np.shape[0] // 3, dtype=np.float32)
        local_weights_np = np.maximum(local_weights_np, 0.05).astype(np.float32)

        mean_flat = torch.from_numpy(mean_flat_np).to(device)
        components = torch.from_numpy(components_np).to(device)
        faces = torch.from_numpy(faces_np.astype(np.int64)).to(device)
        template_normals = torch.from_numpy(vertex_normals_np(mean_flat_np.reshape(-1, 3), faces_np)).to(device)
        local_weights = torch.from_numpy(local_weights_np).to(device)
        coefficient_mean_device = coefficient_mean.to(device).view(1, -1)
        coefficient_std_device = coefficient_std.to(device).view(1, -1)

        model = RegisteredPCAFlowMap(
            latent_dim=int(args.components),
            dynamic_basis=dynamic_basis,
            coefficient_mean=coefficient_mean,
            coefficient_std=coefficient_std,
            cn_velocity_mean=cn_velocity_mean,
            ad_velocity_mean=ad_velocity_mean,
            hidden_dims=[int(value) for value in args.hidden_dims],
            activation=str(args.activation),
            dropout=float(args.dropout),
            latent_condition_dim=int(args.latent_condition_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
        ema_model = copy.deepcopy(model).to(device)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

        resolved = {
            "args": vars(args),
            "run_dir": str(run_dir),
            "task_dir": str(TASK_DIR),
            "pca_model_dir": str(pca_model_dir),
            "basis_path": str(basis_path),
            "rate_targets_path": str(rate_targets_path),
            "local_weights_path": str(local_weights_path),
            "record_summary": {
                "train_base": summarize_pair_records(train_records),
                "val": summarize_pair_records(val_records),
                "train_sequence_count": len(sequence_starts),
            },
        }
        write_json(run_dir / "resolved_config.json", resolved)
        write_json(run_dir / "record_summary.json", resolved["record_summary"])

        history: list[dict[str, Any]] = []
        best_vertex = float("inf")
        best_ad_volume = float("inf")
        best_pareto = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, int(args.epochs) + 1):
            epoch_start = time.time()
            epoch_records = make_balanced_epoch_records(
                train_records,
                samples_per_epoch=int(args.samples_per_epoch),
                seed=int(args.seed),
                epoch=epoch,
                balanced=bool(args.balanced_sampling),
            )
            train_dataset = PairDataset(train_archive, epoch_records, int(args.components))
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
            )
            train_metrics = train_one_epoch(
                model=model,
                ema_model=ema_model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                args=args,
                mean_flat=mean_flat,
                components=components,
                faces=faces,
                template_normals=template_normals,
                local_weights=local_weights,
                rate_targets=rate_targets,
                coefficient_mean=coefficient_mean_device,
                coefficient_std=coefficient_std_device,
                epoch=epoch,
            )
            sequence_metrics = train_sequence_epoch(
                model=model,
                ema_model=ema_model,
                loaders=sequence_loaders,
                optimizer=optimizer,
                device=device,
                args=args,
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - epoch_start,
                "train_epoch_records": len(epoch_records),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"train_{key}": value for key, value in sequence_metrics.items()},
            }

            should_validate = (
                epoch == 1
                or epoch % int(args.validation_frequency) == 0
                or epoch == int(args.epochs)
            )
            if should_validate:
                val_dataset = PairDataset(val_archive, val_records, int(args.components))
                val_metrics = evaluate_records(
                    model=model,
                    dataset=val_dataset,
                    batch_size=int(args.batch_size),
                    device=device,
                    mean_flat=mean_flat,
                    components=components,
                    faces=faces,
                    template_normals=template_normals,
                )
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                pareto = (
                    float(val_metrics["vertex_euclidean_mean"])
                    + float(args.pareto_volume_weight) * float(val_metrics["ad_log_volume_rate_mae"])
                    + float(args.pareto_local_weight) * float(val_metrics["local_normal_rate_mae"])
                )
                row["val_pareto_score"] = pareto

                improved = False
                if float(val_metrics["vertex_euclidean_mean"]) < best_vertex - float(
                    args.early_stopping_min_delta
                ):
                    best_vertex = float(val_metrics["vertex_euclidean_mean"])
                    improved = True
                    save_checkpoint(
                        path=checkpoint_dir / "best_vertex.pth",
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=row,
                        args=args,
                        basis_path=basis_path,
                        rate_targets_path=rate_targets_path,
                        local_weights_path=local_weights_path,
                    )
                if float(val_metrics["ad_log_volume_rate_mae"]) < best_ad_volume - float(
                    args.early_stopping_min_delta
                ):
                    best_ad_volume = float(val_metrics["ad_log_volume_rate_mae"])
                    improved = True
                    save_checkpoint(
                        path=checkpoint_dir / "best_ad_volume_rate.pth",
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=row,
                        args=args,
                        basis_path=basis_path,
                        rate_targets_path=rate_targets_path,
                        local_weights_path=local_weights_path,
                    )
                if pareto < best_pareto - float(args.early_stopping_min_delta):
                    best_pareto = pareto
                    improved = True
                    save_checkpoint(
                        path=checkpoint_dir / "best_pareto.pth",
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=row,
                        args=args,
                        basis_path=basis_path,
                        rate_targets_path=rate_targets_path,
                        local_weights_path=local_weights_path,
                    )
                epochs_without_improvement = 0 if improved else epochs_without_improvement + int(
                    args.validation_frequency
                )

            history.append(row)
            with history_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            write_json(history_path, history)
            if epoch % int(args.latest_frequency) == 0 or epoch == int(args.epochs):
                save_checkpoint(
                    path=checkpoint_dir / "latest.pth",
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    args=args,
                    basis_path=basis_path,
                    rate_targets_path=rate_targets_path,
                    local_weights_path=local_weights_path,
                )
            if epoch % int(args.snapshot_frequency) == 0:
                save_checkpoint(
                    path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    args=args,
                    basis_path=basis_path,
                    rate_targets_path=rate_targets_path,
                    local_weights_path=local_weights_path,
                )
            if epochs_without_improvement >= int(args.early_stopping_patience):
                row["early_stopped"] = True
                break

        write_json(
            status_path,
            {
                "status": "completed",
                "run_name": str(args.run_name),
                "run_dir": str(run_dir),
                "finished_at_unix": time.time(),
                "elapsed_seconds": time.time() - start_wall,
                "best_vertex": best_vertex,
                "best_ad_volume_rate": best_ad_volume,
                "best_pareto": best_pareto,
                "epochs": len(history),
            },
        )
        print(json.dumps({"run_dir": str(run_dir), "epochs": len(history)}, indent=2))
        return 0
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "run_name": str(args.run_name),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - start_wall,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
