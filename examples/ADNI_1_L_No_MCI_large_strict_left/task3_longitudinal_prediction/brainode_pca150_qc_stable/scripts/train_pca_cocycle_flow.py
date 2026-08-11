#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core_brainode_common import TASK_DIR, load_config, resolve_repo_path, write_json
from longitudinal_direct_flow import DirectAgeFlow


DEFAULT_RUN_NAME = "pca150_direct_cocycle_flow_qc_v1"


@dataclass(frozen=True)
class PairRecord:
    source_index: int
    target_index: int
    intermediate_index: int
    source_visit_order: int
    target_visit_order: int
    subject_id: str
    diagnosis: str
    pair_type: str


@dataclass(frozen=True)
class SequenceRecord:
    start_index: int
    end_index: int
    start_visit_order: int
    subject_id: str
    diagnosis: str


class PcaPairDataset(Dataset):
    def __init__(
        self,
        archive: dict[str, np.ndarray],
        records: list[PairRecord],
        components: int,
    ) -> None:
        self.archive = archive
        self.records = records
        self.components = int(components)
        self.latents = pca_latents(archive, self.components)
        self.times = archive["visit_continuous_age_norm"].astype(np.float32)
        self.conditions = archive["visit_cognition"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_index = int(record.source_index)
        target_index = int(record.target_index)
        intermediate_index = int(record.intermediate_index)
        has_intermediate = intermediate_index >= 0
        if has_intermediate:
            intermediate_latent = self.latents[intermediate_index]
            intermediate_time = self.times[intermediate_index]
        else:
            intermediate_latent = np.zeros(self.components, dtype=np.float32)
            intermediate_time = np.float32(0.0)
        return {
            "source_latent": torch.from_numpy(self.latents[source_index].copy()).float(),
            "target_latent": torch.from_numpy(self.latents[target_index].copy()).float(),
            "intermediate_latent": torch.from_numpy(intermediate_latent.copy()).float(),
            "source_time": torch.tensor(self.times[source_index], dtype=torch.float32),
            "target_time": torch.tensor(self.times[target_index], dtype=torch.float32),
            "intermediate_time": torch.tensor(intermediate_time, dtype=torch.float32),
            "condition": torch.tensor(self.conditions[source_index], dtype=torch.float32),
            "has_intermediate": torch.tensor(has_intermediate, dtype=torch.bool),
        }


class PcaSequenceDataset(Dataset):
    def __init__(
        self,
        archive: dict[str, np.ndarray],
        records: list[SequenceRecord],
        components: int,
    ) -> None:
        self.archive = archive
        self.records = records
        self.components = int(components)
        self.latents = pca_latents(archive, self.components)
        self.times = archive["visit_continuous_age_norm"].astype(np.float32)
        self.conditions = archive["visit_cognition"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        start = int(record.start_index)
        end = int(record.end_index)
        return {
            "latents": torch.from_numpy(self.latents[start:end].copy()).float(),
            "times": torch.from_numpy(self.times[start:end].copy()).float(),
            "condition": torch.tensor(self.conditions[start], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a conditional direct/cocycle flow in PCA coefficient space on "
            "the QC-stable ADNI left hippocampus longitudinal dataset."
        )
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-frequency", type=int, default=5)
    parser.add_argument("--snapshot-frequency", type=int, default=25)
    parser.add_argument("--latest-frequency", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-6)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--condition-dim", type=int, default=1)
    parser.add_argument("--include-delta-time-input", action="store_true", default=True)
    parser.add_argument("--no-delta-time-input", dest="include_delta_time_input", action="store_false")
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--observed-cocycle-weight", type=float, default=1.0)
    parser.add_argument("--virtual-cocycle-weight", type=float, default=0.5)
    parser.add_argument("--sequence-rollout-weight", type=float, default=0.5)
    parser.add_argument("--sequence-cocycle-weight", type=float, default=0.25)
    parser.add_argument("--direction-weight", type=float, default=0.1)
    parser.add_argument("--magnitude-weight", type=float, default=0.1)
    parser.add_argument("--latent-guard-weight", type=float, default=0.05)
    parser.add_argument("--speed-guard-weight", type=float, default=0.05)
    parser.add_argument("--virtual-ratio-min", type=float, default=0.1)
    parser.add_argument("--virtual-ratio-max", type=float, default=0.9)
    parser.add_argument("--latent-guard-threshold", type=float, default=3.0)
    parser.add_argument("--speed-guard-percentile", type=float, default=95.0)
    parser.add_argument("--use-gap-weighted-loss", action="store_true", default=True)
    parser.add_argument("--no-gap-weighted-loss", dest="use_gap_weighted_loss", action="store_false")
    parser.add_argument("--gap-weight-min", type=float, default=0.5)
    parser.add_argument("--gap-weight-max", type=float, default=2.0)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def resolve_device(value: str | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    return TASK_DIR.parent / str(args.run_name)


def pca_latents(archive: dict[str, np.ndarray], components: int) -> np.ndarray:
    key = f"visit_pca_{int(components)}"
    if key in archive:
        return archive[key].astype(np.float32)
    if int(components) <= 256 and "visit_pca_256" in archive:
        return archive["visit_pca_256"][:, : int(components)].astype(np.float32)
    raise KeyError(f"Archive does not contain PCA coefficients for {components} components.")


def coefficient_stats(
    archive: dict[str, np.ndarray],
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_key = f"train_coefficient_mean_{int(components)}"
    std_key = f"train_coefficient_std_{int(components)}"
    if mean_key in archive and std_key in archive:
        mean = archive[mean_key].astype(np.float32)
        std = archive[std_key].astype(np.float32)
    else:
        mean = archive["train_coefficient_mean_256"][: int(components)].astype(np.float32)
        std = archive["train_coefficient_std_256"][: int(components)].astype(np.float32)
    std = np.maximum(std, np.float32(1.0e-6))
    return mean, std


def build_pair_records(
    archive: dict[str, np.ndarray],
    *,
    max_pairs: int = 0,
) -> list[PairRecord]:
    offsets = archive["subject_visit_offsets"]
    records: list[PairRecord] = []
    for subject_index in range(len(offsets) - 1):
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        for source_index in range(start, end - 1):
            for target_index in range(source_index + 1, end):
                source_order = int(archive["visit_orders"][source_index])
                target_order = int(archive["visit_orders"][target_index])
                pair_type = "adjacent" if target_order - source_order == 1 else "nonadjacent"
                intermediate_index = -1
                if target_index - source_index > 1:
                    intermediate_index = source_index + (target_index - source_index) // 2
                records.append(
                    PairRecord(
                        source_index=source_index,
                        target_index=target_index,
                        intermediate_index=intermediate_index,
                        source_visit_order=source_order,
                        target_visit_order=target_order,
                        subject_id=str(archive["visit_subject_ids"][source_index]),
                        diagnosis=str(archive["visit_diagnoses"][source_index]),
                        pair_type=pair_type,
                    )
                )
                if max_pairs > 0 and len(records) >= int(max_pairs):
                    return records
    return records


def build_sequence_records(
    archive: dict[str, np.ndarray],
) -> list[SequenceRecord]:
    offsets = archive["subject_visit_offsets"]
    records: list[SequenceRecord] = []
    for subject_index in range(len(offsets) - 1):
        subject_start = int(offsets[subject_index])
        subject_end = int(offsets[subject_index + 1])
        for start_index in range(subject_start, subject_end - 1):
            records.append(
                SequenceRecord(
                    start_index=start_index,
                    end_index=subject_end,
                    start_visit_order=int(archive["visit_orders"][start_index]),
                    subject_id=str(archive["visit_subject_ids"][start_index]),
                    diagnosis=str(archive["visit_diagnoses"][start_index]),
                )
            )
    return records


def grouped_sequence_loaders(
    archive: dict[str, np.ndarray],
    records: list[SequenceRecord],
    components: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> list[tuple[int, DataLoader]]:
    groups: dict[int, list[SequenceRecord]] = {}
    for record in records:
        groups.setdefault(int(record.end_index - record.start_index), []).append(record)
    loaders: list[tuple[int, DataLoader]] = []
    for length in sorted(groups):
        dataset = PcaSequenceDataset(archive, groups[length], components)
        loaders.append(
            (
                length,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                ),
            )
        )
    return loaders


def make_flow(args: argparse.Namespace, latent_dim: int) -> DirectAgeFlow:
    return DirectAgeFlow(
        latent_size=int(latent_dim),
        hidden_dims=[int(value) for value in args.hidden_dims],
        condition_dim=int(args.condition_dim),
        activation=str(args.activation),
        dropout=float(args.dropout),
        zero_initialize_output=True,
        latent_condition_mode="full",
        latent_condition_dim=int(latent_dim),
        include_delta_time_input=bool(args.include_delta_time_input),
    )


def inverse_transform_pca(
    coefficients: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    flat = coefficients.reshape(-1, coefficients.shape[-1])
    reconstructed = flat @ components + mean_flat
    return reconstructed.reshape(*coefficients.shape[:-1], mean_flat.shape[0])


def per_row_latent_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.mean((left - right) ** 2, dim=1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return torch.sum(values * weights) / torch.clamp(weights.sum(), min=1.0e-8)


def gap_weights(
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    *,
    max_gap_norm: float,
    weight_min: float,
    weight_max: float,
) -> torch.Tensor:
    if max_gap_norm <= 0.0:
        return torch.ones_like(source_time)
    ratio = torch.clamp(torch.abs(target_time - source_time) / float(max_gap_norm), 0.0, 1.0)
    return float(weight_min) + ratio * (float(weight_max) - float(weight_min))


def zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def virtual_cocycle_loss(
    flow: DirectAgeFlow,
    source_latent: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    condition: torch.Tensor,
    direct_target: torch.Tensor,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    source_column = source_time.view(-1, 1)
    target_column = target_time.view(-1, 1)
    ratio = torch.rand_like(source_column)
    ratio = float(ratio_min) + ratio * (float(ratio_max) - float(ratio_min))
    intermediate_time = source_column + ratio * (target_column - source_column)
    intermediate_latent = flow.transport(source_latent, source_time, intermediate_time, condition)
    composed_target = flow.transport(intermediate_latent, intermediate_time, target_time, condition)
    return torch.mean((composed_target - direct_target) ** 2)


def observed_cocycle_loss(
    flow: DirectAgeFlow,
    batch: dict[str, torch.Tensor],
    direct_target: torch.Tensor,
) -> torch.Tensor:
    mask = batch["has_intermediate"]
    if not bool(mask.any().item()):
        return zero_loss(direct_target)
    source_latent = batch["source_latent"][mask]
    source_time = batch["source_time"][mask]
    target_time = batch["target_time"][mask]
    intermediate_time = batch["intermediate_time"][mask]
    condition = batch["condition"][mask]
    intermediate_latent = flow.transport(source_latent, source_time, intermediate_time, condition)
    composed_target = flow.transport(intermediate_latent, intermediate_time, target_time, condition)
    return torch.mean((composed_target - direct_target[mask]) ** 2)


def direction_loss(predicted_delta: torch.Tensor, true_delta: torch.Tensor) -> torch.Tensor:
    true_norm = torch.linalg.norm(true_delta, dim=1)
    valid = true_norm > 1.0e-8
    if not bool(valid.any().item()):
        return zero_loss(predicted_delta)
    cosine = F.cosine_similarity(predicted_delta[valid], true_delta[valid], dim=1, eps=1.0e-8)
    return torch.mean(1.0 - cosine)


def magnitude_loss(predicted_delta: torch.Tensor, true_delta: torch.Tensor) -> torch.Tensor:
    true_norm = torch.linalg.norm(true_delta, dim=1)
    predicted_norm = torch.linalg.norm(predicted_delta, dim=1)
    return torch.mean(torch.abs(predicted_norm - true_norm) / torch.clamp(true_norm, min=1.0e-6))


def latent_guard_loss(
    latent: torch.Tensor,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    z_score = torch.abs((latent - train_mean) / train_std)
    excess = torch.relu(z_score - float(threshold))
    return torch.mean(excess**2)


def speed_guard_loss(
    predicted_delta: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if threshold <= 0.0:
        return zero_loss(predicted_delta)
    speed = torch.linalg.norm(predicted_delta, dim=1) / torch.clamp(
        torch.abs(target_time - source_time),
        min=1.0e-6,
    )
    excess = torch.relu(speed - float(threshold))
    return torch.mean((excess / float(threshold)) ** 2)


def compute_speed_threshold(dataset: PcaPairDataset, percentile: float) -> float:
    speeds: list[float] = []
    latents = dataset.latents
    times = dataset.times
    for record in dataset.records:
        dt = abs(float(times[record.target_index] - times[record.source_index]))
        if dt <= 1.0e-8:
            continue
        delta = latents[record.target_index] - latents[record.source_index]
        speeds.append(float(np.linalg.norm(delta) / dt))
    if not speeds:
        return 0.0
    return float(np.percentile(np.asarray(speeds, dtype=np.float64), float(percentile)))


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_pair_epoch(
    *,
    flow: DirectAgeFlow,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    speed_threshold: float,
    max_gap_norm: float,
) -> dict[str, float]:
    flow.train()
    totals: dict[str, float] = {
        "pair_total_loss": 0.0,
        "pair_target_loss": 0.0,
        "pair_observed_cocycle_loss": 0.0,
        "pair_virtual_cocycle_loss": 0.0,
        "pair_direction_loss": 0.0,
        "pair_magnitude_loss": 0.0,
        "pair_latent_guard_loss": 0.0,
        "pair_speed_guard_loss": 0.0,
    }
    total_rows = 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        source_latent = batch["source_latent"]
        target_latent = batch["target_latent"]
        source_time = batch["source_time"]
        target_time = batch["target_time"]
        condition = batch["condition"]
        weights = None
        if bool(args.use_gap_weighted_loss):
            weights = gap_weights(
                source_time,
                target_time,
                max_gap_norm=max_gap_norm,
                weight_min=float(args.gap_weight_min),
                weight_max=float(args.gap_weight_max),
            )

        direct_target = flow.transport(source_latent, source_time, target_time, condition)
        target_loss = weighted_mean(per_row_latent_mse(direct_target, target_latent), weights)
        obs_loss = observed_cocycle_loss(flow, batch, direct_target)
        virt_loss = virtual_cocycle_loss(
            flow,
            source_latent,
            source_time,
            target_time,
            condition,
            direct_target,
            float(args.virtual_ratio_min),
            float(args.virtual_ratio_max),
        )
        predicted_delta = direct_target - source_latent
        true_delta = target_latent - source_latent
        dir_loss = direction_loss(predicted_delta, true_delta)
        mag_loss = magnitude_loss(predicted_delta, true_delta)
        guard_loss = latent_guard_loss(
            direct_target,
            train_mean,
            train_std,
            float(args.latent_guard_threshold),
        )
        speed_loss = speed_guard_loss(
            predicted_delta,
            source_time,
            target_time,
            float(speed_threshold),
        )

        loss = (
            float(args.target_loss_weight) * target_loss
            + float(args.observed_cocycle_weight) * obs_loss
            + float(args.virtual_cocycle_weight) * virt_loss
            + float(args.direction_weight) * dir_loss
            + float(args.magnitude_weight) * mag_loss
            + float(args.latent_guard_weight) * guard_loss
            + float(args.speed_guard_weight) * speed_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.gradient_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.gradient_clip_norm))
        optimizer.step()

        batch_size = int(source_latent.shape[0])
        total_rows += batch_size
        totals["pair_total_loss"] += float(loss.item()) * batch_size
        totals["pair_target_loss"] += float(target_loss.item()) * batch_size
        totals["pair_observed_cocycle_loss"] += float(obs_loss.item()) * batch_size
        totals["pair_virtual_cocycle_loss"] += float(virt_loss.item()) * batch_size
        totals["pair_direction_loss"] += float(dir_loss.item()) * batch_size
        totals["pair_magnitude_loss"] += float(mag_loss.item()) * batch_size
        totals["pair_latent_guard_loss"] += float(guard_loss.item()) * batch_size
        totals["pair_speed_guard_loss"] += float(speed_loss.item()) * batch_size

    return {key: value / max(total_rows, 1) for key, value in totals.items()}


def sequence_predictions(
    flow: DirectAgeFlow,
    latents: torch.Tensor,
    times: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_latent = latents[:, 0, :]
    source_time = times[:, 0]
    direct_steps = [source_latent]
    for step_index in range(1, times.shape[1]):
        direct_steps.append(
            flow.transport(source_latent, source_time, times[:, step_index], condition)
        )
    direct = torch.stack(direct_steps, dim=1)

    current = source_latent
    composed_steps = [source_latent]
    for step_index in range(times.shape[1] - 1):
        current = flow.transport(
            current,
            times[:, step_index],
            times[:, step_index + 1],
            condition,
        )
        composed_steps.append(current)
    composed = torch.stack(composed_steps, dim=1)
    return direct, composed


def train_sequence_epoch(
    *,
    flow: DirectAgeFlow,
    loaders: list[tuple[int, DataLoader]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    if float(args.sequence_rollout_weight) <= 0.0 and float(args.sequence_cocycle_weight) <= 0.0:
        return {
            "sequence_total_loss": 0.0,
            "sequence_rollout_loss": 0.0,
            "sequence_cocycle_loss": 0.0,
            "sequence_rows": 0.0,
        }

    flow.train()
    total_loss = 0.0
    total_rollout = 0.0
    total_cocycle = 0.0
    total_rows = 0
    for _, loader in loaders:
        for raw_batch in loader:
            latents = raw_batch["latents"].to(device)
            times = raw_batch["times"].to(device)
            condition = raw_batch["condition"].to(device)
            direct, composed = sequence_predictions(flow, latents, times, condition)
            rollout_loss = torch.mean((direct[:, 1:, :] - latents[:, 1:, :]) ** 2)
            cocycle_loss = torch.mean((composed[:, 1:, :] - direct[:, 1:, :]) ** 2)
            loss = (
                float(args.sequence_rollout_weight) * rollout_loss
                + float(args.sequence_cocycle_weight) * cocycle_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.gradient_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(flow.parameters(), float(args.gradient_clip_norm))
            optimizer.step()

            batch_size = int(latents.shape[0])
            total_rows += batch_size
            total_loss += float(loss.item()) * batch_size
            total_rollout += float(rollout_loss.item()) * batch_size
            total_cocycle += float(cocycle_loss.item()) * batch_size

    return {
        "sequence_total_loss": total_loss / max(total_rows, 1),
        "sequence_rollout_loss": total_rollout / max(total_rows, 1),
        "sequence_cocycle_loss": total_cocycle / max(total_rows, 1),
        "sequence_rows": float(total_rows),
    }


@torch.no_grad()
def evaluate_pair_dataset(
    *,
    flow: DirectAgeFlow,
    dataset: PcaPairDataset,
    batch_size: int,
    device: torch.device,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    flow.eval()
    pca_losses: list[float] = []
    no_change_pca_losses: list[float] = []
    vertex_maes: list[float] = []
    no_change_vertex_maes: list[float] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        source = batch["source_latent"]
        target = batch["target_latent"]
        prediction = flow.transport(
            source,
            batch["source_time"],
            batch["target_time"],
            batch["condition"],
        )
        pca_loss = per_row_latent_mse(prediction, target)
        no_change_pca = per_row_latent_mse(source, target)
        pred_vertices = inverse_transform_pca(prediction, mean_flat, components)
        source_vertices = inverse_transform_pca(source, mean_flat, components)
        target_vertices = inverse_transform_pca(target, mean_flat, components)
        vertex_mae = torch.mean(torch.abs(pred_vertices - target_vertices), dim=1)
        no_change_vertex_mae = torch.mean(torch.abs(source_vertices - target_vertices), dim=1)
        pca_losses.extend(float(value) for value in pca_loss.detach().cpu().tolist())
        no_change_pca_losses.extend(float(value) for value in no_change_pca.detach().cpu().tolist())
        vertex_maes.extend(float(value) for value in vertex_mae.detach().cpu().tolist())
        no_change_vertex_maes.extend(
            float(value) for value in no_change_vertex_mae.detach().cpu().tolist()
        )

    pca_array = np.asarray(pca_losses, dtype=np.float64)
    no_change_pca_array = np.asarray(no_change_pca_losses, dtype=np.float64)
    vertex_array = np.asarray(vertex_maes, dtype=np.float64)
    no_change_vertex_array = np.asarray(no_change_vertex_maes, dtype=np.float64)
    return {
        "rows": float(len(pca_losses)),
        "endpoint_pca_mse": float(pca_array.mean()) if pca_array.size else float("nan"),
        "no_change_pca_mse": float(no_change_pca_array.mean()) if no_change_pca_array.size else float("nan"),
        "endpoint_pca_improvement": float((no_change_pca_array - pca_array).mean())
        if pca_array.size
        else float("nan"),
        "endpoint_vertex_mae": float(vertex_array.mean()) if vertex_array.size else float("nan"),
        "no_change_vertex_mae": float(no_change_vertex_array.mean())
        if no_change_vertex_array.size
        else float("nan"),
        "endpoint_vertex_mae_improvement": float((no_change_vertex_array - vertex_array).mean())
        if vertex_array.size
        else float("nan"),
        "beats_no_change_fraction": float(np.mean(vertex_array < no_change_vertex_array))
        if vertex_array.size
        else float("nan"),
    }


def summarize_records(
    pair_records: list[PairRecord],
    sequence_records: list[SequenceRecord],
) -> dict[str, Any]:
    def count_by(values: Iterable[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "pair_count": len(pair_records),
        "pair_count_by_diagnosis": count_by(record.diagnosis for record in pair_records),
        "pair_count_by_pair_type": count_by(record.pair_type for record in pair_records),
        "sequence_count": len(sequence_records),
        "sequence_count_by_diagnosis": count_by(record.diagnosis for record in sequence_records),
    }


def save_checkpoint(
    *,
    path: Path,
    flow: DirectAgeFlow,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    speed_threshold: float,
    max_gap_norm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": dict(metrics),
        "components": int(args.components),
        "speed_threshold_norm_units": float(speed_threshold),
        "max_gap_norm": float(max_gap_norm),
        "flow_config": {
            "latent_size": int(args.components),
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "condition_dim": int(args.condition_dim),
            "activation": str(args.activation),
            "dropout": float(args.dropout),
            "include_delta_time_input": bool(args.include_delta_time_input),
            "latent_condition_mode": "full",
            "latent_condition_dim": int(args.components),
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_resume(
    *,
    path: Path,
    flow: DirectAgeFlow,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, float]:
    if not path.is_file():
        return 0, float("inf"), -float("inf")
    payload = torch.load(path, map_location=device)
    flow.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    metrics = dict(payload.get("metrics", {}))
    best_vertex = float(metrics.get("best_val_endpoint_vertex_mae", float("inf")))
    best_improvement = float(metrics.get("best_val_endpoint_vertex_mae_improvement", -float("inf")))
    return int(payload.get("epoch", 0)), best_vertex, best_improvement


def main() -> int:
    args = parse_args()
    if int(args.components) <= 0 or int(args.components) > 256:
        raise ValueError("--components must be in [1, 256].")
    if not 0.0 < float(args.virtual_ratio_min) < float(args.virtual_ratio_max) < 1.0:
        raise ValueError("Virtual ratios must satisfy 0 < min < max < 1.")

    set_random_seed(int(args.seed))
    config = load_config(args.config)
    device = resolve_device(args.device)
    run_dir = output_dir_for(args)
    checkpoint_dir = run_dir / "checkpoints"
    analysis_dir = run_dir / "analysis"
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
        train_archive = load_npz(TASK_DIR / "dataset" / "train_subject_sequences.npz")
        val_archive = load_npz(TASK_DIR / "dataset" / "val_subject_sequences.npz")
        train_pair_records = build_pair_records(train_archive, max_pairs=int(args.max_train_pairs))
        val_pair_records = build_pair_records(val_archive, max_pairs=int(args.max_val_pairs))
        train_sequence_records = build_sequence_records(train_archive)
        val_sequence_records = build_sequence_records(val_archive)
        train_dataset = PcaPairDataset(train_archive, train_pair_records, int(args.components))
        val_dataset = PcaPairDataset(val_archive, val_pair_records, int(args.components))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
        )
        sequence_loaders = grouped_sequence_loaders(
            train_archive,
            train_sequence_records,
            int(args.components),
            int(args.sequence_batch_size),
            True,
            int(args.num_workers),
        )

        pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
        mean_flat = torch.from_numpy(np.load(pca_model_dir / "mean.npy").astype(np.float32)).to(device)
        components = torch.from_numpy(
            np.load(pca_model_dir / "components_256.npy").astype(np.float32)[: int(args.components)]
        ).to(device)
        coefficient_mean, coefficient_std = coefficient_stats(train_archive, int(args.components))
        train_mean = torch.from_numpy(coefficient_mean).to(device).view(1, -1)
        train_std = torch.from_numpy(coefficient_std).to(device).view(1, -1)
        speed_threshold = compute_speed_threshold(
            train_dataset,
            percentile=float(args.speed_guard_percentile),
        )
        max_gap_norm = max(
            (
                abs(
                    float(train_dataset.times[record.target_index])
                    - float(train_dataset.times[record.source_index])
                )
                for record in train_dataset.records
            ),
            default=1.0,
        )

        flow = make_flow(args, int(args.components)).to(device)
        optimizer = torch.optim.AdamW(
            flow.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        start_epoch = 0
        best_val_vertex = float("inf")
        best_val_improvement = -float("inf")
        if bool(args.resume):
            start_epoch, best_val_vertex, best_val_improvement = load_resume(
                path=checkpoint_dir / "latest.pth",
                flow=flow,
                optimizer=optimizer,
                device=device,
            )

        resolved_config = {
            "config": str(Path(args.config).resolve()),
            "run_dir": str(run_dir),
            "task_dir": str(TASK_DIR),
            "pca_model_dir": str(pca_model_dir),
            "args": vars(args),
            "speed_threshold_norm_units": speed_threshold,
            "max_gap_norm": max_gap_norm,
            "record_summary": {
                "train": summarize_records(train_pair_records, train_sequence_records),
                "val": summarize_records(val_pair_records, val_sequence_records),
            },
        }
        write_json(run_dir / "resolved_config.json", resolved_config)
        write_json(run_dir / "record_summary.json", resolved_config["record_summary"])

        history: list[dict[str, Any]] = []
        epochs_without_improvement = 0
        for epoch in range(start_epoch + 1, int(args.epochs) + 1):
            epoch_start = time.time()
            pair_metrics = train_pair_epoch(
                flow=flow,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                args=args,
                train_mean=train_mean,
                train_std=train_std,
                speed_threshold=speed_threshold,
                max_gap_norm=max_gap_norm,
            )
            sequence_metrics = train_sequence_epoch(
                flow=flow,
                loaders=sequence_loaders,
                optimizer=optimizer,
                device=device,
                args=args,
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - epoch_start,
                **pair_metrics,
                **sequence_metrics,
            }

            should_validate = epoch == 1 or epoch % int(args.validation_frequency) == 0 or epoch == int(args.epochs)
            if should_validate:
                train_eval = evaluate_pair_dataset(
                    flow=flow,
                    dataset=train_dataset,
                    batch_size=int(args.batch_size),
                    device=device,
                    mean_flat=mean_flat,
                    components=components,
                )
                val_eval = evaluate_pair_dataset(
                    flow=flow,
                    dataset=val_dataset,
                    batch_size=int(args.batch_size),
                    device=device,
                    mean_flat=mean_flat,
                    components=components,
                )
                row.update({f"train_{key}": value for key, value in train_eval.items()})
                row.update({f"val_{key}": value for key, value in val_eval.items()})

                val_vertex = float(val_eval["endpoint_vertex_mae"])
                val_improvement = float(val_eval["endpoint_vertex_mae_improvement"])
                metrics_best_improvement = max(best_val_improvement, val_improvement)
                improved = val_vertex < best_val_vertex - float(args.early_stopping_min_delta)
                if improved:
                    best_val_vertex = val_vertex
                    epochs_without_improvement = 0
                    save_checkpoint(
                        path=checkpoint_dir / "best_val_endpoint_vertex_mae.pth",
                        flow=flow,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={
                            **row,
                            "best_val_endpoint_vertex_mae": best_val_vertex,
                            "best_val_endpoint_vertex_mae_improvement": metrics_best_improvement,
                        },
                        args=args,
                        speed_threshold=speed_threshold,
                        max_gap_norm=max_gap_norm,
                    )
                else:
                    epochs_without_improvement += int(args.validation_frequency)

                if val_improvement > best_val_improvement:
                    best_val_improvement = val_improvement
                    save_checkpoint(
                        path=checkpoint_dir / "best_val_no_change_improvement.pth",
                        flow=flow,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={
                            **row,
                            "best_val_endpoint_vertex_mae": best_val_vertex,
                            "best_val_endpoint_vertex_mae_improvement": best_val_improvement,
                        },
                        args=args,
                        speed_threshold=speed_threshold,
                        max_gap_norm=max_gap_norm,
                    )

            if epoch % int(args.latest_frequency) == 0 or epoch == int(args.epochs):
                save_checkpoint(
                    path=checkpoint_dir / "latest.pth",
                    flow=flow,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        **row,
                        "best_val_endpoint_vertex_mae": best_val_vertex,
                        "best_val_endpoint_vertex_mae_improvement": best_val_improvement,
                    },
                    args=args,
                    speed_threshold=speed_threshold,
                    max_gap_norm=max_gap_norm,
                )
            if epoch % int(args.snapshot_frequency) == 0:
                save_checkpoint(
                    path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
                    flow=flow,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        **row,
                        "best_val_endpoint_vertex_mae": best_val_vertex,
                        "best_val_endpoint_vertex_mae_improvement": best_val_improvement,
                    },
                    args=args,
                    speed_threshold=speed_threshold,
                    max_gap_norm=max_gap_norm,
                )

            history.append(row)
            append_jsonl(history_jsonl_path, row)
            write_json(history_path, history)
            print(json.dumps(row, sort_keys=True))

            if (
                int(args.early_stopping_patience) > 0
                and epochs_without_improvement >= int(args.early_stopping_patience)
            ):
                break

        write_json(
            status_path,
            {
                "status": "completed",
                "run_name": str(args.run_name),
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.time() - start_wall,
                "best_val_endpoint_vertex_mae": best_val_vertex,
                "best_val_endpoint_vertex_mae_improvement": best_val_improvement,
                "history_path": str(history_path),
                "analysis_dir": str(analysis_dir),
            },
        )
        return 0
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "run_name": str(args.run_name),
                "failed_at_unix": time.time(),
                "elapsed_seconds": time.time() - start_wall,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
