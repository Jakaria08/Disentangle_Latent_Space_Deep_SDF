#!/usr/bin/env python3
"""Train the previous attention BrainODE on the current strict SynthSeg cohort.

The ODE architecture and RK4 integrator are imported unchanged from the prior
PCA-150 BrainODE experiment.  This adaptation changes only the data contract:
each structure uses its own strict CN/AD subject splits, fixed PCA-150 basis,
standardized PCA scores, and fixed stable diagnosis condition.

Training remains shape-only for a faithful BrainODE baseline.  Forward and
backward suffix trajectories are used for training; the primary checkpoint is
selected exclusively by validation first-to-last decoded vertex MAE.  Test
data are never loaded by this trainer.  ``--dry-run`` validates one forward /
backward gradient batch and validation integration without writing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_adni_synthseg_pca_cocycle_v4 import (
    BASE_ROOT,
    STRUCTURES,
    atomic_json,
    atomic_torch_save,
    choose_device,
    load_archive,
    read_json,
    set_seed,
    validate_pca_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_BRAINODE_MODEL = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI_large_strict_left"
    / "task3_longitudinal_prediction"
    / "brainode_pca150_qc_stable"
    / "scripts"
    / "brainode_model.py"
)
if str(PRIOR_BRAINODE_MODEL.parent) not in sys.path:
    sys.path.insert(0, str(PRIOR_BRAINODE_MODEL.parent))
from brainode_model import ODEFuncWithAttention, integrate_sequence_rk4  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_config_path(structure: str) -> Path:
    return (
        BASE_ROOT
        / f"{structure}_pca_cocycle_v4"
        / "brainode"
        / "configs"
        / "brainode_attention_primary.json"
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_name(value: str) -> None:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError("Run name must be one directory-name component")


@dataclass(frozen=True)
class TrajectoryRecord:
    subject_id: str
    diagnosis: str
    direction: str
    start_visit_order: int
    times: np.ndarray
    targets: np.ndarray
    condition: float

    @property
    def length(self) -> int:
        return int(len(self.times))


class TrajectoryDataset(Dataset[TrajectoryRecord]):
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrajectoryRecord:
        return self.records[index]


def collate_records(records: list[TrajectoryRecord]) -> dict[str, Any]:
    return {
        "subject_id": [record.subject_id for record in records],
        "diagnosis": [record.diagnosis for record in records],
        "direction": [record.direction for record in records],
        "start_visit_order": torch.tensor([record.start_visit_order for record in records], dtype=torch.long),
        "times": torch.from_numpy(np.stack([record.times for record in records])).float(),
        "targets": torch.from_numpy(np.stack([record.targets for record in records])).float(),
        "condition": torch.tensor([record.condition for record in records], dtype=torch.float32),
    }


def build_records(
    archive: dict[str, np.ndarray],
    *,
    include_backward: bool,
    only_start_visit_zero: bool,
) -> list[TrajectoryRecord]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    records: list[TrajectoryRecord] = []
    for subject_index in range(len(offsets) - 1):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        times = archive["visit_age_norm_train"][start:end].astype(np.float32)
        targets = archive["visit_pca_standardized_150"][start:end].astype(np.float32)
        diagnoses = archive["visit_diagnoses"][start:end].astype(str)
        labels = archive["visit_label_ad"][start:end].astype(np.float32)
        if len(set(diagnoses)) != 1 or len(set(labels.tolist())) != 1:
            raise ValueError(f"Non-stable diagnosis sequence at subject index {subject_index}")
        subject_id = str(archive["subject_ids"][subject_index])
        diagnosis = str(diagnoses[0])
        condition = float(labels[0])
        for visit in range(len(times)):
            if only_start_visit_zero and visit != 0:
                continue
            forward_times = times[visit:].copy()
            forward_targets = targets[visit:].copy()
            if len(forward_times) > 1:
                records.append(TrajectoryRecord(
                    subject_id, diagnosis, "forward", visit,
                    forward_times, forward_targets, condition,
                ))
            if include_backward:
                backward_times = times[: visit + 1][::-1].copy()
                backward_targets = targets[: visit + 1][::-1].copy()
                if len(backward_times) > 1:
                    records.append(TrajectoryRecord(
                        subject_id, diagnosis, "backward", visit,
                        backward_times, backward_targets, condition,
                    ))
    if not records:
        raise ValueError("No eligible BrainODE trajectories")
    return records


def record_summary(records: list[TrajectoryRecord]) -> dict[str, Any]:
    by_length: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    by_diagnosis: dict[str, int] = {}
    for record in records:
        by_length[str(record.length)] = by_length.get(str(record.length), 0) + 1
        by_direction[record.direction] = by_direction.get(record.direction, 0) + 1
        by_diagnosis[record.diagnosis] = by_diagnosis.get(record.diagnosis, 0) + 1
    return {
        "trajectories": len(records),
        "subjects": len({record.subject_id for record in records}),
        "by_length": dict(sorted(by_length.items(), key=lambda item: int(item[0]))),
        "by_direction": dict(sorted(by_direction.items())),
        "by_diagnosis": dict(sorted(by_diagnosis.items())),
    }


def build_loaders(
    records: list[TrajectoryRecord],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> list[tuple[int, DataLoader]]:
    grouped: dict[int, list[TrajectoryRecord]] = {}
    for record in records:
        grouped.setdefault(record.length, []).append(record)
    output = []
    for length, current in sorted(grouped.items()):
        generator = torch.Generator().manual_seed(seed + length * 1009)
        output.append((
            length,
            DataLoader(
                TrajectoryDataset(current),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_records,
                generator=generator,
            ),
        ))
    return output


class PcaDecoder(nn.Module):
    def __init__(
        self,
        pca_model: dict[str, np.ndarray],
        score_mean: np.ndarray,
        score_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("mean_flat", torch.from_numpy(pca_model["mean"].astype(np.float32)).view(1, -1))
        self.register_buffer("components", torch.from_numpy(pca_model["components"].astype(np.float32)))
        self.register_buffer("score_mean", torch.from_numpy(score_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("score_std", torch.from_numpy(np.maximum(score_std, 1.0e-6).astype(np.float32)).view(1, -1))
        self.register_buffer("faces", torch.from_numpy(pca_model["faces"].astype(np.int64)))

    def vertices(self, standardized: torch.Tensor) -> torch.Tensor:
        original = standardized.shape[:-1]
        flat = standardized.reshape(-1, standardized.shape[-1])
        raw = flat * self.score_std + self.score_mean
        decoded = raw @ self.components + self.mean_flat
        return decoded.reshape(*original, -1, 3)

    def volume(self, vertices: torch.Tensor) -> torch.Tensor:
        original = vertices.shape[:-2]
        flat = vertices.reshape(-1, vertices.shape[-2], 3)
        v0 = flat[:, self.faces[:, 0], :]
        v1 = flat[:, self.faces[:, 1], :]
        v2 = flat[:, self.faces[:, 2], :]
        signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
        return torch.abs(signed).clamp_min(1.0e-8).reshape(*original)


def shared_augmentation(
    targets: torch.Tensor,
    noise_std: float,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    batch = targets.shape[0]
    noise = torch.randn(batch, 1, targets.shape[-1], device=targets.device) * noise_std
    scale = torch.empty(batch, 1, 1, device=targets.device).uniform_(scale_min, scale_max)
    return (targets + noise) * scale


def train_epoch(
    model: nn.Module,
    loaders: list[tuple[int, DataLoader]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: dict[str, Any],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_endpoint = 0.0
    trajectories = 0
    batches = 0
    for _, loader in loaders:
        for batch in loader:
            times = batch["times"].to(device)
            targets = batch["targets"].to(device)
            condition = batch["condition"].to(device)
            augmented = shared_augmentation(
                targets,
                float(training["noise_std"]),
                float(training["scale_min"]),
                float(training["scale_max"]),
            )
            prediction = integrate_sequence_rk4(
                model,
                augmented[:, 0, :],
                times,
                condition,
                substeps=int(training["integration_substeps"]),
            )
            loss = torch.mean((prediction - augmented) ** 2)
            endpoint = torch.mean((prediction[:, -1, :] - augmented[:, -1, :]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            count = int(targets.shape[0])
            trajectories += count
            batches += 1
            total_loss += float(loss.detach().cpu()) * count
            total_endpoint += float(endpoint.detach().cpu()) * count
    return {
        "train_full_pca_mse": total_loss / max(trajectories, 1),
        "train_endpoint_pca_mse": total_endpoint / max(trajectories, 1),
        "train_trajectories": float(trajectories),
        "train_batches": float(batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    decoder: PcaDecoder,
    loaders: list[tuple[int, DataLoader]],
    device: torch.device,
    integration_substeps: int,
) -> dict[str, float]:
    model.eval()
    totals = {"pca": 0.0, "vertex": 0.0, "volume": 0.0}
    records = 0
    for _, loader in loaders:
        for batch in loader:
            times = batch["times"].to(device)
            targets = batch["targets"].to(device)
            condition = batch["condition"].to(device)
            prediction = integrate_sequence_rk4(
                model,
                targets[:, 0, :],
                times,
                condition,
                substeps=integration_substeps,
            )
            predicted_endpoint = prediction[:, -1, :]
            target_endpoint = targets[:, -1, :]
            predicted_vertices = decoder.vertices(predicted_endpoint)
            target_vertices = decoder.vertices(target_endpoint)
            predicted_volume = decoder.volume(predicted_vertices)
            target_volume = decoder.volume(target_vertices)
            count = int(targets.shape[0])
            records += count
            totals["pca"] += float(torch.mean((predicted_endpoint - target_endpoint) ** 2).cpu()) * count
            totals["vertex"] += float(torch.mean(torch.abs(predicted_vertices - target_vertices)).cpu()) * count
            totals["volume"] += float(torch.mean(torch.abs(predicted_volume - target_volume) / target_volume.clamp_min(1.0e-8)).cpu()) * count
    return {
        "records": float(records),
        "endpoint_pca_mse": totals["pca"] / max(records, 1),
        "endpoint_vertex_mae_mm": totals["vertex"] / max(records, 1),
        "endpoint_volume_relative_error": totals["volume"] / max(records, 1),
    }


def checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_value: float,
    config: dict[str, Any],
    input_config: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric_name": "val_first_last_endpoint_vertex_mae_mm",
        "best_metric_value": best_value,
        "config": config,
        "input_config": input_config,
        "history": history,
        "representation": "train-standardized PCA-150",
        "brainode_model_source": str(PRIOR_BRAINODE_MODEL),
        "brainode_model_source_sha256": sha256(PRIOR_BRAINODE_MODEL),
        "test_data_loaded": False,
    }


def validate_config(
    config_path: Path,
    config: dict[str, Any],
    structure: str,
) -> tuple[Path, dict[str, Any]]:
    expected = {"hippocampus": "left_hippocampus", "lateral_ventricle": "left_lateral_ventricle"}[structure]
    if config.get("structure") != expected:
        raise ValueError(f"BrainODE config structure mismatch: {config_path}")
    if config.get("method") != "previous_attention_brainode_current_strict_pca150":
        raise ValueError(f"Not the expected BrainODE adaptation: {config_path}")
    input_path = resolve_path(config["input_config"])
    input_config = read_json(input_path)
    if input_config.get("structure") != expected:
        raise ValueError("BrainODE input config structure mismatch")
    if int(input_config["representation"]["components"]) != 150:
        raise ValueError("BrainODE requires PCA-150")
    return input_path, input_config


def validate_split_isolation(
    train_archive: dict[str, np.ndarray],
    val_archive: dict[str, np.ndarray],
) -> None:
    train_subjects = set(train_archive["subject_ids"].astype(str))
    val_subjects = set(val_archive["subject_ids"].astype(str))
    train_scans = set(train_archive["visit_scan_ids"].astype(str))
    val_scans = set(val_archive["visit_scan_ids"].astype(str))
    if train_subjects & val_subjects or train_scans & val_scans:
        raise ValueError("Train/validation leakage in BrainODE inputs")
    for name, archive in (("train", train_archive), ("val", val_archive)):
        diagnoses = set(archive["visit_diagnoses"].astype(str))
        if diagnoses - {"CN", "AD"}:
            raise ValueError(f"Non-CN/AD diagnosis in BrainODE {name} archive")


def main() -> int:
    args = parse_args()
    if not PRIOR_BRAINODE_MODEL.is_file():
        raise FileNotFoundError(PRIOR_BRAINODE_MODEL)
    config_path = args.config or default_config_path(args.structure)
    config = read_json(config_path)
    input_config_path, input_config = validate_config(config_path, config, args.structure)
    training = dict(config["training"])
    run_name = str(args.run_name or training["run_name"])
    validate_run_name(run_name)
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else training["batch_size"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    num_workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("BrainODE epochs and batch size must be positive")
    device = choose_device(args.device)
    set_seed(seed)

    # Deliberately load train and validation only. Test remains locked.
    train_archive = load_archive(Path(input_config["dataset"]["train_sequences"]), "train", 150)
    val_archive = load_archive(Path(input_config["dataset"]["val_sequences"]), "val", 150)
    validate_split_isolation(train_archive, val_archive)
    pca_model = validate_pca_model(input_config, 150)
    decoder = PcaDecoder(
        pca_model,
        train_archive["train_pca_mean_150"],
        train_archive["train_pca_std_150"],
    ).to(device)
    train_records = build_records(train_archive, include_backward=True, only_start_visit_zero=False)
    val_forward_records = build_records(val_archive, include_backward=False, only_start_visit_zero=False)
    val_first_last_records = build_records(val_archive, include_backward=False, only_start_visit_zero=True)
    train_loaders = build_loaders(train_records, batch_size, True, num_workers, seed)
    val_forward_loaders = build_loaders(val_forward_records, batch_size, False, num_workers, seed)
    val_first_last_loaders = build_loaders(val_first_last_records, batch_size, False, num_workers, seed)
    model = ODEFuncWithAttention(
        latent_dim=150,
        condition_dim=int(config["model"]["condition_dim"]),
        attention_dim=int(config["model"]["attention_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)

    summaries = {
        "train": record_summary(train_records),
        "val_forward": record_summary(val_forward_records),
        "val_first_last": record_summary(val_first_last_records),
    }
    print("=" * 96, flush=True)
    print(f"Previous attention BrainODE | {args.structure} | device={device} | standardized PCA-150", flush=True)
    print(f"Exact prior model source: {PRIOR_BRAINODE_MODEL}", flush=True)
    print(f"Model source SHA256: {sha256(PRIOR_BRAINODE_MODEL)}", flush=True)
    print(f"Train trajectories: {summaries['train']}", flush=True)
    print(f"Validation forward: {summaries['val_forward']}", flush=True)
    print(f"Validation first-last: {summaries['val_first_last']}", flush=True)
    print("Test archive loaded: no", flush=True)

    if args.dry_run:
        _, loader = train_loaders[0]
        batch = next(iter(loader))
        times = batch["times"].to(device)
        targets = batch["targets"].to(device)
        condition = batch["condition"].to(device)
        prediction = integrate_sequence_rk4(
            model, targets[:, 0, :], times, condition,
            substeps=int(training["integration_substeps"]),
        )
        loss = torch.mean((prediction - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite BrainODE dry-run loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("BrainODE gradient audit failed")
        gradient_norm = float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)).cpu())
        validation = evaluate(
            model, decoder, val_first_last_loaders, device,
            int(training["integration_substeps"]),
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0 or not all(math.isfinite(value) for value in validation.values()):
            raise RuntimeError("BrainODE dry-run finite-value audit failed")
        print("DRY RUN PASSED — forward/backward records, RK4 gradients, decoder, and validation are finite; no files written.", flush=True)
        print(json.dumps({"loss": float(loss.detach().cpu()), "gradient_l2_norm": gradient_norm, "val_first_last": validation}, indent=2), flush=True)
        return 0

    output_dir = config_path.parents[1] / "training" / run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing BrainODE run: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(output_dir / "resolved_config.json", {
            "config_path": str(config_path),
            "input_config_path": str(input_config_path),
            "config": config,
            "input_config": input_config,
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "brainode_model_source": str(PRIOR_BRAINODE_MODEL),
            "brainode_model_source_sha256": sha256(PRIOR_BRAINODE_MODEL),
            "test_data_loaded": False,
        })
        atomic_json(output_dir / "record_summary.json", summaries)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_size"]),
        gamma=float(training["scheduler_gamma"]),
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_value = float("inf")
    best_epoch = 0
    latest_path = output_dir / "checkpoints" / "latest.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {latest_path}")
        payload = torch.load(latest_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"]) + 1
        best_value = float(payload["best_metric_value"])
        if history:
            best_epoch = int(min(history, key=lambda row: row["val_first_last_endpoint_vertex_mae_mm"])["epoch"])
    history_path = output_dir / "history.jsonl"
    history_mode = "a" if args.resume else "w"
    started = time.time()
    with history_path.open(history_mode, encoding="utf-8") as history_file:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.time()
            train_metrics = train_epoch(model, train_loaders, optimizer, device, training)
            val_forward = evaluate(model, decoder, val_forward_loaders, device, int(training["integration_substeps"]))
            val_first_last = evaluate(model, decoder, val_first_last_loaders, device, int(training["integration_substeps"]))
            scheduler.step()
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - epoch_started,
                "elapsed_minutes": (time.time() - started) / 60.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **train_metrics,
                **{f"val_forward_{key}": value for key, value in val_forward.items()},
                **{f"val_first_last_{key}": value for key, value in val_first_last.items()},
            }
            history.append(row)
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            current = float(row["val_first_last_endpoint_vertex_mae_mm"])
            payload = checkpoint_payload(epoch, model, optimizer, scheduler, min(best_value, current), config, input_config, history)
            atomic_torch_save(latest_path, payload)
            if epoch % int(training["save_every"]) == 0:
                atomic_torch_save(output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
            if current < best_value:
                best_value, best_epoch = current, epoch
                payload["best_metric_value"] = best_value
                atomic_torch_save(output_dir / "checkpoints" / "best.pt", payload)
            atomic_json(output_dir / "training_status.json", {
                "status": "running",
                "epoch": epoch,
                "epochs_requested": epochs,
                "best_epoch": best_epoch,
                "best_val_first_last_endpoint_vertex_mae_mm": best_value,
                "test_data_loaded": False,
            })
            print(
                f"epoch {epoch:03d}/{epochs} train={row['train_full_pca_mse']:.6f} "
                f"val_first_last_vertex={current:.6f} val_volume={row['val_first_last_endpoint_volume_relative_error']:.6f} "
                f"best={best_value:.6f}@{best_epoch}",
                flush=True,
            )

    final = {
        "status": "complete",
        "structure": config["structure"],
        "method": config["method"],
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_metric": "val_first_last_endpoint_vertex_mae_mm",
        "best_metric_value": best_value,
        "checkpoint": str(output_dir / "checkpoints" / "best.pt"),
        "representation": "train-standardized PCA-150",
        "shape_only_training": True,
        "forward_and_backward_training": True,
        "test_data_loaded": False,
        "strict_no_mci": True,
        "source_meshes_modified": False,
        "pca_refitted": False,
    }
    atomic_json(output_dir / "training_status.json", final)
    atomic_json(output_dir / "final_report.json", final)
    print("=" * 96, flush=True)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
