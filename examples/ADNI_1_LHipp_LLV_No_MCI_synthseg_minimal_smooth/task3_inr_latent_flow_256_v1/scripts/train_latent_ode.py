#!/usr/bin/env python3
"""Train matched INR-256 plain ODE or singleton-attention BrainODE."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import c4_objective as O
import common as C
from inr_geometry import build_geometry
from models import build_ode, integrate_sequence_rk4, transport_rk4


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


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    targets: np.ndarray
    condition: float

    @property
    def length(self) -> int:
        return len(self.times)


class TrajectoryDataset(Dataset):
    def __init__(self, rows: list[Trajectory]) -> None:
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows: list[Trajectory]) -> dict[str, torch.Tensor]:
    return {
        "times": torch.from_numpy(np.stack([row.times for row in rows])).float(),
        "targets": torch.from_numpy(np.stack([row.targets for row in rows])).float(),
        "condition": torch.tensor([row.condition for row in rows], dtype=torch.float32),
    }


def build_trajectories(archive: dict[str, np.ndarray]) -> list[Trajectory]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    output = []
    for subject in range(len(offsets) - 1):
        start, end = int(offsets[subject]), int(offsets[subject + 1])
        times = archive["visit_age_norm_train"][start:end].astype(np.float32)
        targets = archive["visit_latent_standardized_256"][start:end].astype(np.float32)
        condition = float(archive["subject_label_ad"][subject])
        for visit in range(len(times)):
            if len(times[visit:]) > 1:
                output.append(Trajectory(times[visit:].copy(), targets[visit:].copy(), condition))
            if len(times[:visit + 1]) > 1:
                output.append(Trajectory(times[:visit + 1][::-1].copy(), targets[:visit + 1][::-1].copy(), condition))
    if not output:
        raise ValueError("No eligible trajectories")
    return output


def loaders(rows: list[Trajectory], batch_size: int, shuffle: bool, seed: int):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.length, []).append(row)
    return [DataLoader(TrajectoryDataset(current), batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(seed + length)) for length, current in sorted(grouped.items())]


def augment(targets: torch.Tensor, training: dict[str, Any]) -> torch.Tensor:
    noise = torch.randn(targets.shape[0], 1, targets.shape[-1], device=targets.device) * float(training["noise_std"])
    scale = torch.empty(targets.shape[0], 1, 1, device=targets.device).uniform_(float(training["scale_min"]), float(training["scale_max"]))
    return (targets + noise) * scale


def train_epoch(model, grouped, optimizer, device, training, max_batches=None):
    model.train()
    total, count, batches = 0.0, 0, 0
    for loader in grouped:
        for batch in loader:
            targets = augment(batch["targets"].to(device), training)
            prediction = integrate_sequence_rk4(model, targets[:, 0], batch["times"].to(device), batch["condition"].to(device), int(training["integration_substeps"]))
            loss = torch.mean((prediction - targets).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(targets)
            count += len(targets)
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                return {"latent_trajectory_mse": total / count, "trajectories": count, "batches": batches}
    return {"latent_trajectory_mse": total / max(count, 1), "trajectories": count, "batches": batches}


class ODETransport(nn.Module):
    def __init__(self, model: nn.Module, substeps: int) -> None:
        super().__init__()
        self.model, self.substeps = model, int(substeps)

    def transport(self, latent, source_time, target_time, condition, context=None, context_time=None):
        del context, context_time
        return transport_rk4(self.model, latent, source_time, target_time, condition, self.substeps)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("representation") != "inr256" or config.get("method") not in {"plain_ode", "brainode"}:
        raise ValueError("Expected an INR-256 ODE config")
    if int(config["model"].get("latent_dim", -1)) != C.LATENT_DIM:
        raise ValueError("ODE latent dimension must be 256")
    if config.get("loss") != {"latent_trajectory_mse_weight": 1.0}:
        raise ValueError("ODE optimizer objective must be unit latent trajectory MSE only")


def main() -> int:
    args = parse_args()
    config_path = C.resolve_path(args.config)
    config = C.read_json(config_path)
    validate_config(config)
    registry = C.load_registry()
    train_archive, val_archive = C.load_archive("train", registry), C.load_archive("val", registry)
    device = C.choose_device(args.device)
    training = config["training"]
    seed = int(args.seed if args.seed is not None else training["seed"])
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    if args.smoke:
        epochs = 1
    C.set_seed(seed)
    model = build_ode(config).to(device)
    geometry = build_geometry(train_archive, device, registry)
    trajectories = build_trajectories(train_archive)
    grouped = loaders(trajectories, int(training["batch_size"]), True, seed)
    val_values = C.values_on_device(val_archive, device)
    val_first_last = C.first_last_pairs(val_archive)
    transport = ODETransport(model, int(training["integration_substeps"]))
    print(f"{config['method']} | inr256 | device={device} | params={C.parameter_count(model):,} | trajectories={len(trajectories)}")
    print(f"attention={getattr(model, 'attention_contract', 'none')} decoder_in_optimizer_loss=no test_loaded=no")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    if args.dry_run:
        metrics = train_epoch(model, grouped, optimizer, device, training, max_batches=1)
        report = O.evaluate_pairs(transport, geometry, val_values, val_first_last[: int(training["smoke_val_records"])], {"field_samples_per_pair": 128, "decoder_point_chunk": 65536, "volume_samples": 128, "volume_temperature": 0.01}, min(8, int(training["batch_size"])))
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        gradient_norm = math.sqrt(sum(float(torch.sum(value.detach().square()).cpu()) for value in gradients))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("ODE gradient audit failed")
        if any(parameter.grad is not None for parameter in geometry.decoder.parameters()):
            raise RuntimeError("Decoder entered ODE optimizer graph")
        if config["method"] == "brainode":
            model.eval()
            z = torch.randn(3, C.LATENT_DIM, device=device)
            t = torch.linspace(0.1, 0.3, 3, device=device)
            d = torch.tensor([0.0, 1.0, 0.0], device=device)
            if not torch.allclose(model(t[:1], z[:1], d[:1]), model(t, z, d)[:1], atol=1.0e-6, rtol=1.0e-6):
                raise RuntimeError("BrainODE batch-invariance audit failed")
        print("DRY RUN PASSED — latent-only gradients, frozen decoder evaluation, and attention contract; no files written.")
        print(json.dumps({"train": metrics, "validation": report, "gradient_l2_norm": gradient_norm}, indent=2, sort_keys=True))
        return 0
    run_name = str(args.run_name or training["run_name"])
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    C.validate_run_name(run_name)
    run_dir = C.output_root(registry) / "training" / "inr256" / config["method"] / run_name
    checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(checkpoint_dir / "latest.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_json(run_dir / "resolved_config.json", {"config_path": str(config_path), "config": config, "device": str(device), "seed": seed, "epochs": epochs, "decoder_in_optimizer_loss": False, "test_data_loaded": False})
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(training["scheduler_step_size"]), gamma=float(training["scheduler_gamma"]))
    start_epoch, best_epoch, best_score, history = 1, 0, float("inf"), []
    if args.resume:
        payload = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch, best_epoch, best_score, history = int(payload["epoch"]) + 1, int(payload["best_epoch"]), float(payload["best_selection_score"]), list(payload["history"])
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train = train_epoch(model, grouped, optimizer, device, training, int(training["smoke_train_batches"]) if args.smoke else None)
        val = O.evaluate_pairs(transport, geometry, val_values, val_first_last[: int(training["smoke_val_records"])] if args.smoke else val_first_last, {"field_samples_per_pair": 128, "decoder_point_chunk": 65536, "volume_samples": 256, "volume_temperature": 0.01}, min(16, int(training["batch_size"])))
        ratios = [val["groups"][diagnosis]["sdf_mean"] / max(val["groups"][diagnosis]["nochange_sdf_mean"], 1.0e-8) for diagnosis in ("CN", "AD")]
        score = float(np.mean(ratios))
        scheduler.step()
        if score < best_score:
            best_score, best_epoch = score, epoch
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0, "train": train, "validation": val, "selection_score": score}
        history.append(row)
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "history": history, "best_selection_score": best_score, "best_epoch": best_epoch, "decoder_in_optimizer_loss": False, "test_data_loaded": False}
        C.atomic_torch_save(checkpoint_dir / "latest.pt", payload)
        if best_epoch == epoch:
            C.atomic_torch_save(checkpoint_dir / "best.pt", payload)
        C.atomic_json(run_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_selection_score": best_score, "decoder_in_optimizer_loss": False, "test_data_loaded": False})
        print(f"epoch {epoch:03d}/{epochs} mse={train['latent_trajectory_mse']:.6f} val={score:.6f} best={best_score:.6f}@{best_epoch}", flush=True)
    C.atomic_json(run_dir / "training_status.json", {"status": "complete", "epoch": epochs, "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_selection_score": best_score, "selected_checkpoint": str(checkpoint_dir / "best.pt"), "decoder_in_optimizer_loss": False, "test_data_loaded": False})
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
