#!/usr/bin/env python3
"""Training / evaluation for the deterministic spiral AE.

The whole cohort is ~67 MB of float32, so every split lives on the GPU for the entire run and
there is no DataLoader in the hot path. Loss is L1 on normalised coordinates (matching
guided_vae's loss_function with beta=0); model selection is on validation vertex_rmse_mm in mm.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import spiral_common as sc


@dataclass
class MeshTensors:
    """Normalised splits resident on `device`, plus the stats needed to get back to mm."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    faces: np.ndarray
    device: torch.device

    def denormalize(self, x):
        return x * self.std + self.mean

    def split(self, name):
        return {"train": self.train, "val": self.val, "test": self.test}[name]


def load_mesh_tensors(device, rows=None) -> MeshTensors:
    rows = rows if rows is not None else sc.read_manifest()
    train = sc.load_split_vertices("train", rows=rows)
    val = sc.load_split_vertices("val", rows=rows)
    test = sc.load_split_vertices("test", rows=rows)
    faces = sc.load_faces(rows=rows)
    mean, std = sc.train_normalization(train)

    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    def prep(arr):
        return ((torch.from_numpy(arr).to(device) - mean_t) / std_t).contiguous()

    return MeshTensors(prep(train), prep(val), prep(test), mean_t, std_t, faces, device)


@torch.no_grad()
def predict(model, data: MeshTensors, split: str, batch_size: int = 64) -> torch.Tensor:
    """Returns predicted vertices in mm."""
    model.eval()
    x = data.split(split)
    chunks = [model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return data.denormalize(torch.cat(chunks))


@torch.no_grad()
def encode_split(model, data: MeshTensors, split: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    x = data.split(split)
    chunks = [model.encode(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(chunks).detach().cpu().numpy()


def evaluate(model, data: MeshTensors, split: str, with_volume: bool = False) -> dict:
    pred_mm = predict(model, data, split)
    gt_mm = data.denormalize(data.split(split))
    faces = data.faces if with_volume else None
    return sc.reconstruction_metrics(pred_mm, gt_mm, faces=faces)


def val_rmse(model, data: MeshTensors) -> float:
    pred_mm = predict(model, data, "val")
    gt_mm = data.denormalize(data.val)
    return float(sc.vertex_rmse_mm(pred_mm, gt_mm).mean())


def train_model(
    model,
    data: MeshTensors,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_decay: float,
    decay_step: int,
    weight_decay: float,
    device,
    eval_every: int = 10,
    log_fn=None,
    report_fn=None,
    should_prune=None,
    seed: int = 1,
    time_budget_s: float | None = None,
    noise_std: float = 0.0,
    scheduler_type: str = "step",
    patience: int = 0,
    min_epochs: int = 0,
):
    """Returns (best_val_rmse_mm, best_state_dict, history).

    v1 studies overfit (train L1 falling while val RMSE rose), so three levers were added:
    `noise_std` trains as a denoising AE (Gaussian noise on the normalised input, clean target),
    `patience` stops a trial after that many consecutive evaluations without a new best, and
    `scheduler_type` allows a cosine decay as an alternative to StepLR.

    `min_epochs` protects slow-starting configurations: neither the pruner nor patience may
    fire before it. Without this, a config that begins badly but converges well is killed at
    the first evaluation -- which is exactly how the v1 pure-architecture runs lost their best
    candidates.

    `time_budget_s` caps wall-clock per trial. Trial cost varies ~10x across the search space
    (a wide adaptive config at 600 epochs is hours; a narrow spiral one is minutes), so without
    a cap a single unlucky trial can stall a study. Hitting the cap is not a failure: the best
    validation score reached so far is returned and recorded, with `stopped_on_time_budget`.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if str(scheduler_type).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, max(1, int(decay_step)), gamma=lr_decay
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x_train = data.train
    n = len(x_train)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    stopped_on_budget = False
    stopped_on_patience = False
    stopped_on_prune = False
    stale_evals = 0
    start = time.time()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            batch = x_train[perm[i : i + batch_size]]
            noisy = batch if noise_std <= 0 else batch + noise_std * torch.randn_like(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(noisy), batch, reduction="mean")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        train_loss = total / max(1, batches)

        if epoch % eval_every == 0 or epoch == int(epochs):
            current = val_rmse(model, data)
            if current < best_val - 1e-9:
                best_val = current
                best_epoch = epoch
                stale_evals = 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                )
            else:
                stale_evals += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_l1": train_loss,
                    "val_rmse_mm": current,
                    "best_val_rmse_mm": best_val,
                    "elapsed_s": time.time() - start,
                }
            )
            if log_fn is not None:
                log_fn(
                    f"    epoch {epoch:4d}/{int(epochs)}  train_l1={train_loss:.5f}  "
                    f"val_rmse_mm={current:.6f}  best={best_val:.6f}  "
                    f"({time.time() - start:.0f}s)"
                )
            if report_fn is not None:
                report_fn(current, epoch)
            if epoch >= int(min_epochs) and should_prune is not None and should_prune():
                stopped_on_prune = True
                if log_fn is not None:
                    log_fn(f"    pruner signalled at epoch {epoch}")
                break

            if epoch >= int(min_epochs) and patience and stale_evals >= int(patience):
                stopped_on_patience = True
                if log_fn is not None:
                    log_fn(
                        f"    early stop at epoch {epoch}: {stale_evals} evals without "
                        f"improvement (best={best_val:.6f} @ {best_epoch})"
                    )
                break

            if time_budget_s is not None and (time.time() - start) > time_budget_s:
                stopped_on_budget = True
                if log_fn is not None:
                    log_fn(
                        f"    stopped at epoch {epoch}/{int(epochs)} on the "
                        f"{time_budget_s:.0f}s trial budget (best={best_val:.6f})"
                    )
                break

    return (
        best_val,
        best_state,
        {
            "history": history,
            "best_epoch": best_epoch,
            "stopped_on_time_budget": stopped_on_budget,
            "stopped_on_patience": stopped_on_patience,
            "stopped_on_prune": stopped_on_prune,
            "epochs_completed": history[-1]["epoch"] if history else 0,
        },
    )
