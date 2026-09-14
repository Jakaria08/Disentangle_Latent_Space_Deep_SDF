#!/usr/bin/env python3
"""Train the Rubanova Latent ODE (rubanova_latent_ode.py) on a protocol view.

Each epoch draws ``samples_per_epoch`` training subjects with replacement (seeded). A subject with n
visits shows the encoder a uniformly sampled prefix of k in 1..n-1 visits, and the ELBO scores
all n visits: reconstruct the prefix, extrapolate the rest. The KL weight ramps linearly to its
final value over ``kl_anneal_epochs``, with free bits per latent dimension.

Checkpoint selection uses the same validation quantity as the ODE arms: macro CN/AD first-to-
last decoded shape ratio against no-change, with the encoder given only the first visit.
Only train and val are opened. Test is evaluator-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmark_common as bc
import dynamics_core as D
import rubanova_latent_ode as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--representation", required=True, choices=bc.ALL_REPRESENTATIONS)
    parser.add_argument("--method", default="latent_ode", choices=("latent_ode", "latent_ode_residual"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class Sequences:
    """Per-subject standardized codes, times (visit_age_norm_train) and labels on a device."""

    def __init__(self, archive: dict[str, np.ndarray], device: torch.device) -> None:
        offsets = archive["subject_visit_offsets"].astype(np.int64)
        self.starts = offsets[:-1]
        self.lengths = np.diff(offsets)
        self.codes = torch.from_numpy(archive["visit_latent_standardized_128"].astype(np.float32)).to(device)
        self.times = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
        self.labels = torch.from_numpy(archive["subject_label_ad"].astype(np.float32)).to(device)
        self.diagnoses = archive["subject_diagnoses"].astype(str)
        self.device = device

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, subjects: np.ndarray, prefix_sizes: np.ndarray) -> dict[str, torch.Tensor]:
        width = int(self.lengths[subjects].max())
        count = len(subjects)
        obs = torch.zeros(count, width, self.codes.shape[1], device=self.device)
        times = torch.zeros(count, width, device=self.device)
        visit_mask = torch.zeros(count, width, dtype=torch.bool, device=self.device)
        prefix_mask = torch.zeros_like(visit_mask)
        for row, (subject, k) in enumerate(zip(subjects, prefix_sizes)):
            start, n = int(self.starts[subject]), int(self.lengths[subject])
            obs[row, :n] = self.codes[start : start + n]
            times[row, :n] = self.times[start : start + n]
            times[row, n:] = self.times[start + n - 1]  # zero-length padded steps
            visit_mask[row, :n] = True
            prefix_mask[row, : int(k)] = True
        return {"obs": obs, "times": times, "visit_mask": visit_mask, "prefix_mask": prefix_mask,
                "condition": self.labels[torch.as_tensor(subjects, device=self.device)]}


def train_epoch(model, data: Sequences, optimizer, training: dict[str, Any], epoch: int, seed: int, max_batches: int | None) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(seed * 1_000_003 + epoch)
    # samples_per_epoch subject draws with replacement, like the cocycle's 4096 pair draws. One
    # pass over 475 ADNI subjects is only 8 optimizer steps, far too few for the encoder-decoder
    # to learn the codes (first search attempt: code MSE ~1 after 20 epochs for every trial).
    order = rng.integers(0, len(data), size=int(training.get("samples_per_epoch", len(data))))
    kl_weight = float(model_kl_weight(training, epoch))
    totals: dict[str, float] = {}
    batches = 0
    batch_size = int(training["batch_size"])
    for start in range(0, len(order), batch_size):
        subjects = order[start : start + batch_size]
        prefix = np.asarray([rng.integers(1, int(data.lengths[s])) for s in subjects])  # k in 1..n-1
        batch = data.batch(subjects, prefix)
        loss, parts = model.loss(batch["obs"], batch["times"], batch["visit_mask"], batch["prefix_mask"], batch["condition"],
                                 kl_weight, float(training["free_bits_per_dim"]))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite Latent ODE loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        batches += 1
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.cpu())
        if max_batches is not None and batches >= max_batches:
            break
    return {name: value / max(batches, 1) for name, value in totals.items()} | {"kl_weight": kl_weight, "batches": float(batches)}


def model_kl_weight(training: dict[str, Any], epoch: int) -> float:
    anneal = max(int(training["kl_anneal_epochs"]), 1)
    return min(1.0, epoch / anneal) * float(training.get("kl_weight_final", 1.0))


@torch.no_grad()
def first_last_validation(model, geometry, archive: dict[str, np.ndarray], raw_vertices: np.ndarray, device, max_records: int | None) -> dict[str, Any]:
    """Same quantities and selection score as task3 train_latent_ode.evaluate on first-to-last rows."""
    model.eval()
    data = Sequences(archive, device)
    subjects = np.arange(len(data)) if max_records is None else np.arange(min(len(data), max_records))
    first = torch.as_tensor(data.starts[subjects], device=device)
    last = torch.as_tensor(data.starts[subjects] + data.lengths[subjects] - 1, device=device)
    groups = {name: {key: [] for key in ("transport_coordinate_mae", "transport_euclidean", "end_to_end_coordinate_mae",
                                          "end_to_end_euclidean", "nochange_transport_coordinate_mae", "nochange_transport_euclidean")}
              for name in ("CN", "AD", "overall")}
    for chunk in range(0, len(subjects), 64):
        f, l = first[chunk : chunk + 64], last[chunk : chunk + 64]
        condition = data.labels[torch.as_tensor(subjects[chunk : chunk + 64], device=device)]
        predicted = model.transport(data.codes[f], data.times[f], data.times[l], condition)
        predicted_v, target_v, source_v = geometry.vertices(predicted), geometry.vertices(data.codes[l]), geometry.vertices(data.codes[f])
        real = torch.from_numpy(np.asarray(raw_vertices[l.cpu().numpy()], dtype=np.float32)).to(device)
        values = {
            "transport_coordinate_mae": torch.mean(torch.abs(predicted_v - target_v), dim=(1, 2)),
            "transport_euclidean": torch.linalg.vector_norm(predicted_v - target_v, dim=2).mean(dim=1),
            "end_to_end_coordinate_mae": torch.mean(torch.abs(predicted_v - real), dim=(1, 2)),
            "end_to_end_euclidean": torch.linalg.vector_norm(predicted_v - real, dim=2).mean(dim=1),
            "nochange_transport_coordinate_mae": torch.mean(torch.abs(source_v - target_v), dim=(1, 2)),
            "nochange_transport_euclidean": torch.linalg.vector_norm(source_v - target_v, dim=2).mean(dim=1),
        }
        for offset, subject in enumerate(subjects[chunk : chunk + 64]):
            for bucket in (data.diagnoses[subject], "overall"):
                for key, tensor in values.items():
                    groups[bucket][key].append(float(tensor[offset].cpu()))
    summary = {"records": int(len(subjects)), "groups": {g: {f"{k}_mean": float(np.mean(v)) if v else float("nan") for k, v in items.items()} for g, items in groups.items()}}
    ratios = []
    for diagnosis in ("CN", "AD"):
        values = summary["groups"][diagnosis]
        if np.isnan(values["transport_coordinate_mae_mean"]):
            continue
        ratios.append(0.5 * (values["transport_coordinate_mae_mean"] / max(values["nochange_transport_coordinate_mae_mean"], 1e-12)
                             + values["transport_euclidean_mean"] / max(values["nochange_transport_euclidean_mean"], 1e-12)))
    summary["selection_score"] = float(np.mean(ratios))
    return summary


def main() -> int:
    args = parse_args()
    overrides = {key: json.loads(raw) for key, _, raw in (item.partition("=") for item in args.override)}
    config = D.load_recipe(args.method, args.representation, overrides)
    if bool(config["model"].get("residual", False)) != (args.method == "latent_ode_residual"):
        raise ValueError("model.residual must be true exactly for latent_ode_residual")
    parts = D.core()
    C = parts["C"]
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
    model = R.LatentODE.from_config(config).to(device)
    train_data = Sequences(train_archive, device)
    raw_val = D.view_vertices(val_archive)
    max_batches = int(training.get("smoke_train_batches", 2)) if (args.smoke or args.dry_run) else None
    max_records = int(training.get("smoke_val_records", 8)) if (args.smoke or args.dry_run) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(training["scheduler_step_size"]), gamma=float(training["scheduler_gamma"]))

    print("=" * 96)
    print(f"{args.method} | view={args.view} | {representation} | seed={seed} | device={device} | parameters={C.parameter_count(model)}")
    print(f"train subjects={len(train_data)} val subjects={len(val_archive['subject_ids'])} | test loaded: no")

    if args.dry_run:
        metrics = train_epoch(model, train_data, optimizer, training, 1, seed, 1)
        validation = first_last_validation(model, geometry, val_archive, raw_val, device, max_records)
        print("DRY RUN PASSED - nothing written.")
        print(json.dumps({"train": metrics, "validation_score": validation["selection_score"]}, indent=2))
        return 0

    run_name = args.run_name or f"{representation}_{args.method}_s{seed}"
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    output_dir = D.run_directory(args.view, representation, args.method, run_name, args.runs_root)
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"cannot resume without {checkpoint_dir / 'latest.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_json(output_dir / "resolved_config.json", D.provenance(args.view, representation, config, seed, {
        "trainer": "train_rubanova_latent_ode.py", "epochs": epochs, "device": str(device), "smoke": bool(args.smoke),
    }))
    start_epoch, best_score, best_epoch = 1, float("inf"), 0
    if args.resume:
        saved = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_score, best_epoch = float(saved["best_selection_score"]), int(saved["best_epoch"])
    started = time.time()
    patience = int(training.get("early_stopping_patience", 0))
    completed = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_epoch(model, train_data, optimizer, training, epoch, seed, max_batches)
        validation = first_last_validation(model, geometry, val_archive, raw_val, device, max_records)
        scheduler.step()
        score = float(validation["selection_score"])
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "config": config, "representation": representation,
                 "method": args.method, "best_selection_score": best_score, "best_epoch": best_epoch, "view": args.view,
                 "seed": seed, "decoder_in_optimizer_loss": False, "test_data_loaded": False}
        C.atomic_torch_save(checkpoint_dir / "latest.pt", state)
        if improved:
            C.atomic_torch_save(checkpoint_dir / "best.pt", state)
        row = {"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60.0, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               **{f"train_{k}": v for k, v in train_metrics.items()}, "val_first_last": validation, "val_selection_score": score}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=bc._json_default) + "\n")
        bc.atomic_json(output_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs,
                       "best_epoch": best_epoch, "best_validation_selection_score": best_score, "test_data_loaded": False})
        print(f"epoch {epoch:03d}/{epochs} loss={train_metrics['loss']:.4f} nll={train_metrics['nll']:.4f} kl={train_metrics['kl']:.3f} "
              f"code_mse={train_metrics['code_mse']:.4f} val_score={score:.6f} best={best_score:.6f}@{best_epoch}", flush=True)
        completed = epoch
        if patience > 0 and not args.smoke and epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch}: no validation improvement for {patience} epochs")
            break
    bc.atomic_json(output_dir / "training_status.json", {
        "status": "complete", "epoch": completed, "epochs_requested": epochs, "best_epoch": best_epoch,
        "best_validation_selection_score": best_score, "selected_checkpoint": str(checkpoint_dir / "best.pt"),
        "elapsed_minutes": (time.time() - started) / 60.0, "test_data_loaded": False,
    })
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
