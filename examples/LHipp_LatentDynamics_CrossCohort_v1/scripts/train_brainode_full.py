#!/usr/bin/env python3
"""Stage 5 step 2: BrainODE-full for the converter line (PCA only; a PLAN Part 5B comparator).

BrainODE-full = BrainODE-core's vector field whose condition is re-estimated from the predicted shape at every RK4
substep by a shape-only cognition estimator, trained with pseudo-cognitive sampling:

* field: the P3 pooled BrainODE-core checkpoint (pca128, same seed), unchanged; it was trained on stable subjects
  with fixed labels, like the paper's core model;
* estimator: task3's VoxelCognitionCNN on solid 32^3 voxelizations of decoded PCA-128 shapes (what it sees during
  feedback), trained on stable train visits (CN 0 / AD 1, subject-balanced BCE) plus pseudo-cognitive samples: a
  CN-stable and an AD-stable train visit of similar age are mixed in code space, z = (1 - a) z_CN + a z_AD with
  a ~ U(0, 1), and the decoded mixture gets the soft target a; the two BCE terms are averaged;
* selection on validation stable subjects with task3's rule (subject AUROC - 0.05 Brier); one temperature is fitted
  on validation logits.

Converter subjects are never used for supervision. Only train and val are opened. Settings:
configs/converter_line.json -> brainode_full.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import benchmark_common as bc
import converter_line as L
import dynamics_core as D

CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")
SETTINGS = CONFIG["brainode_full"]
REPRESENTATION = SETTINGS["representation"]
METHOD = "brainode_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=CONFIG["seeds"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--workers", type=int, default=int(SETTINGS["voxel_workers"]))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def task3_modules():
    """task3 brainode_cognition, prepare_cognition_voxels and train_brainode_cognition (after core() fixes names)."""
    D.core()
    import brainode_cognition as B
    import prepare_cognition_voxels as PV
    import train_brainode_cognition as TB

    for module in (B, PV, TB):
        if Path(module.__file__).resolve().parent != D.T3_SCRIPTS.resolve():
            raise ImportError(f"{module.__name__} resolved to {module.__file__}")
    return B, PV, TB


def ode_checkpoint(seed: int) -> Path:
    path = D.RUNS_ROOT / CONFIG["base_view"] / REPRESENTATION / "brainode" / f"{REPRESENTATION}_brainode_s{seed}" / "checkpoints" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"P3 BrainODE-core checkpoint missing: {path}")
    return path


@torch.no_grad()
def decode(geometry, codes: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    return torch.cat([geometry.vertices(codes[start:start + batch_size]) for start in range(0, len(codes), batch_size)]).cpu().numpy().astype(np.float32)


def stable_indices(archive: dict[str, np.ndarray], per_class_limit: int | None = None) -> np.ndarray:
    groups = L.stable_groups(archive)
    if per_class_limit is None:
        return np.flatnonzero(groups >= 0)
    return np.concatenate([np.flatnonzero(groups == L.STABLE_CN)[:per_class_limit], np.flatnonzero(groups == L.STABLE_AD)[:per_class_limit]])


def pseudo_pairs(ages: np.ndarray, labels: np.ndarray, count: int, max_age_difference: float, rng: np.random.Generator):
    """(CN positions, AD positions, alpha): each CN visit is paired with a random AD visit within the age window."""
    cn, ad = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    order = np.argsort(ages[ad], kind="stable")
    sorted_ages = ages[ad][order]
    first, second = [], []
    while len(first) < count:
        candidate = int(rng.choice(cn))
        low = int(np.searchsorted(sorted_ages, ages[candidate] - max_age_difference, side="left"))
        high = int(np.searchsorted(sorted_ages, ages[candidate] + max_age_difference, side="right"))
        if high > low:
            first.append(candidate)
            second.append(int(ad[order[int(rng.integers(low, high))]]))
    return np.asarray(first), np.asarray(second), rng.uniform(0.0, 1.0, count)


def estimator_loss(logits, labels, weights, pseudo_logits, pseudo_targets):
    observed = torch.sum(F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights) / weights.sum().clamp_min(1e-8)
    pseudo = F.binary_cross_entropy_with_logits(pseudo_logits, pseudo_targets)
    return 0.5 * (observed + pseudo), observed, pseudo


def main() -> int:
    args = parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)  # voxel workers must not fork a CUDA parent
    B, PV, TB = task3_modules()
    parts = D.core()
    C = parts["C"]
    device = D.device(args.device)
    C.set_seed(args.seed)
    view = CONFIG["view"]
    registry = D.view_registry(view)
    train = C.load_archive(REPRESENTATION, "train", registry)
    val = C.load_archive(REPRESENTATION, "val", registry)
    D.assert_no_test_leakage(view, [train, val])
    geometry = C.build_geometry(REPRESENTATION, train, device, registry)
    faces = geometry.faces.cpu().numpy()
    smoke = args.smoke or args.dry_run
    rng = np.random.default_rng(args.seed)
    estimator = SETTINGS["estimator_settings"]

    train_index, val_index = stable_indices(train, 16 if smoke else None), stable_indices(val, 16 if smoke else None)
    train_codes = torch.from_numpy(train["visit_latent_standardized_128"]).to(device)[train_index]
    val_codes = torch.from_numpy(val["visit_latent_standardized_128"]).to(device)[val_index]
    started = time.time()
    train_vertices = decode(geometry, train_codes)
    grid = B.VoxelGrid.from_training_vertices(train_vertices, int(SETTINGS["voxel_resolution"]), int(SETTINGS["padding_voxels"]))
    train_masks, train_quality = PV.voxelize_split(train_vertices, faces, grid, args.workers)
    val_masks, val_quality = PV.voxelize_split(decode(geometry, val_codes), faces, grid, args.workers)
    train_labels = train["visit_label_ad"][train_index].astype(np.int64)
    val_labels = val["visit_label_ad"][val_index].astype(np.int64)
    bank = 256 if smoke else int(SETTINGS["pseudo_bank_size"])
    cn, ad, alpha = pseudo_pairs(train["visit_age_years"][train_index].astype(np.float64), train_labels, bank,
                                 float(SETTINGS["pseudo_max_age_difference_years"]), rng)
    weights_alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)[:, None]
    mixed = (1.0 - weights_alpha) * train_codes[torch.as_tensor(cn, device=device)] + weights_alpha * train_codes[torch.as_tensor(ad, device=device)]
    pseudo_masks, pseudo_quality = PV.voxelize_split(decode(geometry, mixed), faces, grid, args.workers)
    print(f"BrainODE-full estimator | s{args.seed} | observed train {len(train_index)} val {len(val_index)} | pseudo bank {bank} | "
          f"voxelized in {(time.time() - started) / 60:.1f} min", flush=True)

    model = B.VoxelCognitionCNN(int(estimator["base_channels"]), float(estimator["dropout"])).to(device)
    weights = B.subject_balanced_weights(train["visit_subject_ids"][train_index], train_labels)
    loader = DataLoader(TB.VoxelDataset(train_masks, train_labels, weights), batch_size=int(estimator["batch_size"]), shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(TB.VoxelDataset(val_masks, val_labels, np.ones(len(val_labels), np.float32)),
                            batch_size=int(estimator["evaluation_batch_size"]), shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(estimator["learning_rate"]), weight_decay=float(estimator["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=int(estimator["scheduler_patience"]),
                                                           min_lr=float(estimator["minimum_learning_rate"]))
    epochs = 1 if smoke else int(args.epochs if args.epochs is not None else estimator["epochs"])
    per_epoch = min(bank, int(SETTINGS["pseudo_samples_per_epoch"]))

    run_name = args.run_name or f"{REPRESENTATION}_{METHOD}_s{args.seed}"
    run_dir = D.run_directory(view, REPRESENTATION, METHOD, run_name, args.runs_root)
    if not args.dry_run and run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    checkpoints = run_dir / "checkpoints"
    config = {"method": METHOD, "representation": REPRESENTATION, "seed": args.seed, "settings": SETTINGS, "grid": grid.mapping(),
              "ode_checkpoint": str(ode_checkpoint(args.seed)), "ode_checkpoint_sha256": bc.sha256_file(ode_checkpoint(args.seed))}
    best_score, best_epoch, stale, history = -math.inf, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(bank)[:per_epoch]
        pseudo_batches = np.array_split(order, max(1, len(loader)))
        totals = {"loss": 0.0, "observed_bce": 0.0, "pseudo_bce": 0.0}
        steps = 0
        for step, batch in enumerate(loader):
            pick = pseudo_batches[step % len(pseudo_batches)]
            logits = model(batch["mask"].to(device, dtype=torch.float32))
            pseudo_logits = model(torch.from_numpy(pseudo_masks[pick]).to(device, dtype=torch.float32))
            loss, observed, pseudo = estimator_loss(logits, batch["label"].to(device), batch["weight"].to(device), pseudo_logits,
                                                    torch.as_tensor(alpha[pick], dtype=torch.float32, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(estimator["gradient_clip_norm"]))
            optimizer.step()
            for name, value in (("loss", loss), ("observed_bce", observed), ("pseudo_bce", pseudo)):
                totals[name] += float(value.detach())
            steps += 1
            if args.dry_run:
                norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
                print(f"DRY RUN PASSED: loss={float(loss):.5f} observed={float(observed):.5f} pseudo={float(pseudo):.5f} gradient={norm:.3e}")
                return 0 if math.isfinite(norm) and norm > 0 else 1
            if smoke and steps >= 2:
                break
        logits = TB.cognition_predictions(model, val_loader, device)
        metrics = B.scan_and_subject_metrics(val_labels, 1.0 / (1.0 + np.exp(-logits)), val["visit_subject_ids"][val_index])
        score = TB.cognition_selection(metrics)
        scheduler.step(score)
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}, "validation": metrics, "selection_score": score,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        improved = score > best_score
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "history": history, "view": view,
                   "seed": args.seed, "input_contract": model.input_contract, "test_data_loaded": False, "pseudo_cognition_sampling": True}
        C.atomic_torch_save(checkpoints / "estimator_latest.pt", payload)
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            C.atomic_torch_save(checkpoints / "estimator_best.pt", payload)
        else:
            stale += 1
        bc.atomic_json(run_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch,
                                                          "best_selection_score": best_score, "test_data_loaded": False})
        print(f"estimator epoch {epoch:03d}/{epochs} loss={row['loss']:.4f} observed={row['observed_bce']:.4f} pseudo={row['pseudo_bce']:.4f} "
              f"val subject AUROC={metrics['subject']['auroc']:.4f} score={score:.4f} best={best_score:.4f}@{best_epoch}", flush=True)
        if not smoke and stale >= int(estimator["early_stopping_patience"]):
            break

    best = torch.load(checkpoints / "estimator_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    logits = TB.cognition_predictions(model, val_loader, device)
    temperature = B.fit_temperature(logits, val_labels)
    best["temperature"] = temperature
    best["calibrated_validation"] = B.scan_and_subject_metrics(val_labels, 1.0 / (1.0 + np.exp(-logits / temperature)), val["visit_subject_ids"][val_index])
    best["voxel_quality"] = {"train": PV.quality_summary(train_quality), "val": PV.quality_summary(val_quality), "pseudo": PV.quality_summary(pseudo_quality)}
    C.atomic_torch_save(checkpoints / "estimator_best.pt", best)
    bc.atomic_json(run_dir / "training_status.json", {
        "status": "complete", "epoch": history[-1]["epoch"], "epochs_requested": epochs, "best_epoch": best_epoch, "best_selection_score": best_score,
        "temperature": temperature, "calibrated_validation_subject": best["calibrated_validation"]["subject"],
        "selected_checkpoint": str(checkpoints / "estimator_best.pt"), "elapsed_minutes": (time.time() - started) / 60, "test_data_loaded": False})
    print(f"COMPLETE: {checkpoints / 'estimator_best.pt'} (temperature {temperature:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
