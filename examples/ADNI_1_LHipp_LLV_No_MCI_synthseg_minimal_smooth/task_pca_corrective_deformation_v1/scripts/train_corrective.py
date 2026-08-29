#!/usr/bin/env python3
"""Train the PCA-conditioned corrective mesh decoder.

The training process never loads the test split. Epoch zero is exact PCA and is saved as the
initial best checkpoint, so the selected model cannot be worse than PCA on validation RMSE.
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

import common
from model import PCACorrectiveModel, build_model, unique_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two epochs on 64 train and 32 validation meshes; still uses full train-only statistics.",
    )
    return parser.parse_args()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    first = vertices[:, faces[:, 0]]
    second = vertices[:, faces[:, 1]]
    third = vertices[:, faces[:, 2]]
    return F.normalize(torch.cross(second - first, third - first, dim=-1), dim=-1, eps=1e-12)


def laplacian_coordinates(
    vertices: torch.Tensor, directed_source: torch.Tensor, directed_target: torch.Tensor, degree: torch.Tensor
) -> torch.Tensor:
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_sum.index_add_(1, directed_target, vertices.index_select(1, directed_source))
    return vertices - neighbor_sum / degree.view(1, -1, 1)


class GeometryLoss:
    def __init__(self, faces: np.ndarray, device: torch.device):
        self.faces = torch.from_numpy(np.asarray(faces, dtype=np.int64)).to(device)
        edges = torch.from_numpy(unique_edges(faces)).to(device)
        self.edge_first = edges[:, 0]
        self.edge_second = edges[:, 1]
        self.directed_source = torch.cat((self.edge_first, self.edge_second))
        self.directed_target = torch.cat((self.edge_second, self.edge_first))
        degree = torch.bincount(self.directed_target, minlength=int(np.max(faces)) + 1)
        self.degree = degree.to(device=device, dtype=torch.float32).clamp_min(1.0)

    def terms(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        delta: torch.Tensor,
        residual_rms_mm: torch.Tensor,
        weights: dict,
    ) -> dict[str, torch.Tensor]:
        scale_squared = residual_rms_mm.square().clamp_min(1e-12)
        coordinate = F.mse_loss(prediction, target) / scale_squared
        correction = delta.square().mean() / scale_squared
        zero = coordinate.new_zeros(())
        edge, normal, laplacian = zero, zero, zero
        if float(weights["edge_weight"]) > 0.0:
            predicted_edge = torch.linalg.vector_norm(
                prediction[:, self.edge_first] - prediction[:, self.edge_second], dim=-1
            )
            target_edge = torch.linalg.vector_norm(
                target[:, self.edge_first] - target[:, self.edge_second], dim=-1
            )
            edge = F.mse_loss(predicted_edge, target_edge) / scale_squared
        if float(weights["normal_weight"]) > 0.0:
            predicted_normals = face_normals(prediction, self.faces)
            target_normals = face_normals(target, self.faces)
            normal = (1.0 - (predicted_normals * target_normals).sum(dim=-1)).mean()
        if float(weights["laplacian_weight"]) > 0.0:
            predicted_laplacian = laplacian_coordinates(
                prediction, self.directed_source, self.directed_target, self.degree
            )
            target_laplacian = laplacian_coordinates(
                target, self.directed_source, self.directed_target, self.degree
            )
            laplacian = F.mse_loss(predicted_laplacian, target_laplacian) / scale_squared
        return {
            "coordinate": coordinate,
            "correction": correction,
            "edge": edge,
            "normal": normal,
            "laplacian": laplacian,
        }


def weighted_loss(terms: dict[str, torch.Tensor], weights: dict) -> torch.Tensor:
    return (
        float(weights["coordinate_weight"]) * terms["coordinate"]
        + float(weights["correction_weight"]) * terms["correction"]
        + float(weights["edge_weight"]) * terms["edge"]
        + float(weights["normal_weight"]) * terms["normal"]
        + float(weights["laplacian_weight"]) * terms["laplacian"]
    )


@torch.no_grad()
def split_predictions(
    model: PCACorrectiveModel, coefficients: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, pca = [], []
    for start in range(0, len(coefficients), int(batch_size)):
        details = model.decode_with_details(coefficients[start : start + int(batch_size)])
        predictions.append(details["prediction"])
        pca.append(details["pca_mesh"])
    return torch.cat(predictions), torch.cat(pca)


@torch.no_grad()
def per_scan_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((prediction - target).square().flatten(1).mean(dim=1))


@torch.no_grad()
def evaluate_rmse(
    model: PCACorrectiveModel,
    split: common.SplitTensors,
    batch_size: int,
) -> dict:
    prediction, pca = split_predictions(model, split.coefficients, batch_size)
    corrected = per_scan_rmse(prediction, split.vertices_mm)
    baseline = per_scan_rmse(pca, split.vertices_mm)
    return {
        "corrected_rmse_mm": float(corrected.mean()),
        "pca_rmse_mm": float(baseline.mean()),
        "corrected_minus_pca_rmse_mm": float(corrected.mean() - baseline.mean()),
        "corrected_per_scan": corrected.detach().cpu().numpy(),
        "pca_per_scan": baseline.detach().cpu().numpy(),
    }


def checkpoint_payload(
    *,
    model: PCACorrectiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    contract: common.PCAContract,
    epoch: int,
    best_epoch: int,
    best_val_rmse_mm: float,
    best_model_state: dict[str, torch.Tensor],
    history: list[dict],
    permutation_generator: torch.Generator,
) -> dict:
    return {
        "schema_version": 1,
        "architecture": "PCA-conditioned orthogonal corrective Spiral decoder",
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_rmse_mm": float(best_val_rmse_mm),
        "model_state": cpu_state_dict(model),
        "best_model_state": best_model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "permutation_generator_state": permutation_generator.get_state(),
        "config": config,
        "data_contract": contract.summary(),
        "train_split_loaded": True,
        "validation_split_loaded": True,
        "test_split_loaded": False,
    }


def train(args: argparse.Namespace) -> Path:
    config = common.load_config(args.config)
    device = common.choose_device(args.device)
    training = dict(config["training"])
    smoke = bool(args.smoke)
    if smoke:
        training.update(
            {
                "epochs": 2,
                "batch_size": 16,
                "eval_every": 1,
                "early_stopping_evaluations": 0,
                "min_epochs": 0,
            }
        )
    run_dir = common.run_directory(config, args.output_dir, smoke=smoke)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use --resume or a new --output-dir."
        )
    common.makedirs(run_dir)
    common.set_seed(int(training["seed"]))

    print(f"[device] {device}", flush=True)
    print("[data] loading PCA contract from the full train split only", flush=True)
    contract = common.load_pca_contract(config)
    common.assert_subject_disjoint(contract.rows)
    train_split = common.load_split_tensors(
        "train", contract, device, limit=64 if smoke else None
    )
    val_split = common.load_split_tensors("val", contract, device, limit=32 if smoke else None)
    print(
        f"[data] train={len(train_split)} val={len(val_split)} test=NOT_LOADED "
        f"vertices={contract.n_vertices} residual_rms={contract.residual_rms_mm:.6f} mm",
        flush=True,
    )

    model = build_model(config, contract, device)
    geometry = GeometryLoss(contract.faces, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    permutation_generator = torch.Generator(device="cpu").manual_seed(int(training["seed"]))

    resolved_config = copy.deepcopy(config)
    resolved_config["training"] = training
    resolved_config["resolved_device"] = str(device)
    resolved_config["smoke"] = smoke
    common.atomic_write_json(run_dir / "resolved_config.json", resolved_config)
    common.atomic_write_json(run_dir / "data_contract.json", contract.summary())

    history: list[dict] = []
    start_epoch = 1
    best_epoch = 0
    initial = evaluate_rmse(model, val_split, int(training["batch_size"]))
    best_val = float(initial["corrected_rmse_mm"])
    if abs(initial["corrected_minus_pca_rmse_mm"]) > 2e-6:
        raise RuntimeError(
            "Zero-initialized model is not exact PCA: "
            f"delta={initial['corrected_minus_pca_rmse_mm']:.3e} mm"
        )
    best_state = cpu_state_dict(model)
    print(f"[epoch 0] exact PCA validation RMSE={best_val:.9f} mm", flush=True)

    if args.resume:
        latest_path = run_dir / "last.pt"
        if not latest_path.exists():
            raise FileNotFoundError(f"Cannot resume without {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        common.checkpoint_contract_matches(checkpoint, contract)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_val = float(checkpoint["best_val_rmse_mm"])
        best_state = checkpoint["best_model_state"]
        if "permutation_generator_state" in checkpoint:
            permutation_generator.set_state(checkpoint["permutation_generator_state"].cpu())
        print(
            f"[resume] epoch={start_epoch - 1} best={best_val:.9f} @ {best_epoch}", flush=True
        )
    else:
        initial_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=resolved_config,
            contract=contract,
            epoch=0,
            best_epoch=0,
            best_val_rmse_mm=best_val,
            best_model_state=best_state,
            history=history,
            permutation_generator=permutation_generator,
        )
        common.atomic_torch_save(run_dir / "best.pt", initial_payload)
        common.atomic_torch_save(run_dir / "last.pt", initial_payload)

    stale_evaluations = 0
    started = time.time()
    weights = config["loss"]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_split), generator=permutation_generator).to(device)
        accumulated = {key: 0.0 for key in ("total", "coordinate", "correction", "edge", "normal", "laplacian")}
        batches = 0
        for start in range(0, len(train_split), batch_size):
            indices = permutation[start : start + batch_size]
            coefficients = train_split.coefficients.index_select(0, indices)
            target = train_split.vertices_mm.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            details = model.decode_with_details(coefficients)
            terms = geometry.terms(
                details["prediction"],
                target,
                details["delta"],
                model.residual_rms_mm,
                weights,
            )
            loss = weighted_loss(terms, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            accumulated["total"] += float(loss.detach())
            for key, value in terms.items():
                accumulated[key] += float(value.detach())
            batches += 1
        scheduler.step()

        should_evaluate = epoch % int(training["eval_every"]) == 0 or epoch == epochs
        if not should_evaluate:
            continue
        validation = evaluate_rmse(model, val_split, batch_size)
        improved = validation["corrected_rmse_mm"] < (
            best_val - float(training["minimum_improvement_mm"])
        )
        if improved:
            best_val = float(validation["corrected_rmse_mm"])
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value / max(1, batches) for key, value in accumulated.items()},
            "val_corrected_rmse_mm": validation["corrected_rmse_mm"],
            "val_pca_rmse_mm": validation["pca_rmse_mm"],
            "val_corrected_minus_pca_rmse_mm": validation["corrected_minus_pca_rmse_mm"],
            "best_val_rmse_mm": best_val,
            "best_epoch": best_epoch,
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=resolved_config,
            contract=contract,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_rmse_mm=best_val,
            best_model_state=best_state,
            history=history,
            permutation_generator=permutation_generator,
        )
        common.atomic_torch_save(run_dir / "last.pt", payload)
        if improved:
            best_payload = dict(payload)
            best_payload["model_state"] = best_state
            common.atomic_torch_save(run_dir / "best.pt", best_payload)
        common.atomic_write_csv(run_dir / "history.csv", history)
        print(
            f"[epoch {epoch:4d}] loss={row['train_total']:.6f} "
            f"val={validation['corrected_rmse_mm']:.9f} mm "
            f"delta_vs_pca={validation['corrected_minus_pca_rmse_mm']:+.9f} "
            f"best={best_val:.9f} @ {best_epoch}",
            flush=True,
        )
        patience = int(training["early_stopping_evaluations"])
        if (
            patience > 0
            and epoch >= int(training["min_epochs"])
            and stale_evaluations >= patience
        ):
            print(f"[early-stop] {stale_evaluations} evaluations without improvement", flush=True)
            break

    model.load_state_dict(best_state)
    train_metrics = evaluate_rmse(model, train_split, batch_size)
    val_metrics = evaluate_rmse(model, val_split, batch_size)
    summary = {
        "status": "complete",
        "smoke": smoke,
        "device": str(device),
        "parameters_trainable": model.num_parameters(),
        "best_epoch": int(best_epoch),
        "selected_model_is_exact_pca_fallback": bool(best_epoch == 0),
        "best_validation": {
            key: value for key, value in val_metrics.items() if not key.endswith("per_scan")
        },
        "selected_train": {
            key: value for key, value in train_metrics.items() if not key.endswith("per_scan")
        },
        "test_split_loaded": False,
        "data_contract": contract.summary(),
        "checkpoint_best": str(run_dir / "best.pt"),
        "checkpoint_last": str(run_dir / "last.pt"),
    }
    common.atomic_write_json(run_dir / "training_summary.json", summary)
    print(f"[done] {run_dir}", flush=True)
    return run_dir


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
