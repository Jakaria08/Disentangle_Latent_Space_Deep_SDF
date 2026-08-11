#!/usr/bin/env python3
"""Train the paper-core BrainODE protocol on current strict SynthSeg data.

This is an isolated, reproducible implementation of the *core* BrainODE
algorithm: raw PCA-150 coefficients, one subject trajectory per optimization
step, all forward suffixes and backward prefixes, PCA-space L2 loss, and RK4
integration.  It deliberately does not claim to implement the paper's
pseudo-cognitive embedding or 3-D cognition estimator, because the current
cohort contains stable CN and stable AD trajectories only.

The current QC-passed subject splits, all ages, topology, and fixed train-only
PCA models are retained.  Test archives are never loaded by this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from train_adni_synthseg_pca_brainode import integrate_sequence_rk4
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
OFFICIAL_MODEL_URL = "https://github.com/PWonjung/BrainODE/blob/main/model.py"
OFFICIAL_TRAINING_URL = "https://github.com/PWonjung/BrainODE/blob/main/train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=STRUCTURES)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_config_path(structure: str) -> Path:
    return (
        BASE_ROOT
        / f"{structure}_pca_cocycle_v4"
        / "brainode_paper_core"
        / "configs"
        / "paper_core_primary.json"
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


class PaperCoreODEFuncWithAttention(nn.Module):
    """Released BrainODE Q/K/V architecture with enforced one-case semantics.

    The official implementation applies Q @ K.T over its batch dimension.  Its
    released training/evaluation loop passes one subject state at a time.  This
    class therefore rejects batch sizes other than one, preventing unrelated
    participants from becoming attention tokens.
    """

    def __init__(self, latent_dim: int = 150, condition_dim: int = 1) -> None:
        super().__init__()
        input_dim = latent_dim + 1 + condition_dim
        self.query = nn.Linear(input_dim, 256)
        self.key = nn.Linear(input_dim, 256)
        self.value = nn.Linear(input_dim, 256)
        self.scale = 256 ** -0.5
        self.fc1 = nn.Linear(256, 512)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(512, latent_dim)
        self.condition_dim = int(condition_dim)

    def forward(
        self,
        time_value: torch.Tensor,
        latent_state: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latent_state.ndim != 2 or latent_state.shape[0] != 1 or latent_state.shape[1] != 150:
            raise ValueError(f"Paper-core BrainODE requires latent state [1, 150], got {tuple(latent_state.shape)}")
        if time_value.ndim == 0:
            time_feature = time_value.reshape(1, 1)
        elif time_value.shape == (1,):
            time_feature = time_value.reshape(1, 1)
        else:
            raise ValueError(f"Paper-core BrainODE requires scalar or [1] time, got {tuple(time_value.shape)}")
        if condition.shape == (1,):
            condition_feature = condition.reshape(1, 1)
        elif condition.shape == (1, 1):
            condition_feature = condition
        else:
            raise ValueError(f"Paper-core BrainODE requires condition [1], got {tuple(condition.shape)}")
        x = torch.cat((latent_state, time_feature.to(latent_state.dtype), condition_feature.to(latent_state.dtype)), dim=-1)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = torch.softmax(query @ key.transpose(0, 1) * self.scale, dim=-1)
        hidden = attention @ value
        return self.fc2(self.gelu(self.fc1(hidden)))


@dataclass(frozen=True)
class SubjectSequence:
    subject_id: str
    diagnosis: str
    times: np.ndarray
    values: np.ndarray
    condition: float


def build_subjects(archive: dict[str, np.ndarray]) -> list[SubjectSequence]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    output: list[SubjectSequence] = []
    for index in range(len(offsets) - 1):
        start, end = int(offsets[index]), int(offsets[index + 1])
        diagnoses = archive["visit_diagnoses"][start:end].astype(str)
        labels = archive["visit_label_ad"][start:end].astype(np.float32)
        if len(set(diagnoses)) != 1 or len(set(labels.tolist())) != 1:
            raise ValueError(f"Non-stable diagnosis sequence at subject index {index}")
        times = archive["visit_age_norm_train"][start:end].astype(np.float32)
        values = archive["visit_pca_150"][start:end].astype(np.float32)
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"Invalid time sequence at subject index {index}")
        output.append(SubjectSequence(
            subject_id=str(archive["subject_ids"][index]),
            diagnosis=str(diagnoses[0]),
            times=times,
            values=values,
            condition=float(labels[0]),
        ))
    if not output:
        raise ValueError("No eligible subject sequences")
    return output


def subject_summary(subjects: list[SubjectSequence]) -> dict[str, Any]:
    by_diagnosis: dict[str, int] = {}
    by_visits: dict[str, int] = {}
    trajectories = 0
    for subject in subjects:
        by_diagnosis[subject.diagnosis] = by_diagnosis.get(subject.diagnosis, 0) + 1
        by_visits[str(len(subject.times))] = by_visits.get(str(len(subject.times)), 0) + 1
        trajectories += 2 * (len(subject.times) - 1)
    return {
        "subjects": len(subjects),
        "forward_plus_backward_trajectories": trajectories,
        "by_diagnosis": dict(sorted(by_diagnosis.items())),
        "by_visits": dict(sorted(by_visits.items(), key=lambda item: int(item[0]))),
        "all_qc_passed_current_subjects_retained": True,
        "age_filter_applied": False,
        "representation": "raw train-PCA-150 coefficients",
    }


class RawPcaDecoder(nn.Module):
    def __init__(self, pca_model: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.register_buffer("mean_flat", torch.from_numpy(pca_model["mean"].astype(np.float32)).reshape(1, -1))
        self.register_buffer("components", torch.from_numpy(pca_model["components"].astype(np.float32)))
        self.register_buffer("faces", torch.from_numpy(pca_model["faces"].astype(np.int64)))

    def vertices(self, raw_scores: torch.Tensor) -> torch.Tensor:
        original = raw_scores.shape[:-1]
        values = raw_scores.reshape(-1, 150)
        vertices = values @ self.components + self.mean_flat
        return vertices.reshape(*original, -1, 3)

    def volume(self, vertices: torch.Tensor) -> torch.Tensor:
        original = vertices.shape[:-2]
        flat = vertices.reshape(-1, vertices.shape[-2], 3)
        v0 = flat[:, self.faces[:, 0], :]
        v1 = flat[:, self.faces[:, 1], :]
        v2 = flat[:, self.faces[:, 2], :]
        signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
        return torch.abs(signed).clamp_min(1.0e-8).reshape(*original)


def augmented_trajectory(values: torch.Tensor, training: dict[str, Any]) -> torch.Tensor:
    noise = torch.randn(1, values.shape[-1], device=values.device, dtype=values.dtype) * float(training["noise_std"])
    scale = torch.empty(1, device=values.device, dtype=values.dtype).uniform_(
        float(training["scale_min"]), float(training["scale_max"])
    )
    return (values + noise) * scale


def integrate(
    model: nn.Module,
    values: torch.Tensor,
    times: torch.Tensor,
    condition: torch.Tensor,
    substeps: int,
) -> torch.Tensor:
    return integrate_sequence_rk4(
        model,
        values[:1],
        times.reshape(1, -1),
        condition.reshape(1),
        substeps=substeps,
    ).squeeze(0)


def subject_training_loss(
    model: nn.Module,
    subject: SubjectSequence,
    device: torch.device,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    values = torch.from_numpy(subject.values).to(device)
    times = torch.from_numpy(subject.times).to(device)
    condition = torch.tensor([subject.condition], device=device, dtype=torch.float32)
    losses: list[torch.Tensor] = []
    endpoint_losses: list[torch.Tensor] = []
    count = 0
    # Algorithm 1, lines 5–7: every nontrivial forward suffix.
    for start in range(len(times) - 1):
        target = augmented_trajectory(values[start:], training)
        predicted = integrate(model, target, times[start:], condition, int(training["integration_substeps"]))
        losses.append(torch.mean((predicted - target) ** 2))
        endpoint_losses.append(torch.mean((predicted[-1] - target[-1]) ** 2))
        count += 1
    # Algorithm 1, lines 9–11: every nontrivial backward prefix.
    for start in range(1, len(times)):
        target = augmented_trajectory(torch.flip(values[: start + 1], dims=(0,)), training)
        target_times = torch.flip(times[: start + 1], dims=(0,))
        predicted = integrate(model, target, target_times, condition, int(training["integration_substeps"]))
        losses.append(torch.mean((predicted - target) ** 2))
        endpoint_losses.append(torch.mean((predicted[-1] - target[-1]) ** 2))
        count += 1
    if not losses:
        raise RuntimeError("Subject has no paper-core trajectories")
    return torch.stack(losses).mean(), {
        "trajectories": float(count),
        "endpoint_pca_mse": float(torch.stack(endpoint_losses).mean().detach().cpu()),
    }


def train_epoch(
    model: nn.Module,
    subjects: list[SubjectSequence],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: dict[str, Any],
    epoch: int,
    seed: int,
) -> dict[str, float]:
    model.train()
    order = np.random.default_rng(seed + epoch * 1009).permutation(len(subjects))
    total_loss = 0.0
    total_endpoint = 0.0
    trajectories = 0.0
    for position, index in enumerate(order, start=1):
        loss, detail = subject_training_loss(model, subjects[int(index)], device, training)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite paper-core loss for subject {subjects[int(index)].subject_id}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_endpoint += detail["endpoint_pca_mse"]
        trajectories += detail["trajectories"]
        if position % int(training["progress_every_subjects"]) == 0 or position == len(subjects):
            print(
                f"  subject {position:03d}/{len(subjects)} mean_subject_loss={total_loss / position:.6f}",
                flush=True,
            )
    return {
        "train_subject_loss": total_loss / len(subjects),
        "train_subject_endpoint_pca_mse": total_endpoint / len(subjects),
        "train_subjects": float(len(subjects)),
        "train_trajectories": trajectories,
        "optimization_batch_size": 1.0,
    }


@torch.no_grad()
def evaluate_one_shot(
    model: nn.Module,
    subjects: list[SubjectSequence],
    decoder: RawPcaDecoder,
    device: torch.device,
    substeps: int,
) -> dict[str, float]:
    model.eval()
    totals = {"pca": 0.0, "coordinate": 0.0, "euclidean": 0.0, "volume": 0.0}
    by_diagnosis: dict[str, dict[str, float]] = {}
    for subject in subjects:
        values = torch.from_numpy(subject.values).to(device)
        times = torch.from_numpy(subject.times).to(device)
        condition = torch.tensor([subject.condition], device=device, dtype=torch.float32)
        prediction = integrate(model, values, times, condition, substeps)[-1:]
        target = values[-1:]
        predicted_vertices = decoder.vertices(prediction)
        target_vertices = decoder.vertices(target)
        euclidean = torch.linalg.vector_norm(predicted_vertices - target_vertices, dim=-1).mean()
        coordinate = torch.abs(predicted_vertices - target_vertices).mean()
        predicted_volume = decoder.volume(predicted_vertices)
        target_volume = decoder.volume(target_vertices)
        detail = {
            "pca": float(torch.mean((prediction - target) ** 2).cpu()),
            "coordinate": float(coordinate.cpu()),
            "euclidean": float(euclidean.cpu()),
            "volume": float(torch.mean(torch.abs(predicted_volume - target_volume) / target_volume).cpu()),
        }
        for key, value in detail.items():
            totals[key] += value
        bucket = by_diagnosis.setdefault(subject.diagnosis, {"subjects": 0.0, **{key: 0.0 for key in detail}})
        bucket["subjects"] += 1.0
        for key, value in detail.items():
            bucket[key] += value
    result = {
        "subjects": float(len(subjects)),
        "one_shot_endpoint_pca_mse": totals["pca"] / len(subjects),
        "one_shot_endpoint_vertex_coordinate_mae_mm": totals["coordinate"] / len(subjects),
        "one_shot_endpoint_vertex_euclidean_mean_mm": totals["euclidean"] / len(subjects),
        "one_shot_endpoint_volume_relative_error": totals["volume"] / len(subjects),
    }
    for diagnosis, bucket in by_diagnosis.items():
        count = bucket.pop("subjects")
        for key, value in bucket.items():
            result[f"{diagnosis}_{key}"] = value / count
        result[f"{diagnosis}_subjects"] = count
    return result


def validate_config(config_path: Path, config: dict[str, Any], structure: str) -> tuple[Path, dict[str, Any]]:
    expected = {"hippocampus": "left_hippocampus", "lateral_ventricle": "left_lateral_ventricle"}[structure]
    if config.get("structure") != expected:
        raise ValueError(f"Paper-core config structure mismatch: {config_path}")
    if config.get("method") != "brainode_paper_core_stable_cn_ad_raw_pca150":
        raise ValueError(f"Unexpected paper-core method: {config_path}")
    if config["representation"].get("training_key") != "visit_pca_150":
        raise ValueError("Paper-core BrainODE must use raw PCA coefficients")
    if config["representation"].get("age_policy") != "retain_all_current_qc_passed_subjects_train_range_normalization":
        raise ValueError("Paper-core config does not preserve the requested all-age cohort")
    contract = config.get("scientific_contract", {})
    if contract.get("strict_no_mci") is not True or contract.get("test_loaded_during_training") is not False:
        raise ValueError("Paper-core cohort/test contract failed")
    input_path = resolve_path(config["input_config"])
    input_config = read_json(input_path)
    if input_config.get("structure") != expected or int(input_config["representation"]["components"]) != 150:
        raise ValueError("Paper-core input configuration mismatch")
    return input_path, input_config


def validate_split_isolation(train_archive: dict[str, np.ndarray], val_archive: dict[str, np.ndarray]) -> None:
    for name, archive in (("train", train_archive), ("val", val_archive)):
        if set(archive["visit_diagnoses"].astype(str)) - {"CN", "AD"}:
            raise ValueError(f"Non-CN/AD diagnosis in paper-core {name} archive")
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Paper-core train/validation subject leakage")
    if set(train_archive["visit_scan_ids"].astype(str)) & set(val_archive["visit_scan_ids"].astype(str)):
        raise ValueError("Paper-core train/validation scan leakage")


def checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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
        "best_metric_name": "val_one_shot_endpoint_vertex_euclidean_mean_mm",
        "best_metric_value": best_value,
        "config": config,
        "input_config": input_config,
        "history": history,
        "representation": "raw train-PCA-150 coefficients",
        "training_semantics": "one subject trajectory per optimizer update",
        "official_model_url": OFFICIAL_MODEL_URL,
        "official_training_url": OFFICIAL_TRAINING_URL,
        "test_data_loaded": False,
    }


def main() -> int:
    args = parse_args()
    config_path = args.config or default_config_path(args.structure)
    config = read_json(config_path)
    input_config_path, input_config = validate_config(config_path, config, args.structure)
    training = dict(config["training"])
    run_name = str(args.run_name or training["run_name"])
    validate_run_name(run_name)
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    if epochs <= 0:
        raise ValueError("Epochs must be positive")
    device = choose_device(args.device)
    set_seed(seed)

    # Test is deliberately not opened by this trainer.
    train_archive = load_archive(Path(input_config["dataset"]["train_sequences"]), "train", 150)
    val_archive = load_archive(Path(input_config["dataset"]["val_sequences"]), "val", 150)
    validate_split_isolation(train_archive, val_archive)
    train_subjects = build_subjects(train_archive)
    val_subjects = build_subjects(val_archive)
    pca_model = validate_pca_model(input_config, 150)
    decoder = RawPcaDecoder(pca_model).to(device)
    model = PaperCoreODEFuncWithAttention().to(device)
    summaries = {"train": subject_summary(train_subjects), "val": subject_summary(val_subjects)}

    print("=" * 96, flush=True)
    print(f"BrainODE paper-core | {args.structure} | device={device} | raw PCA-150 | subject batch=1", flush=True)
    print("Age policy: all current QC-passed subjects; existing train-range normalization retained", flush=True)
    print(f"Train: {summaries['train']}", flush=True)
    print(f"Validation: {summaries['val']}", flush=True)
    print("Test archive loaded: no", flush=True)

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

    if args.dry_run:
        loss, detail = subject_training_loss(model, train_subjects[0], device, training)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite paper-core dry-run loss")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Paper-core gradient audit failed")
        gradient_norm = float(torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)).cpu())
        validation = evaluate_one_shot(model, val_subjects, decoder, device, int(training["integration_substeps"]))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0 or not all(math.isfinite(float(value)) for value in validation.values()):
            raise RuntimeError("Paper-core dry-run finite-value audit failed")
        print("DRY RUN PASSED — raw PCA, one-subject Algorithm-1 loss, RK4 gradients, and validation are finite; no files written.", flush=True)
        print(json.dumps({"loss": float(loss.detach().cpu()), "detail": detail, "gradient_l2_norm": gradient_norm, "val_one_shot": validation}, indent=2), flush=True)
        return 0

    output_dir = config_path.parents[1] / "training" / run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing paper-core BrainODE run: {output_dir}")
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
            "test_data_loaded": False,
        })
        atomic_json(output_dir / "subject_summary.json", summaries)

    history: list[dict[str, Any]] = []
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
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
            best_epoch = int(min(history, key=lambda row: row["val_one_shot_endpoint_vertex_euclidean_mean_mm"])["epoch"])

    started = time.time()
    history_path = output_dir / "history.jsonl"
    with history_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.time()
            train_metrics = train_epoch(model, train_subjects, optimizer, device, training, epoch, seed)
            validation = evaluate_one_shot(model, val_subjects, decoder, device, int(training["integration_substeps"]))
            scheduler.step()
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - epoch_started,
                "elapsed_minutes": (time.time() - started) / 60.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **train_metrics,
                **{f"val_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            current = float(row["val_one_shot_endpoint_vertex_euclidean_mean_mm"])
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
                "best_val_one_shot_endpoint_vertex_euclidean_mean_mm": best_value,
                "test_data_loaded": False,
            })
            print(
                f"epoch {epoch:03d}/{epochs} train={row['train_subject_loss']:.6f} "
                f"val_one_shot_euclidean={current:.6f} best={best_value:.6f}@{best_epoch}",
                flush=True,
            )

    final = {
        "status": "complete",
        "structure": config["structure"],
        "method": config["method"],
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_metric": "val_one_shot_endpoint_vertex_euclidean_mean_mm",
        "best_metric_value": best_value,
        "checkpoint": str(output_dir / "checkpoints" / "best.pt"),
        "representation": "raw train-PCA-150 coefficients",
        "training_semantics": "one subject trajectory per optimizer update",
        "forward_and_backward_training": True,
        "pseudo_cognitive_embedding": False,
        "cognition_estimator": False,
        "strict_no_mci": True,
        "all_current_qc_passed_subjects_retained": True,
        "test_data_loaded": False,
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
