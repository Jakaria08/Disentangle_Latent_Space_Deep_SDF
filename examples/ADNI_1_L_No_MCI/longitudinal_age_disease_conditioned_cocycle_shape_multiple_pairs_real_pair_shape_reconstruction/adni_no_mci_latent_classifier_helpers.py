from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import adni_no_mci_longitudinal_model_helpers as model_helpers

DEFAULT_CHECKPOINT = "1000"
DEFAULT_FEATURE_OBSERVATION_MODE = "all"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "analysis" / "latent_classifier"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_output_dir(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_bundle(checkpoint: str | int = DEFAULT_CHECKPOINT, device: str | torch.device | None = "auto"):
    return model_helpers.load_model_bundle(EXPERIMENT_DIR, checkpoint, device=device)


def _split_frame(bundle, split: str) -> pd.DataFrame:
    frame = model_helpers.split_frame(bundle.metadata, split)
    if len(frame) == 0:
        raise RuntimeError(f"No rows found for split='{split}'.")
    return frame.reset_index(drop=True)


def _subject_level_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = model_helpers.grouped_subject_rows(frame)
    for subject_id, subject_frame in grouped.items():
        diagnoses = subject_frame["diagnosis"].astype(str).unique().tolist()
        labels = subject_frame["label_ad"].astype(int).unique().tolist()
        if len(diagnoses) != 1:
            raise RuntimeError(
                f"Subject {subject_id} in split='{split}' has mixed diagnoses: {diagnoses}"
            )
        if len(labels) != 1:
            raise RuntimeError(
                f"Subject {subject_id} in split='{split}' has mixed label_ad values: {labels}"
            )
        rows.append(
            {
                "subject_id": str(subject_id),
                "split": split,
                "diagnosis": str(diagnoses[0]),
                "label_ad": int(labels[0]),
                "n_scans": int(len(subject_frame)),
                "baseline_age_years": float(subject_frame["baseline_age_years"].iloc[0]),
                "age_span_years": float(
                    subject_frame["continuous_age_years"].iloc[-1]
                    - subject_frame["continuous_age_years"].iloc[0]
                ),
                "scan_ids": "|".join(subject_frame["scan_id"].astype(str).tolist()),
                "latent_source": None,
                "latent_fit_loss_start": float("nan"),
                "latent_fit_loss_end": float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["diagnosis", "subject_id"]).reset_index(drop=True)


def _subject_scan_ids(subject_frame: pd.DataFrame, observation_mode: str) -> List[str]:
    subject_frame = subject_frame.sort_values(["visit_order", "continuous_age_norm", "scan_id"])
    scan_ids = subject_frame["scan_id"].astype(str).tolist()
    mode = str(observation_mode).strip().lower()
    if mode == "all":
        return scan_ids
    if mode == "baseline_only":
        return scan_ids[:1]
    if mode == "first_two":
        return scan_ids[:2] if len(scan_ids) >= 2 else scan_ids
    raise ValueError(
        f"Unknown observation_mode='{observation_mode}'. Use one of: all, baseline_only, first_two."
    )


def load_train_split_latents(checkpoint: str | int = DEFAULT_CHECKPOINT, device: str | torch.device | None = "auto") -> Tuple[pd.DataFrame, np.ndarray]:
    bundle = load_bundle(checkpoint=checkpoint, device=device)
    train_frame = _split_frame(bundle, "train")
    subject_frame = _subject_level_frame(train_frame, "train")
    latents = bundle.train_latents
    if latents is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint} does not contain a latent table under LatentCodes/."
        )
    print(
        f"[latent] loaded train latents from checkpoint {checkpoint}: "
        f"{latents.shape[0]} subjects x {latents.shape[1]} dims"
    )
    latents = latents.detach().cpu().float()
    expected_subject_ids = sorted(train_frame["subject_id"].astype(str).unique().tolist())
    if len(expected_subject_ids) != latents.shape[0]:
        raise RuntimeError(
            "Latent table size does not match the number of training subjects: "
            f"{latents.shape[0]} vs {len(expected_subject_ids)}"
        )

    subject_frame = subject_frame.set_index("subject_id").loc[expected_subject_ids].reset_index()
    subject_frame["latent_source"] = f"trained_latent_checkpoint_{checkpoint}"
    subject_frame["latent_fit_loss_start"] = np.nan
    subject_frame["latent_fit_loss_end"] = np.nan
    feature_matrix = latents.numpy()
    return subject_frame, feature_matrix


def fit_split_latents(
    checkpoint: str | int = DEFAULT_CHECKPOINT,
    split: str = "val",
    observation_mode: str = DEFAULT_FEATURE_OBSERVATION_MODE,
    device: str | torch.device | None = "auto",
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
    anchor_fit_lr: float = 5e-3,
    anchor_fit_init_std: float = 1e-2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    bundle = load_bundle(checkpoint=checkpoint, device=device)
    frame = _split_frame(bundle, split)
    subject_frame = _subject_level_frame(frame, split)
    grouped = model_helpers.grouped_subject_rows(frame)
    rows: List[pd.Series] = []
    features: List[np.ndarray] = []

    print(
        f"[latent] fitting {split} latents from checkpoint {checkpoint}: "
        f"{len(grouped)} subjects, observation_mode={observation_mode}"
    )

    subject_ids = sorted(grouped.keys())
    total_subjects = len(subject_ids)
    for index, subject_id in enumerate(subject_ids, start=1):
        print(f"[latent:{split}] {index}/{total_subjects} subject={subject_id}")
        subject_rows = grouped[subject_id]
        scan_ids = _subject_scan_ids(subject_rows, observation_mode=observation_mode)
        anchor, loss_hist, _ = model_helpers.fit_subject_anchor(
            bundle,
            scan_ids,
            seed=seed,
            num_iterations=int(anchor_fit_steps),
            num_samples=int(anchor_fit_samples),
            lr=float(anchor_fit_lr),
            init_std=float(anchor_fit_init_std),
        )
        row = subject_frame.loc[subject_frame["subject_id"] == subject_id]
        if len(row) != 1:
            raise RuntimeError(f"Unexpected subject row count for subject_id={subject_id}: {len(row)}")
        row = row.iloc[0].copy()
        row["latent_source"] = f"fitted_{observation_mode}_checkpoint_{checkpoint}"
        row["latent_fit_loss_start"] = float(loss_hist[0]) if loss_hist else float("nan")
        row["latent_fit_loss_end"] = float(loss_hist[-1]) if loss_hist else float("nan")
        rows.append(row)
        features.append(anchor.detach().cpu().numpy().reshape(-1))

    feature_matrix = np.stack(features, axis=0) if features else np.zeros((0, int(bundle.specs["CodeLength"])), dtype=np.float32)
    subject_frame = pd.DataFrame(rows).sort_values(["diagnosis", "subject_id"]).reset_index(drop=True)
    return subject_frame, feature_matrix


def prepare_latent_splits(
    checkpoint: str | int = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = "auto",
    observation_mode: str = DEFAULT_FEATURE_OBSERVATION_MODE,
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
    anchor_fit_lr: float = 5e-3,
    anchor_fit_init_std: float = 1e-2,
    seed: int = 0,
) -> Dict[str, Dict[str, object]]:
    train_frame, train_X = load_train_split_latents(checkpoint=checkpoint, device=device)
    val_frame, val_X = fit_split_latents(
        checkpoint=checkpoint,
        split="val",
        observation_mode=observation_mode,
        device=device,
        anchor_fit_steps=anchor_fit_steps,
        anchor_fit_samples=anchor_fit_samples,
        anchor_fit_lr=anchor_fit_lr,
        anchor_fit_init_std=anchor_fit_init_std,
        seed=seed,
    )
    test_frame, test_X = fit_split_latents(
        checkpoint=checkpoint,
        split="test",
        observation_mode=observation_mode,
        device=device,
        anchor_fit_steps=anchor_fit_steps,
        anchor_fit_samples=anchor_fit_samples,
        anchor_fit_lr=anchor_fit_lr,
        anchor_fit_init_std=anchor_fit_init_std,
        seed=seed,
    )

    return {
        "train": {
            "frame": train_frame,
            "X": train_X,
            "y": train_frame["label_ad"].to_numpy(dtype=np.int64),
        },
        "val": {
            "frame": val_frame,
            "X": val_X,
            "y": val_frame["label_ad"].to_numpy(dtype=np.int64),
        },
        "test": {
            "frame": test_frame,
            "X": test_X,
            "y": test_frame["label_ad"].to_numpy(dtype=np.int64),
        },
    }


class LatentBinaryMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor_features(X: np.ndarray, y: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    return X_t, y_t


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return metrics


def _standardize_feature_sets(train_X: np.ndarray, *other_X: np.ndarray) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    train_X_std = scaler.fit_transform(train_X)
    transformed = [train_X_std]
    for X in other_X:
        transformed.append(scaler.transform(X))
    return tuple(transformed)


def train_seeded_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    device: str | torch.device = "auto",
    train_frame: Optional[pd.DataFrame] = None,
    val_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 25,
    threshold: float = 0.5,
) -> Dict[str, object]:
    seed_everything(seed)
    device_obj = model_helpers.determine_device(device)
    train_X_std, val_X_std, test_X_std = _standardize_feature_sets(train_X, val_X, test_X)

    train_X_t, train_y_t = _to_tensor_features(train_X_std, train_y, device_obj)
    val_X_t, val_y_t = _to_tensor_features(val_X_std, val_y, device_obj)
    test_X_t, test_y_t = _to_tensor_features(test_X_std, test_y, device_obj)

    pos_count = float(train_y.sum())
    neg_count = float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device_obj)

    dataset = TensorDataset(train_X_t, train_y_t)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=generator)

    model = LatentBinaryMLP(input_dim=train_X.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_size_actual = int(batch_X.shape[0])
            epoch_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual
        train_loss = epoch_loss / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss = float(criterion(val_logits, val_y_t).item())
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
        val_metrics = compute_binary_metrics(val_y, val_prob, threshold=threshold)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_auc": val_metrics["roc_auc"],
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(train_X_t)).detach().cpu().numpy()
        val_prob = torch.sigmoid(model(val_X_t)).detach().cpu().numpy()
        test_prob = torch.sigmoid(model(test_X_t)).detach().cpu().numpy()

    train_metrics = compute_binary_metrics(train_y, train_prob, threshold=threshold)
    val_metrics = compute_binary_metrics(val_y, val_prob, threshold=threshold)
    test_metrics = compute_binary_metrics(test_y, test_prob, threshold=threshold)

    def _prediction_frame(
        split_name: str,
        frame: Optional[pd.DataFrame],
        y: np.ndarray,
        prob: np.ndarray,
    ) -> pd.DataFrame:
        pred = (prob >= threshold).astype(int)
        payload = {
            "split": split_name,
            "label_ad_true": y.astype(int),
            "label_ad_prob": prob.astype(float),
            "label_ad_pred": pred.astype(int),
        }
        if frame is not None and len(frame) == len(y):
            payload["subject_id"] = frame["subject_id"].astype(str).to_numpy()
            payload["diagnosis"] = frame["diagnosis"].astype(str).to_numpy()
            payload["n_scans"] = frame["n_scans"].to_numpy(dtype=int)
            payload["baseline_age_years"] = frame["baseline_age_years"].to_numpy(dtype=float)
            payload["age_span_years"] = frame["age_span_years"].to_numpy(dtype=float)
            payload["latent_source"] = frame["latent_source"].astype(str).to_numpy()
            payload["latent_fit_loss_end"] = frame["latent_fit_loss_end"].to_numpy(dtype=float)
        return pd.DataFrame(payload)

    return {
        "seed": int(seed),
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "prediction_frames": {
            "train": _prediction_frame("train", train_frame, train_y, train_prob),
            "val": _prediction_frame("val", val_frame, val_y, val_prob),
            "test": _prediction_frame("test", test_frame, test_y, test_prob),
        },
    }


def run_three_seed_latent_classifier(
    checkpoint: str | int = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = "auto",
    observation_mode: str = DEFAULT_FEATURE_OBSERVATION_MODE,
    anchor_fit_steps: int = 50,
    anchor_fit_samples: int = 512,
    anchor_fit_lr: float = 5e-3,
    anchor_fit_init_std: float = 1e-2,
    seeds: Sequence[int] = (0, 1, 2),
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 25,
    threshold: float = 0.5,
) -> Dict[str, object]:
    splits = prepare_latent_splits(
        checkpoint=checkpoint,
        device=device,
        observation_mode=observation_mode,
        anchor_fit_steps=anchor_fit_steps,
        anchor_fit_samples=anchor_fit_samples,
        anchor_fit_lr=anchor_fit_lr,
        anchor_fit_init_std=anchor_fit_init_std,
        seed=0,
    )
    return run_three_seed_latent_classifier_from_splits(
        splits=splits,
        checkpoint=checkpoint,
        observation_mode=observation_mode,
        seeds=seeds,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        threshold=threshold,
        device=device,
    )


def run_three_seed_latent_classifier_from_splits(
    splits: Dict[str, Dict[str, object]],
    checkpoint: str | int = DEFAULT_CHECKPOINT,
    observation_mode: str = DEFAULT_FEATURE_OBSERVATION_MODE,
    seeds: Sequence[int] = (0, 1, 2),
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 25,
    threshold: float = 0.5,
    device: str | torch.device | None = "auto",
) -> Dict[str, object]:

    train_X = splits["train"]["X"]
    train_y = splits["train"]["y"]
    val_X = splits["val"]["X"]
    val_y = splits["val"]["y"]
    test_X = splits["test"]["X"]
    test_y = splits["test"]["y"]

    seed_rows: List[Dict[str, object]] = []
    history_rows: List[pd.DataFrame] = []
    prediction_rows: List[pd.DataFrame] = []
    models: Dict[int, nn.Module] = {}

    for seed in seeds:
        print(f"[classifier] training seed={seed}")
        result = train_seeded_classifier(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            seed=int(seed),
            device=device,
            train_frame=splits["train"]["frame"],
            val_frame=splits["val"]["frame"],
            test_frame=splits["test"]["frame"],
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            threshold=threshold,
        )
        models[int(seed)] = result["model"]
        history = result["history"].copy()
        history["seed"] = int(seed)
        history_rows.append(history)

        for split_name in ("train", "val", "test"):
            frame = result["prediction_frames"][split_name].copy()
            frame["seed"] = int(seed)
            prediction_rows.append(frame)

        row = {
            "seed": int(seed),
            "best_epoch": int(result["best_epoch"]),
            "best_val_loss": float(result["best_val_loss"]),
        }
        for prefix, metrics in (
            ("train", result["train_metrics"]),
            ("val", result["val_metrics"]),
            ("test", result["test_metrics"]),
        ):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = float(value)
        seed_rows.append(row)
        print(
            f"[classifier] seed={seed} done | best_epoch={result['best_epoch']} "
            f"| test_acc={result['test_metrics']['accuracy']:.4f} "
            f"| test_auc={result['test_metrics']['roc_auc']:.4f}"
        )

    seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    prediction_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    metric_cols = [col for col in seed_df.columns if col.startswith("test_") or col.startswith("val_") or col.startswith("train_")]
    summary_rows: List[Dict[str, object]] = []
    for col in metric_cols:
        summary_rows.append(
            {
                "metric": col,
                "mean": float(seed_df[col].mean()),
                "std": float(seed_df[col].std(ddof=0)),
                "min": float(seed_df[col].min()),
                "max": float(seed_df[col].max()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)

    return {
        "checkpoint": str(checkpoint),
        "observation_mode": str(observation_mode),
        "splits": splits,
        "seed_results": seed_df,
        "summary": summary_df,
        "history": history_df,
        "predictions": prediction_df,
        "models": models,
        "seeds": [int(seed) for seed in seeds],
        "feature_dim": int(train_X.shape[1]) if len(train_X.shape) == 2 else None,
    }


def save_latent_classifier_outputs(
    bundle: Dict[str, object],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    seed_df: pd.DataFrame = bundle["seed_results"]
    summary_df: pd.DataFrame = bundle["summary"]
    history_df: pd.DataFrame = bundle["history"]
    prediction_df: pd.DataFrame = bundle["predictions"]
    splits = bundle["splits"]

    seed_df.to_csv(output_dir / "seed_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)

    split_summary = {}
    for split_name in ("train", "val", "test"):
        frame = splits[split_name]["frame"]
        split_summary[split_name] = {
            "subjects": int(len(frame)),
            "diagnosis_counts": {
                str(k): int(v) for k, v in frame["diagnosis"].value_counts().sort_index().items()
            },
            "label_counts": {
                str(k): int(v) for k, v in frame["label_ad"].value_counts().sort_index().items()
            },
        }

    payload = {
        "checkpoint": bundle["checkpoint"],
        "observation_mode": bundle["observation_mode"],
        "seeds": bundle["seeds"],
        "feature_dim": bundle["feature_dim"],
        "split_summary": split_summary,
        "seed_results_csv": str(output_dir / "seed_results.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "history_csv": str(output_dir / "training_history.csv"),
        "predictions_csv": str(output_dir / "predictions.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir
