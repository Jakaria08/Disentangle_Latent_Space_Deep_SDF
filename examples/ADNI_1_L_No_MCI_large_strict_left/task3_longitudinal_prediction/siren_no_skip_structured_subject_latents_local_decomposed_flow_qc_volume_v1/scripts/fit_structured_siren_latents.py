#!/usr/bin/env python3
"""Fit structured subject/visit SIREN latents for the ADNI flow experiment.

The exported archives intentionally keep the existing direct-flow contract:

    scan_ids: [N]
    latents:  [N, 256]

Internally each scan latent is represented as

    z_struct(i, k) = a_i + B q_{i,k}

where ``a_i`` is a subject anchor, ``B`` is a train-fitted low-rank basis, and
``q_{i,k}`` is a visit coordinate.  The SIREN decoder stays frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_deep_sdf_longitudinal_direct_flow import (  # noqa: E402
    load_frozen_decoder,
    resolve_path,
)


SPLITS = ("train", "val", "test")


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_specs(experiment_dir: Path) -> dict[str, object]:
    path = experiment_dir / "specs.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing specs.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if args.gpus:
        first = next(value.strip() for value in args.gpus.split(",") if value.strip())
        return torch.device(f"cuda:{int(first)}")
    if args.gpu is not None:
        return torch.device(f"cuda:{int(args.gpu)}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_gpu_ids(args: argparse.Namespace, device: torch.device) -> list[int]:
    if args.gpus:
        return [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if device.type == "cuda":
        return [int(device.index or 0)]
    return []


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "split",
        "subject_id",
        "scan_id",
        "diagnosis",
        "label_ad",
        "visit_order",
        "continuous_age_norm",
        "continuous_age_years",
        "sdf_npz_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["scan_id"] = frame["scan_id"].astype(str)
    frame["diagnosis"] = frame["diagnosis"].astype(str)
    frame["label_ad"] = frame["label_ad"].astype(int)
    frame["visit_order"] = frame["visit_order"].astype(int)
    frame["continuous_age_norm"] = frame["continuous_age_norm"].astype(float)
    frame["continuous_age_years"] = frame["continuous_age_years"].astype(float)
    for column in ("left_mesh_volume_mm3", "left_mask_volume_mm3"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(
        ["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)


def load_latent_archive(path: Path, expected_dim: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "scan_ids" not in payload or "latents" not in payload:
            raise ValueError(f"Latent archive lacks scan_ids/latents: {path}")
        scan_ids = payload["scan_ids"].astype(str)
        latents = np.asarray(payload["latents"], dtype=np.float32)
    if latents.ndim != 2 or latents.shape[1] != int(expected_dim):
        raise ValueError(
            f"Expected latent archive [N,{expected_dim}], got {latents.shape} at {path}"
        )
    result: dict[str, np.ndarray] = {}
    for scan_id, latent in zip(scan_ids, latents):
        key = Path(str(scan_id)).stem
        if key in result:
            raise ValueError(f"Duplicate scan ID {key!r} in {path}")
        result[key] = np.asarray(latent, dtype=np.float32)
    return result


def load_raw_latents(
    specs: Mapping[str, object],
    experiment_dir: Path,
    expected_dim: int,
) -> dict[str, dict[str, np.ndarray]]:
    raw_files = specs.get("StructuredLatentRawFiles")
    if raw_files is None:
        raw_files = specs.get("FrozenLatentFiles")
    if not isinstance(raw_files, Mapping):
        raise TypeError("StructuredLatentRawFiles/FrozenLatentFiles must be a mapping")
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        if split not in raw_files:
            raise KeyError(f"Missing raw latent archive for split {split}")
        result[split] = load_latent_archive(
            resolve_path(str(raw_files[split]), experiment_dir),
            expected_dim,
        )
    return result


def load_sdf_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "pos" not in archive or "neg" not in archive:
            raise ValueError(f"SDF archive lacks pos/neg arrays: {path}")
        pos = np.asarray(archive["pos"], dtype=np.float32)
        neg = np.asarray(archive["neg"], dtype=np.float32)
    pos = pos[np.isfinite(pos).all(axis=1)]
    neg = neg[np.isfinite(neg).all(axis=1)]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"SDF archive has empty pos/neg after filtering: {path}")
    return pos, neg


def balanced_sdf_samples(
    path: str | Path,
    count: int,
    rng: np.random.Generator,
    cached: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    if int(count) <= 0:
        raise ValueError("SDF sample count must be positive")
    pos, neg = cached if cached is not None else load_sdf_arrays(path)
    half = int(count) // 2
    pos_index = rng.integers(0, len(pos), size=half, endpoint=False)
    neg_index = rng.integers(0, len(neg), size=int(count) - half, endpoint=False)
    samples = np.concatenate((pos[pos_index], neg[neg_index]), axis=0)
    rng.shuffle(samples)
    return np.asarray(samples, dtype=np.float32)


@dataclass(frozen=True)
class SplitArrays:
    split: str
    frame: pd.DataFrame
    scan_ids: list[str]
    subject_ids: list[str]
    subject_index: np.ndarray
    first_scan_indices: np.ndarray
    raw_latents: np.ndarray
    first_raw_by_subject: np.ndarray
    sdf_paths: list[str]
    times: np.ndarray
    age_years: np.ndarray
    volumes: np.ndarray
    diagnoses: list[str]
    labels: np.ndarray
    adjacent_pairs: np.ndarray
    adjacent_dt: np.ndarray
    adjacent_log_volume_ratio: np.ndarray
    acceleration_triples: np.ndarray


def split_arrays(
    frame: pd.DataFrame,
    split: str,
    raw_latents: Mapping[str, np.ndarray],
    expected_dim: int,
    *,
    limit_subjects: int = 0,
) -> SplitArrays:
    split_frame = frame.loc[frame["split"] == split].copy()
    if limit_subjects > 0:
        keep_subjects = sorted(split_frame["subject_id"].unique())[: int(limit_subjects)]
        split_frame = split_frame.loc[split_frame["subject_id"].isin(keep_subjects)].copy()
    split_frame = split_frame.sort_values(
        ["subject_id", "continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)
    if split_frame.empty:
        raise ValueError(f"No rows for split {split}")
    missing = sorted(set(split_frame["scan_id"]).difference(raw_latents))
    if missing:
        raise KeyError(f"{len(missing)} {split} scans lack raw latents; first={missing[0]}")

    subject_ids = sorted(split_frame["subject_id"].unique())
    subject_to_index = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    subject_index = split_frame["subject_id"].map(subject_to_index).to_numpy(dtype=np.int64)
    raw = np.stack(
        [raw_latents[str(scan_id)] for scan_id in split_frame["scan_id"]],
        axis=0,
    ).astype(np.float32)
    if raw.shape[1] != int(expected_dim):
        raise ValueError(f"Expected raw latent dim {expected_dim}, got {raw.shape[1]}")

    first_scan_indices: list[int] = []
    first_raw_by_subject: list[np.ndarray] = []
    adjacent_pairs: list[tuple[int, int]] = []
    adjacent_dt: list[float] = []
    adjacent_log_volume_ratio: list[float] = []
    triples: list[tuple[int, int, int]] = []
    volumes = (
        split_frame["left_mesh_volume_mm3"].to_numpy(dtype=np.float64)
        if "left_mesh_volume_mm3" in split_frame.columns
        else np.full(len(split_frame), np.nan, dtype=np.float64)
    )
    times = split_frame["continuous_age_norm"].to_numpy(dtype=np.float64)
    age_years = split_frame["continuous_age_years"].to_numpy(dtype=np.float64)

    for subject_id, group in split_frame.groupby("subject_id", sort=True):
        group = group.sort_values(
            ["continuous_age_norm", "visit_order", "scan_id"]
        )
        indices = group.index.to_numpy(dtype=np.int64)
        subject_times = times[indices]
        if any(curr <= prev for prev, curr in zip(subject_times[:-1], subject_times[1:])):
            raise ValueError(f"Non-increasing time in split={split}, subject={subject_id}")
        if group["diagnosis"].nunique() != 1:
            raise ValueError(f"Diagnosis changes in split={split}, subject={subject_id}")
        first = int(indices[0])
        first_scan_indices.append(first)
        first_raw_by_subject.append(raw[first])
        for left, right in zip(indices[:-1], indices[1:]):
            gap = float(age_years[right] - age_years[left])
            adjacent_pairs.append((int(left), int(right)))
            adjacent_dt.append(max(gap, 1.0e-6))
            left_volume = float(volumes[left])
            right_volume = float(volumes[right])
            if (
                math.isfinite(left_volume)
                and math.isfinite(right_volume)
                and left_volume > 0.0
                and right_volume > 0.0
            ):
                adjacent_log_volume_ratio.append(math.log(right_volume / left_volume))
            else:
                adjacent_log_volume_ratio.append(float("nan"))
        for left, middle, right in zip(indices[:-2], indices[1:-1], indices[2:]):
            triples.append((int(left), int(middle), int(right)))

    return SplitArrays(
        split=split,
        frame=split_frame,
        scan_ids=[str(value) for value in split_frame["scan_id"].tolist()],
        subject_ids=subject_ids,
        subject_index=subject_index,
        first_scan_indices=np.asarray(first_scan_indices, dtype=np.int64),
        raw_latents=raw,
        first_raw_by_subject=np.stack(first_raw_by_subject, axis=0).astype(np.float32),
        sdf_paths=[str(value) for value in split_frame["sdf_npz_path"].tolist()],
        times=times.astype(np.float32),
        age_years=age_years.astype(np.float32),
        volumes=volumes.astype(np.float32),
        diagnoses=[str(value) for value in split_frame["diagnosis"].tolist()],
        labels=split_frame["label_ad"].to_numpy(dtype=np.int64),
        adjacent_pairs=np.asarray(adjacent_pairs, dtype=np.int64).reshape(-1, 2),
        adjacent_dt=np.asarray(adjacent_dt, dtype=np.float32),
        adjacent_log_volume_ratio=np.asarray(adjacent_log_volume_ratio, dtype=np.float32),
        acceleration_triples=np.asarray(triples, dtype=np.int64).reshape(-1, 3),
    )


class StructuredScanDataset(Dataset):
    def __init__(
        self,
        arrays: SplitArrays,
        samples_per_scan: int,
        *,
        seed: int,
        load_sdf_into_ram: bool = False,
    ) -> None:
        self.arrays = arrays
        self.samples_per_scan = int(samples_per_scan)
        self.seed = int(seed)
        self.cached_sdf: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
        if load_sdf_into_ram:
            self.cached_sdf = [load_sdf_arrays(path) for path in arrays.sdf_paths]

    def __len__(self) -> int:
        return len(self.arrays.scan_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scan_id = self.arrays.scan_ids[index]
        rng = np.random.default_rng(stable_seed(f"{scan_id}:{np.random.randint(0, 2**31)}", self.seed))
        samples = balanced_sdf_samples(
            self.arrays.sdf_paths[index],
            self.samples_per_scan,
            rng,
            cached=None if self.cached_sdf is None else self.cached_sdf[index],
        )
        return {
            "scan_index": torch.tensor(index, dtype=torch.long),
            "subject_index": torch.tensor(self.arrays.subject_index[index], dtype=torch.long),
            "raw_latent": torch.from_numpy(self.arrays.raw_latents[index]),
            "samples": torch.from_numpy(samples),
        }


class StructuredLatents(nn.Module):
    def __init__(
        self,
        arrays: SplitArrays,
        basis: np.ndarray,
        q_init: np.ndarray,
        *,
        train_basis: bool,
    ) -> None:
        super().__init__()
        if basis.ndim != 2:
            raise ValueError(f"Expected basis [D,R], got {basis.shape}")
        if q_init.shape != (len(arrays.scan_ids), basis.shape[1]):
            raise ValueError(
                f"Expected q_init [{len(arrays.scan_ids)},{basis.shape[1]}], got {q_init.shape}"
            )
        self.latent_dim = int(basis.shape[0])
        self.rank = int(basis.shape[1])
        self.anchors = nn.Parameter(torch.from_numpy(arrays.first_raw_by_subject.copy()))
        self.q = nn.Parameter(torch.from_numpy(q_init.astype(np.float32, copy=True)))
        if train_basis:
            self.basis = nn.Parameter(torch.from_numpy(basis.astype(np.float32, copy=True)))
        else:
            self.register_buffer("basis", torch.from_numpy(basis.astype(np.float32, copy=True)))
        self.register_buffer(
            "scan_subject_index",
            torch.from_numpy(arrays.subject_index.astype(np.int64, copy=True)),
        )
        self.register_buffer(
            "first_scan_indices",
            torch.from_numpy(arrays.first_scan_indices.astype(np.int64, copy=True)),
        )
        self.register_buffer(
            "first_raw_by_subject",
            torch.from_numpy(arrays.first_raw_by_subject.astype(np.float32, copy=True)),
        )
        self.enforce_baseline_zero()

    def enforce_baseline_zero(self) -> None:
        with torch.no_grad():
            self.q[self.first_scan_indices].zero_()

    def forward(
        self,
        scan_index: torch.Tensor,
        subject_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scan_index = scan_index.to(device=self.q.device, dtype=torch.long)
        if subject_index is None:
            subject_index = self.scan_subject_index[scan_index]
        else:
            subject_index = subject_index.to(device=self.q.device, dtype=torch.long)
        return self.anchors[subject_index] + self.q[scan_index] @ self.basis.T

    def all_latents(self) -> torch.Tensor:
        return self.anchors[self.scan_subject_index] + self.q @ self.basis.T


def pca_basis_from_train(arrays: SplitArrays, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    residuals = arrays.raw_latents - arrays.first_raw_by_subject[arrays.subject_index]
    if len(residuals) <= 1:
        raise ValueError("Need more than one scan to initialize structured basis")
    try:
        _u, _s, vt = np.linalg.svd(residuals.astype(np.float64), full_matrices=False)
        components = vt[: min(rank, vt.shape[0])].astype(np.float32)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        components = rng.normal(size=(min(rank, arrays.raw_latents.shape[1]), arrays.raw_latents.shape[1])).astype(np.float32)
    if components.shape[0] < int(rank):
        rng = np.random.default_rng(seed)
        extra = rng.normal(size=(int(rank) - components.shape[0], arrays.raw_latents.shape[1]))
        components = np.concatenate((components, extra.astype(np.float32)), axis=0)
    q_mat, _ = np.linalg.qr(components.T)
    basis = q_mat[:, : int(rank)].astype(np.float32)
    q_init = residuals @ basis
    q_init[arrays.first_scan_indices] = 0.0
    return basis.astype(np.float32), q_init.astype(np.float32)


def q_from_basis(arrays: SplitArrays, basis: np.ndarray) -> np.ndarray:
    residuals = arrays.raw_latents - arrays.first_raw_by_subject[arrays.subject_index]
    q_init = residuals @ basis
    q_init[arrays.first_scan_indices] = 0.0
    return q_init.astype(np.float32)


def decode_sdf_loss(
    decoder: nn.Module,
    latents: torch.Tensor,
    samples: torch.Tensor,
    *,
    clamp_distance: float,
    max_decoder_batch: int,
) -> torch.Tensor:
    if latents.ndim != 2 or samples.ndim != 3 or samples.shape[2] < 4:
        raise ValueError(
            f"Expected latents [B,D] and samples [B,N,>=4], got {tuple(latents.shape)}, {tuple(samples.shape)}"
        )
    batch_size, sample_count, _ = samples.shape
    xyz = samples[:, :, :3].reshape(batch_size * sample_count, 3)
    target = samples[:, :, 3:4].reshape(batch_size * sample_count, 1)
    target = target.clamp(-float(clamp_distance), float(clamp_distance))
    expanded = latents.unsqueeze(1).expand(batch_size, sample_count, latents.shape[1])
    expanded = expanded.reshape(batch_size * sample_count, latents.shape[1])
    total = target.sum() * 0.0
    count = 0
    for start in range(0, int(target.shape[0]), int(max_decoder_batch)):
        stop = min(start + int(max_decoder_batch), int(target.shape[0]))
        pred = decoder(torch.cat((expanded[start:stop], xyz[start:stop]), dim=1))
        pred = pred.clamp(-float(clamp_distance), float(clamp_distance))
        total = total + torch.abs(pred - target[start:stop]).sum()
        count += int(stop - start)
    return total / max(1, count)


def soft_volume_proxy(
    decoder: nn.Module,
    latents: torch.Tensor,
    probe_points: torch.Tensor,
    *,
    temperature: float,
    inside_sdf_sign: float,
    max_decoder_batch: int,
) -> torch.Tensor:
    if latents.numel() == 0:
        return torch.empty(0, device=latents.device, dtype=latents.dtype)
    total_occ = torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype)
    total_count = 0
    points = probe_points.to(device=latents.device, dtype=latents.dtype)
    latent_dim = latents.shape[1]
    chunk_points = max(1, int(max_decoder_batch) // max(1, int(latents.shape[0])))
    for start in range(0, int(points.shape[0]), chunk_points):
        xyz = points[start : start + chunk_points]
        count = int(xyz.shape[0])
        expanded_xyz = xyz.unsqueeze(0).expand(latents.shape[0], count, 3)
        expanded_latents = latents.unsqueeze(1).expand(latents.shape[0], count, latent_dim)
        flat = torch.cat(
            (
                expanded_latents.reshape(-1, latent_dim),
                expanded_xyz.reshape(-1, 3),
            ),
            dim=1,
        )
        pred = decoder(flat).reshape(latents.shape[0], count, -1)
        occ = torch.sigmoid(float(inside_sdf_sign) * pred / float(temperature))
        total_occ = total_occ + occ.mean(dim=2).sum(dim=1)
        total_count += count
    return (total_occ / max(1, total_count)).clamp_min(1.0e-8)


def make_volume_probe_points(
    specs: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    count = int(specs.get("VolumeProbePoints", 4096))
    bounds = np.asarray(specs.get("VolumeProbeBounds", [-1.05, 1.05]), dtype=np.float32)
    if bounds.shape == (2,):
        low = np.full(3, float(bounds[0]), dtype=np.float32)
        high = np.full(3, float(bounds[1]), dtype=np.float32)
    elif bounds.shape == (3, 2):
        low = bounds[:, 0].astype(np.float32)
        high = bounds[:, 1].astype(np.float32)
    else:
        raise ValueError(f"Invalid VolumeProbeBounds shape: {bounds.shape}")
    rng = np.random.default_rng(int(specs.get("VolumeProbeSeed", 12345)))
    points = low[None, :] + rng.random(size=(count, 3), dtype=np.float32) * (high - low)[None, :]
    return torch.from_numpy(points.astype(np.float32)).to(device)


def sample_rows(total: int, count: int, device: torch.device) -> torch.Tensor:
    if total <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    count = min(int(count), int(total))
    return torch.randint(0, int(total), (count,), device=device)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


@torch.no_grad()
def latent_summary(arrays: SplitArrays, model: StructuredLatents) -> dict[str, float]:
    latents = model.all_latents().detach().cpu().numpy()
    raw = arrays.raw_latents.astype(np.float64)
    structured = latents.astype(np.float64)
    diff = structured - raw
    adjacent_norms = []
    adjacent_speeds = []
    cosines = []
    for left, right in arrays.adjacent_pairs:
        delta = structured[right] - structured[left]
        adjacent_norms.append(float(np.linalg.norm(delta)))
        dt = float(max(arrays.times[right] - arrays.times[left], 1.0e-6))
        adjacent_speeds.append(float(np.linalg.norm(delta) / dt))
    deltas = [
        structured[right] - structured[left]
        for left, right in arrays.adjacent_pairs
    ]
    for first, second in zip(deltas[:-1], deltas[1:]):
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom > 0.0:
            cosines.append(float(np.dot(first, second) / denom))
    return {
        "latent_count": int(len(latents)),
        "mean_raw_norm": float(np.linalg.norm(raw, axis=1).mean()),
        "mean_structured_norm": float(np.linalg.norm(structured, axis=1).mean()),
        "mean_raw_to_structured_l2": float(np.linalg.norm(diff, axis=1).mean()),
        "median_raw_to_structured_l2": float(np.median(np.linalg.norm(diff, axis=1))),
        "mean_adjacent_structured_delta_norm": finite_mean(adjacent_norms),
        "mean_adjacent_structured_speed_norm": finite_mean(adjacent_speeds),
        "mean_consecutive_delta_cosine": finite_mean(cosines),
    }


def compute_regularizers(
    *,
    model: StructuredLatents,
    arrays: SplitArrays,
    decoder: nn.Module,
    specs: Mapping[str, object],
    probe_points: Optional[torch.Tensor],
    device: torch.device,
    max_decoder_batch: int,
    pair_batch_size: int,
    triple_batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = model.q.sum() * 0.0
    metrics: dict[str, torch.Tensor] = {}

    anchor_target = model.first_raw_by_subject.to(device=device, dtype=model.anchors.dtype)
    anchor_loss = F.mse_loss(model.anchors, anchor_target)
    metrics["anchor"] = anchor_loss.detach()
    total = float(specs.get("StructuredLatentAnchorLossLambda", 0.01)) * anchor_loss

    basis = model.basis
    eye = torch.eye(model.rank, device=device, dtype=basis.dtype)
    ortho = F.mse_loss(basis.T @ basis, eye)
    metrics["basis_ortho"] = ortho.detach()
    total = total + float(specs.get("StructuredBasisOrthoLambda", 0.001)) * ortho

    pair_indices = sample_rows(len(arrays.adjacent_pairs), pair_batch_size, device)
    if pair_indices.numel() > 0:
        pairs = torch.from_numpy(arrays.adjacent_pairs).to(device=device)[pair_indices]
        dt = torch.from_numpy(arrays.adjacent_dt).to(device=device, dtype=model.q.dtype)[pair_indices]
        left = pairs[:, 0]
        right = pairs[:, 1]
        q_speed = (model.q[right] - model.q[left]) / dt.view(-1, 1).clamp_min(1.0e-6)
        velocity_smooth = torch.mean(q_speed.pow(2))
        metrics["velocity_smooth"] = velocity_smooth.detach()
        total = total + float(specs.get("StructuredLatentVelocitySmoothnessLambda", 0.005)) * velocity_smooth

        if (
            bool(specs.get("StructuredLatentUseRelativeVolumeLoss", True))
            and float(specs.get("StructuredLatentRelativeVolumeLambda", 0.0)) > 0.0
            and probe_points is not None
        ):
            real_log = torch.from_numpy(arrays.adjacent_log_volume_ratio).to(
                device=device,
                dtype=model.q.dtype,
            )[pair_indices]
            valid = torch.isfinite(real_log)
            if bool(valid.any().item()):
                subject_left = model.scan_subject_index[left]
                subject_right = model.scan_subject_index[right]
                left_latents = model(left, subject_left)
                right_latents = model(right, subject_right)
                left_proxy = soft_volume_proxy(
                    decoder,
                    left_latents[valid],
                    probe_points,
                    temperature=float(specs.get("VolumeProbeTemperature", 0.01)),
                    inside_sdf_sign=float(specs.get("VolumeProbeInsideSDFSign", -1.0)),
                    max_decoder_batch=max_decoder_batch,
                )
                right_proxy = soft_volume_proxy(
                    decoder,
                    right_latents[valid],
                    probe_points,
                    temperature=float(specs.get("VolumeProbeTemperature", 0.01)),
                    inside_sdf_sign=float(specs.get("VolumeProbeInsideSDFSign", -1.0)),
                    max_decoder_batch=max_decoder_batch,
                )
                pred_log = torch.log(right_proxy / left_proxy)
                volume = F.smooth_l1_loss(
                    pred_log,
                    real_log[valid],
                    beta=float(specs.get("StructuredLatentRelativeVolumeSmoothL1Beta", 0.02)),
                )
            else:
                volume = zero
            metrics["relative_volume"] = volume.detach()
            total = total + float(specs.get("StructuredLatentRelativeVolumeLambda", 0.10)) * volume
        else:
            metrics["relative_volume"] = zero.detach()
    else:
        metrics["velocity_smooth"] = zero.detach()
        metrics["relative_volume"] = zero.detach()

    triple_indices = sample_rows(len(arrays.acceleration_triples), triple_batch_size, device)
    if triple_indices.numel() > 0:
        triples = torch.from_numpy(arrays.acceleration_triples).to(device=device)[triple_indices]
        left = triples[:, 0]
        middle = triples[:, 1]
        right = triples[:, 2]
        times = torch.from_numpy(arrays.age_years).to(device=device, dtype=model.q.dtype)
        dt1 = (times[middle] - times[left]).view(-1, 1).clamp_min(1.0e-6)
        dt2 = (times[right] - times[middle]).view(-1, 1).clamp_min(1.0e-6)
        v1 = (model.q[middle] - model.q[left]) / dt1
        v2 = (model.q[right] - model.q[middle]) / dt2
        acceleration = F.mse_loss(v2, v1)
        metrics["acceleration"] = acceleration.detach()
        total = total + float(specs.get("StructuredLatentAccelerationLambda", 0.01)) * acceleration
    else:
        metrics["acceleration"] = zero.detach()

    return total, metrics


def optimize_split(
    *,
    split: str,
    arrays: SplitArrays,
    basis: np.ndarray,
    q_init: np.ndarray,
    decoder: nn.Module,
    specs: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
    train_basis: bool,
    epochs: int,
    args: argparse.Namespace,
) -> tuple[StructuredLatents, list[dict[str, object]]]:
    model = StructuredLatents(arrays, basis, q_init, train_basis=train_basis).to(device)
    dataset = StructuredScanDataset(
        arrays,
        samples_per_scan=int(args.samples_per_scan),
        seed=int(args.seed),
        load_sdf_into_ram=bool(args.load_sdf_into_ram),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_scans),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    probe_points = (
        make_volume_probe_points(specs, device)
        if bool(specs.get("StructuredLatentUseRelativeVolumeLoss", True))
        and float(specs.get("StructuredLatentRelativeVolumeLambda", 0.0)) > 0.0
        else None
    )
    log_rows: list[dict[str, object]] = []
    best_value = math.inf
    best_state: Optional[dict[str, torch.Tensor]] = None
    patience = int(args.early_stop_patience)
    no_improvement = 0
    started = time.time()
    decoder.eval()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        sums: dict[str, float] = {
            "total": 0.0,
            "sdf": 0.0,
            "raw": 0.0,
            "anchor": 0.0,
            "basis_ortho": 0.0,
            "velocity_smooth": 0.0,
            "acceleration": 0.0,
            "relative_volume": 0.0,
        }
        count = 0
        for batch_id, batch in enumerate(loader, start=1):
            scan_index = batch["scan_index"].to(device=device, dtype=torch.long)
            subject_index = batch["subject_index"].to(device=device, dtype=torch.long)
            raw_latent = batch["raw_latent"].to(device=device, dtype=torch.float32)
            samples = batch["samples"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            structured = model(scan_index, subject_index)
            sdf = decode_sdf_loss(
                decoder,
                structured,
                samples,
                clamp_distance=float(specs.get("ClampingDistance", 0.1)),
                max_decoder_batch=int(args.max_decoder_batch),
            )
            raw = F.mse_loss(structured, raw_latent)
            regularizer, reg_metrics = compute_regularizers(
                model=model,
                arrays=arrays,
                decoder=decoder,
                specs=specs,
                probe_points=probe_points,
                device=device,
                max_decoder_batch=int(args.max_decoder_batch),
                pair_batch_size=int(args.pair_batch_size),
                triple_batch_size=int(args.triple_batch_size),
            )
            total = (
                float(specs.get("StructuredLatentSDFLossLambda", 1.0)) * sdf
                + float(specs.get("StructuredLatentRawLossLambda", 0.05)) * raw
                + regularizer
            )
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(args.gradient_clip_norm),
            )
            optimizer.step()
            model.enforce_baseline_zero()

            sums["total"] += float(total.detach().cpu())
            sums["sdf"] += float(sdf.detach().cpu())
            sums["raw"] += float(raw.detach().cpu())
            for key in ("anchor", "basis_ortho", "velocity_smooth", "acceleration", "relative_volume"):
                sums[key] += float(reg_metrics[key].detach().cpu())
            count += 1
            if args.smoke_batches > 0 and batch_id >= int(args.smoke_batches):
                break

        row: dict[str, object] = {
            "split": split,
            "epoch": epoch,
            "elapsed_sec": round(time.time() - started, 3),
            "train_basis": bool(train_basis),
            "gradient_norm": float(grad_norm.detach().cpu())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm),
        }
        for key, value in sums.items():
            row[key] = value / max(1, count)
        log_rows.append(row)
        value = float(row["total"])
        print(
            f"[{split}] epoch {epoch:04d}/{epochs} total={value:.6f} "
            f"sdf={float(row['sdf']):.6f} raw={float(row['raw']):.6f} "
            f"vol={float(row['relative_volume']):.6f}"
        )
        if value < best_value - float(args.early_stop_min_delta):
            best_value = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if args.smoke_batches > 0:
            break
        if patience > 0 and no_improvement >= patience:
            print(f"[{split}] early stop after epoch {epoch}; best_total={best_value:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split}_optimization_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    return model, log_rows


@torch.no_grad()
def export_split(
    arrays: SplitArrays,
    model: StructuredLatents,
    output_latent_path: Path,
    output_fit_dir: Path,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    latents = model.all_latents().detach().cpu().numpy().astype(np.float32)
    output_latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_latent_path,
        scan_ids=np.asarray(arrays.scan_ids),
        latents=latents,
    )
    q = model.q.detach().cpu().numpy().astype(np.float32)
    anchors = model.anchors.detach().cpu().numpy().astype(np.float32)
    rows: list[dict[str, object]] = []
    for index, scan_id in enumerate(arrays.scan_ids):
        rows.append(
            {
                "split": arrays.split,
                "scan_id": scan_id,
                "subject_id": arrays.frame.iloc[index]["subject_id"],
                "diagnosis": arrays.diagnoses[index],
                "label_ad": int(arrays.labels[index]),
                "visit_order": int(arrays.frame.iloc[index]["visit_order"]),
                "continuous_age_norm": float(arrays.times[index]),
                "continuous_age_years": float(arrays.age_years[index]),
                "raw_latent_norm": float(np.linalg.norm(arrays.raw_latents[index])),
                "structured_latent_norm": float(np.linalg.norm(latents[index])),
                "raw_to_structured_l2": float(
                    np.linalg.norm(latents[index] - arrays.raw_latents[index])
                ),
                "q_norm": float(np.linalg.norm(q[index])),
                "anchor_norm": float(np.linalg.norm(anchors[arrays.subject_index[index]])),
            }
        )
    q_path = output_fit_dir / f"{arrays.split}_visit_coordinates.npy"
    np.save(q_path, q)
    return latents, rows


def write_visit_coordinate_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_model_payload(
    path: Path,
    *,
    split: str,
    arrays: SplitArrays,
    model: StructuredLatents,
    train_basis: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": split,
            "train_basis": bool(train_basis),
            "scan_ids": arrays.scan_ids,
            "subject_ids": arrays.subject_ids,
            "subject_index": arrays.subject_index,
            "first_scan_indices": arrays.first_scan_indices,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        },
        path,
    )


def validate_inputs(
    *,
    specs: Mapping[str, object],
    experiment_dir: Path,
    frame: pd.DataFrame,
    raw_latents: Mapping[str, Mapping[str, np.ndarray]],
    latent_dim: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "experiment_dir": str(experiment_dir),
        "latent_dim": int(latent_dim),
        "metadata_rows": int(len(frame)),
        "metadata_subjects": int(frame["subject_id"].nunique()),
        "splits": {},
    }
    for split in SPLITS:
        split_frame = frame.loc[frame["split"] == split]
        scan_ids = set(split_frame["scan_id"].astype(str))
        latent_ids = set(raw_latents[split])
        missing = sorted(scan_ids.difference(latent_ids))
        extras = sorted(latent_ids.difference(scan_ids))
        missing_sdf = [
            str(path)
            for path in split_frame["sdf_npz_path"]
            if not Path(str(path)).is_file()
        ]
        report["splits"][split] = {
            "scans": int(len(split_frame)),
            "subjects": int(split_frame["subject_id"].nunique()),
            "diagnosis_subjects": {
                key: int(value)
                for key, value in split_frame.groupby("diagnosis")["subject_id"].nunique().items()
            },
            "missing_latents": int(len(missing)),
            "extra_latents": int(len(extras)),
            "missing_sdf_files": int(len(missing_sdf)),
            "first_missing_latent": missing[0] if missing else None,
            "first_extra_latent": extras[0] if extras else None,
            "first_missing_sdf": missing_sdf[0] if missing_sdf else None,
        }
    output_report = experiment_dir / "structured_latent_fit" / "input_validation_report.json"
    write_json(output_report, report)
    if any(
        entry["missing_latents"] or entry["extra_latents"] or entry["missing_sdf_files"]
        for entry in report["splits"].values()  # type: ignore[union-attr]
    ):
        raise RuntimeError(f"Input validation failed; see {output_report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit structured subject/visit latents and export direct-flow NPZ archives."
    )
    parser.add_argument(
        "--experiment",
        default=str(EXPERIMENT_DIR),
        help="Experiment directory containing specs.json.",
    )
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--gpu", type=int, default=None, help="Single CUDA device index.")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Optional comma-separated CUDA ids for DataParallel decoder, e.g. 0,1,2.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Train-basis epochs.")
    parser.add_argument(
        "--inference-epochs",
        type=int,
        default=None,
        help="Val/test anchor/q inference epochs with basis frozen.",
    )
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--batch-scans", type=int, default=None)
    parser.add_argument("--samples-per-scan", type=int, default=None)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--triple-batch-size", type=int, default=None)
    parser.add_argument("--max-decoder-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit-subjects-per-split", type=int, default=0)
    parser.add_argument(
        "--load-sdf-into-ram",
        action="store_true",
        help="Cache SDF arrays in RAM for repeated epochs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check paths/metadata/latent contract without fitting.",
    )
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=0,
        help="Run this many batches per split and write outputs for a quick smoke test.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Optional suffix for output directories.  If omitted, smoke/subset "
            "runs use 'smoke' and full runs use the production latents directory."
        ),
    )
    return parser.parse_args()


def apply_spec_defaults(args: argparse.Namespace, specs: Mapping[str, object]) -> None:
    defaults = {
        "epochs": int(specs.get("StructuredLatentFitEpochs", 120)),
        "inference_epochs": int(specs.get("StructuredLatentInferenceEpochs", 80)),
        "rank": int(specs.get("StructuredRank", 32)),
        "batch_scans": int(specs.get("StructuredLatentFitBatchScans", 12)),
        "samples_per_scan": int(specs.get("StructuredLatentFitSamplesPerScan", 2048)),
        "pair_batch_size": int(specs.get("StructuredLatentFitPairBatchSize", 48)),
        "triple_batch_size": int(specs.get("StructuredLatentFitTripleBatchSize", 48)),
        "max_decoder_batch": int(specs.get("StructuredLatentFitMaxDecoderBatch", 65536)),
        "learning_rate": float(specs.get("StructuredLatentFitLearningRate", 0.001)),
        "weight_decay": float(specs.get("StructuredLatentFitWeightDecay", 0.0)),
        "gradient_clip_norm": float(specs.get("StructuredLatentFitGradientClipNorm", 1.0)),
        "early_stop_patience": int(specs.get("StructuredLatentFitEarlyStoppingPatience", 25)),
        "early_stop_min_delta": float(specs.get("StructuredLatentFitEarlyStoppingMinDelta", 1.0e-5)),
        "workers": int(specs.get("StructuredLatentFitWorkers", 4)),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment).resolve()
    specs = read_specs(experiment_dir)
    apply_spec_defaults(args, specs)
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    latent_dim = int(specs["CodeLength"])
    metadata_path = resolve_path(str(specs["LongitudinalMetadataFile"]), experiment_dir)
    frame = load_metadata(metadata_path)
    raw_latents = load_raw_latents(specs, experiment_dir, latent_dim)
    validation_report = validate_inputs(
        specs=specs,
        experiment_dir=experiment_dir,
        frame=frame,
        raw_latents=raw_latents,
        latent_dim=latent_dim,
    )
    print(json.dumps(validation_report, indent=2, sort_keys=True))
    if args.validate_only:
        print("Validation-only completed; no latents were fitted or exported.")
        return 0

    device = choose_device(args)
    decoder, decoder_epoch = load_frozen_decoder(specs, experiment_dir, device)
    gpu_ids = parse_gpu_ids(args, device)
    if len(gpu_ids) > 1:
        decoder = torch.nn.DataParallel(decoder, device_ids=gpu_ids)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    decoder.eval()

    output_suffix = args.output_suffix
    if output_suffix is None and (
        int(args.smoke_batches) > 0 or int(args.limit_subjects_per_split) > 0
    ):
        output_suffix = "smoke"
    if output_suffix:
        fit_dir = experiment_dir / f"structured_latent_fit_{output_suffix}"
        latent_dir = experiment_dir / f"latents_{output_suffix}"
    else:
        fit_dir = experiment_dir / "structured_latent_fit"
        latent_dir = experiment_dir / "latents"
    fit_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    arrays_by_split = {
        split: split_arrays(
            frame,
            split,
            raw_latents[split],
            latent_dim,
            limit_subjects=int(args.limit_subjects_per_split),
        )
        for split in SPLITS
    }

    train_arrays = arrays_by_split["train"]
    basis, train_q = pca_basis_from_train(train_arrays, int(args.rank), seed)
    all_visit_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    all_fit_logs: list[dict[str, object]] = []
    models: dict[str, StructuredLatents] = {}

    train_model, train_log = optimize_split(
        split="train",
        arrays=train_arrays,
        basis=basis,
        q_init=train_q,
        decoder=decoder,
        specs=specs,
        output_dir=fit_dir,
        device=device,
        train_basis=True,
        epochs=int(args.epochs),
        args=args,
    )
    models["train"] = train_model
    save_model_payload(
        fit_dir / "train_structured_latents.pth",
        split="train",
        arrays=train_arrays,
        model=train_model,
        train_basis=True,
    )
    basis = train_model.basis.detach().cpu().numpy().astype(np.float32)
    np.save(fit_dir / "shared_basis.npy", basis)
    all_fit_logs.extend(train_log)

    for split in ("val", "test"):
        arrays = arrays_by_split[split]
        q_init = q_from_basis(arrays, basis)
        model, log_rows = optimize_split(
            split=split,
            arrays=arrays,
            basis=basis,
            q_init=q_init,
            decoder=decoder,
            specs=specs,
            output_dir=fit_dir,
            device=device,
            train_basis=False,
            epochs=int(args.inference_epochs),
            args=args,
        )
        models[split] = model
        save_model_payload(
            fit_dir / f"{split}_structured_latents.pth",
            split=split,
            arrays=arrays,
            model=model,
            train_basis=False,
        )
        all_fit_logs.extend(log_rows)

    subject_anchor_rows: list[dict[str, object]] = []
    for split in SPLITS:
        arrays = arrays_by_split[split]
        model = models[split]
        _latents, visit_rows = export_split(
            arrays,
            model,
            latent_dir / f"{split}_latents.npz",
            fit_dir,
        )
        all_visit_rows.extend(visit_rows)
        summary = latent_summary(arrays, model)
        summary["split"] = split
        all_summary_rows.append(summary)
        anchors = model.anchors.detach().cpu().numpy()
        for subject_index, subject_id in enumerate(arrays.subject_ids):
            subject_anchor_rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "anchor_norm": float(np.linalg.norm(anchors[subject_index])),
                    "scan_count": int(np.sum(arrays.subject_index == subject_index)),
                }
            )

    write_visit_coordinate_csv(fit_dir / "visit_coordinates.csv", all_visit_rows)
    write_visit_coordinate_csv(fit_dir / "subject_anchors.csv", subject_anchor_rows)
    with (fit_dir / "fit_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)
    with (fit_dir / "combined_optimization_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_fit_logs[0].keys()))
        writer.writeheader()
        writer.writerows(all_fit_logs)
    write_json(
        fit_dir / "fit_summary.json",
        {
            "experiment_dir": str(experiment_dir),
            "decoder_epoch": int(decoder_epoch),
            "rank": int(args.rank),
            "train_basis_split": "train",
            "splits": all_summary_rows,
            "raw_latent_source": specs.get("StructuredLatentRawFiles"),
            "exported_latents": {
                split: str((latent_dir / f"{split}_latents.npz").resolve())
                for split in SPLITS
            },
            "smoke_batches": int(args.smoke_batches),
        },
    )
    print(f"Structured latent fitting complete: {fit_dir}")
    print(f"Exported latent archives: {latent_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
