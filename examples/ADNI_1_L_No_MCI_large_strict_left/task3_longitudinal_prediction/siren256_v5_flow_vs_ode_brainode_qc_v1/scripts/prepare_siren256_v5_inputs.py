#!/usr/bin/env python3
"""Create the small, auditable point-1 input contract without copying source data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from siren256_common import decode_sdf, load_config, load_frozen_decoder, read_json, resolve, root_dir, sample_sdf, sha256, write_json


def robust_slope(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Huber IRLS slope; one equal-weight velocity vector per subject."""
    centered = times.astype(np.float64) - float(np.mean(times))
    design = np.column_stack((np.ones(len(times)), centered))
    coefficients = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)[0]
    for _ in range(12):
        residual = values - design @ coefficients
        magnitude = np.sqrt(np.mean(residual**2, axis=1))
        scale = max(float(np.median(np.abs(magnitude - np.median(magnitude)))) * 1.4826, 1.0e-8)
        cutoff = 1.345 * scale
        weight = np.minimum(1.0, cutoff / np.maximum(magnitude, 1.0e-12))
        weighted = design * weight[:, None]
        coefficients = np.linalg.solve(design.T @ weighted + 1.0e-10 * np.eye(2), weighted.T @ values)
    return coefficients[1].astype(np.float32)


def forward_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (split, subject_id), visits in frame.groupby(["split", "subject_id"], sort=True):
        visits = visits.sort_values(["continuous_age_norm", "visit_order", "scan_id"]).reset_index(drop=True)
        if len(visits) < 2 or visits.label_ad.nunique() != 1 or np.any(np.diff(visits.continuous_age_norm) <= 0):
            raise ValueError(f"Invalid stable chronological trajectory: {subject_id}")
        for source_index in range(len(visits) - 1):
            for target_index in range(source_index + 1, len(visits)):
                source, target = visits.iloc[source_index], visits.iloc[target_index]
                middle = (source_index + target_index) // 2
                has_middle = target_index - source_index > 1
                records.append({
                    "split": str(split), "subject_id": str(subject_id), "diagnosis": str(source.diagnosis), "label_ad": int(source.label_ad),
                    "source_cache_index": int(source.cache_index), "target_cache_index": int(target.cache_index),
                    "source_scan_id": str(source.scan_id), "target_scan_id": str(target.scan_id),
                    "source_time": float(source.continuous_age_norm), "target_time": float(target.continuous_age_norm),
                    "source_age_years": float(source.continuous_age_years), "target_age_years": float(target.continuous_age_years),
                    "gap_years": float(target.continuous_age_years - source.continuous_age_years),
                    "source_visit_order": int(source.visit_order), "target_visit_order": int(target.visit_order),
                    "is_adjacent": bool(target_index - source_index == 1), "is_first_last": bool(source_index == 0 and target_index == len(visits) - 1),
                    "observed_cache_index": int(visits.iloc[middle].cache_index) if has_middle else -1,
                    "observed_time": float(visits.iloc[middle].continuous_age_norm) if has_middle else float((source.continuous_age_norm + target.continuous_age_norm) / 2.0),
                    "is_reverse": False,
                })
    return pd.DataFrame.from_records(records)


def backward_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    reverse = pairs.copy()
    for left, right in (("source_cache_index", "target_cache_index"), ("source_scan_id", "target_scan_id"), ("source_time", "target_time"), ("source_age_years", "target_age_years"), ("source_visit_order", "target_visit_order")):
        reverse[left], reverse[right] = pairs[right].to_numpy(), pairs[left].to_numpy()
    reverse["gap_years"] = -pairs.gap_years.to_numpy()
    reverse["is_reverse"] = True
    return reverse


def write_sequences(frame: pd.DataFrame, destination: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ids: list[str] = []
        offsets, indices, times, labels = [0], [], [], []
        for subject_id, visits in frame.loc[frame.split == split].groupby("subject_id", sort=True):
            visits = visits.sort_values(["continuous_age_norm", "visit_order", "scan_id"])
            ids.append(str(subject_id))
            indices.extend(visits.cache_index.astype(np.int64).tolist())
            times.extend(visits.continuous_age_norm.astype(np.float32).tolist())
            labels.append(int(visits.label_ad.iloc[0]))
            offsets.append(len(indices))
        np.savez_compressed(destination / f"subject_sequences_{split}.npz", subject_ids=np.asarray(ids, dtype="U"), offsets=np.asarray(offsets, dtype=np.int64), cache_indices=np.asarray(indices, dtype=np.int64), times=np.asarray(times, dtype=np.float32), labels=np.asarray(labels, dtype=np.int64))
        counts[split] = len(ids)
    return counts


def fit_basis(frame: pd.DataFrame, cache: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = frame.loc[frame.split == "train"].copy()
    latents = cache["latents"]
    slopes: list[np.ndarray] = []
    for _, visits in train.groupby("subject_id", sort=True):
        visits = visits.sort_values(["continuous_age_norm", "visit_order", "scan_id"])
        slopes.append(robust_slope(visits.continuous_age_norm.to_numpy(), latents[visits.cache_index.to_numpy(dtype=int)]))
    adjacent = forward_pairs(train)
    adjacent = adjacent.loc[adjacent.is_adjacent]
    velocity_by_label: list[np.ndarray] = []
    class_counts: dict[str, dict[str, int]] = {}
    subject_velocities: dict[int, list[np.ndarray]] = {}
    for label in (0, 1):
        pairs = adjacent.loc[adjacent.label_ad == label].reset_index(drop=True)
        subject_rows = []
        for _, subject_pairs in pairs.groupby("subject_id", sort=True):
            delta = (subject_pairs.target_time.to_numpy() - subject_pairs.source_time.to_numpy())[:, None]
            velocity = (latents[subject_pairs.target_cache_index.to_numpy(dtype=int)] - latents[subject_pairs.source_cache_index.to_numpy(dtype=int)]) / delta
            # Every subject contributes exactly one robust mean adjacent velocity.
            subject_rows.append(np.median(velocity, axis=0))
        subject_velocities[label] = subject_rows
        class_counts["CN" if label == 0 else "AD"] = {"adjacent_pairs": int(len(pairs)), "subjects": int(len(subject_rows))}
    maximum_subjects = max(len(values) for values in subject_velocities.values())
    for label in (0, 1):
        rows = np.stack(subject_velocities[label]).astype(np.float64)
        # Equal total CN/AD contribution while retaining equal weight within
        # each diagnosis.  Weighted SVD avoids duplicate-pair over-weighting.
        velocity_by_label.append(rows * np.sqrt(float(maximum_subjects) / len(rows)))
    motions = np.concatenate((np.stack(slopes), *velocity_by_label), axis=0).astype(np.float64)
    _, singular_values, vh = np.linalg.svd(motions, full_matrices=False)
    rank = int(config["VelocityRank"])
    velocity_basis = vh[:rank].T.astype(np.float32)
    train_latents = latents[train.cache_index.to_numpy(dtype=int)].astype(np.float64)
    feature_mean = train_latents.mean(axis=0, keepdims=True)
    _, feature_singular, feature_vh = np.linalg.svd(train_latents - feature_mean, full_matrices=False)
    feature_components = feature_vh[: int(config["FeatureDimensions"])].astype(np.float32)
    feature_scores = (train_latents - feature_mean) @ feature_components.T
    feature_scale = np.maximum(feature_scores.std(axis=0, keepdims=True), 1.0e-6).astype(np.float32)
    report = {"fit_split": "train", "rank": rank, "feature_dimensions": int(config["FeatureDimensions"]), "subject_slopes": len(slopes), "adjacent_velocity_counts": class_counts, "adjacent_velocity_weighting": "one median adjacent velocity per subject; diagnosis totals equalized by sqrt(max_subjects/n_subjects) before SVD", "balanced_velocity_subject_rows_per_diagnosis": maximum_subjects, "velocity_explained_energy": float(np.sum(singular_values[:rank] ** 2) / np.sum(singular_values ** 2)), "feature_explained_energy": float(np.sum(feature_singular[: int(config["FeatureDimensions"])] ** 2) / np.sum(feature_singular ** 2))}
    return {"velocity_basis": velocity_basis, "velocity_singular_values": singular_values.astype(np.float32), "feature_mean": feature_mean.astype(np.float32), "feature_components": feature_components, "feature_scale": feature_scale, "feature_singular_values": feature_singular.astype(np.float32)}, report


def loss_scales(frame: pd.DataFrame, cache: dict[str, np.ndarray], pairs: pd.DataFrame, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    train = pairs.loc[pairs.split == "train"].reset_index(drop=True)
    rng = np.random.default_rng(int(config["Seed"]) + 7)
    if len(train) > int(config["LossScalePairCap"]):
        train = train.iloc[np.sort(rng.choice(len(train), int(config["LossScalePairCap"]), replace=False))].reset_index(drop=True)
    source, target = train.source_cache_index.to_numpy(dtype=int), train.target_cache_index.to_numpy(dtype=int)
    source_latents, target_latents = cache["latents"][source], cache["latents"][target]
    latent = np.mean((target_latents - source_latents) ** 2, axis=1)
    target_displacement = ((cache["vertices"][target] - cache["vertices"][source]) * cache["normals"][source]).sum(axis=-1)
    volume_ratio = np.log(np.maximum(cache["volumes"][target], 1e-8) / np.maximum(cache["volumes"][source], 1e-8))
    rate = volume_ratio / np.maximum(np.abs(train.gap_years.to_numpy()), 0.05)
    slopes = []
    for _, visits in frame.loc[frame.split == "train"].groupby("subject_id"):
        visits = visits.sort_values(["continuous_age_norm", "visit_order"])
        times = visits.continuous_age_norm.to_numpy()
        logs = np.log(cache["volumes"][visits.cache_index.to_numpy(dtype=int)].clip(1e-8))
        slopes.append(np.polyfit(times, logs, 1)[0])
    decoder = load_frozen_decoder(config, device)
    sdf_errors = []
    count = int(config["SdfScaleSamples"])
    for start in range(0, len(train), 8):
        rows = train.iloc[start : start + 8]
        samples = np.stack([sample_sdf(str(frame.loc[frame.cache_index == int(index), "sdf_npz_path"].iloc[0]), count, int(index)) for index in rows.target_cache_index])
        with torch.no_grad():
            predicted = decode_sdf(decoder, torch.from_numpy(cache["latents"][rows.source_cache_index.to_numpy(dtype=int)]).to(device), torch.from_numpy(samples[:, :, :3]).to(device)).cpu().numpy()
        sdf_errors.extend(np.mean(np.abs(predicted - samples[:, :, 3]), axis=1).tolist())
    def median(values: np.ndarray | list[float]) -> float:
        return max(float(np.median(np.abs(values))), 1.0e-8)
    return {"latent": median(latent), "target_sdf": median(sdf_errors), "registered_normal": median(target_displacement), "volume": median(volume_ratio), "rate": median(rate), "slope": median(slopes), "scale_pair_count": int(len(train)), "sdf_scale_samples": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root_dir() / "configs" / "common_qc_siren256.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="Deliberately overwrite only generated metadata files.")
    args = parser.parse_args()
    config, root = load_config(args.config), root_dir()
    outputs = [root / "metadata" / name for name in ("input_contract.json", "scan_manifest.csv", "train_only_transport_basis.npz", "transport_basis_report.json", "loss_scales.json", "pair_records_train.csv", "pair_records_val.csv", "pair_records_test.csv", "pair_records_train_backward.csv", "subject_sequences_train.npz", "subject_sequences_val.npz", "subject_sequences_test.npz")]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Refusing to overwrite prepared inputs. Use --force deliberately:\n" + "\n".join(existing))
    metadata_path, cache_path = resolve(config["MetadataCsv"]), resolve(config["RegisteredMeshCache"])
    frame = pd.read_csv(metadata_path)
    cache_archive = np.load(cache_path, allow_pickle=False)
    cache = {key: cache_archive[key] for key in cache_archive.files}
    cache_indices = {str(scan): index for index, scan in enumerate(cache["scan_ids"].astype(str))}
    frame["scan_id"] = frame.scan_id.astype(str)
    if set(frame.scan_id) != set(cache_indices):
        raise ValueError("QC metadata and registered mesh cache scan IDs differ.")
    frame["cache_index"] = frame.scan_id.map(cache_indices).astype(int)
    frame["subject_id"] = frame.subject_id.astype(str)
    for split, archive_value in config["LatentArchives"].items():
        archive = np.load(resolve(archive_value), allow_pickle=False)
        ids, latents = archive["scan_ids"].astype(str), archive["latents"]
        split_indices = frame.loc[frame.split == split, "cache_index"].to_numpy(dtype=int)
        if set(ids) != set(frame.loc[frame.split == split, "scan_id"]):
            raise ValueError(f"{split} latent archive is not exactly the QC split.")
        mapped = {scan: latent for scan, latent in zip(ids, latents)}
        if not np.allclose(np.stack([mapped[scan] for scan in frame.loc[frame.split == split, "scan_id"]]), cache["latents"][split_indices], rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{split} source cache does not contain the exact frozen latent archive.")
    frame = frame.sort_values(["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]).reset_index(drop=True)
    pairs = forward_pairs(frame)
    root.joinpath("metadata").mkdir(exist_ok=True)
    frame.to_csv(root / "metadata" / "scan_manifest.csv", index=False)
    pair_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_pairs = pairs.loc[pairs.split == split].reset_index(drop=True)
        split_pairs.to_csv(root / "metadata" / f"pair_records_{split}.csv", index=False)
        pair_counts[split] = len(split_pairs)
    backward_pairs(pairs.loc[pairs.split == "train"]).to_csv(root / "metadata" / "pair_records_train_backward.csv", index=False)
    sequence_counts = write_sequences(frame, root / "metadata")
    basis, report = fit_basis(frame, cache, config)
    np.savez_compressed(root / "metadata" / "train_only_transport_basis.npz", **basis)
    write_json(root / "metadata" / "transport_basis_report.json", report)
    scales = loss_scales(frame, cache, pairs, config, torch.device(args.device))
    write_json(root / "metadata" / "loss_scales.json", scales)
    inputs = {"metadata": metadata_path, "registered_mesh_cache": cache_path, "decoder": resolve(config["DecoderCheckpoint"]), **{f"latents_{key}": resolve(value) for key, value in config["LatentArchives"].items()}}
    write_json(root / "metadata" / "input_contract.json", {"experiment": config["ExperimentName"], "immutable_source_paths": {key: str(value) for key, value in inputs.items()}, "sha256": {key: sha256(value) for key, value in inputs.items()}, "latent_dimension": int(config["LatentSize"]), "scan_counts": {key: int(value) for key, value in frame.groupby("split").size().items()}, "subject_counts": {key: int(value) for key, value in frame.groupby("split").subject_id.nunique().items()}, "forward_pair_counts": pair_counts, "backward_train_pairs": int(len(pairs.loc[pairs.split == "train"])), "sequence_subject_counts": sequence_counts, "registered_vertex_count": int(cache["vertices"].shape[1]), "registered_face_count": int(cache["faces"].shape[0]), "basis_fit_split": "train_only", "training_mesh_access": "train_val_only_direct_registered_OBJ; all_split_registered_cache_is_evaluator_only", "training_must_not_load_test": True})
    print(f"Prepared {len(frame)} scans; forward pairs={pair_counts}; train-only rank={config['VelocityRank']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
