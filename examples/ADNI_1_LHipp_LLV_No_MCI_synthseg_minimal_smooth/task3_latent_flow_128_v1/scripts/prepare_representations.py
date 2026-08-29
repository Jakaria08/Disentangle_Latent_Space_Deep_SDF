#!/usr/bin/env python3
"""Materialize safe PCA/SpiralNet++/Adaptive 128-D longitudinal archives.

The command writes only below this task's new bulk output root.  Source PCA,
AE checkpoints, AE caches, pair CSVs, and active studies are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=None,
        help="Registry names to export; default is every registered representation.",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Check registry, checkpoints, and a tiny encode/decode batch without writing.")
    return parser.parse_args()


def common_arrays(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    excluded = {
        "visit_pca_150",
        "visit_pca_standardized_150",
        "train_pca_mean_150",
        "train_pca_std_150",
    }
    return {key: np.asarray(value) for key, value in source.items() if key not in excluded}


def pca_codes(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = source["visit_pca_150"].astype(np.float32)[:, : C.LATENT_DIM]
    mean = source["train_pca_mean_150"].astype(np.float32)[: C.LATENT_DIM]
    std = np.maximum(source["train_pca_std_150"].astype(np.float32)[: C.LATENT_DIM], np.float32(1.0e-8))
    return raw, mean, std


@torch.no_grad()
def ae_codes(
    model: torch.nn.Module,
    vertices_mm: np.ndarray,
    mesh_mean: np.ndarray,
    mesh_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    mean = torch.from_numpy(mesh_mean).to(device)
    std = torch.from_numpy(mesh_std).to(device)
    for start in range(0, len(vertices_mm), batch_size):
        vertices = torch.from_numpy(
            np.asarray(vertices_mm[start : start + batch_size], dtype=np.float32).copy()
        ).to(device)
        normalized = (vertices - mean) / std
        output.append(model.encode(normalized).detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(output, axis=0)
    if result.shape != (len(vertices_mm), C.LATENT_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid AE latent array: {result.shape}")
    return result


@torch.no_grad()
def reconstruction_metrics(
    geometry: C.FrozenGeometry,
    standardized: np.ndarray,
    target_mm: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    euclidean: list[np.ndarray] = []
    for start in range(0, len(standardized), batch_size):
        latent = torch.from_numpy(standardized[start : start + batch_size].astype(np.float32)).to(device)
        prediction = geometry.vertices(latent).detach().cpu().numpy()
        target = np.asarray(target_mm[start : start + batch_size], dtype=np.float32)
        delta = prediction - target
        squared.append(np.mean(delta * delta, axis=(1, 2)))
        absolute.append(np.mean(np.abs(delta), axis=(1, 2)))
        euclidean.append(np.linalg.norm(delta, axis=2).mean(axis=1))
    mse = np.concatenate(squared)
    return {
        "scans": int(len(standardized)),
        "coordinate_rmse_mm_mean": float(np.sqrt(mse).mean()),
        "coordinate_mae_mm_mean": float(np.concatenate(absolute).mean()),
        "vertex_euclidean_mm_mean": float(np.concatenate(euclidean).mean()),
    }


def verify_registry(registry: dict[str, Any], names: list[str]) -> None:
    for name in names:
        spec = C.representation_spec(name, registry)
        if spec["kind"] != "pca":
            checkpoint = C.resolve_path(spec["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            actual = C.sha256(checkpoint)
            if actual != spec["checkpoint_sha256"]:
                raise ValueError(f"{name} checkpoint hash mismatch: {actual}")


def dry_run(registry: dict[str, Any], names: list[str], device: torch.device, batch_size: int) -> int:
    verify_registry(registry, names)
    source = C.load_source_archive("train", registry)
    target = C.cached_vertices("train", registry)[:2]
    for name in names:
        spec = C.representation_spec(name, registry)
        if spec["kind"] == "pca":
            raw, mean, std = pca_codes(source)
            archive = common_arrays(source) | {
                "visit_latent_raw_128": raw,
                "visit_latent_standardized_128": ((raw - mean) / std).astype(np.float32),
                "train_latent_mean_128": mean,
                "train_latent_std_128": std,
            }
        else:
            model, _ = C.load_ae_model(name, device, registry)
            mesh_mean, mesh_std = C.ae_normalization(registry)
            raw = ae_codes(model, target, mesh_mean, mesh_std, device, min(batch_size, 2))
            mean = raw.mean(axis=0)
            std = np.maximum(raw.std(axis=0), np.float32(1.0e-8))
            archive = common_arrays(source) | {
                "visit_latent_raw_128": raw,
                "visit_latent_standardized_128": ((raw - mean) / std).astype(np.float32),
                "train_latent_mean_128": mean,
                "train_latent_std_128": std,
            }
        geometry = C.build_geometry(name, archive, device, registry)
        prediction = geometry.vertices(torch.zeros(1, C.LATENT_DIM, device=device))
        if prediction.shape != (1, target.shape[1], 3) or not torch.isfinite(prediction).all():
            raise RuntimeError(f"{name} decoder dry-run failed: {tuple(prediction.shape)}")
        print(f"[dry-run] {name}: finite 128-D encode/decode; decoder trainable parameters={C.parameter_count(geometry)}")
        del geometry
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("DRY RUN PASSED — no files written.")
    return 0


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    registry = C.load_registry()
    requested = args.representations or list(registry["representations"])
    names = list(dict.fromkeys(requested))
    unknown = set(names).difference(registry["representations"])
    if unknown:
        raise ValueError(f"Unknown representations: {sorted(unknown)}")
    device = C.choose_device(args.device)
    if args.dry_run:
        return dry_run(registry, names, device, args.batch_size)

    verify_registry(registry, names)
    sources = {split: C.load_source_archive(split, registry) for split in C.SPLITS}
    vertices = {split: C.cached_vertices(split, registry) for split in C.SPLITS}
    mesh_mean, mesh_std = C.ae_normalization(registry)
    output = C.output_root(registry) / "representations"

    for name in names:
        destination = output / name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite prepared representation: {destination}. "
                "Use a new task/output root when adopting a later checkpoint."
            )
        spec = C.representation_spec(name, registry)
        raw_by_split: dict[str, np.ndarray] = {}
        model = None
        payload = None
        if spec["kind"] == "pca":
            for split in C.SPLITS:
                raw_by_split[split], _, _ = pca_codes(sources[split])
        else:
            model, payload = C.load_ae_model(name, device, registry)
            for split in C.SPLITS:
                raw_by_split[split] = ae_codes(
                    model, vertices[split], mesh_mean, mesh_std, device, args.batch_size
                )
        latent_mean = raw_by_split["train"].mean(axis=0).astype(np.float32)
        latent_std = np.maximum(raw_by_split["train"].std(axis=0), np.float32(1.0e-8)).astype(np.float32)
        archive_by_split: dict[str, dict[str, np.ndarray]] = {}
        latent_split = [int(value) for value in spec.get("expected_latent_split", [C.LATENT_DIM])]
        if sum(latent_split) != C.LATENT_DIM or any(value <= 0 for value in latent_split):
            raise ValueError(f"Invalid latent split for {name}: {latent_split}")
        latent_offsets = np.asarray([0, *np.cumsum(latent_split).tolist()], dtype=np.int64)
        latent_scale_names = [str(value) for value in spec.get("latent_scale_names", ["full"])]
        if len(latent_scale_names) != len(latent_split):
            raise ValueError(f"Latent scale-name mismatch for {name}: {latent_scale_names}")
        for split in C.SPLITS:
            raw = raw_by_split[split].astype(np.float32)
            standardized = ((raw - latent_mean) / latent_std).astype(np.float32)
            archive = common_arrays(sources[split]) | {
                "visit_latent_raw_128": raw,
                "visit_latent_standardized_128": standardized,
                "train_latent_mean_128": latent_mean,
                "train_latent_std_128": latent_std,
                "representation_name": np.asarray(name),
                "representation_kind": np.asarray(spec["kind"]),
                "latent_scale_offsets_128": latent_offsets,
                "latent_scale_names_128": np.asarray(latent_scale_names),
            }
            archive_by_split[split] = archive

        geometry = C.build_geometry(name, archive_by_split["train"], device, registry)
        metrics = {
            split: reconstruction_metrics(
                geometry,
                archive_by_split[split]["visit_latent_standardized_128"],
                vertices[split],
                device,
                args.batch_size,
            )
            for split in C.SPLITS
        }
        destination.mkdir(parents=True, exist_ok=False)
        for split in C.SPLITS:
            C.atomic_npz(destination / f"{split}_subject_sequences_128.npz", archive_by_split[split])
        checkpoint_path = C.resolve_path(spec["checkpoint"]) if "checkpoint" in spec else None
        C.atomic_json(destination / "manifest.json", {
            "representation": name,
            "kind": spec["kind"],
            "latent_dim": C.LATENT_DIM,
            "latent_scale_split": latent_split,
            "latent_scale_offsets": latent_offsets.tolist(),
            "latent_scale_names": latent_scale_names,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": C.sha256(checkpoint_path) if checkpoint_path else None,
            "checkpoint_trial": payload.get("trial") if payload else None,
            "checkpoint_validation_rmse_mm": payload.get("best_val_rmse_mm") if payload else spec.get("source_validation_rmse_mm"),
            "train_only_standardization": True,
            "decoder_frozen": True,
            "decoder_trainable_parameters": C.parameter_count(geometry),
            "split_counts": {split: int(len(archive_by_split[split]["visit_scan_ids"])) for split in C.SPLITS},
            "reconstruction": metrics,
            "source_meshes_modified": False,
        })
        print(json.dumps({"representation": name, "reconstruction": metrics}, indent=2, sort_keys=True))
        del geometry, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prepared = {name: {split: C.load_archive(name, split, registry) for split in C.SPLITS} for name in names}
    for name, archives in prepared.items():
        C.validate_split_isolation(archives)
        print(f"[verified] {name}: safe archives, exact scan order, disjoint subject splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
