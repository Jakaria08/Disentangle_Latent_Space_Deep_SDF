#!/usr/bin/env python3
"""Build the self-contained mesh cache and train-only bases for v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inr_anchored_common import (
    ensure_paths_exist, experiment_dir, load_frozen_base_flow,
    load_metadata_and_latents, make_forward_pair_table, mesh_normals_and_areas,
    read_obj_mesh, resolve_path, reverse_pair_table, write_json,
)


def _fit_train_only_bases(
    config: dict, frame: pd.DataFrame, latents: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    train_indices = frame.loc[frame.split == "train", "global_index"].to_numpy(dtype=int)
    train_latents = latents[train_indices].astype(np.float64)
    feature_mean = train_latents.mean(axis=0, keepdims=True)
    _, feature_singular, feature_vh = np.linalg.svd(train_latents - feature_mean, full_matrices=False)
    feature_dim = min(int(config["FeaturePcaDimensions"]), feature_vh.shape[0])
    feature_components = feature_vh[:feature_dim].astype(np.float32)
    adjacent = make_forward_pair_table(frame, "adjacent")
    ad_pairs = adjacent[(adjacent.split == "train") & (adjacent.label_ad == 1)].reset_index(drop=True)
    if len(ad_pairs) < 2:
        raise ValueError("Need at least two train AD adjacent pairs for the residual basis.")
    base = load_frozen_base_flow(config, device)
    residuals: list[np.ndarray] = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, len(ad_pairs), batch_size):
            batch = ad_pairs.iloc[start:start + batch_size]
            source_i = batch.source_index.to_numpy(dtype=int)
            target_i = batch.target_index.to_numpy(dtype=int)
            source = torch.from_numpy(latents[source_i]).to(device)
            source_time = torch.tensor(batch.source_time.to_numpy(), device=device, dtype=torch.float32)
            target_time = torch.tensor(batch.target_time.to_numpy(), device=device, dtype=torch.float32)
            ad = base.transport(source, source_time, target_time, torch.ones(len(batch), 1, device=device))
            target = torch.from_numpy(latents[target_i]).to(device)
            delta = (target_time - source_time).reshape(-1, 1).abs().clamp_min(1.0e-6)
            residuals.append(((target - ad) / delta).cpu().numpy())
    residuals_array = np.concatenate(residuals, axis=0).astype(np.float64)
    residual_mean = residuals_array.mean(axis=0, keepdims=True)
    centered = residuals_array - residual_mean
    _, residual_singular, residual_vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(config["ResidualBasisRank"]), residual_vh.shape[0])
    mean_direction = residual_mean / np.maximum(np.linalg.norm(residual_mean), 1.0e-12)
    residual_basis = np.concatenate([mean_direction, residual_vh[:rank]], axis=0).astype(np.float32)
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_components": feature_components,
        "feature_singular_values": feature_singular.astype(np.float32),
        "residual_mean_velocity": residual_mean.astype(np.float32),
        "residual_basis": residual_basis,
        "residual_singular_values": residual_singular.astype(np.float32),
        "train_ad_adjacent_pair_count": np.asarray([len(ad_pairs)], dtype=np.int64),
        "train_scan_count": np.asarray([len(train_indices)], dtype=np.int64),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", help="Only needed to fit the frozen-base residual basis.")
    parser.add_argument("--force", action="store_true", help="Regenerate only this experiment's cache files.")
    parser.add_argument("--resume", action="store_true", help="Reuse a complete mesh/pair cache after an interrupted basis fit.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = experiment_dir()
    paths = ensure_paths_exist(config, root)
    outputs = [
        root / "metadata" / "registered_meshes.npz",
        root / "metadata" / "scan_manifest.csv",
        root / "metadata" / "pair_records_train.csv",
        root / "metadata" / "pair_records_val.csv",
        root / "metadata" / "pair_records_test.csv",
        root / "basis" / "train_only_basis.npz",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not (args.force or args.resume):
        raise FileExistsError("Cache exists; refusing to overwrite. Use --force only deliberately:\n" + "\n".join(existing))
    device = torch.device(args.device)
    frame, latents = load_metadata_and_latents(config, root)
    missing_mesh = [path for path in frame.mesh_path.astype(str) if not Path(path).is_file()]
    missing_sdf = [path for path in frame.sdf_npz_path.astype(str) if not Path(path).is_file()]
    if missing_mesh or missing_sdf:
        raise FileNotFoundError(f"Missing mesh={len(missing_mesh)} sdf={len(missing_sdf)}")
    cache_path = root / "metadata" / "registered_meshes.npz"
    if args.resume and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(cached["scan_ids"], frame.scan_id.to_numpy(dtype="U")) or not np.array_equal(cached["latents"], latents):
            raise ValueError("Existing cache is not aligned to the current QC inputs; use --force to rebuild it.")
        vertices_array = cached["vertices"]
        normals_array = cached["normals"]
        areas_array = cached["areas"]
        shared_faces = cached["faces"]
        computed_volumes = cached["volumes"]
    else:
        vertices: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        areas: list[np.ndarray] = []
        shared_faces: np.ndarray | None = None
        for row in frame.itertuples(index=False):
            mesh_vertices, mesh_faces = read_obj_mesh(row.mesh_path)
            if shared_faces is None:
                shared_faces = mesh_faces
            elif mesh_vertices.shape != vertices[0].shape or not np.array_equal(mesh_faces, shared_faces):
                raise ValueError(f"Registered-mesh topology mismatch at {row.scan_id}")
            mesh_normals, mesh_areas = mesh_normals_and_areas(mesh_vertices, mesh_faces)
            vertices.append(mesh_vertices)
            normals.append(mesh_normals)
            areas.append(mesh_areas)
        assert shared_faces is not None
        vertices_array = np.asarray(vertices, dtype=np.float32)
        normals_array = np.asarray(normals, dtype=np.float32)
        areas_array = np.asarray(areas, dtype=np.float32)
        computed_volumes = np.abs(
            np.einsum(
                "nfi,nfi->n", vertices_array[:, shared_faces[:, 0]],
                np.cross(vertices_array[:, shared_faces[:, 1]], vertices_array[:, shared_faces[:, 2]]),
            )
        ) / 6.0
    if "left_mesh_volume_mm3" in frame.columns and frame.left_mesh_volume_mm3.notna().all():
        metadata_volumes = frame.left_mesh_volume_mm3.to_numpy(dtype=np.float32)
        coordinate_to_mm3 = metadata_volumes / np.maximum(computed_volumes, 1.0e-12)
        coordinate_to_mm3_median = float(np.median(coordinate_to_mm3))
        scaled_volume_relative_error = np.abs(
            computed_volumes * coordinate_to_mm3_median - metadata_volumes
        ) / np.maximum(np.abs(metadata_volumes), 1.0e-6)
        volume_contract = {
            "volume_unit": "mesh_coordinates_with_documented_mm3_scale",
            "mesh_coordinate_to_mm3_scale_median": coordinate_to_mm3_median,
            "mesh_coordinate_to_mm3_scale_relative_std": float(np.std(coordinate_to_mm3) / max(coordinate_to_mm3_median, 1.0e-12)),
            "max_scaled_mesh_volume_relative_error": float(scaled_volume_relative_error.max()),
        }
    else:
        volume_contract = {
            "volume_unit": "mesh_coordinates; relative-volume trends/rates are scale-invariant",
        }
    root.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    root.joinpath("basis").mkdir(parents=True, exist_ok=True)
    if not (args.resume and cache_path.exists()):
        np.savez_compressed(
            cache_path,
            scan_ids=frame.scan_id.to_numpy(dtype="U"), latents=latents.astype(np.float32),
            vertices=vertices_array, normals=normals_array, areas=areas_array, faces=shared_faces.astype(np.int64),
            volumes=computed_volumes.astype(np.float32), times=frame.continuous_age_norm.to_numpy(dtype=np.float32),
            ages_years=frame.continuous_age_years.to_numpy(dtype=np.float32), labels=frame.label_ad.to_numpy(dtype=np.int64),
        )
        frame.to_csv(root / "metadata" / "scan_manifest.csv", index=False)
    pair_counts: dict[str, int] = {}
    pair_mode = config.get("TrainPairSourceMode", "all_forward_starts")
    pairs = make_forward_pair_table(frame, pair_mode)
    for split in ("train", "val", "test"):
        split_pairs = pairs[pairs.split == split].reset_index(drop=True)
        if not (args.resume and (root / "metadata" / f"pair_records_{split}.csv").exists()):
            split_pairs.to_csv(root / "metadata" / f"pair_records_{split}.csv", index=False)
        if split == "train" and bool(config.get("UseBackwardPairs", True)) and not (args.resume and (root / "metadata" / "pair_records_train_backward.csv").exists()):
            reverse_pair_table(split_pairs).to_csv(root / "metadata" / "pair_records_train_backward.csv", index=False)
        pair_counts[split] = int(len(split_pairs))
    bases = _fit_train_only_bases(config, frame, latents, device)
    np.savez_compressed(root / "basis" / "train_only_basis.npz", **bases)
    write_json(root / "metadata" / "input_contract.json", {
        "experiment": config["ExperimentName"], "source_experiment": str(paths["source_experiment"]),
        "metadata": str(paths["metadata"]), "decoder": str(paths["decoder"]), "base_flow": str(paths["base_flow"]),
        "scan_counts": {str(k): int(v) for k, v in frame.groupby("split").size().items()},
        "pair_counts": pair_counts, "backward_train_pairs": bool(config.get("UseBackwardPairs", True)),
        "registered_vertex_count": int(vertices_array.shape[1]), "registered_face_count": int(shared_faces.shape[0]),
        **volume_contract,
        "basis_fit_split": "train", "basis_pair_type": "adjacent", "train_ad_adjacent_pair_count": int(bases["train_ad_adjacent_pair_count"][0]),
    })
    print(f"Wrote cache for {len(frame)} scans, {vertices_array.shape[1]} vertices, {len(shared_faces)} faces.")
    print(f"Forward pairs: {pair_counts}; train AD adjacent basis pairs: {int(bases['train_ad_adjacent_pair_count'][0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
