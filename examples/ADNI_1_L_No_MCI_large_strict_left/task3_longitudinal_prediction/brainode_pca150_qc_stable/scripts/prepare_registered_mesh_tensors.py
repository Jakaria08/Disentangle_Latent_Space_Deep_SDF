#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from registered_pca_flowmap_utils import (
    DEFAULT_EXPERIMENT_NAME,
    TASK_DIR,
    decode_pca_np,
    experiment_root,
    load_all_archive,
    load_pca_model,
    mesh_area_np,
    mesh_volume_np,
    pca_latents,
    vertex_normals_np,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute registered PCA vertices, volumes, areas, faces, and template normals."
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, pca_model_dir, mean_flat, components, faces = load_pca_model(
        args.config,
        int(args.components),
    )
    archive = load_all_archive()
    latents = pca_latents(archive, int(args.components))
    chunk_size = max(1, int(args.chunk_size))
    vertices_chunks: list[np.ndarray] = []
    volumes: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for start in range(0, latents.shape[0], chunk_size):
        chunk = decode_pca_np(latents[start : start + chunk_size], mean_flat, components)
        vertices_chunks.append(chunk.astype(np.float32))
        volumes.append(mesh_volume_np(chunk, faces).astype(np.float32))
        areas.append(mesh_area_np(chunk, faces).astype(np.float32))
    vertices = np.concatenate(vertices_chunks, axis=0)
    volume_array = np.concatenate(volumes, axis=0)
    area_array = np.concatenate(areas, axis=0)
    template_vertices = mean_flat.reshape(-1, 3).astype(np.float32)
    template_normals = vertex_normals_np(template_vertices, faces)

    output = (
        Path(args.output).expanduser()
        if args.output
        else experiment_root(args.experiment_name) / "metadata" / "registered_mesh_tensors.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scan_ids=archive["visit_scan_ids"],
        subject_ids=archive["visit_subject_ids"],
        splits=archive["visit_splits"],
        diagnoses=archive["visit_diagnoses"],
        label_ad=archive["visit_label_ad"],
        visit_orders=archive["visit_orders"],
        age_years=archive["visit_continuous_age_years"],
        age_norm=archive["visit_continuous_age_norm"],
        vertices=vertices,
        volumes=volume_array,
        surface_areas=area_array,
        faces=faces.astype(np.int64),
        template_vertex_normals=template_normals,
    )
    summary = {
        "output": str(output),
        "components": int(args.components),
        "pca_model_dir": str(pca_model_dir),
        "scans": int(latents.shape[0]),
        "vertices_per_mesh": int(vertices.shape[1]),
        "faces": int(faces.shape[0]),
        "volume_mean": float(volume_array.mean()),
        "surface_area_mean": float(area_array.mean()),
    }
    summary_path = output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

