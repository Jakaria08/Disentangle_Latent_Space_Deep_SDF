#!/usr/bin/env python3
"""Reconstruct meshes from a shared-grid decoder and an exported latent table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from shared_grid_common import (
    choose_device,
    decode_latent_to_mesh,
    load_config,
    load_decoder_checkpoint,
    resolve_repo_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--scan-ids", nargs="+", default=None)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, payload, checkpoint_path = load_decoder_checkpoint(config, args.checkpoint, device)
    latent_dir = Path(args.latents_dir).resolve()
    latents = np.load(latent_dir / "latents.npy")
    with (latent_dir / "latent_index.csv").open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != len(latents):
        raise ValueError("Latent matrix and index table lengths differ.")
    selected = index_rows
    if args.splits:
        selected = [row for row in selected if row["split"] in args.splits]
    if args.scan_ids:
        requested = set(args.scan_ids)
        selected = [row for row in selected if row["scan_id"] in requested]
        missing = requested.difference(row["scan_id"] for row in selected)
        if missing:
            raise KeyError(f"Requested scan IDs are absent: {sorted(missing)}")
    if args.max_scans is not None:
        selected = selected[: args.max_scans]

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repo_path(config["output_dir"]) / "reconstructions" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(args.resolution or config["reconstruction"]["resolution"])
    max_batch = int(args.max_batch or config["reconstruction"]["max_batch"])
    reports = []
    for number, row in enumerate(selected, start=1):
        mesh_path = output_dir / row["split"] / f"{row['scan_id']}.ply"
        if mesh_path.exists() and not args.overwrite:
            raise FileExistsError(f"Mesh exists; pass --overwrite: {mesh_path}")
        report = decode_latent_to_mesh(
            decoder,
            latents[int(row["latent_index"])],
            mesh_path,
            resolution,
            max_batch,
            device,
        )
        report.update({"scan_id": row["scan_id"], "split": row["split"]})
        reports.append(report)
        print(f"[{number:04d}/{len(selected):04d}] {row['scan_id']} -> {mesh_path}", flush=True)
    write_json(
        output_dir / "reconstruction_report.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": payload.get("epoch"),
            "resolution": resolution,
            "max_batch": max_batch,
            "count": len(reports),
            "meshes": reports,
        },
    )


if __name__ == "__main__":
    main()
