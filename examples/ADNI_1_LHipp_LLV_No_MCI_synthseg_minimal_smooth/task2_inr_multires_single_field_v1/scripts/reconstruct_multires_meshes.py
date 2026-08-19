#!/usr/bin/env python3
"""Decode an exported multiresolution latent table to meshes on the bulk disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from multires_common import (
    choose_device,
    decode_latent_to_mesh,
    load_config,
    load_decoder_checkpoint,
    read_scaling,
    require_bulk_path,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--latents", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, _payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    archive_path = require_bulk_path(args.latents, "latent input")
    archive = torch.load(archive_path, map_location="cpu")
    vectors = archive["latents"].numpy()
    scan_ids = list(archive["scan_ids"])
    splits = list(archive["splits"])
    if len(vectors) != len(scan_ids) or len(vectors) != len(splits):
        raise ValueError("Latent table metadata lengths do not match.")
    output = require_bulk_path(
        args.output_dir
        or require_bulk_path(config["output_dir"]) / "reconstructions" / checkpoint.stem
    )
    resolution = int(args.resolution or config["reconstruction"]["resolution"])
    scaling = read_scaling(config["periodic_evaluation"]["rescale_details_csv"])
    rows = []
    selected = [(scan_id, split, vector) for scan_id, split, vector in zip(scan_ids, splits, vectors) if split in args.splits]
    for number, (scan_id, split, vector) in enumerate(selected, start=1):
        path = output / split / f"{scan_id}.ply"
        report = decode_latent_to_mesh(
            decoder,
            np.asarray(vector, dtype=np.float32),
            path,
            resolution,
            int(config["reconstruction"]["max_batch"]),
            device,
            scaling=scaling,
        )
        rows.append({"scan_id": scan_id, "split": split, **report, "coordinate_space": "physical_mm"})
        print(f"[{number:04d}/{len(selected):04d}] {split} {scan_id}", flush=True)
    write_csv(output / "reconstruction_manifest.csv", rows)
    write_json(output / "summary.json", {"checkpoint": str(checkpoint), "latent_archive": str(archive_path), "count": len(rows), "resolution": resolution, "output": str(output)})


if __name__ == "__main__":
    main()
