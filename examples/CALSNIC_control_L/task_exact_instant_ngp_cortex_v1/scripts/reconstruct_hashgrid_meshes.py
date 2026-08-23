#!/usr/bin/env python3
"""Decode exported latents into physical-millimetre meshes, one per readout variant.

For the two-branch model this writes ``global_only``, ``local_only``,
``fused_hard_band`` and ``fused_smooth_gate`` meshes from a single checkpoint and
a single fitted code, which is the whole fusion ablation.  For the single-field
Instant-NGP runs it writes one ``single`` mesh, matching the multires task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate_hashgrid_variants import decode_mm
from hashgrid_common import (
    architecture_name,
    choose_device,
    load_config,
    load_decoder_checkpoint,
    read_manifest,
    require_bulk_path,
    variants_for,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--latents", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val",))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in args.splits and not args.confirm_test:
        raise PermissionError("Test reconstruction is locked; add --confirm-test after selection.")
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, _payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    latent_path = require_bulk_path(args.latents, "latent input")
    archive = torch.load(latent_path, map_location="cpu")
    rows_by_id = {row["scan_id"]: row for row in read_manifest(config["manifest"])}
    vectors = archive["latents"].numpy()
    scan_ids, splits = list(archive["scan_ids"]), list(archive["splits"])
    output = require_bulk_path(
        args.output_dir or Path(config["output_dir"]) / "reconstructions_mm" / checkpoint.stem
    )
    resolution = int(args.resolution or config["reconstruction"]["resolution"])
    max_batch = int(config["reconstruction"]["max_batch"])
    band_cells = int(config["periodic_evaluation"].get("band_cells", 3))
    variants = variants_for(config, args.variants)

    selected = [
        (scan_id, split, vector)
        for scan_id, split, vector in zip(scan_ids, splits, vectors)
        if split in args.splits
    ]
    reports = []
    for number, (scan_id, split, vector) in enumerate(selected, start=1):
        row = rows_by_id[scan_id]
        for variant in variants:
            path = output / variant / split / f"{scan_id}.ply"
            mesh, decode_report = decode_mm(
                decoder,
                np.asarray(vector, dtype=np.float32),
                row,
                path,
                resolution,
                max_batch,
                device,
                variant,
                band_cells,
            )
            reports.append(
                {
                    "scan_id": scan_id,
                    "split": split,
                    "variant": variant,
                    "mesh_path": str(path),
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "watertight": bool(mesh.is_watertight),
                    "connected_components": int(len(mesh.split(only_watertight=False))),
                    "coordinate_space": "physical_mm",
                    "band_lattice_points": decode_report.get("band_lattice_points", ""),
                    "band_fraction": decode_report.get("band_fraction", ""),
                }
            )
        print(f"[{number}/{len(selected)}] {split} {scan_id}", flush=True)

    write_csv(output / "reconstruction_manifest.csv", reports)
    write_json(
        output / "summary.json",
        {
            "architecture": architecture_name(config),
            "checkpoint": str(checkpoint),
            "latent_archive": str(latent_path),
            "resolution": resolution,
            "marching_cubes_voxel_mm": 2.0
            / (resolution - 1)
            * float(config["mm_per_normalized_unit"]),
            "variants": variants,
            "band_cells": band_cells,
            "count": len(reports),
            "test_confirmed": bool(args.confirm_test),
        },
    )


if __name__ == "__main__":
    main()
