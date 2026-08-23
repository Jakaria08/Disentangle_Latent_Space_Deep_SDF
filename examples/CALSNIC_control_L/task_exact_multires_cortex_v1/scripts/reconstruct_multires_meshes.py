#!/usr/bin/env python3
"""Decode exported CALSNIC latents and save only physical-mm triangle meshes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from calsnic_common import read_manifest
from delegate_multires import GENERIC_SCRIPTS
from periodic_evaluate_multires import decode_mm

sys.path.insert(0, str(GENERIC_SCRIPTS))
from multires_common import (  # noqa: E402
    choose_device,
    load_config,
    load_decoder_checkpoint,
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
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val",))
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
    reports = []
    selected = [(scan_id, split, vector) for scan_id, split, vector in zip(scan_ids, splits, vectors) if split in args.splits]
    for number, (scan_id, split, vector) in enumerate(selected, start=1):
        row = rows_by_id[scan_id]
        path = output / split / f"{scan_id}.ply"
        mesh = decode_mm(
            decoder,
            np.asarray(vector, dtype=np.float32),
            row,
            path,
            resolution,
            int(config["reconstruction"]["max_batch"]),
            device,
        )
        reports.append(
            {
                "scan_id": scan_id,
                "split": split,
                "mesh_path": str(path),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "coordinate_space": "physical_mm",
            }
        )
        print(f"[{number}/{len(selected)}] {split} {scan_id}", flush=True)
    write_csv(output / "reconstruction_manifest.csv", reports)
    write_json(output / "summary.json", {"checkpoint": str(checkpoint), "latent_archive": str(latent_path), "resolution": resolution, "count": len(reports)})


if __name__ == "__main__":
    main()
