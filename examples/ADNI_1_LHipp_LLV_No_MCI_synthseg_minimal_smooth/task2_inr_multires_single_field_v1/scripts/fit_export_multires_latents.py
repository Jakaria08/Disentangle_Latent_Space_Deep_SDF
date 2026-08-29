#!/usr/bin/env python3
"""Export learned train codes and fitted validation/test codes to the bulk disk."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from multires_common import (
    atomic_torch_save,
    choose_device,
    fit_single_latent,
    load_config,
    load_decoder_checkpoint,
    load_manifest,
    read_scaling,
    require_bulk_path,
    sha256_file,
    stable_seed,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    output = require_bulk_path(
        args.output_dir
        or require_bulk_path(config["output_dir"]) / "latent_exports" / checkpoint.stem
    )
    rows = [row for row in load_manifest(config["manifest"]) if row["split"] in args.splits]
    train_ids = list(payload["training_scan_ids"])
    train_table = payload["latent_codes"].detach().cpu().numpy()
    learned = {scan_id: train_table[index] for index, scan_id in enumerate(train_ids)}
    vectors = []
    metadata = []
    metrics = []
    for number, row in enumerate(rows, start=1):
        if row["split"] == "train":
            latent = np.asarray(learned[row["scan_id"]], dtype=np.float32)
            fit_metrics = {"source": "learned_embedding", "steps_completed": 0}
        else:
            latent, values = fit_single_latent(
                decoder,
                row["sdf_npz_path"],
                int(config["latent_size"]),
                config["latent_fit"],
                float(config["clamp_distance"]),
                device,
                stable_seed(row["scan_id"], int(config["seed"])),
                config["network_specs"],
                steps_override=args.steps,
            )
            fit_metrics = {"source": "frozen_decoder_fit", **values}
        vectors.append(latent)
        metadata.append({key: row.get(key, "") for key in ("scan_id", "subject_id", "split", "diagnosis", "visit_month", "age_years")})
        metrics.append({"scan_id": row["scan_id"], "split": row["split"], **fit_metrics})
        print(f"[{number:04d}/{len(rows):04d}] {row['split']} {row['scan_id']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    latent_path = require_bulk_path(output / "latents.pth", "latent export")
    checkpoint_sha256 = sha256_file(checkpoint)
    split_counts = {
        split: sum(row["split"] == split for row in metadata)
        for split in ("train", "val", "test")
    }
    scaling_path = config["periodic_evaluation"]["rescale_details_csv"]
    atomic_torch_save(
        {
            "architecture": "single_field_dense_multiresolution_sdf",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": int(payload.get("epoch", 0)),
            "manifest": str(config["manifest"]),
            "latent_size": int(config["latent_size"]),
            "latents": torch.from_numpy(np.stack(vectors).astype(np.float32)),
            "scan_ids": [row["scan_id"] for row in metadata],
            "subject_ids": [row["subject_id"] for row in metadata],
            "splits": [row["split"] for row in metadata],
            "split_counts": split_counts,
            "fit_steps": int(args.steps or config["latent_fit"]["steps"]),
            "latent_fit_config": config["latent_fit"],
            "normalization": {
                "rescale_details_csv": str(scaling_path),
                "values": read_scaling(scaling_path),
            },
            "sdf_supervision": config.get("sdf_supervision", {}),
        },
        latent_path,
    )
    write_csv(output / "latent_metadata.csv", metadata)
    write_csv(output / "latent_fit_metrics.csv", metrics)
    write_json(
        output / "summary.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": int(payload.get("epoch", 0)),
            "count": len(vectors),
            "split_counts": split_counts,
            "latent_size": int(config["latent_size"]),
            "fit_steps": int(args.steps or config["latent_fit"]["steps"]),
            "normalization": {
                "rescale_details_csv": str(scaling_path),
                "values": read_scaling(scaling_path),
            },
            "output": str(latent_path),
        },
    )


if __name__ == "__main__":
    main()
