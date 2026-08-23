#!/usr/bin/env python3
"""Export learned train codes and fitted validation/test codes to the bulk disk.

One 256-D code per subject is produced regardless of architecture: the
two-branch model shares the same code between its generalization and
overfitting branches, so the exported representation size is identical to
MR64/MR128 and to a PCA coefficient vector of the same length.

The test split is locked behind ``--confirm-test``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from hashgrid_common import (
    architecture_name,
    choose_device,
    fit_single_latent,
    load_config,
    load_decoder_checkpoint,
    load_manifest,
    require_bulk_path,
    stable_seed,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val")
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in args.splits and not args.confirm_test:
        raise PermissionError(
            "Test latent fitting is locked; add --confirm-test after model selection."
        )
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

    vectors, metadata, metrics = [], [], []
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
        metadata.append(
            {
                key: row.get(key, "")
                for key in ("scan_id", "subject_id", "split", "diagnosis", "study", "site", "sex", "age_years")
            }
        )
        metrics.append({"scan_id": row["scan_id"], "split": row["split"], **fit_metrics})
        print(f"[{number:04d}/{len(rows):04d}] {row['split']} {row['scan_id']}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    latent_path = require_bulk_path(output / "latents.pth", "latent export")
    torch.save(
        {
            "architecture": architecture_name(config),
            "checkpoint": str(checkpoint),
            "latent_size": int(config["latent_size"]),
            "latents": torch.from_numpy(np.stack(vectors).astype(np.float32)),
            "scan_ids": [row["scan_id"] for row in metadata],
            "splits": [row["split"] for row in metadata],
        },
        latent_path,
    )
    write_csv(output / "latent_metadata.csv", metadata)
    write_csv(output / "latent_fit_metrics.csv", metrics)
    write_json(
        output / "summary.json",
        {
            "architecture": architecture_name(config),
            "checkpoint": str(checkpoint),
            "count": len(vectors),
            "latent_size": int(config["latent_size"]),
            "splits": list(args.splits),
            "test_confirmed": bool(args.confirm_test),
            "output": str(latent_path),
        },
    )


if __name__ == "__main__":
    main()
