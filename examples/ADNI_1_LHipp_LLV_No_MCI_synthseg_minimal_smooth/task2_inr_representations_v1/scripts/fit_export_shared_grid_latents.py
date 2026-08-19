#!/usr/bin/env python3
"""Export learned train codes and fit held-out codes with a frozen decoder."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from shared_grid_common import (
    choose_device,
    fit_single_latent,
    load_config,
    load_checkpoint_training_scan_ids,
    load_decoder_checkpoint,
    load_manifest,
    resolve_repo_path,
    stable_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override held-out latent fit steps.")
    parser.add_argument("--max-scans", type=int, default=None, help="Debug-only per-split limit.")
    parser.add_argument("--refit-train", action="store_true", help="Fit train codes again instead of exporting learned embeddings.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, payload, checkpoint_path = load_decoder_checkpoint(config, args.checkpoint, device)
    manifest_rows = load_manifest(config["manifest"])
    rows = [row for row in manifest_rows if row["split"] in args.splits]
    source_training_ids = load_checkpoint_training_scan_ids(
        config["global_checkpoint"], str(config.get("global_checkpoint_scan_id_suffix", ""))
    )
    source_seen_subjects = {
        row["subject_id"] for row in manifest_rows if row["scan_id"] in source_training_ids
    }
    if args.max_scans is not None:
        limited = []
        for split in args.splits:
            limited.extend([row for row in rows if row["split"] == split][: args.max_scans])
        rows = limited

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repo_path(config["output_dir"]) / "latents" / checkpoint_path.stem
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = payload.get("training_scan_ids", [])
    learned_codes = payload.get("latent_codes")
    learned_by_scan = {}
    if learned_codes is not None:
        learned_array = learned_codes.detach().cpu().numpy().astype(np.float32)
        if len(train_ids) != len(learned_array):
            raise ValueError("Checkpoint training_scan_ids and latent_codes lengths differ.")
        learned_by_scan = dict(zip(train_ids, learned_array))

    vectors = []
    index_rows = []
    metric_rows = []
    started = time.time()
    for row in rows:
        use_learned = row["split"] == "train" and not args.refit_train
        if use_learned:
            if row["scan_id"] not in learned_by_scan:
                raise KeyError(f"No learned code for training scan {row['scan_id']}")
            vector = learned_by_scan[row["scan_id"]]
            metrics = {
                "steps_requested": 0,
                "steps_completed": 0,
                "initial_data_l1": None,
                "final_data_l1": None,
                "best_objective": None,
                "heldout_sdf_l1": None,
                "heldout_global_l1": None,
                "heldout_local_l1": None,
                "heldout_fused_l1": None,
                "latent_norm": float(np.linalg.norm(vector)),
                "finite": bool(np.isfinite(vector).all()),
            }
            source = "learned_embedding"
        else:
            vector, metrics = fit_single_latent(
                decoder,
                row["sdf_npz_path"],
                int(config["latent_size"]),
                config["latent_fit"],
                float(config["clamp_distance"]),
                device,
                stable_seed(row["scan_id"], int(config["seed"])),
                steps_override=args.steps,
                mesh_path=row["mesh_path"],
                network_specs=config["network_specs"],
            )
            source = "frozen_decoder_fit"
        vector_index = len(vectors)
        vectors.append(vector)
        index_rows.append(
            {
                "latent_index": vector_index,
                "scan_id": row["scan_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "structure": row.get("structure", ""),
                "VISCODE": row.get("VISCODE", ""),
                "visit_month": row.get("visit_month", ""),
                "visit_order": row.get("visit_order", ""),
                "diagnosis": row.get("diagnosis", ""),
                "age_norm_train": row.get("age_norm_train", ""),
                "source": source,
                "source_pretrain_scan_seen": row["scan_id"] in source_training_ids,
                "source_pretrain_subject_seen": row["subject_id"] in source_seen_subjects,
            }
        )
        metric_rows.append({"latent_index": vector_index, "scan_id": row["scan_id"], "split": row["split"], **metrics})
        print(f"[{len(vectors):04d}/{len(rows):04d}] {row['scan_id']} {source}", flush=True)

    matrix = np.stack(vectors).astype(np.float32)
    np.save(output_dir / "latents.npy", matrix)
    torch.save(torch.from_numpy(matrix), output_dir / "latents.pt")
    write_csv(output_dir / "latent_index.csv", index_rows)
    write_csv(output_dir / "latent_fit_metrics.csv", metric_rows)
    write_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": payload.get("epoch"),
            "latent_size": int(matrix.shape[1]),
            "latent_count": int(matrix.shape[0]),
            "splits": list(args.splits),
            "refit_train": args.refit_train,
            "seconds": time.time() - started,
            "finite": bool(np.isfinite(matrix).all()),
            "mean_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        },
    )
    print(f"Exported {len(matrix)} latent vectors to {output_dir}")


if __name__ == "__main__":
    main()
