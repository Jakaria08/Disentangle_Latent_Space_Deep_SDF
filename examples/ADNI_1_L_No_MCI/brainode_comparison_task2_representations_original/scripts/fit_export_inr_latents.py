#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from task2_common import (
    choose_device,
    fit_single_latent,
    load_config,
    load_decoder_checkpoint,
    load_json,
    load_manifest,
    resolve_repo_path,
    sha256_file,
    stable_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit frozen-decoder INR latents for train, validation, and test scans."
    )
    parser.add_argument("--config", required=True, help="INR JSON configuration.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint name/path; defaults to validation-selected best.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to process.",
    )
    parser.add_argument("--device", default=None, help="Device such as cuda:0 or cpu.")
    parser.add_argument("--steps", type=int, default=None, help="Override latent-fit steps.")
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Additional attempts for non-converged fits.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing per-scan latent and JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config output_dir for exported latents and reports.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional display name override for summaries.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit scans for testing.")
    return parser.parse_args()


def converged(stats: dict) -> bool:
    initial = stats.get("initial_data_l1")
    final = stats.get("final_data_l1")
    return bool(
        stats.get("finite")
        and initial is not None
        and final is not None
        and np.isfinite(stats["heldout_sdf_l1"])
        and final < initial
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.name:
        config["name"] = args.name
    device = choose_device(args.device)
    decoder, checkpoint_payload, checkpoint_path = load_decoder_checkpoint(
        config, args.checkpoint, device
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    rows = load_manifest(config["manifest"])
    requested_splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    }
    invalid = requested_splits.difference({"train", "val", "test"})
    if invalid:
        raise ValueError(f"Unknown splits: {sorted(invalid)}")
    selected = [row for row in rows if row["split"] in requested_splits]
    if args.limit > 0:
        selected = selected[: args.limit]

    output_dir = resolve_repo_path(config["output_dir"])
    per_scan_dir = output_dir / "latents" / "per_scan"
    metric_dir = output_dir / "latents" / "fit_metrics"
    per_scan_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    fit_config = dict(config["latent_fit"])
    if args.steps is not None:
        fit_config["steps"] = int(args.steps)
    seed = int(config["seed"])

    print(
        f"Fitting {len(selected)} {config['name']} latents from {checkpoint_path} "
        f"on {device}."
    )
    for index, row in enumerate(selected, start=1):
        latent_path = per_scan_dir / f"{row['scan_id']}.npy"
        stats_path = metric_dir / f"{row['scan_id']}.json"
        if args.skip_existing and latent_path.is_file() and stats_path.is_file():
            print(f"[{index}/{len(selected)}] existing {row['scan_id']}")
            continue

        attempts = []
        best = None
        base_seed = stable_seed(row["scan_id"], seed)
        for attempt in range(args.retries + 1):
            latent, stats = fit_single_latent(
                decoder=decoder,
                sdf_path=row["sdf_npz_path"],
                latent_size=int(config["latent_size"]),
                fit_config=fit_config,
                clamp_distance=float(config["clamp_distance"]),
                device=device,
                seed=(base_seed + attempt) % (2**32),
            )
            stats["attempt"] = attempt
            attempts.append((latent, stats))
            if best is None or stats["heldout_sdf_l1"] < best[1]["heldout_sdf_l1"]:
                best = (latent, stats)
            if converged(stats):
                break
        if best is None:
            raise RuntimeError(f"No latent result for {row['scan_id']}")
        latent, stats = best
        stats.update(
            {
                "scan_id": row["scan_id"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "attempt_count": len(attempts),
                "converged": converged(stats),
            }
        )
        np.save(latent_path, latent)
        write_json(stats_path, stats)
        print(
            f"[{index}/{len(selected)}] {row['scan_id']} "
            f"heldout_l1={stats['heldout_sdf_l1']:.7f} "
            f"steps={stats['steps_completed']} converged={stats['converged']}"
        )

    export_rows = []
    for row in rows:
        if row["split"] not in requested_splits:
            continue
        latent_path = per_scan_dir / f"{row['scan_id']}.npy"
        stats_path = metric_dir / f"{row['scan_id']}.json"
        if not latent_path.is_file() or not stats_path.is_file():
            continue
        stats = load_json(stats_path)
        latent = np.load(latent_path)
        export_rows.append(
            {
                "scan_id": row["scan_id"],
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "label_ad": row["label_ad"],
                "visit_order": row["visit_order"],
                "age_norm": row["age_norm"],
                "latent_dimension": int(latent.size),
                "latent_norm": float(np.linalg.norm(latent)),
                "heldout_sdf_l1": stats["heldout_sdf_l1"],
                "converged": stats["converged"],
                "latent_path": str(latent_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
            }
        )

    csv_path = output_dir / "latents" / "inr_latents.csv"
    if export_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(export_rows[0].keys()))
            writer.writeheader()
            writer.writerows(export_rows)

    for split in ("train", "val", "test"):
        split_rows = [row for row in export_rows if row["split"] == split]
        expected_rows = [row for row in rows if row["split"] == split]
        if split not in requested_splits or len(split_rows) != len(expected_rows):
            continue
        np.savez_compressed(
            output_dir / "latents" / f"{split}_latents.npz",
            scan_ids=np.asarray([row["scan_id"] for row in split_rows]),
            latents=np.stack([np.load(row["latent_path"]) for row in split_rows]),
        )

    summary = {
        "name": config["name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_sha256": checkpoint_hash,
        "requested_splits": sorted(requested_splits),
        "exported_scan_count": len(export_rows),
        "converged_scan_count": sum(
            str(row["converged"]).lower() == "true" for row in export_rows
        ),
        "mean_heldout_sdf_l1": (
            float(np.mean([float(row["heldout_sdf_l1"]) for row in export_rows]))
            if export_rows
            else None
        ),
    }
    write_json(output_dir / "latents" / "latent_export_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
