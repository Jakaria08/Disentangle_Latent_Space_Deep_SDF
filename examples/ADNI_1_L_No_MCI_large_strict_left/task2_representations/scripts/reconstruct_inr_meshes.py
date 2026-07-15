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
    decode_latent_to_mesh,
    load_config,
    load_decoder_checkpoint,
    load_json,
    load_manifest,
    resolve_repo_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode exported INR latents to PLY meshes."
    )
    parser.add_argument("--config", required=True, help="INR JSON configuration.")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name/path.")
    parser.add_argument(
        "--splits", default="train,val,test", help="Comma-separated splits."
    )
    parser.add_argument("--device", default=None, help="Device such as cuda:0 or cpu.")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config output_dir for latent inputs and mesh outputs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional display name override for summaries.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.name:
        config["name"] = args.name
    device = choose_device(args.device)
    decoder, _payload, checkpoint_path = load_decoder_checkpoint(
        config, args.checkpoint, device
    )
    rows = load_manifest(config["manifest"])
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    selected = [row for row in rows if row["split"] in splits]
    if args.limit > 0:
        selected = selected[: args.limit]

    output_dir = resolve_repo_path(config["output_dir"])
    latent_dir = output_dir / "latents" / "per_scan"
    mesh_dir = output_dir / "reconstructed_meshes"
    report_dir = output_dir / "reconstruction_reports"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(args.resolution or config["reconstruction"]["resolution"])
    max_batch = int(args.max_batch or config["reconstruction"]["max_batch"])

    failures = []
    for index, row in enumerate(selected, start=1):
        latent_path = latent_dir / f"{row['scan_id']}.npy"
        mesh_path = mesh_dir / f"{row['scan_id']}.ply"
        report_path = report_dir / f"{row['scan_id']}.json"
        if not latent_path.is_file():
            failures.append(
                {"scan_id": row["scan_id"], "error": "missing latent"}
            )
            continue
        if args.skip_existing and mesh_path.is_file() and report_path.is_file():
            print(f"[{index}/{len(selected)}] existing {row['scan_id']}")
            continue
        try:
            stats = decode_latent_to_mesh(
                decoder=decoder,
                latent=np.load(latent_path),
                output_path=mesh_path,
                resolution=resolution,
                max_batch=max_batch,
                device=device,
            )
            stats.update(
                {
                    "scan_id": row["scan_id"],
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "resolution": resolution,
                    "checkpoint": str(checkpoint_path),
                }
            )
            write_json(report_path, stats)
            print(
                f"[{index}/{len(selected)}] {row['scan_id']} "
                f"vertices={stats['vertex_count']} watertight={stats['watertight']}"
            )
        except Exception as exc:
            failures.append({"scan_id": row["scan_id"], "error": str(exc)})
            print(f"[{index}/{len(selected)}] FAILED {row['scan_id']}: {exc}")

    completed = []
    for row in rows:
        mesh_path = mesh_dir / f"{row['scan_id']}.ply"
        report_path = report_dir / f"{row['scan_id']}.json"
        if mesh_path.is_file() and report_path.is_file():
            stats = load_json(report_path)
            completed.append(
                {
                    "scan_id": row["scan_id"],
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "mesh_path": str(mesh_path),
                    "vertex_count": stats["vertex_count"],
                    "face_count": stats["face_count"],
                    "watertight": stats["watertight"],
                    "winding_consistent": stats["winding_consistent"],
                }
            )
    if completed:
        with (output_dir / "reconstructed_meshes.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completed[0].keys()))
            writer.writeheader()
            writer.writerows(completed)

    summary = {
        "name": config["name"],
        "checkpoint": str(checkpoint_path),
        "resolution": resolution,
        "requested_scan_count": len(selected),
        "completed_total": len(completed),
        "failure_count_this_run": len(failures),
        "failures": failures,
    }
    write_json(output_dir / "reconstruction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
