#!/usr/bin/env python3
"""Read-only integrity audit for a final multiresolution latent export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPTS = SCRIPT_DIR.parent.parent / "task2_inr_multires_single_field_v1" / "scripts"
for path in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_common import require_bulk_path, sha256_file  # noqa: E402
from multires_common import load_config, load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--require-all-splits", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    export = require_bulk_path(args.export_dir, "latent export")
    paths = {
        "latents": export / "latents.pth",
        "metadata": export / "latent_metadata.csv",
        "metrics": export / "latent_fit_metrics.csv",
        "summary": export / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete latent export: {missing}")
    payload = torch.load(paths["latents"], map_location="cpu")
    metadata = read_csv(paths["metadata"])
    metrics = read_csv(paths["metrics"])
    with paths["summary"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    latents = payload.get("latents")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        raise ValueError("latents.pth does not contain a rank-2 latent tensor.")
    if latents.shape[1] != int(config["latent_size"]):
        raise ValueError("Exported latent width differs from the decoder config.")
    if not torch.isfinite(latents).all():
        raise ValueError("Exported latent table contains non-finite values.")
    count = int(latents.shape[0])
    if len(metadata) != count or len(metrics) != count:
        raise ValueError("Latent, metadata, and metric row counts differ.")
    scan_ids = list(payload.get("scan_ids", []))
    splits = list(payload.get("splits", []))
    subject_ids = list(payload.get("subject_ids", []))
    if not (len(scan_ids) == len(splits) == len(subject_ids) == count):
        raise ValueError("Latent payload ID metadata lengths differ.")
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Exported scan IDs are not unique.")
    if scan_ids != [row["scan_id"] for row in metadata]:
        raise ValueError("Payload and CSV scan order differ.")
    if splits != [row["split"] for row in metadata]:
        raise ValueError("Payload and CSV split order differ.")
    manifest = load_manifest(config["manifest"])
    included_splits = set(splits)
    expected = [row for row in manifest if row["split"] in included_splits]
    if scan_ids != [row["scan_id"] for row in expected]:
        raise ValueError("Exported codes do not follow exact manifest order.")
    if args.require_all_splits:
        expected_counts = {"train": 401, "val": 100, "test": 100}
        observed_counts = {split: splits.count(split) for split in expected_counts}
        if observed_counts != expected_counts or count != 601:
            raise ValueError(f"Expected the final 601-code export, found {observed_counts}.")
    checkpoint = Path(payload.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Exported decoder checkpoint is missing: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    if payload.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Latent export checkpoint checksum is missing or stale.")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    learned = {
        scan_id: checkpoint_payload["latent_codes"][index].detach().cpu()
        for index, scan_id in enumerate(checkpoint_payload["training_scan_ids"])
    }
    train_positions = [index for index, split in enumerate(splits) if split == "train"]
    for index in train_positions:
        if not torch.equal(latents[index].cpu(), learned[scan_ids[index]]):
            raise ValueError(f"Exported learned train code changed for {scan_ids[index]}.")
    nontrain_metrics = [row for row in metrics if row["split"] != "train"]
    if any(row.get("source") != "frozen_decoder_fit" for row in nontrain_metrics):
        raise ValueError("At least one validation/test code was not fitted to the final frozen decoder.")
    array = latents.float().numpy()
    report = {
        "passed": True,
        "export_dir": str(export),
        "latent_path": str(paths["latents"]),
        "latent_file_sha256": sha256_file(paths["latents"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", 0)),
        "shape": list(latents.shape),
        "split_counts": {split: splits.count(split) for split in ("train", "val", "test")},
        "scan_ids_unique": True,
        "manifest_order_exact": True,
        "train_codes_equal_checkpoint": True,
        "nontrain_codes_fitted_to_final_decoder": True,
        "finite": True,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "max_abs": float(np.abs(array).max()),
        "summary_consistent": int(summary["count"]) == count,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
