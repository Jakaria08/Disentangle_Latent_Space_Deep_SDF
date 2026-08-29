#!/usr/bin/env python3
"""Build immutable 601-scan INR-256 trajectory archives and pair tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=C.REGISTRY_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with C.resolve_path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_hashes(registry: dict) -> dict[str, str]:
    fields = {
        "latent_export": "latent_export_sha256",
        "latent_metadata": "latent_metadata_sha256",
        "decoder_checkpoint": "decoder_checkpoint_sha256",
    }
    actual = {}
    for path_key, hash_key in fields.items():
        path = C.resolve_path(registry[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[path_key] = C.sha256(path)
        if actual[path_key] != registry[hash_key]:
            raise ValueError(f"Pinned hash mismatch for {path_key}: {actual[path_key]}")
    return actual


def sample_field(row: dict[str, str], count: int, clamp: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(C.resolve_path(row["sdf_npz_path"]), allow_pickle=False) as loaded:
        positive = np.asarray(loaded["pos"], dtype=np.float32)
        negative = np.asarray(loaded["neg"], dtype=np.float32)
    generator = np.random.default_rng(C.stable_seed(row["scan_id"], seed))
    positive_count = count // 2
    negative_count = count - positive_count
    pos_index = generator.choice(len(positive), positive_count, replace=len(positive) < positive_count)
    neg_index = generator.choice(len(negative), negative_count, replace=len(negative) < negative_count)
    samples = np.concatenate((positive[pos_index], negative[neg_index]), axis=0)
    generator.shuffle(samples, axis=0)
    return samples[:, :3].astype(np.float32), np.clip(samples[:, 3], -clamp, clamp).astype(np.float32)


def build_archive(rows: list[dict[str, str]], latent_by_scan: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, registry: dict, dry_run: bool) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda row: (row["subject_id"], float(row["visit_month"]), int(row["visit_order"])))
    subject_ids = []
    offsets = [0]
    for subject in sorted({row["subject_id"] for row in rows}):
        current = [row for row in rows if row["subject_id"] == subject]
        subject_ids.append(subject)
        offsets.append(offsets[-1] + len(current))
    # Reorder by the same subject order used to construct offsets.
    by_subject = {subject: [] for subject in subject_ids}
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    rows = [row for subject in subject_ids for row in by_subject[subject]]
    count = int(registry["field_samples_per_scan"])
    clamp = float(registry["field_clamp_distance"])
    seed = int(registry["field_sampling_seed"])
    xyz, sdf = [], []
    for index, row in enumerate(rows):
        if dry_run and index >= 2:
            xyz.append(np.zeros((count, 3), dtype=np.float32))
            sdf.append(np.zeros(count, dtype=np.float32))
        else:
            points, values = sample_field(row, count, clamp, seed)
            xyz.append(points)
            sdf.append(values)
    raw = np.stack([latent_by_scan[row["scan_id"]] for row in rows]).astype(np.float32)
    subject_first = {subject: by_subject[subject][0] for subject in subject_ids}
    return {
        "subject_ids": np.asarray(subject_ids),
        "subject_diagnoses": np.asarray([subject_first[subject]["diagnosis"] for subject in subject_ids]),
        "subject_label_ad": np.asarray([int(subject_first[subject]["label_ad"]) for subject in subject_ids], dtype=np.int64),
        "subject_visit_offsets": np.asarray(offsets, dtype=np.int64),
        "visit_scan_ids": np.asarray([row["scan_id"] for row in rows]),
        "visit_subject_ids": np.asarray([row["subject_id"] for row in rows]),
        "visit_splits": np.asarray([row["split"] for row in rows]),
        "visit_diagnoses": np.asarray([row["diagnosis"] for row in rows]),
        "visit_label_ad": np.asarray([int(row["label_ad"]) for row in rows], dtype=np.int64),
        "visit_orders": np.asarray([int(row["visit_order"]) for row in rows], dtype=np.int64),
        "visit_months_from_baseline": np.asarray([float(row["months_from_baseline"]) for row in rows], dtype=np.float32),
        "visit_time_years_from_baseline": np.asarray([float(row["months_from_baseline"]) / 12.0 for row in rows], dtype=np.float32),
        "visit_age_years": np.asarray([float(row["age_years"]) for row in rows], dtype=np.float32),
        "visit_age_norm_train": np.asarray([float(row["age_norm_train"]) for row in rows], dtype=np.float32),
        "visit_volume_mm3": np.asarray([float(row["correspondence_volume_mm3"]) for row in rows], dtype=np.float32),
        "visit_latent_raw_256": raw,
        "visit_latent_standardized_256": ((raw - mean) / std).astype(np.float32),
        "train_latent_mean_256": mean.astype(np.float32),
        "train_latent_std_256": std.astype(np.float32),
        "visit_sdf_xyz": np.stack(xyz).astype(np.float32),
        "visit_sdf_values": np.stack(sdf).astype(np.float32),
        "visit_mesh_path_mm": np.asarray([row["mesh_path_mm"] for row in rows]),
        "visit_mesh_path_scaled": np.asarray([row["mesh_path"] for row in rows]),
        "visit_sdf_npz_path": np.asarray([row["sdf_npz_path"] for row in rows]),
    }


def pair_rows(split: str, archive: dict[str, np.ndarray]) -> list[dict[str, object]]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    output = []
    for subject_index in range(len(offsets) - 1):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        for source in range(start, end - 1):
            for target in range(source + 1, end):
                output.append({
                    "split": split,
                    "diagnosis": str(archive["visit_diagnoses"][source]),
                    "label_ad": int(archive["visit_label_ad"][source]),
                    "subject_id": str(archive["visit_subject_ids"][source]),
                    "source_index": source,
                    "target_index": target,
                    "intermediate_index": source + 1 if target - source > 1 else -1,
                    "source_scan_id": str(archive["visit_scan_ids"][source]),
                    "target_scan_id": str(archive["visit_scan_ids"][target]),
                    "source_visit_order": int(archive["visit_orders"][source]),
                    "target_visit_order": int(archive["visit_orders"][target]),
                    "pair_type": "adjacent" if target - source == 1 else "nonadjacent",
                    "delta_years": float(archive["visit_time_years_from_baseline"][target] - archive["visit_time_years_from_baseline"][source]),
                })
    return output


def main() -> int:
    args = parse_args()
    registry = C.load_registry(args.registry)
    hashes = verify_hashes(registry)
    payload = torch.load(C.resolve_path(registry["latent_export"]), map_location="cpu", weights_only=False)
    latents = torch.as_tensor(payload["latents"], dtype=torch.float32).cpu().numpy()
    scan_ids = [str(value) for value in payload["scan_ids"]]
    splits = [str(value) for value in payload["splits"]]
    if latents.shape != (601, C.LATENT_DIM) or len(set(scan_ids)) != 601 or not np.isfinite(latents).all():
        raise ValueError(f"Invalid latent export contract: {latents.shape}")
    metadata = {row["scan_id"]: row for row in read_csv(registry["latent_metadata"])}
    manifest = {row["scan_id"]: row for row in read_csv(registry["source_manifest"])}
    if set(scan_ids) != set(metadata) or set(scan_ids) != set(manifest):
        raise ValueError("Latent, metadata, and exact-SDF manifest scan IDs differ")
    for scan_id, split in zip(scan_ids, splits):
        if metadata[scan_id]["split"] != split or manifest[scan_id]["split"] != split:
            raise ValueError(f"Split mismatch for {scan_id}")
        if metadata[scan_id]["subject_id"] != manifest[scan_id]["subject_id"]:
            raise ValueError(f"Subject mismatch for {scan_id}")
    latent_by_scan = dict(zip(scan_ids, latents))
    train = latents[np.asarray(splits) == "train"]
    mean = train.mean(axis=0).astype(np.float32)
    std = train.std(axis=0).astype(np.float32)
    if float(std.min()) <= 1.0e-3:
        raise ValueError(f"Unsafe train latent standard deviation: {std.min()}")
    archives = {}
    for split in C.SPLITS:
        rows = [manifest[scan_id] for scan_id in scan_ids if manifest[scan_id]["split"] == split]
        archives[split] = build_archive(rows, latent_by_scan, mean, std, registry, args.dry_run)
        expected_scans = int(registry["expected_split_counts"][split])
        expected_subjects = int(registry["expected_subject_counts"][split])
        if len(archives[split]["visit_scan_ids"]) != expected_scans or len(archives[split]["subject_ids"]) != expected_subjects:
            raise ValueError(f"Unexpected {split} scan/subject counts")
    C.validate_split_isolation(archives)
    audit = {
        "status": "verified",
        "latent_shape": list(latents.shape),
        "split_scans": {split: int(len(archives[split]["visit_scan_ids"])) for split in C.SPLITS},
        "split_subjects": {split: int(len(archives[split]["subject_ids"])) for split in C.SPLITS},
        "latent_norm": {"mean": float(np.linalg.norm(latents, axis=1).mean()), "max": float(np.linalg.norm(latents, axis=1).max())},
        "train_std": {"min": float(std.min()), "median": float(np.median(std)), "max": float(std.max())},
        "hashes": hashes,
        "decoder_checkpoint_epoch": int(torch.load(C.resolve_path(registry["decoder_checkpoint"]), map_location="cpu", weights_only=False)["epoch"]),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print("DRY RUN PASSED — exact IDs, splits, hashes, standardization, and sample schema; no files written.")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    representation = C.output_root(registry) / "representations" / "inr256"
    if representation.exists() or (C.output_root(registry) / "pairs").exists():
        raise FileExistsError(f"Refusing to overwrite prepared task under {C.output_root(registry)}")
    for split in C.SPLITS:
        C.atomic_npz(C.archive_path(split, registry), archives[split])
        C.write_csv(C.pair_path(split, registry), pair_rows(split, archives[split]))
    C.atomic_json(representation / "manifest.json", audit | {
        "decoder_frozen": True,
        "train_only_standardization": True,
        "test_evaluator_only": True,
        "registry": str(C.resolve_path(args.registry)),
    })
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"PREPARED {representation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
