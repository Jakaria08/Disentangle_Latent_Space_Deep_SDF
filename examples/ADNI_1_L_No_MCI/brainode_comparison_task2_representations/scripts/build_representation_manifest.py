#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from task2_common import TASK_DIR, load_config, load_manifest, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one scan-level manifest joining PCA and all configured INR exports."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless all latent and reconstruction paths exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rows = load_manifest(config["manifest"])
    inr_models = config.get("inr_models", {})
    output_rows = []
    missing = []
    for row in rows:
        paths = {
            "pca_coeff_path": TASK_DIR
            / "pca"
            / "coefficients"
            / "per_scan"
            / f"{row['scan_id']}.npy",
            "pca_150_mesh_path": TASK_DIR
            / "pca"
            / "reconstructed_meshes"
            / "k150"
            / f"{row['scan_id']}.ply",
            "pca_256_mesh_path": TASK_DIR
            / "pca"
            / "reconstructed_meshes"
            / "k256"
            / f"{row['scan_id']}.ply",
        }
        for method, spec in inr_models.items():
            prefix = spec.get("manifest_prefix", method)
            output_dir = resolve_repo_path(spec["output_dir"])
            paths[f"{prefix}_latent_path"] = (
                output_dir / "latents" / "per_scan" / f"{row['scan_id']}.npy"
            )
            paths[f"{prefix}_mesh_path"] = (
                output_dir / "reconstructed_meshes" / f"{row['scan_id']}.ply"
            )
        for name, path in paths.items():
            if not path.is_file():
                missing.append(f"{row['scan_id']}:{name}")
        output_rows.append(
            {
                "scan_id": row["scan_id"],
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "label_ad": row["label_ad"],
                "visit_order": row["visit_order"],
                "age_norm": row["age_norm"],
                "ground_truth_mesh_path": row["mesh_path"],
                "sdf_npz_path": row["sdf_npz_path"],
                **{name: str(path) for name, path in paths.items()},
            }
        )

    output_path = TASK_DIR / "metadata" / "representation_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"Wrote {len(output_rows)} rows to {output_path}; "
        f"missing referenced artifacts: {len(missing)}"
    )
    if args.require_complete and missing:
        print(f"First missing artifact: {missing[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
