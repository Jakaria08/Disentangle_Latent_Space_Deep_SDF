#!/usr/bin/env python3
"""Export AE latents for the downstream flow network.

Output schema deliberately mirrors hippocampus_pca_cocycle_v4/pca/coefficients/*.npz so the
cocycle / Brain-ODE code can swap PCA coefficients for AE latents without an adapter:

    scan_ids, subject_ids, splits, diagnoses, label_ad, visit_orders, visit_months,
    age_years, age_norm_train,
    ae_<K>, ae_standardized_<K>, train_ae_mean_<K>, train_ae_std_<K>

Standardisation statistics come from the train split only, matching train_pca_mean/std.
Row order is manifest order within each split -- identical to the PCA files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spiral_common as sc
import train_eval as te
from network import build_model

METADATA_FIELDS = {
    "scan_ids": ("scan_id", str),
    "subject_ids": ("subject_id", str),
    "splits": ("split", str),
    "diagnoses": ("diagnosis", str),
    "label_ad": ("label_ad", np.int8),
    "visit_orders": ("visit_order", np.int16),
    "visit_months": ("visit_month", np.float32),
    "age_years": ("age_years", np.float32),
    "age_norm_train": ("age_norm_train", np.float32),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", nargs="+", default=sorted(sc.EXPERIMENTS))
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def metadata_arrays(rows, split):
    selected = sc.split_rows(rows, split)
    out = {}
    for key, (column, dtype) in METADATA_FIELDS.items():
        values = [r[column] for r in selected]
        if dtype is str:
            out[key] = np.array(values, dtype=object)
        else:
            out[key] = np.array([dtype(float(v)) for v in values], dtype=dtype)
    return out


def export_one(exp_name, rows, data, device):
    dirs = sc.experiment_dirs(exp_name)
    best_fp = dirs["best"] / "best_model.pt"
    if not best_fp.exists():
        print(f"[skip] {exp_name}: no best_model.pt yet")
        return None

    payload = torch.load(best_fp, map_location="cpu")
    latent = int(payload["latent_channels"])
    transform = sc.get_transform(payload["ds_factors"])
    spirals, dynamic, down, up = sc.build_spiral_stack(
        transform,
        payload["seq_length"],
        payload["dilation"],
        payload["dynamic_seq_lengths"],
        device,
    )
    model = build_model(
        transform=transform,
        spiral_indices=spirals,
        dynamic_spiral_indices=dynamic,
        down_transform=down,
        up_transform=up,
        out_channels=payload["out_channels"],
        latent_channels=latent,
        conv_type=payload["conv_type"],
        adaptive_levels=payload["adaptive_levels"],
        conv_types=payload["conv_types"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    codes = {split: te.encode_split(model, data, split) for split in sc.SPLITS}
    train_mean = codes["train"].mean(axis=0)
    train_std = codes["train"].std(axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)

    for split in sc.SPLITS:
        z = codes[split].astype(np.float32)
        arrays = metadata_arrays(rows, split)
        if len(z) != len(arrays["scan_ids"]):
            raise ValueError(
                f"{exp_name}/{split}: {len(z)} latents vs {len(arrays['scan_ids'])} manifest rows"
            )
        arrays[f"ae_{latent}"] = z
        arrays[f"ae_standardized_{latent}"] = ((z - train_mean) / train_std).astype(np.float32)
        arrays[f"train_ae_mean_{latent}"] = train_mean.astype(np.float32)
        arrays[f"train_ae_std_{latent}"] = train_std.astype(np.float32)
        out_fp = dirs["latents"] / f"{split}_coefficients.npz"
        np.savez(out_fp, **arrays)
        print(f"[latents] {exp_name}/{split}: {z.shape} -> {out_fp}")

    del model
    torch.cuda.empty_cache()
    return {"experiment": exp_name, "latent": latent, "dir": str(dirs["latents"])}


def main():
    args = parse_args()
    device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    rows = sc.read_manifest()
    data = te.load_mesh_tensors(device, rows=rows)

    exported = [r for r in (export_one(e, rows, data, device) for e in args.exp) if r]
    if exported:
        sc.atomic_write_json(sc.OUTPUT_ROOT / "latents" / "export_summary.json", exported)
    print(f"[latents] exported {len(exported)}/{len(args.exp)} experiments")


if __name__ == "__main__":
    main()
