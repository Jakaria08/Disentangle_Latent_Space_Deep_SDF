#!/usr/bin/env python3
"""Phase 4/5: SpiralNet++, Adaptive-Spiral and LAMM under the same protocol matrix.

Architectures and hyperparameters are frozen at the values ADNI's Optuna search selected
(``configs/hyperparameters_z128.json``); only the weights are retrained.  Re-searching per
cohort would cost roughly twenty hours per study and would make cohorts incomparable.

The protocols differ in exactly one respect - where the fitted parameters come from:

``internal``  train on the cohort's own train split, using that cohort's own template,
              decimation hierarchy and per-vertex normalisation.
``external``  the reference (ADNI) checkpoint evaluated on the target cohort, using the
              *reference* template, hierarchy and normalisation.  This is the subtle one:
              ``spiral_common`` derives the template and normalisation from whichever
              manifest it is pointed at, so re-deriving them from the target would feed the
              network a different input distribution than it was trained on and quietly
              turn an external validation into a partial refit.  This script therefore
              keeps the reference environment and streams target vertices through it.
``pooled`` / ``loco``  train on a merged manifest built from several cohorts' train splits,
              evaluate per cohort.

The spiral trainers are driven through their own library functions, and LAMM through its
CLI, so the training code here is the same code that produced the ADNI results.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import xcohort_common as xc

PYTORCH_GEO_PYTHON = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["spiralnet_z128", "adaptive_z128", "lamm_z128"])
    parser.add_argument("--protocol", required=True, choices=["internal", "external", "pooled", "loco"])
    parser.add_argument("--cohorts", nargs="+", default=None, help="Cohorts to evaluate.")
    parser.add_argument("--held-out", default=None, help="loco only: the cohort to hold out.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reference-checkpoint", type=Path, default=None,
                        help="external only: the reference checkpoint to evaluate.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke", type=int, default=0,
        help="Train for this many epochs instead of the configured budget, to prove the path end to end.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# manifests for multi-cohort training
# --------------------------------------------------------------------------------------


def merged_manifest(specs: list[xc.CohortSpec], config: xc.Config, destination: Path,
                    train_only_from: list[str] | None = None) -> Path:
    """Write one manifest spanning several cohorts, tagging each row with its cohort.

    Rows keep their original split, so a pooled model still trains on train rows only and is
    still evaluated on held-out subjects.  Subject ids are prefixed with the cohort name
    because two cohorts can legitimately use the same numeric id.
    """
    rows_out: list[dict[str, str]] = []
    for spec in specs:
        rows = xc.read_manifest(spec, config)
        for row in rows:
            if train_only_from and spec.name in train_only_from and row["split"] != "train":
                continue
            merged = dict(row)
            merged["cohort"] = spec.name
            merged["subject_id"] = f"{spec.name}:{row['subject_id']}"
            merged["scan_id"] = f"{spec.name}:{row['scan_id']}"
            rows_out.append(merged)
    if not rows_out:
        raise ValueError("merged manifest is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(destination, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    counts: dict[str, int] = {}
    for row in rows_out:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(f"[manifest] {len(rows_out)} rows ({counts}) -> {destination}", flush=True)
    return destination


def resolve_device(requested: int):
    """Map a requested GPU index onto a device that actually exists in this process.

    CUDA_VISIBLE_DEVICES renumbers the visible cards from zero, so a run pinned to physical
    card 2 sees exactly one device and it is called cuda:0.  Passing both the pin and the
    physical index - the natural thing to write - fails with 'invalid device ordinal'.
    """
    import torch

    if not torch.cuda.is_available():
        print("[device] CUDA unavailable, using CPU", flush=True)
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if requested < count:
        return torch.device(f"cuda:{requested}")
    print(f"[device] cuda:{requested} not visible ({count} device(s) visible); using cuda:0", flush=True)
    return torch.device("cuda:0")


def visible_gpu_index(requested: int) -> int:
    """The index a subprocess should be given, accounting for a CUDA_VISIBLE_DEVICES pin."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return 0
    return requested


def spiral_env(tag: str, manifest: Path, output_root: Path) -> dict[str, str]:
    """Point spiral_common (and therefore LAMM, which imports it) at one cohort."""
    env = dict(os.environ)
    env.update({
        "SPIRAL_COHORT_TAG": tag,
        "SPIRAL_MANIFEST_FP": str(manifest),
        "SPIRAL_OUTPUT_ROOT": str(output_root),
    })
    return env


# --------------------------------------------------------------------------------------
# spiral / adaptive
# --------------------------------------------------------------------------------------


def osr_cached_stack(transform, ds_key: str, seq_length: int, dilation: int, dyn_lengths, device):
    """The spiral index stack, built by the ADNI search code so it matches exactly."""
    sys.path.insert(0, str(xc.SPIRAL_SCRIPTS))
    import optuna_search as osr

    return osr.cached_spiral_stack(transform, ds_key, seq_length, dilation, dyn_lengths, device)


def build_spiral_model(spec_cfg: dict, transform, device):
    """Rebuild the searched architecture exactly as optuna_search.py did."""
    sys.path.insert(0, str(xc.SPIRAL_SCRIPTS))
    import optuna_search as osr
    from network import build_model

    latent = int(spec_cfg["latent_channels"])
    base = int(spec_cfg["base_channels"])
    ds_factors = list(spec_cfg["ds_factors"])
    n_levels = len(ds_factors)
    out_channels = [base] * (n_levels - 1) + [2 * base]
    sizes = osr.level_sizes(transform)
    conv_types = osr.plan_conv_types(n_levels, spec_cfg["conv_type"], int(spec_cfg["adaptive_levels"]), sizes)
    dyn_lengths = osr.plan_dynamic_lengths(conv_types, sizes, float(spec_cfg["dynamic_seq_frac"]))
    spirals, dynamic, down, up = osr.cached_spiral_stack(
        transform, xc.spiral_common().ds_tag(ds_factors), int(spec_cfg["seq_length"]),
        int(spec_cfg["dilation"]), dyn_lengths, device,
    )
    model = build_model(
        transform=transform, spiral_indices=spirals, dynamic_spiral_indices=dynamic,
        down_transform=down, up_transform=up, out_channels=out_channels,
        latent_channels=latent, conv_type=spec_cfg["conv_type"],
        adaptive_levels=int(spec_cfg["adaptive_levels"]), conv_types=conv_types,
        dropout=float(spec_cfg["dropout"]),
    )
    return model, {"conv_types": conv_types, "dynamic_seq_lengths": dyn_lengths, "out_channels": out_channels}


def train_spiral(model_name: str, spec_cfg: dict, tag: str, manifest: Path, output_root: Path,
                 gpu: int, seed: int) -> dict:
    """Train one spiral/adaptive model on whatever manifest the environment points at."""
    import torch

    os.environ.update(spiral_env(tag, manifest, output_root))
    sys.path.insert(0, str(xc.SPIRAL_SCRIPTS))
    import spiral_common as sc
    import train_eval as te

    device = resolve_device(gpu)
    rows = sc.read_manifest(manifest)
    data = te.load_mesh_tensors(device, rows=rows)
    transform = sc.get_transform(spec_cfg["ds_factors"], rows=rows)
    model, plan = build_spiral_model(spec_cfg, transform, device)
    model = model.to(device)

    best_val, best_state, history = te.train_model(
        model, data, epochs=int(spec_cfg["epochs"]), batch_size=int(spec_cfg["batch_size"]),
        lr=float(spec_cfg["lr"]), lr_decay=float(spec_cfg["lr_decay"]),
        decay_step=int(spec_cfg["decay_step"]), weight_decay=float(spec_cfg["weight_decay"]),
        device=device, seed=seed, noise_std=float(spec_cfg["noise_std"]),
        scheduler_type=spec_cfg["scheduler"],
    )
    model.load_state_dict(best_state)
    metrics = {split: te.evaluate(model, data, split, with_volume=True) for split in ("val", "test")}
    checkpoint = output_root / "checkpoints" / f"{model_name}_{tag}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": spec_cfg, "plan": plan, "tag": tag}, checkpoint)
    return {"best_val_rmse_mm": float(best_val), "metrics": metrics,
            "checkpoint": str(checkpoint), "epochs_run": len(history)}


def evaluate_external_spiral(model_name: str, spec_cfg: dict, checkpoint: Path, target: xc.CohortSpec,
                            config: xc.Config, gpu: int) -> dict:
    """Reference checkpoint, reference normalisation, target vertices.

    The reference environment is kept in place so the template, hierarchy and per-vertex
    statistics are the ones the weights were trained with.
    """
    import torch

    reference = config.reference()
    os.environ.update(spiral_env("adni", reference.keep_manifest, xc.BULK_ROOT / "adni"))
    sys.path.insert(0, str(xc.SPIRAL_SCRIPTS))
    import spiral_common as sc

    device = resolve_device(gpu)
    ref_rows = sc.read_manifest(reference.keep_manifest)
    ref_train = sc.load_split_vertices("train", rows=ref_rows)
    mean, std = sc.train_normalization(ref_train)          # reference statistics, not the target's
    payload = torch.load(checkpoint, map_location=device)
    # Prefer the architecture stored inside the checkpoint over the config: it is what the
    # weights were actually trained with, so the two can never drift apart.
    stored = {key: payload[key] for key in
              ("conv_type", "conv_types", "out_channels", "ds_factors", "seq_length", "dilation",
               "adaptive_levels", "dynamic_seq_lengths", "dropout", "latent_channels")
              if key in payload}
    effective = {**spec_cfg, **stored}
    transform = sc.get_transform(effective["ds_factors"], rows=ref_rows)
    if stored.get("conv_types") and stored.get("out_channels"):
        from network import build_model

        spirals, dynamic, down, up = osr_cached_stack(
            transform, sc.ds_tag(effective["ds_factors"]), int(effective["seq_length"]),
            int(effective["dilation"]), list(effective["dynamic_seq_lengths"]), device,
        )
        model = build_model(
            transform=transform, spiral_indices=spirals, dynamic_spiral_indices=dynamic,
            down_transform=down, up_transform=up, out_channels=list(effective["out_channels"]),
            latent_channels=int(effective["latent_channels"]), conv_type=effective["conv_type"],
            adaptive_levels=int(effective["adaptive_levels"]),
            conv_types=list(effective["conv_types"]), dropout=float(effective.get("dropout", 0.0)),
        )
    else:
        model, _plan = build_spiral_model(effective, transform, device)
    state = payload.get("model_state_dict") or payload.get("state_dict") or payload
    model.load_state_dict(state)
    model = model.to(device).eval()

    target_rows = xc.read_manifest(target, config)
    faces = xc.load_faces(target_rows)
    out: dict[str, dict] = {}
    for split in xc.SPLITS:
        vertices = xc.load_vertices(target, target_rows, split)
        normalised = (vertices - mean) / std
        preds = []
        with torch.no_grad():
            for start in range(0, len(normalised), 64):
                batch = torch.as_tensor(normalised[start:start + 64], dtype=torch.float32, device=device)
                preds.append(model(batch).cpu().numpy())
        pred_mm = np.concatenate(preds, axis=0) * std + mean
        out[split] = xc.reconstruction_metrics(pred_mm.astype(np.float64), vertices.astype(np.float64), faces=faces)
    return out


# --------------------------------------------------------------------------------------
# LAMM
# --------------------------------------------------------------------------------------


def train_lamm(spec_cfg: dict, tag: str, manifest: Path, output_root: Path, gpu: int, seed: int,
               dry_run: bool) -> dict:
    """LAMM ships a CLI, so it is driven as a subprocess with the cohort environment set."""
    script = xc.LAMM_SCRIPTS / "train_lamm.py"
    run_name = f"{tag}_lamm_z128_s{seed}"
    command = [
        PYTORCH_GEO_PYTHON, str(script),
        "--run-name", run_name, "--out-root", str(output_root / "lamm"),
        "--latent", str(spec_cfg["latent"]), "--dim", str(spec_cfg["dim"]),
        "--dim-head", str(spec_cfg["dim_head"]), "--heads", str(spec_cfg["heads"]),
        "--enc-depth", str(spec_cfg["enc_depth"]), "--dec-depth", str(spec_cfg["dec_depth"]),
        "--dropout", str(spec_cfg["dropout"]), "--scales", str(spec_cfg["scales"]),
        "--patch-level", str(spec_cfg["patch_level"]), "--region-mode", str(spec_cfg["region_mode"]),
        "--latent-mode", str(spec_cfg["latent_mode"]), "--norm-mode", str(spec_cfg["norm_mode"]),
        "--loss", str(spec_cfg["loss"]), "--huber-delta", str(spec_cfg["huber_delta"]),
        "--batch-size", str(spec_cfg["batch_size"]), "--epochs", str(spec_cfg["epochs"]),
        "--min-epochs", str(spec_cfg["min_epochs"]), "--patience", str(spec_cfg["patience"]),
        "--warmup-epochs", str(spec_cfg["warmup_epochs"]), "--lr", str(spec_cfg["lr"]),
        "--lr-final", str(spec_cfg["lr_final"]), "--weight-decay", str(spec_cfg["weight_decay"]),
        "--ema-decay", str(spec_cfg["ema_decay"]), "--mixup-alpha", str(spec_cfg["mixup_alpha"]),
        "--mixup-prob", str(spec_cfg["mixup_prob"]), "--time-budget", str(spec_cfg["time_budget"]),
        "--gpu", str(visible_gpu_index(gpu)), "--seed", str(seed),
    ]
    if spec_cfg.get("residual"):
        command.append("--residual")
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return {"dry_run": True, "command": command}
    subprocess.run(command, check=True, env=spiral_env(tag, manifest, output_root), cwd=str(xc.REPO_ROOT))
    summary = output_root / "lamm" / "studies" / run_name / "summary.json"
    import json
    return json.loads(summary.read_text()) if summary.is_file() else {"summary_missing": str(summary)}


# --------------------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    config = xc.load_config()
    hparams = xc.load_hyperparameters()
    reference = config.reference()
    names = args.cohorts or list(config.cohorts)
    rows_out: list[dict] = []
    run_root = xc.BULK_ROOT / "learned" / args.protocol

    for model_name in args.models:
        spec_cfg = dict(hparams["models"][model_name])
        if args.smoke:
            spec_cfg["epochs"] = args.smoke
            spec_cfg["min_epochs"] = min(int(spec_cfg.get("min_epochs", 0)), args.smoke)
            spec_cfg["time_budget"] = 600.0
            print(f"[smoke] {model_name}: {args.smoke} epochs only - results are not meaningful", flush=True)

        if args.protocol == "internal":
            for name in names:
                spec = config.cohorts[name]
                tag = f"{name}"
                output_root = run_root / name
                print(f"\n=== {model_name} | internal | {name} ===", flush=True)
                if args.dry_run:
                    print(f"  would train on {spec.keep_manifest}")
                    continue
                if spec_cfg["trainer"] == "lamm":
                    result = train_lamm(spec_cfg, tag, spec.keep_manifest, output_root, args.gpu, args.seed, False)
                    val = result.get("best_val_rmse_mm", "")
                    rows_out.append(xc.result_row(
                        protocol="internal", model=model_name, latent_k=hparams["primary_k"], fit_cohorts=name,
                        eval_cohort=name, split="val", n_scans="", vertex_rmse_mm_mean=val,
                        vertex_euclidean_mm_mean=(val * xc.EUCLIDEAN_FACTOR) if val else "",
                        normalization_source=f"self:{name}", seed=args.seed))
                else:
                    result = train_spiral(model_name, spec_cfg, tag, spec.keep_manifest, output_root,
                                          args.gpu, args.seed)
                    for split, metrics in result["metrics"].items():
                        rows_out.append(xc.result_row(
                            protocol="internal", model=model_name, latent_k=hparams["primary_k"],
                            fit_cohorts=name, eval_cohort=name, split=split, n_scans="",
                            vertex_rmse_mm_mean=metrics.get("vertex_rmse_mm_mean", ""),
                            vertex_euclidean_mm_mean=metrics.get("vertex_rmse_mm_mean", 0) * xc.EUCLIDEAN_FACTOR,
                            volume_abs_relative_error_pct_mean=metrics.get("volume_abs_relative_error_pct_mean", ""),
                            normalization_source=f"self:{name}", seed=args.seed))

        elif args.protocol == "external":
            if spec_cfg["trainer"] == "lamm":
                print(f"[{model_name}] external evaluation for LAMM needs its own decoder entry point; "
                      "run internal first and see RUNBOOK.md", flush=True)
                continue
            checkpoint = args.reference_checkpoint or hparams.get("reference_checkpoints", {}).get(model_name)
            if checkpoint is None or not Path(checkpoint).is_file():
                raise SystemExit(
                    f"external {model_name}: no reference checkpoint. Pass --reference-checkpoint or fix "
                    f"configs/hyperparameters_z128.json (looked for {checkpoint})"
                )
            for name in names:
                if name == reference.name:
                    continue
                print(f"\n=== {model_name} | external | {reference.name} -> {name} ===", flush=True)
                if args.dry_run:
                    continue
                metrics = evaluate_external_spiral(model_name, spec_cfg, Path(checkpoint),
                                                   config.cohorts[name], config, args.gpu)
                for split, values in metrics.items():
                    rows_out.append(xc.result_row(
                        protocol="external", model=model_name, latent_k=hparams["primary_k"],
                        fit_cohorts=reference.name, eval_cohort=name, split=split, n_scans="",
                        vertex_rmse_mm_mean=values["vertex_rmse_mm_mean"],
                        vertex_euclidean_mm_mean=values["vertex_euclidean_mm_mean"],
                        volume_abs_relative_error_pct_mean=values.get("volume_abs_relative_error_pct_mean", ""),
                        normalization_source=f"reference:{reference.name}", seed=args.seed))

        else:  # pooled / loco
            members = [c for c in config.pooled_members()]
            if args.protocol == "loco":
                if not args.held_out:
                    raise SystemExit("--held-out is required for the loco protocol")
                members = [c for c in members if c.name != args.held_out]
                tag = f"loco_without_{args.held_out}"
            else:
                tag = "pooled_" + "_".join(c.name for c in members)
            manifest = merged_manifest(members, config, run_root / tag / "manifest.csv")
            print(f"\n=== {model_name} | {args.protocol} | fit on {[c.name for c in members]} ===", flush=True)
            if args.dry_run:
                continue
            if spec_cfg["trainer"] == "lamm":
                train_lamm(spec_cfg, tag, manifest, run_root / tag, args.gpu, args.seed, False)
            else:
                train_spiral(model_name, spec_cfg, tag, manifest, run_root / tag, args.gpu, args.seed)

    if rows_out:
        xc.write_results(xc.TASK_ROOT / "reports" / f"phase4_{args.protocol}.csv", rows_out, append=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
