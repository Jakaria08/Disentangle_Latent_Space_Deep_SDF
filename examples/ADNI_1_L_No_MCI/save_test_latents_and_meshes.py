#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import deep_sdf
from deep_sdf import mesh, metrics
import deep_sdf.workspace as ws
import reconstruct


def _load_decoder_weights(exp_dir, checkpoint, model, device):
    model_dir = ws.get_model_params_dir(exp_dir)
    path = os.path.join(model_dir, checkpoint + ".pth")
    data = torch.load(path, map_location=device)
    state = data.get("model_state_dict", data)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return data.get("epoch")

def _parse_csv_list(v: str):
    if v is None:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _infer_val_split_from_train(train_split_path: str) -> str | None:
    """Best-effort inference of a validation split file from a train split path."""
    if not train_split_path:
        return None
    p = Path(train_split_path)
    if "train_split" in p.name:
        cand = p.with_name(p.name.replace("train_split", "val_split"))
        if cand.is_file():
            return str(cand)
    # Fallback: any val_split*.json in the same directory.
    for cand in sorted(p.parent.glob("val_split*.json")):
        if cand.is_file():
            return str(cand)
    return None


def _resolve_split_files(specs: dict) -> dict:
    split_files = {}
    if "TrainSplit" in specs:
        split_files["train"] = specs["TrainSplit"]
    if "TestSplit" in specs:
        split_files["test"] = specs["TestSplit"]

    val = (
        specs.get("ValSplit")
        or specs.get("ValidationSplit")
        or specs.get("ValidSplit")
        or specs.get("ValidationSplitFile")
    )
    if val:
        split_files["val"] = val
    else:
        inferred = _infer_val_split_from_train(split_files.get("train"))
        if inferred:
            split_files["val"] = inferred

    return split_files


def parse_args():
    default_base = Path(__file__).resolve().parent
    default_experiment = default_base / "minimal_eikonal_gmm"
    default_gt_mesh = Path(
        "/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/minimal_scaled_obj_files"
    )
    parser = argparse.ArgumentParser(
        description="Reconstruct DeepSDF latents for train/test/val splits (optionally save meshes)."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_dir",
        default=str(default_experiment),
        help="Experiment directory containing specs.json and ModelParameters/latest.pth",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output dir for latents/meshes (default: <experiment_dir>)",
    )
    parser.add_argument(
        "--gt_mesh_dir",
        default=str(default_gt_mesh),
        help="Directory containing ground-truth meshes",
    )
    parser.add_argument("--gt_ext", default=".obj", help="GT mesh extension")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint name")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument(
        "--splits",
        default="train,test,val",
        help="Comma-separated split names to process: train,test,val (default: all three)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit samples per split (0 = no limit)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")

    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument(
        "--iters_train",
        type=int,
        default=None,
        help="Override num_iterations for train split only (default: num_iterations).",
    )
    parser.add_argument(
        "--iters_test",
        type=int,
        default=None,
        help="Override num_iterations for test split only (default: 2 * num_iterations).",
    )
    parser.add_argument(
        "--iters_val",
        type=int,
        default=None,
        help="Override num_iterations for val split only (default: 2 * num_iterations).",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--latent_lr", type=float, default=None)
    parser.add_argument("--latent_l2reg", action="store_true")
    parser.add_argument("--latent_init_std", type=float, default=None)
    parser.add_argument(
        "--no_latent_init_from_train",
        action="store_true",
        help="Disable initializing latents from train mean/std (when available).",
    )
    parser.add_argument("--grid_res", type=int, default=None)
    parser.add_argument("--max_batch", type=int, default=None)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip shapes already present in the output <split>_latents.pt (if it exists).",
    )

    # Training-like latent constraints (recommended).
    parser.add_argument(
        "--code_reg_lambda",
        type=float,
        default=None,
        help="Latent L2 regularization weight (default: 1e-3).",
    )
    parser.add_argument(
        "--code_reg_type",
        default="l2_norm",
        help="Latent regularizer type: l2_norm (train-like) or l2_sq.",
    )
    parser.add_argument(
        "--code_bound",
        type=float,
        default=None,
        help="Hard L2 norm bound for the latent code (default: CodeBound from specs).",
    )

    # Optional distribution-matching penalty (use train latent stats as a prior).
    parser.add_argument(
        "--dist_match",
        action="store_true",
        help="Enable distribution-matching penalty (default: enabled).",
    )
    parser.add_argument(
        "--no_dist_match",
        action="store_true",
        help="Disable distribution-matching penalty.",
    )
    parser.add_argument(
        "--dist_match_weight",
        type=float,
        default=2e-2,
        help="Weight for distribution-matching penalty (used only if --dist_match).",
    )
    parser.add_argument(
        "--dist_match_type",
        default="zscore_l2",
        help="Penalty type: zscore_l2 (default), l2, or l1.",
    )
    parser.add_argument(
        "--dist_match_from",
        default=None,
        help=(
            "Path to train_latents.pt to compute target mean/std. "
            "If omitted, tries <output_dir>/train_latents/train_latents.pt "
            "or uses the train split computed in this run."
        ),
    )
    parser.add_argument(
        "--dist_match_splits",
        default="test,val",
        help="Comma-separated splits to apply the penalty to (default: test,val).",
    )

    # Mesh generation is optional and OFF by default.
    parser.add_argument(
        "--save_meshes",
        action="store_true",
        help="Also reconstruct meshes (and chamfer if GT meshes exist). Default: off.",
    )
    parser.add_argument(
        "--mesh_splits",
        default="test",
        help="Comma-separated split names to mesh when --save_meshes is set (default: test).",
    )
    parser.add_argument(
        "--sweep_presets",
        action="store_true",
        help=(
            "Run two preset reconstruction settings for comparison and save outputs in "
            "separate subfolders under output_dir."
        ),
    )

    return parser.parse_args()


def _latent_stats(z: torch.Tensor) -> dict:
    z_np = z.detach().cpu().numpy()
    return {
        "shape": list(z_np.shape),
        "global_mean": float(z_np.mean()),
        "global_std": float(z_np.std()),
        "global_min": float(z_np.min()),
        "global_max": float(z_np.max()),
        "per_dim_mean_abs": float(np.mean(np.abs(z_np.mean(axis=0)))),
        "per_dim_std_avg": float(np.mean(z_np.std(axis=0))),
    }


def _compare_splits(train_z: torch.Tensor, other_z: torch.Tensor) -> dict:
    mean_diff = (train_z.mean(dim=0) - other_z.mean(dim=0)).cpu().numpy()
    std_diff = (train_z.std(dim=0) - other_z.std(dim=0)).cpu().numpy()
    l2_mean = float(np.linalg.norm(mean_diff))
    l2_std = float(np.linalg.norm(std_diff))
    mean_train = train_z.mean(dim=0)
    mean_other = other_z.mean(dim=0)
    cos_sim = float((mean_train * mean_other).sum() / (mean_train.norm() * mean_other.norm() + 1e-8))
    return {
        "l2_mean": l2_mean,
        "l2_std": l2_std,
        "cosine_mean": cos_sim,
    }


def _latent_map_to_tensor(latent_map: dict) -> torch.Tensor | None:
    if not isinstance(latent_map, dict) or not latent_map:
        return None
    keys = sorted(latent_map.keys())
    return torch.stack([latent_map[k] for k in keys], dim=0)


def _load_latent_map(path: str) -> dict | None:
    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _compute_latent_mean_std(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = latents.mean(dim=0)
    std = latents.std(dim=0)
    return mean, std


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = args.experiment_dir
    base_output_dir = args.output_dir or experiment_dir

    specs = ws.load_experiment_specifications(experiment_dir)
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs.get("CodeLength", 256)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    _load_decoder_weights(experiment_dir, args.checkpoint, decoder, device)
    decoder.eval()

    clamp_dist = specs.get("ClampingDistance", 0.1)

    data_source = specs["DataSource"]
    split_files = _resolve_split_files(specs)
    requested_splits = [s.lower() for s in _parse_csv_list(args.splits)]
    if not requested_splits:
        raise ValueError("--splits must include at least one of: train,test,val")

    num_iterations = args.num_iterations if args.num_iterations is not None else 300
    default_iters_train = int(args.iters_train) if args.iters_train is not None else int(num_iterations)
    default_iters_test = int(args.iters_test) if args.iters_test is not None else int(num_iterations * 2)
    default_iters_val = int(args.iters_val) if args.iters_val is not None else int(num_iterations * 2)
    num_samples = args.num_samples or specs.get(
        "EvalTestNumSamples", specs.get("SamplesPerScene", 16384)
    )
    latent_lr = args.latent_lr or specs.get("EvalTestLatentLR", 5e-3)
    latent_init_std = args.latent_init_std or specs.get("EvalTestLatentInitStd", 0.01)
    grid_res = args.grid_res or specs.get("EvalGridResolution", 256)
    max_batch = args.max_batch or int(2 ** 18)

    mesh_splits = {s.lower() for s in _parse_csv_list(args.mesh_splits)}

    dist_match_enabled = not bool(args.no_dist_match)
    dist_match_weight = float(args.dist_match_weight)
    dist_match_type = args.dist_match_type
    dist_match_splits = {s.lower() for s in _parse_csv_list(args.dist_match_splits)}
    if dist_match_enabled and dist_match_weight <= 0:
        print("dist_match enabled but dist_match_weight <= 0; disabling dist_match.")
        dist_match_enabled = False
    latent_init_from_train = dist_match_enabled and (not args.no_latent_init_from_train)
    if dist_match_enabled and "train" in requested_splits:
        requested_splits = ["train"] + [s for s in requested_splits if s != "train"]

    # Training-like defaults (used unless overridden per setting).
    default_code_reg_lambda = float(args.code_reg_lambda) if args.code_reg_lambda is not None else 1e-3
    default_code_reg_type = args.code_reg_type
    default_code_bound = specs.get("CodeBound", None) if args.code_bound is None else float(args.code_bound)

    # Legacy L2 reg toggle (kept for compatibility). If code_reg_lambda==0, this can still apply.
    latent_l2reg = bool(args.latent_l2reg or specs.get("EvalTestLatentL2Reg", True))

    # Settings sweep (optional)
    if args.sweep_presets:
        settings = [
            {
                "name": "lambda_5e-4_iters_500",
                "code_reg_lambda": 5e-4,
                "num_iterations": 500,
                "latent_lr": latent_lr,
                "latent_init_std": latent_init_std,
                "code_reg_type": "l2_norm",
                "code_bound": default_code_bound,
            },
            {
                "name": "lambda_1e-3_iters_300",
                "code_reg_lambda": 1e-3,
                "num_iterations": 300,
                "latent_lr": latent_lr,
                "latent_init_std": latent_init_std,
                "code_reg_type": "l2_norm",
                "code_bound": default_code_bound,
            },
        ]
    else:
        settings = [
            {
                "name": "default",
                "code_reg_lambda": default_code_reg_lambda if args.code_reg_lambda is None else float(args.code_reg_lambda),
                "num_iterations": int(num_iterations),
                "latent_lr": latent_lr,
                "latent_init_std": latent_init_std,
                "code_reg_type": default_code_reg_type,
                "code_bound": default_code_bound,
            }
        ]

    def _iters_for_split(split_name: str, base_iters: int) -> int:
        if split_name == "train":
            return default_iters_train
        if split_name == "test":
            return default_iters_test
        if split_name == "val":
            return default_iters_val
        return int(base_iters)

    sweep_summary = {}

    for setting in settings:
        setting_name = setting["name"]
        output_dir = os.path.join(base_output_dir, setting_name) if args.sweep_presets else base_output_dir
        os.makedirs(output_dir, exist_ok=True)
        sweep_summary[setting_name] = {}

        print(
            f"== Setting: {setting_name} | iters={setting['num_iterations']} "
            f"lambda={setting['code_reg_lambda']} type={setting['code_reg_type']} bound={setting['code_bound']} =="
        )
        if dist_match_enabled:
            print(
                f"[{setting_name}] dist_match enabled: weight={dist_match_weight} "
                f"type={dist_match_type} splits={sorted(dist_match_splits)}"
            )
        if latent_init_from_train:
            print(f"[{setting_name}] latent_init_from_train: per-dim mean/std (when available)")

        per_split_latents = {}
        dist_mean = None
        dist_std = None
        dist_source = None

        if dist_match_enabled:
            dist_source = args.dist_match_from
            if dist_source is None:
                cand = os.path.join(output_dir, "train_latents", "train_latents.pt")
                if os.path.isfile(cand):
                    dist_source = cand
            if dist_source is not None:
                latent_map = _load_latent_map(dist_source)
                latents_tensor = _latent_map_to_tensor(latent_map or {})
                if latents_tensor is None:
                    raise ValueError(f"dist_match_from is not a latent map: {dist_source}")
                dist_mean, dist_std = _compute_latent_mean_std(latents_tensor)
                dist_mean = dist_mean.to(device=device)
                dist_std = dist_std.to(device=device)
                stats = _latent_stats(latents_tensor)
                print(
                    f"[{setting_name}] dist_match stats from {dist_source}: "
                    f"mean={stats['global_mean']:.6f} std={stats['global_std']:.6f} "
                    f"per_dim_std_avg={stats['per_dim_std_avg']:.6f}"
                )
                stats_path = os.path.join(output_dir, "dist_match_stats.json")
                with open(stats_path, "w") as f:
                    json.dump(
                        {
                            "source": dist_source,
                            "global": stats,
                            "mean": dist_mean.cpu().tolist(),
                            "std": dist_std.cpu().tolist(),
                        },
                        f,
                        indent=2,
                    )

        for split_name in requested_splits:
            if split_name not in split_files:
                raise ValueError(
                    f"Requested split '{split_name}' but no split file is defined (or inferable) in specs.json."
                )
            split_path = split_files[split_name]
            if not os.path.isfile(split_path):
                raise FileNotFoundError(f"Split file not found: {split_path}")

            with open(split_path) as f:
                split_list = json.load(f)

            instance_files = deep_sdf.data.get_instance_filenames(data_source, split_list)
            if args.limit and args.limit > 0 and args.limit < len(instance_files):
                random.seed(args.seed)
                instance_files = random.sample(instance_files, args.limit)

            latent_out_dir = os.path.join(output_dir, f"{split_name}_latents")
            os.makedirs(latent_out_dir, exist_ok=True)
            latents_path = os.path.join(latent_out_dir, f"{split_name}_latents.pt")

            latent_map = {}
            if args.skip_existing and os.path.isfile(latents_path):
                try:
                    latent_map = torch.load(latents_path, map_location="cpu")
                    if not isinstance(latent_map, dict):
                        latent_map = {}
                except Exception:
                    latent_map = {}

            # Optional mesh output per split.
            do_meshes = bool(args.save_meshes and (split_name in mesh_splits))
            chamfer_rows = []
            loss_rows = []
            mesh_out_dir = os.path.join(output_dir, f"reconstructed_{split_name}_shapes")
            if do_meshes:
                os.makedirs(mesh_out_dir, exist_ok=True)

            split_iters = _iters_for_split(split_name, int(setting["num_iterations"]))
            print(
                f"[{setting_name}:{split_name}] shapes={len(instance_files)} iters={split_iters} "
                f"samples={int(num_samples)} lr={setting['latent_lr']} init_std={setting['latent_init_std']} "
                f"code_reg_lambda={setting['code_reg_lambda']} code_reg_type={setting['code_reg_type']} "
                f"code_bound={setting['code_bound']} meshes={'on' if do_meshes else 'off'}"
            )

            for i, npz_path in enumerate(instance_files):
                if not os.path.isabs(npz_path):
                    npz_path = os.path.join(data_source, npz_path)
                base_name = os.path.splitext(os.path.basename(npz_path))[0]

                if args.skip_existing and base_name in latent_map:
                    if (i + 1) % 50 == 0 or i == 0:
                        print(
                            f"[{setting_name}:{split_name}] {i + 1}/{len(instance_files)} "
                            f"skipping existing: {base_name}"
                        )
                    continue

                sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(npz_path)
                if isinstance(sdf_samples, (list, tuple)) and len(sdf_samples) >= 2:
                    sdf_samples[0] = sdf_samples[0][torch.randperm(sdf_samples[0].shape[0])]
                    sdf_samples[1] = sdf_samples[1][torch.randperm(sdf_samples[1].shape[0])]

                apply_dist = bool(dist_match_enabled and (split_name in dist_match_splits))
                if apply_dist and dist_mean is None:
                    raise RuntimeError(
                        "dist_match enabled but no train stats are available. "
                        "Include 'train' in --splits or pass --dist_match_from."
                    )

                if latent_init_from_train and dist_mean is not None and (split_name in dist_match_splits):
                    init_stat = (dist_mean, dist_std)
                else:
                    init_stat = setting["latent_init_std"]

                loss, latent = reconstruct.reconstruct(
                    decoder,
                    split_iters,
                    latent_size,
                    sdf_samples,
                    init_stat,
                    clamp_dist,
                    num_samples=int(num_samples),
                    lr=setting["latent_lr"],
                    l2reg=latent_l2reg,
                    code_reg_lambda=setting["code_reg_lambda"],
                    code_reg_type=setting["code_reg_type"],
                    code_bound=setting["code_bound"],
                    return_loss_hist=False,
                    dist_mean=dist_mean if apply_dist else None,
                    dist_std=dist_std if apply_dist else None,
                    dist_weight=dist_match_weight if apply_dist else 0.0,
                    dist_type=dist_match_type,
                )

                latent_map[base_name] = latent.detach().cpu()
                loss_rows.append((base_name, float(loss)))

                if do_meshes:
                    mesh_path = os.path.join(mesh_out_dir, base_name)
                    gen_mesh = mesh.create_mesh(
                        decoder,
                        latent,
                        filename=mesh_path,
                        N=grid_res,
                        max_batch=max_batch,
                        return_trimesh=True,
                    )

                    gt_path = os.path.join(args.gt_mesh_dir, base_name + args.gt_ext)
                    if os.path.isfile(gt_path) and gen_mesh is not None:
                        cd, _ = metrics.compute_metric(
                            gt_mesh=gt_path,
                            gen_mesh=gen_mesh,
                            metric="chamfer",
                        )
                        chamfer_rows.append((base_name, float(cd)))
                        print(
                            f"[{setting_name}:{split_name}] {i + 1}/{len(instance_files)} {base_name} "
                            f"loss={float(loss):.6f} chamfer={float(cd):.6f}"
                        )
                    else:
                        reason = []
                        if not os.path.isfile(gt_path):
                            reason.append("missing_gt")
                        if gen_mesh is None:
                            reason.append("mesh_failed")
                        reason_str = ",".join(reason) if reason else "unknown"
                        print(
                            f"[{setting_name}:{split_name}] {i + 1}/{len(instance_files)} {base_name} "
                            f"loss={float(loss):.6f} chamfer=n/a ({reason_str})"
                        )
                else:
                    if (i + 1) % 25 == 0 or i == 0:
                        print(
                            f"[{setting_name}:{split_name}] {i + 1}/{len(instance_files)} "
                            f"{base_name} loss={float(loss):.6f}"
                        )

            torch.save(latent_map, latents_path)
            print(f"[{setting_name}:{split_name}] Saved latents: {latents_path} (count={len(latent_map)})")

            # Save per-shape loss values.
            loss_csv_path = os.path.join(latent_out_dir, f"{split_name}_loss.csv")
            with open(loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "loss"])
                writer.writerows(loss_rows)
            if loss_rows:
                mean_loss = float(np.mean([row[1] for row in loss_rows]))
                print(f"[{setting_name}:{split_name}] Mean reconstruction loss: {mean_loss:.6f}")
            print(f"[{setting_name}:{split_name}] Saved losses: {loss_csv_path}")

            if do_meshes:
                csv_path = os.path.join(mesh_out_dir, f"{split_name}_chamfer.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["name", "chamfer"])
                    writer.writerows(chamfer_rows)

                if chamfer_rows:
                    mean_cd = np.mean([row[1] for row in chamfer_rows])
                    print(f"[{setting_name}:{split_name}] Mean Chamfer: {float(mean_cd):.6f}")
                else:
                    print(
                        f"[{setting_name}:{split_name}] No Chamfer computed "
                        f"(missing GT meshes or meshes failed)."
                    )

                print(f"[{setting_name}:{split_name}] Saved meshes: {mesh_out_dir}")
                print(f"[{setting_name}:{split_name}] Saved chamfer CSV: {csv_path}")

            if dist_match_enabled and split_name == "train" and dist_mean is None:
                latents_tensor = _latent_map_to_tensor(latent_map)
                if latents_tensor is None:
                    raise RuntimeError("Failed to compute dist_match stats: train latents are empty.")
                dist_mean, dist_std = _compute_latent_mean_std(latents_tensor)
                dist_mean = dist_mean.to(device=device)
                dist_std = dist_std.to(device=device)
                stats = _latent_stats(latents_tensor)
                dist_source = latents_path
                print(
                    f"[{setting_name}] dist_match stats from {dist_source}: "
                    f"mean={stats['global_mean']:.6f} std={stats['global_std']:.6f} "
                    f"per_dim_std_avg={stats['per_dim_std_avg']:.6f}"
                )
                stats_path = os.path.join(output_dir, "dist_match_stats.json")
                with open(stats_path, "w") as f:
                    json.dump(
                        {
                            "source": dist_source,
                            "global": stats,
                            "mean": dist_mean.cpu().tolist(),
                            "std": dist_std.cpu().tolist(),
                        },
                        f,
                        indent=2,
                    )

            # Cache tensor for stats
            keys = sorted(latent_map.keys())
            if keys:
                per_split_latents[split_name] = torch.stack([latent_map[k] for k in keys], dim=0)

        # Stats and comparison (per setting)
        if "train" in per_split_latents:
            sweep_summary[setting_name]["train"] = _latent_stats(per_split_latents["train"])
        if "test" in per_split_latents:
            sweep_summary[setting_name]["test"] = _latent_stats(per_split_latents["test"])
        if "val" in per_split_latents:
            sweep_summary[setting_name]["val"] = _latent_stats(per_split_latents["val"])

        if "train" in per_split_latents and "test" in per_split_latents:
            sweep_summary[setting_name]["train_vs_test"] = _compare_splits(
                per_split_latents["train"], per_split_latents["test"]
            )
        if "train" in per_split_latents and "val" in per_split_latents:
            sweep_summary[setting_name]["train_vs_val"] = _compare_splits(
                per_split_latents["train"], per_split_latents["val"]
            )

        # Print compact comparison for the setting
        if "train_vs_test" in sweep_summary[setting_name]:
            tt = sweep_summary[setting_name]["train_vs_test"]
            print(
                f"[{setting_name}] train-vs-test: "
                f"L2(mean)={tt['l2_mean']:.4f} L2(std)={tt['l2_std']:.4f} cos={tt['cosine_mean']:.4f}"
            )
        if "train_vs_val" in sweep_summary[setting_name]:
            tv = sweep_summary[setting_name]["train_vs_val"]
            print(
                f"[{setting_name}] train-vs-val:  "
                f"L2(mean)={tv['l2_mean']:.4f} L2(std)={tv['l2_std']:.4f} cos={tv['cosine_mean']:.4f}"
            )

    # Save summary if sweeping
    if args.sweep_presets:
        summary_path = os.path.join(base_output_dir, "latent_sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print("Saved sweep summary to:", summary_path)


if __name__ == "__main__":
    main()
