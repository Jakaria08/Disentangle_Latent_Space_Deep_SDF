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


def parse_args():
    default_base = Path(__file__).resolve().parent
    default_experiment = default_base / "minimal_eikonal_gmm"
    default_gt_mesh = Path(
        "/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/minimal_scaled_obj_files"
    )
    parser = argparse.ArgumentParser(
        description="Reconstruct test latents, save meshes, compute chamfer."
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
    parser.add_argument("--limit", type=int, default=0, help="Limit test samples")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")

    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--latent_lr", type=float, default=None)
    parser.add_argument("--latent_l2reg", action="store_true")
    parser.add_argument("--latent_init_std", type=float, default=None)
    parser.add_argument("--grid_res", type=int, default=None)
    parser.add_argument("--max_batch", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = args.experiment_dir
    output_dir = args.output_dir or experiment_dir

    specs = ws.load_experiment_specifications(experiment_dir)
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs.get("CodeLength", 256)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    _load_decoder_weights(experiment_dir, args.checkpoint, decoder, device)
    decoder.eval()

    clamp_dist = specs.get("ClampingDistance", 0.1)

    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    num_iterations = args.num_iterations or specs.get("EvalTestOptimizationSteps", 1000)
    num_samples = args.num_samples or specs.get("EvalTestNumSamples", 16384)
    latent_lr = args.latent_lr or specs.get("EvalTestLatentLR", 5e-3)
    latent_init_std = args.latent_init_std or specs.get("EvalTestLatentInitStd", 0.01)
    grid_res = args.grid_res or specs.get("EvalGridResolution", 256)
    max_batch = args.max_batch or int(2 ** 18)
    latent_l2reg = args.latent_l2reg or specs.get("EvalTestLatentL2Reg", True)

    with open(test_split_file) as f:
        test_split = json.load(f)

    test_npy_files = deep_sdf.data.get_instance_filenames(data_source, test_split)

    if args.limit and args.limit > 0 and args.limit < len(test_npy_files):
        random.seed(args.seed)
        test_npy_files = random.sample(test_npy_files, args.limit)

    latent_out_dir = os.path.join(output_dir, "test_latents")
    mesh_out_dir = os.path.join(output_dir, "reconstructed_test_shapes")
    os.makedirs(latent_out_dir, exist_ok=True)
    os.makedirs(mesh_out_dir, exist_ok=True)

    latent_map = {}
    chamfer_rows = []

    for i, npy_path in enumerate(test_npy_files):
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(data_source, npy_path)
        base_name = os.path.splitext(os.path.basename(npy_path))[0]

        sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(npy_path)
        if isinstance(sdf_samples, (list, tuple)) and len(sdf_samples) >= 2:
            sdf_samples[0] = sdf_samples[0][torch.randperm(sdf_samples[0].shape[0])]
            sdf_samples[1] = sdf_samples[1][torch.randperm(sdf_samples[1].shape[0])]

        loss_hist, latent = reconstruct.reconstruct(
            decoder,
            int(num_iterations),
            latent_size,
            sdf_samples,
            latent_init_std,
            clamp_dist,
            num_samples=int(num_samples),
            lr=latent_lr,
            l2reg=latent_l2reg,
            return_loss_hist=True,
        )

        latent_map[base_name] = latent.detach().cpu()

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
            print(f"{i + 1}/{len(test_npy_files)} {base_name} chamfer={float(cd):.6f}")
        else:
            reason = []
            if not os.path.isfile(gt_path):
                reason.append("missing_gt")
            if gen_mesh is None:
                reason.append("mesh_failed")
            reason_str = ",".join(reason) if reason else "unknown"
            print(f"{i + 1}/{len(test_npy_files)} {base_name} chamfer=n/a ({reason_str})")

    latents_path = os.path.join(latent_out_dir, "test_latents.pt")
    torch.save(latent_map, latents_path)

    csv_path = os.path.join(mesh_out_dir, "test_chamfer.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "chamfer"])
        writer.writerows(chamfer_rows)

    if chamfer_rows:
        mean_cd = np.mean([row[1] for row in chamfer_rows])
        print("Mean Chamfer:", mean_cd)
    else:
        print("No Chamfer computed (missing GT meshes or meshes failed).")

    print("Saved latents to:", latents_path)
    print("Saved meshes to:", mesh_out_dir)
    print("Saved chamfer CSV to:", csv_path)


if __name__ == "__main__":
    main()
