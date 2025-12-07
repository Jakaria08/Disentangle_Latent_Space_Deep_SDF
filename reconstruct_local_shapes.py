#!/usr/bin/env python3
"""
DEPRECATED: This script is outdated and missing global code support.
Please use reconstruct_local_shapes_fixed.py instead.

Reconstruction script for Deep Local Shapes.
Optimizes local latent codes to reconstruct test shapes.

WARNING: This script does NOT load or optimize global codes,
which will result in poor reconstruction quality.
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf import mesh

# Print deprecation warning
print("\n" + "="*70)
print("WARNING: reconstruct_local_shapes.py is DEPRECATED!")
print("This script does not support global codes and will produce poor results.")
print("Please use: reconstruct_local_shapes_fixed.py")
print("="*70 + "\n")
logging.warning("This script is deprecated. Use reconstruct_local_shapes_fixed.py instead.")


def reconstruct_local(
    decoder,
    num_iterations,
    local_code_size,
    num_local_codes,
    test_sdf,
    clamp_dist,
    num_samples=30000,
    lr=5e-3,
    l2reg=True,
):
    """
    Reconstruct a shape by optimizing its local latent codes.
    
    Args:
        decoder: Local shapes decoder network
        num_iterations: Number of optimization steps
        local_code_size: Dimension of each local code (e.g., 32)
        num_local_codes: Number of local codes per shape (e.g., 8^3 = 512)
        test_sdf: Test SDF samples
        clamp_dist: Clamping distance for SDF
        num_samples: Number of samples per iteration
        lr: Learning rate
        l2reg: Whether to use L2 regularization
    
    Returns:
        (loss, latent_codes) where latent_codes is [num_local_codes, local_code_size]
    """
    
    def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    
    # Initialize local codes with small random values
    latent = torch.ones(num_local_codes, local_code_size).normal_(mean=0, std=0.01).cuda()
    latent.requires_grad = True
    
    optimizer = torch.optim.Adam([latent], lr=lr)
    
    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    
    for e in range(num_iterations):
        decoder.eval()
        
        # Sample SDF data
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(test_sdf, num_samples).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
        
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        
        optimizer.zero_grad()
        
        # Prepare inputs for decoder
        # Create fake batch where all points belong to shape 0
        batch_size = xyz.shape[0]
        shape_indices = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        # Reshape latent to [1, num_local_codes, local_code_size] (1 shape)
        all_local_codes = latent.unsqueeze(0)
        
        # Forward pass
        pred_sdf = decoder(xyz, all_local_codes, shape_indices)
        
        if e == 0:
            pred_sdf = decoder(xyz, all_local_codes, shape_indices)
        
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        
        loss = loss_l1(pred_sdf, sdf_gt)
        
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        
        loss.backward()
        optimizer.step()
        
        if e % 50 == 0:
            logging.debug(f"Iteration {e}, loss: {loss.cpu().data.numpy()}, latent norm: {latent.norm()}")
        
        loss_num = loss.cpu().data.numpy()
    
    return loss_num, latent


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(
        description="Reconstruct shapes using trained Deep Local Shapes decoder"
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="Experiment directory with trained model",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="Checkpoint to use (epoch number or 'latest')",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="Data source directory",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="Split file to reconstruct",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        type=int,
        help="Number of optimization iterations",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip already reconstructed meshes",
    )
    
    deep_sdf.add_common_args(arg_parser)
    
    args = arg_parser.parse_args()
    
    deep_sdf.configure_logging(args)
    
    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    
    if not os.path.isfile(specs_filename):
        raise Exception(
            'Experiment directory does not include specifications file "specs.json"'
        )
    
    import json
    specs = json.load(open(specs_filename))
    
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    
    local_code_size = specs.get("LocalCodeSize", 32)
    grid_size = specs.get("GridSize", 8)
    num_local_codes = grid_size ** 3
    
    logging.info(f"Grid size: {grid_size}x{grid_size}x{grid_size} = {num_local_codes} local codes")
    logging.info(f"Local code size: {local_code_size}")
    
    decoder = arch.Decoder(local_code_size, **specs["NetworkSpecs"])
    
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.cuda()
    
    with open(args.split_filename, "r") as f:
        split = json.load(f)
    
    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)
    
    logging.debug(decoder)
    
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )
    
    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)
    
    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)
    
    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)
    
    err_sum = 0.0
    
    for ii, npz in enumerate(npz_filenames):
        
        if "npz" not in npz:
            continue
        
        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        
        logging.info(f"Reconstructing {npz} ({ii+1}/{len(npz_filenames)})")
        
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        
        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
        latent_filename = os.path.join(
            reconstruction_codes_dir, npz[:-4] + ".pth"
        )
        
        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
            and os.path.isfile(latent_filename)
        ):
            logging.info(f"Skipping {npz}")
            continue
        
        # Shuffle data
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
        
        import time
        start = time.time()
        
        err, latent = reconstruct_local(
            decoder.local_decoder,  # Use the actual decoder part
            int(args.iterations),
            local_code_size,
            num_local_codes,
            data_sdf,
            specs["ClampingDistance"],
            num_samples=8000,
            lr=5e-3,
            l2reg=True,
        )
        
        logging.info(f"Reconstruction time: {time.time() - start:.2f}s")
        err_sum += err
        logging.info(f"Average error: {err_sum / (ii + 1):.6f}")
        logging.debug(f"Latent norm: {latent.norm().item()}")
        
        decoder.eval()
        
        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        
        start = time.time()
        with torch.no_grad():
            # Create mesh using the optimized local codes
            # We need a wrapper class for the decoder
            class DecoderWrapper(torch.nn.Module):
                """Wrapper to make local decoder compatible with mesh creation."""
                def __init__(self, decoder_module, latent_codes, local_code_size):
                    super().__init__()
                    self.decoder = decoder_module
                    self.latent = latent_codes
                    self.local_code_size = local_code_size
                
                def forward(self, samples):
                    """
                    samples: [N, latent_size + 3] - we ignore latent part, use xyz only
                    Returns: [N, 1] SDF values
                    """
                    xyz = samples[:, -3:]  # Extract xyz coordinates
                    batch_size = xyz.shape[0]
                    
                    # All points belong to shape index 0
                    shape_indices = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
                    
                    # Reshape our local codes: [num_local_codes, local_code_size] -> [1, num_local_codes, local_code_size]
                    all_local_codes = self.latent.unsqueeze(0)
                    
                    # Forward pass through local decoder
                    return self.decoder.local_decoder(xyz, all_local_codes, shape_indices)
            
            decoder_wrapper = DecoderWrapper(decoder, latent, local_code_size).cuda()
            
            # Create a dummy latent vector (required by mesh.create_mesh signature but ignored)
            dummy_latent = torch.zeros(1, local_code_size).cuda()
            
            mesh.create_mesh(
                decoder_wrapper,
                dummy_latent,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18)
            )
        
        logging.info(f"Mesh creation time: {time.time() - start:.2f}s")
        
        if not os.path.exists(os.path.dirname(latent_filename)):
            os.makedirs(os.path.dirname(latent_filename))
        
        # Save local codes: [num_local_codes, local_code_size]
        torch.save(latent.unsqueeze(0), latent_filename)
    
    logging.info(f"Reconstruction complete. Average error: {err_sum / len(npz_filenames):.6f}")
