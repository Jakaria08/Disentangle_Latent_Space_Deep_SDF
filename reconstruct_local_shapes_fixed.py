#!/usr/bin/env python3
"""
Fixed reconstruction script for Deep Local Shapes.
Jointly optimizes BOTH global and local latent codes for best reconstruction quality.
"""

import argparse
import logging
import os
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf import mesh


def reconstruct_local_global(
    decoder,
    num_iterations,
    global_code_size,
    local_code_size,
    num_local_codes,
    test_sdf,
    clamp_dist,
    num_samples=30000,
    global_lr=5e-4,
    local_lr=5e-3,
    l2reg=True,
    init_global_code=None,
):
    """
    Reconstruct a shape by jointly optimizing BOTH global and local latent codes.
    
    Args:
        decoder: Local shapes decoder network
        num_iterations: Number of optimization steps
        global_code_size: Dimension of global code (e.g., 256)
        local_code_size: Dimension of each local code (e.g., 32)
        num_local_codes: Number of local codes per shape (e.g., 16^3 = 4096)
        test_sdf: Test SDF samples
        clamp_dist: Clamping distance for SDF
        num_samples: Number of samples per iteration
        global_lr: Learning rate for global code
        local_lr: Learning rate for local codes
        l2reg: Whether to use L2 regularization
        init_global_code: Optional initial global code [global_code_size]
    
    Returns:
        (loss, global_code, local_codes)
    """
    
    def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    
    # Initialize GLOBAL code
    if init_global_code is not None:
        global_code = init_global_code.clone().cuda()
    else:
        global_code = torch.ones(global_code_size).normal_(mean=0, std=0.01).cuda()
    global_code.requires_grad = True
    
    # Initialize LOCAL codes with small random values
    local_codes = torch.ones(num_local_codes, local_code_size).normal_(mean=0, std=0.01).cuda()
    local_codes.requires_grad = True
    
    # Separate optimizers with different learning rates
    optimizer = torch.optim.Adam([
        {'params': [global_code], 'lr': global_lr},
        {'params': [local_codes], 'lr': local_lr}
    ])
    
    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    
    for e in range(num_iterations):
        decoder.eval()
        
        # Sample SDF data
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(test_sdf, num_samples).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
        
        adjust_learning_rate(global_lr, optimizer, e, decreased_by, adjust_lr_every)
        
        optimizer.zero_grad()
        
        # Prepare inputs for decoder
        batch_size = xyz.shape[0]
        shape_indices = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        # Expand global code for all points: [batch_size, global_code_size]
        batch_global_codes = global_code.unsqueeze(0).expand(batch_size, -1)
        
        # Reshape local codes to [1, num_local_codes, local_code_size] (1 shape)
        all_local_codes = local_codes.unsqueeze(0)
        
        # Forward pass with BOTH global and local codes
        pred_sdf = decoder(xyz, batch_global_codes, all_local_codes, shape_indices)
        
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        
        loss = loss_l1(pred_sdf, sdf_gt)
        
        if l2reg:
            # Regularize both global and local codes
            global_reg = 1e-4 * torch.mean(global_code.pow(2))
            local_reg = 1e-4 * torch.mean(local_codes.pow(2))
            loss = loss + global_reg + local_reg
        
        loss.backward()
        optimizer.step()
        
        if e % 50 == 0:
            logging.info(f"Iter {e}/{num_iterations}, loss: {loss.item():.6f}, "
                        f"global_norm: {global_code.norm().item():.4f}, "
                        f"local_norm: {local_codes.norm().item():.4f}")
        
        loss_num = loss.cpu().data.numpy()
    
    return loss_num, global_code, local_codes


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(
        description="Reconstruct shapes using trained Deep Local Shapes decoder (FIXED VERSION)"
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
    arg_parser.add_argument(
        "--use_trained_codes",
        dest="use_trained_codes",
        action="store_true",
        help="Initialize with trained global codes if available (from training split)",
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
    global_code_size = specs.get("GlobalCodeSize", 256)
    grid_size = specs.get("GridSize", 8)
    num_local_codes = grid_size ** 3
    
    logging.info(f"Grid size: {grid_size}x{grid_size}x{grid_size} = {num_local_codes} local codes")
    logging.info(f"Local code size: {local_code_size}")
    logging.info(f"Global code size: {global_code_size}")
    
    decoder = arch.Decoder(local_code_size, **specs["NetworkSpecs"])
    
    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.cuda()
    
    # Try to load trained latent codes for initialization (if requested)
    trained_global_codes = None
    if args.use_trained_codes:
        global_codes_path = os.path.join(
            args.experiment_directory,
            ws.latent_codes_subdir,
            args.checkpoint + "_global.pth"
        )
        if os.path.exists(global_codes_path):
            logging.info(f"Loading trained global codes from {global_codes_path}")
            trained_global_codes = torch.load(global_codes_path)["latent_codes"]
            logging.info(f"Loaded {trained_global_codes.shape[0]} trained global codes")
        else:
            logging.warning(f"Trained global codes not found at {global_codes_path}")
    
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
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Reconstructing {npz} ({ii+1}/{len(npz_filenames)})")
        logging.info(f"{'='*60}")
        
        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        
        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
        latent_filename_local = os.path.join(
            reconstruction_codes_dir, npz[:-4] + "_local.pth"
        )
        latent_filename_global = os.path.join(
            reconstruction_codes_dir, npz[:-4] + "_global.pth"
        )
        
        if (
            args.skip
            and os.path.isfile(mesh_filename + ".ply")
            and os.path.isfile(latent_filename_local)
            and os.path.isfile(latent_filename_global)
        ):
            logging.info(f"Skipping {npz}")
            continue
        
        # Shuffle data
        data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
        data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]
        
        # Try to find initialization for global code
        init_global = None
        if trained_global_codes is not None and args.use_trained_codes:
            # Try to match by filename in the training set
            # This is a heuristic - you may need to adjust based on your dataset structure
            shape_name = npz[:-4]
            # You would need to implement proper matching here
            # For now, just use random initialization
            pass
        
        import time
        start = time.time()
        
        err, global_code, local_codes = reconstruct_local_global(
            decoder,  # Use wrapper, not inner module
            int(args.iterations),
            global_code_size,
            local_code_size,
            num_local_codes,
            data_sdf,
            specs["ClampingDistance"],
            num_samples=8000,
            global_lr=5e-4,  # Lower LR for global (more stable)
            local_lr=5e-3,   # Higher LR for local (more details)
            l2reg=True,
            init_global_code=init_global,
        )
        
        logging.info(f"Reconstruction time: {time.time() - start:.2f}s")
        err_sum += err
        logging.info(f"Reconstruction loss: {err:.6f}")
        logging.info(f"Average loss: {err_sum / (ii + 1):.6f}")
        logging.info(f"Global code norm: {global_code.norm().item():.4f}")
        logging.info(f"Local codes norm: {local_codes.norm().item():.4f}")
        
        decoder.eval()
        
        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))
        
        start = time.time()
        with torch.no_grad():
            # Create mesh using the optimized global + local codes
            class DecoderWrapper(torch.nn.Module):
                """Wrapper to make local decoder compatible with mesh creation."""
                def __init__(self, decoder_module, global_code, local_codes):
                    super().__init__()
                    self.decoder = decoder_module
                    self.global_code = global_code
                    self.local_codes = local_codes
                
                def forward(self, samples):
                    """
                    samples: [N, latent_size + 3] - we ignore latent part, use xyz only
                    Returns: [N, 1] SDF values
                    """
                    xyz = samples[:, -3:]  # Extract xyz coordinates
                    batch_size = xyz.shape[0]
                    
                    # All points belong to shape index 0
                    shape_indices = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
                    
                    # Expand global code for all points
                    batch_global_codes = self.global_code.unsqueeze(0).expand(batch_size, -1)
                    
                    # Reshape local codes: [num_local_codes, local_code_size] -> [1, num_local_codes, local_code_size]
                    all_local_codes = self.local_codes.unsqueeze(0)
                    
                    # Forward pass through decoder wrapper with BOTH codes
                    return self.decoder(xyz, batch_global_codes, all_local_codes, shape_indices)
            
            decoder_wrapper = DecoderWrapper(decoder, global_code, local_codes).cuda()
            
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
        
        if not os.path.exists(os.path.dirname(latent_filename_local)):
            os.makedirs(os.path.dirname(latent_filename_local))
        
        # Save BOTH global and local codes
        torch.save({"latent": local_codes.unsqueeze(0).cpu()}, latent_filename_local)
        torch.save({"latent": global_code.unsqueeze(0).cpu()}, latent_filename_global)
        
        logging.info(f"Saved codes to:")
        logging.info(f"  Global: {latent_filename_global}")
        logging.info(f"  Local: {latent_filename_local}")
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Reconstruction complete!")
    logging.info(f"Average loss: {err_sum / len(npz_filenames):.6f}")
    logging.info(f"{'='*60}")
