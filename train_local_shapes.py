#!/usr/bin/env python3
"""
Training script for Deep Local Shapes approach.
Uses spatial grid of local latent codes instead of single global code.
"""

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
import os
import logging
import math
import json
import time
import random
import numpy as np

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, plotting, utils
import deep_sdf.workspace as ws

# Import regular functions from original train script
from train_deep_sdf import (
    save_model, save_optimizer, load_optimizer,
    save_logs, load_logs, clip_logs,
    get_spec_with_default, append_parameter_magnitudes
)


def save_local_latent_vectors(experiment_directory, filename, local_lat_vecs, epoch):
    """Save local latent codes: [num_shapes, num_local_codes, local_code_size]"""
    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
    
    torch.save(
        {"epoch": epoch, "latent_codes": local_lat_vecs.state_dict()},
        os.path.join(latent_codes_dir, filename),
    )


def load_local_latent_vectors(experiment_directory, filename, local_lat_vecs):
    """Load local latent codes"""
    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )
    
    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))
    
    data = torch.load(full_filename)
    local_lat_vecs.load_state_dict(data["latent_codes"])
    
    return data["epoch"]


def get_mean_local_latent_magnitude(local_lat_vecs):
    """Compute mean magnitude across all local codes"""
    # local_lat_vecs.weight: [num_shapes * num_local_codes, local_code_size]
    return torch.mean(torch.norm(local_lat_vecs.weight.data.detach(), dim=1))


def main_function(experiment_directory, continue_from, batch_split):
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Running LOCAL SHAPES training")
    logging.info("Experiment directory: " + experiment_directory)
    
    specs = ws.load_experiment_specifications(experiment_directory)
    
    # Handle Description as list or string
    description = specs.get("Description", ["No description"])
    if isinstance(description, list):
        description = " ".join(description)
    logging.info("Experiment description: \n" + description)
    
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    
    # Get local shapes specific parameters
    grid_size = get_spec_with_default(specs, "GridSize", 8)
    local_code_size = get_spec_with_default(specs, "LocalCodeSize", 16)
    global_code_size = get_spec_with_default(specs, "GlobalCodeSize", 256)
    num_local_codes = grid_size ** 3
    
    logging.info(f"Global code size: {global_code_size}")
    logging.info(f"Grid size: {grid_size}x{grid_size}x{grid_size} = {num_local_codes} local codes")
    logging.info(f"Local code size: {local_code_size}")
    
    # Load network architecture
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    
    logging.debug(specs["NetworkSpecs"])
    
    # Training parameters
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()
    
    lr_schedules = deep_sdf.lr_scheduling.get_learning_rate_schedules(specs)
    
    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))
    
    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_local_latent_vectors(experiment_directory, "latest.pth", local_lat_vecs, epoch)
        # Save global codes in the same file
        latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
        torch.save(
            {"epoch": epoch, "latent_codes": global_lat_vecs.state_dict()},
            os.path.join(latent_codes_dir, "latest_global.pth"),
        )
    
    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_local_latent_vectors(experiment_directory, str(epoch) + ".pth", local_lat_vecs, epoch)
        # Save global codes
        latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
        torch.save(
            {"epoch": epoch, "latent_codes": global_lat_vecs.state_dict()},
            os.path.join(latent_codes_dir, str(epoch) + "_global.pth"),
        )
    
    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)
    
    def adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            if isinstance(lr_schedules[i], lr_scheduling.StepLearningRateOnPlateauSchedule):
                param_group["lr"] = lr_schedules[i].get_learning_rate(epoch, loss_log_epoch)
            else:
                param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Training hyperparameters
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True
    
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    
    use_eikonal = get_spec_with_default(specs, "UseEikonal", False)
    
    # Create decoder
    decoder = arch.Decoder(local_code_size, **specs["NetworkSpecs"]).cuda()
    
    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    
    # Load train split
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    # Setup data
    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )
    
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 16)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))
    
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    
    num_scenes = len(sdf_dataset)
    logging.info("There are {} scenes".format(num_scenes))
    logging.debug(decoder)
    
    # Create GLOBAL latent codes: [num_scenes, global_code_size]
    # One global code per shape for disentangled features
    # Note: max_norm can prevent learning by forcing saturation - make it optional
    use_max_norm = get_spec_with_default(specs, "UseMaxNorm", False)
    if use_max_norm:
        global_lat_vecs = torch.nn.Embedding(num_scenes, global_code_size, max_norm=code_bound).cuda()
        logging.info(f"Using max_norm={code_bound} for global codes")
    else:
        global_lat_vecs = torch.nn.Embedding(num_scenes, global_code_size).cuda()
        logging.info("max_norm disabled for global codes (better for learning)")
    
    # Create LOCAL latent codes: [num_scenes * num_local_codes, local_code_size]
    # We'll treat this as [num_scenes, num_local_codes, local_code_size] conceptually
    total_codes = num_scenes * num_local_codes
    if use_max_norm:
        local_lat_vecs = torch.nn.Embedding(total_codes, local_code_size, max_norm=code_bound).cuda()
        logging.info(f"Using max_norm={code_bound} for local codes")
    else:
        local_lat_vecs = torch.nn.Embedding(total_codes, local_code_size).cuda()
        logging.info("max_norm disabled for local codes (better for learning)")
    
    # Initialize with small random values
    init_std = get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(local_code_size)
    global_init_std = get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(global_code_size)
    torch.nn.init.normal_(global_lat_vecs.weight.data, 0.0, global_init_std)
    torch.nn.init.normal_(local_lat_vecs.weight.data, 0.0, init_std / 2)  # Smaller init for local refinements
    
    logging.info(
        f"Initialized {num_scenes} global codes of size {global_code_size}"
    )
    logging.info(
        f"Initialized {num_scenes} shapes x {num_local_codes} local codes = {total_codes} total codes"
    )
    logging.debug(
        "initialized with mean global magnitude {}, local magnitude {}".format(
            torch.mean(torch.norm(global_lat_vecs.weight.data.detach(), dim=1)),
            get_mean_local_latent_magnitude(local_lat_vecs)
        )
    )
    
    loss_l1 = torch.nn.L1Loss(reduction="sum")
    
    # Optimizer with 3 parameter groups: decoder, global codes, local codes
    optimizer_all = torch.optim.Adam(
        [{
            "params": decoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": global_lat_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        },
        {
            "params": local_lat_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0) * 0.1,  # Lower LR for local codes
        }]
    )
    
    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))
    
    loss_log = []
    loss_log_epoch = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}
    
    start_epoch = 1
    
    if continue_from is not None:
        logging.info('continuing from "{}"'.format(continue_from))
        
        lat_epoch = load_local_latent_vectors(
            experiment_directory, continue_from + ".pth", local_lat_vecs
        )
        
        # Load global latent vectors
        global_codes_file = os.path.join(
            ws.get_latent_codes_dir(experiment_directory),
            continue_from + "_global.pth"
        )
        if os.path.isfile(global_codes_file):
            data = torch.load(global_codes_file)
            global_lat_vecs.load_state_dict(data["latent_codes"])
            global_epoch = data["epoch"]
            logging.info("Loaded global codes from epoch {}".format(global_epoch))
        else:
            logging.warning("No global codes found, using random initialization")
        
        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )
        
        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )
        
        for i, lrs in enumerate(lr_schedules):
            if isinstance(lrs, lr_scheduling.StepLearningRateOnPlateauSchedule):
                lrs.last_lr = optimizer_all.param_groups[i]["lr"]
        
        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )
        
        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )
        
        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )
        
        start_epoch = model_epoch + 1
        logging.debug("loaded")
    
    logging.info("starting from epoch {}".format(start_epoch))
    
    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of global code parameters: {} ({} shapes x {} dims)".format(
            num_scenes * global_code_size,
            num_scenes,
            global_code_size,
        )
    )
    logging.info(
        "Number of local code parameters: {} ({} shapes x {} codes x {} dims)".format(
            total_codes * local_code_size,
            num_scenes,
            num_local_codes,
            local_code_size,
        )
    )
    
    # Training loop
    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    
    for epoch in range(start_epoch, num_epochs + 1):
        
        epoch_time_start = time.time()
        logging.info("epoch {}...".format(epoch))
        
        decoder.train()
        
        epoch_losses = []
        epoch_sdf_losses = []
        epoch_reg_losses = []
        epoch_eikonal_losses = []
        
        adjust_learning_rate(lr_schedules, optimizer_all, epoch, loss_log_epoch)
        
        for sdf_data, indices in sdf_loader:
            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            
            sdf_data.requires_grad = False
            
            xyz = sdf_data[:, 0:3].cuda()
            if use_eikonal:
                xyz.requires_grad = True
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            
            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
            
            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )
            sdf_gt = torch.chunk(sdf_gt, batch_split)
            
            batch_loss_tb = 0.0
            sdf_loss_tb = 0.0
            reg_loss_tb = 0.0
            eikonal_loss_tb = 0.0
            
            optimizer_all.zero_grad()
            
            for i in range(batch_split):
                # Get GLOBAL codes for this batch
                batch_global_codes = global_lat_vecs(indices[i].cuda())  # [N, global_code_size]
                
                # Get LOCAL codes for this batch
                # indices[i]: [N] contains shape indices
                # We need to reshape local_lat_vecs to [num_shapes, num_local_codes, local_code_size]
                all_local_codes = local_lat_vecs.weight.view(num_scenes, num_local_codes, local_code_size)
                
                # Forward pass with BOTH global and local codes
                # decoder expects: (xyz, global_codes, all_local_codes, shape_indices)
                # xyz should already have requires_grad=True from earlier
                # Request touched indices for sparse regularization
                use_sparse_reg = get_spec_with_default(specs, "UseSparseRegularization", True)
                if do_code_regularization and use_sparse_reg:
                    pred_sdf, touched_indices_per_shape = decoder(
                        xyz[i], batch_global_codes, all_local_codes, indices[i].cuda(),
                        return_touched_indices=True
                    )
                else:
                    pred_sdf = decoder(xyz[i], batch_global_codes, all_local_codes, indices[i].cuda())
                    touched_indices_per_shape = None
                
                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                sdf_loss_tb += chunk_loss.item()
                
                # Regularization on BOTH global and local codes
                if do_code_regularization:
                    # Get unique indices to avoid double-counting
                    unique_shape_indices = torch.unique(indices[i])
                    
                    # Regularize GLOBAL codes (encourage disentanglement)
                    # Reuse batch_global_codes instead of fetching again to avoid in-place issues
                    # But need to map back to unique indices
                    unique_mask = torch.zeros(len(indices[i]), dtype=torch.bool, device=indices[i].device)
                    for idx in unique_shape_indices:
                        unique_mask[torch.where(indices[i] == idx)[0][0]] = True
                    used_global_codes = batch_global_codes[unique_mask]
                    
                    global_l2 = torch.sum(torch.norm(used_global_codes, dim=1))
                    global_reg_loss = code_reg_lambda * global_l2 / len(unique_shape_indices)
                    
                    # Regularize LOCAL codes with SPARSE or DENSE strategy
                    if use_sparse_reg and touched_indices_per_shape is not None:
                        # SPARSE: Only regularize codes that were accessed in this batch
                        local_l2 = 0.0
                        total_touched = 0
                        for shape_idx in unique_shape_indices:
                            shape_idx_int = shape_idx.item()
                            if shape_idx_int in touched_indices_per_shape:
                                touched_indices = list(touched_indices_per_shape[shape_idx_int])
                                # Get the touched codes for this shape
                                shape_codes = all_local_codes[shape_idx].view(-1, local_code_size)
                                touched_codes = shape_codes[touched_indices]
                                local_l2 += torch.sum(torch.norm(touched_codes, dim=1))
                                total_touched += len(touched_indices)
                        
                        local_sparsity_weight = get_spec_with_default(specs, "LocalSparsityWeight", 10.0)
                        if total_touched > 0:
                            local_reg_loss = (
                                code_reg_lambda * local_sparsity_weight * min(1, epoch / 100) * local_l2
                            ) / total_touched
                        else:
                            local_reg_loss = 0.0
                    else:
                        # DENSE: Regularize ALL local codes (original behavior, not recommended)
                        used_local_codes = []
                        for shape_idx in unique_shape_indices:
                            shape_codes = all_local_codes[shape_idx].view(-1, local_code_size)
                            used_local_codes.append(shape_codes)
                        used_local_codes = torch.cat(used_local_codes, dim=0)
                        
                        local_l2 = torch.sum(torch.norm(used_local_codes, dim=1))
                        local_sparsity_weight = get_spec_with_default(specs, "LocalSparsityWeight", 10.0)
                        local_reg_loss = (
                            code_reg_lambda * local_sparsity_weight * min(1, epoch / 100) * local_l2
                        ) / (len(unique_shape_indices) * num_local_codes)
                    
                    reg_loss = global_reg_loss + local_reg_loss
                    chunk_loss = chunk_loss + reg_loss
                    reg_loss_tb += reg_loss.item()
                
                # Eikonal loss
                if use_eikonal:
                    grad_outputs = torch.ones_like(pred_sdf, requires_grad=True)
                    gradients = torch.autograd.grad(
                        pred_sdf, [xyz[i]], 
                        grad_outputs=grad_outputs, 
                        create_graph=True, 
                        allow_unused=True, 
                        retain_graph=True
                    )[0]
                    eikonal_loss = 0.002 * ((1. - torch.linalg.vector_norm(gradients, dim=1))**2).mean()
                    chunk_loss += eikonal_loss
                    eikonal_loss_tb += eikonal_loss.item()
                
                chunk_loss.backward()
                batch_loss_tb += chunk_loss.item()
            
            logging.debug("loss = {}".format(batch_loss_tb))
            
            loss_log.append(batch_loss_tb)
            epoch_losses.append(batch_loss_tb)
            epoch_sdf_losses.append(sdf_loss_tb)
            epoch_reg_losses.append(reg_loss_tb)
            epoch_eikonal_losses.append(eikonal_loss_tb)
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip, norm_type=2)
            
            optimizer_all.step()
        
        # Log epoch
        seconds_elapsed = time.time() - epoch_time_start
        timing_log.append(seconds_elapsed)
        
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_log_epoch.append(epoch_loss)
        
        summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
        summary_writer.add_scalar("Loss/train_sdf", sum(epoch_sdf_losses)/len(epoch_sdf_losses), global_step=epoch)
        summary_writer.add_scalar("Loss/train_reg", sum(epoch_reg_losses)/len(epoch_reg_losses), global_step=epoch)
        if use_eikonal:
            summary_writer.add_scalar("Loss/train_eikonal", sum(epoch_eikonal_losses)/len(epoch_eikonal_losses), global_step=epoch)
        
        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        summary_writer.add_scalar("Learning Rate/Params", lr_log[-1][0], global_step=epoch)
        summary_writer.add_scalar("Learning Rate/Global", lr_log[-1][1], global_step=epoch)
        if len(lr_log[-1]) > 2:
            summary_writer.add_scalar("Learning Rate/Local", lr_log[-1][2], global_step=epoch)
        
        mlm = get_mean_local_latent_magnitude(local_lat_vecs)
        mgm = torch.mean(torch.norm(global_lat_vecs.weight.data.detach(), dim=1))
        lat_mag_log.append(mlm)  # Store local for backward compatibility
        summary_writer.add_scalar("Mean Latent Magnitude/local", mlm, global_step=epoch)
        summary_writer.add_scalar("Mean Latent Magnitude/global", mgm, global_step=epoch)
        
        append_parameter_magnitudes(param_mag_log, decoder)
        
        # Save checkpoints
        if epoch in checkpoints:
            save_checkpoints(epoch)
        
        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )
        
        logging.info(f"Epoch {epoch} completed in {seconds_elapsed:.2f}s, loss: {epoch_loss:.6f}")
    
    summary_writer.close()


if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Train Deep Local Shapes autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory with specs.json",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="Epoch to continue from (e.g., 'latest' or '1000')",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="Split batch for memory efficiency",
        type=int,
    )
    
    deep_sdf.add_common_args(arg_parser)
    
    args = arg_parser.parse_args()
    
    deep_sdf.configure_logging(args)
    
    main_function(args.experiment_directory, args.continue_from, args.batch_split)
