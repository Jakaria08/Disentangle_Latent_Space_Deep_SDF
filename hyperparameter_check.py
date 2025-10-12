#!/usr/bin/env python3
"""
Hyperparameter sweep for SNNL regression loss
Tests different combinations of w_cls, threshold, w_code_reg, and temp_reg
"""

import torch
import torch.utils.data as data_utils
import os
import logging
import math
import json
import time
import itertools
import numpy as np

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, loss, data
import deep_sdf.workspace as ws
import networks.sdf_vae as vae


def run_sweep_experiment(experiment_directory, sweep_epochs=20, improvement_threshold=0.10):
    """
    Run hyperparameter sweep to find best configuration
    
    Args:
        experiment_directory: Path to experiment directory
        sweep_epochs: Number of epochs to test each configuration
        improvement_threshold: Minimum improvement required (default 10%)
    """
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting hyperparameter sweep...")
    
    # Load specs
    specs = ws.load_experiment_specifications(experiment_directory)
    
    # Define hyperparameter search space (wider range)
    w_cls_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    threshold_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    w_code_reg_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    temp_reg_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Create all combinations
    param_combinations = list(itertools.product(
        w_cls_values,
        threshold_values, 
        w_code_reg_values,
        temp_reg_values
    ))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    print(f"Testing each for {sweep_epochs} epochs with {improvement_threshold*100}% improvement threshold")
    
    # Load data
    data_source = specs["DataSource"]
    data_source_mesh = specs["DataSourceMesh"]
    train_split_file = specs["TrainSplit"]
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    latent_size = specs["CodeLength"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    
    load_ram = False
    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, data_source_mesh, train_split, num_samp_per_scene, load_ram=load_ram
    )
    
    num_data_loader_threads = 1
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    
    # Initialize model architecture
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder_specs = specs["NetworkSpecs"]
    
    # Track successful configurations
    successful_configs = []
    results_file = os.path.join(experiment_directory, "hyperparam_sweep_results.json")
    
    # Loss function
    loss_l1 = torch.nn.L1Loss(reduction="sum")
    
    # Hyperparameter sweep loop
    for sweep_idx, (w_cls_test, threshold_test, w_code_reg_test, temp_reg_test) in enumerate(param_combinations):
        
        print(f"\n{'='*80}")
        print(f"SWEEP {sweep_idx + 1}/{len(param_combinations)}")
        print(f"Testing: w_cls={w_cls_test}, threshold={threshold_test}, "
              f"w_code_reg={w_code_reg_test}, temp_reg={temp_reg_test}")
        print(f"{'='*80}\n")
        
        # Clear CUDA cache before starting new configuration
        torch.cuda.empty_cache()
        
        # Initialize fresh model for this sweep
        decoder = vae.SDFVAE(latent_size, num_samp_per_scene, decoder_specs, kl_div_loss=False).cuda()
        
        # Initialize latent vectors
        lat_vecs = torch.nn.Embedding(len(train_split), latent_size, max_norm=None)
        torch.nn.init.normal_(
            lat_vecs.weight.data,
            0.0,
            1.0 / math.sqrt(latent_size),
        )
        
        # Initialize optimizer with lower learning rate for sweep
        optimizer_all = torch.optim.Adam(
            [
                {"params": decoder.parameters(), "lr": 0.0001},  # Lower LR for stability
                {"params": lat_vecs.parameters(), "lr": 0.0005},
            ]
        )
        
        # Track losses
        loss_history = []
        snnl_reg_history = []
        initial_loss = None
        final_loss = None
        
        # Mini training loop for this configuration
        try:
            for epoch in range(sweep_epochs):
                
                decoder.train()
                epoch_loss = 0.0
                epoch_snnl_reg = 0.0
                num_batches = 0
                
                for sdf_data, indices, labels, filenames, surface_points in sdf_loader:
                    
                    # Process data
                    sdf_data = sdf_data.reshape(-1, 4)
                    num_sdf_samples = sdf_data.shape[0]
                    
                    xyz = sdf_data[:, 0:3]
                    sdf_gt = sdf_data[:, 3].unsqueeze(1)
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)
                    
                    # Extract labels
                    labels_cls = labels[:, 1].to(torch.float32).cuda()  # disease
                    labels_reg = labels[:, 0].to(torch.float32).cuda()  # age
                    
                    # Get latent codes
                    indices_batch = indices.flatten()
                    
                    optimizer_all.zero_grad()
                    
                    try:
                        # Forward pass
                        pred_sdf, z = decoder(surface_points.cuda(), xyz.cuda())
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                        
                        # Main reconstruction loss
                        batch_loss = loss_l1(pred_sdf, sdf_gt.cuda()) / num_sdf_samples
                        
                        # SNNL regression loss
                        SNN_Loss_Reg = loss.SNNRegLoss(temp_reg_test, threshold_test)
                        loss_snn_reg = SNN_Loss_Reg(z, labels_reg)
                        
                        # Check for NaN or Inf
                        if torch.isnan(loss_snn_reg) or torch.isinf(loss_snn_reg):
                            print(f"  Warning: SNNL loss is NaN/Inf, skipping batch")
                            continue
                        
                        batch_loss += loss_snn_reg * w_cls_test
                        
                        # Code regularization
                        if w_code_reg_test > 0:
                            l2_size_loss = torch.sum(torch.norm(z, dim=1))
                            reg_loss = l2_size_loss / z.shape[0]
                            batch_loss += reg_loss * w_code_reg_test
                        
                        # Check total loss
                        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                            print(f"  Warning: Total loss is NaN/Inf, skipping batch")
                            continue
                        
                        batch_loss.backward()
                        
                        # Gradient clipping to prevent explosion
                        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(lat_vecs.parameters(), max_norm=1.0)
                        
                        optimizer_all.step()
                        
                        epoch_loss += batch_loss.item()
                        epoch_snnl_reg += loss_snn_reg.item()
                        num_batches += 1
                        
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "out of memory" in str(e):
                            print(f"  CUDA/Memory error in batch: {str(e)[:100]}...")
                            # Clear CUDA cache and skip this batch
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                    
                    # Limit batches per epoch to speed up sweep (only process 5 batches)
                    if num_batches >= 5:
                        break
                
                # Check if we got any successful batches
                if num_batches == 0:
                    print(f"  No successful batches in epoch {epoch+1}, skipping configuration...")
                    raise RuntimeError("No successful batches")
                
                # Calculate average losses
                avg_loss = epoch_loss / num_batches
                avg_snnl_reg = epoch_snnl_reg / num_batches
                
                loss_history.append(avg_loss)
                snnl_reg_history.append(avg_snnl_reg)
                
                # Track initial and final loss
                if epoch == 0:
                    initial_loss = avg_loss
                if epoch == sweep_epochs - 1:
                    final_loss = avg_loss
                
                # Print progress every 5 epochs
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{sweep_epochs}: Loss={avg_loss:.6f}, SNNL_reg={avg_snnl_reg:.6f}")
        
        except Exception as e:
            print(f"  Error during training: {str(e)[:200]}")
            print(f"  Skipping this configuration...")
        
        finally:
            # CRITICAL: Always clean up, even if there's an error
            print(f"  Cleaning up memory...")
            del decoder, lat_vecs, optimizer_all
            if 'SNN_Loss_Reg' in locals():
                del SNN_Loss_Reg
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU to finish
            time.sleep(1)  # Small delay to ensure cleanup
            continue
        
        # Check if improvement criterion is met
        if initial_loss is not None and final_loss is not None and initial_loss > 0:
            improvement = (initial_loss - final_loss) / initial_loss
            
            print(f"\nResults for this configuration:")
            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Final loss: {final_loss:.6f}")
            print(f"  Improvement: {improvement*100:.2f}%")
            
            if improvement >= improvement_threshold:
                print(f"  ✓ SUCCESS: Improvement >= {improvement_threshold*100}%")
                
                config_result = {
                    "w_cls": w_cls_test,
                    "threshold": threshold_test,
                    "w_code_reg": w_code_reg_test,
                    "temp_reg": temp_reg_test,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "improvement": improvement,
                    "loss_history": loss_history,
                    "snnl_reg_history": snnl_reg_history
                }
                successful_configs.append(config_result)
                
                # Save results incrementally
                with open(results_file, 'w') as f:
                    json.dump(successful_configs, f, indent=2)
                
                print(f"  Saved to {results_file}")
            else:
                print(f"  ✗ FAILED: Improvement < {improvement_threshold*100}%")
        
        # Clean up after successful configuration
        print(f"  Cleaning up memory...")
        del decoder, lat_vecs, optimizer_all
        if 'SNN_Loss_Reg' in locals():
            del SNN_Loss_Reg
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)  # Ensure cleanup completes
        
        # Early stopping if we found enough good configurations
        if len(successful_configs) >= 10:
            print(f"\nFound {len(successful_configs)} successful configurations. Stopping sweep.")
            break
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {min(sweep_idx + 1, len(param_combinations))} configurations")
    print(f"Found {len(successful_configs)} successful configurations")
    
    if successful_configs:
        print("\nTop 5 configurations by improvement:")
        sorted_configs = sorted(successful_configs, key=lambda x: x['improvement'], reverse=True)
        
        for i, config in enumerate(sorted_configs[:5]):
            print(f"\n{i+1}. Improvement: {config['improvement']*100:.2f}%")
            print(f"   w_cls={config['w_cls']}, threshold={config['threshold']}, "
                  f"w_code_reg={config['w_code_reg']}, temp_reg={config['temp_reg']}")
            print(f"   Initial loss: {config['initial_loss']:.6f} → Final loss: {config['final_loss']:.6f}")
        
        # Save best config to a separate file
        best_config_file = os.path.join(experiment_directory, "best_hyperparam_config.json")
        with open(best_config_file, 'w') as f:
            json.dump(sorted_configs[0], f, indent=2)
        
        print(f"\nBest configuration saved to: {best_config_file}")
        print(f"All results saved to: {results_file}")
        
        return sorted_configs[0]
    else:
        print("\nNo successful configurations found.")
        return None


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Hyperparameter sweep for SNNL loss")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory containing specs.json",
    )
    arg_parser.add_argument(
        "--epochs",
        dest="sweep_epochs",
        type=int,
        default=20,
        help="Number of epochs to test each configuration (default: 20)",
    )
    arg_parser.add_argument(
        "--threshold",
        dest="improvement_threshold",
        type=float,
        default=0.10,
        help="Minimum improvement threshold (default: 0.10 for 10%%)",
    )

    args = arg_parser.parse_args()
    
    best_config = run_sweep_experiment(
        args.experiment_directory,
        sweep_epochs=args.sweep_epochs,
        improvement_threshold=args.improvement_threshold
    )
    
    if best_config:
        print("\n" + "="*80)
        print("RECOMMENDED HYPERPARAMETERS:")
        print("="*80)
        print(f"w_cls = {best_config['w_cls']}")
        print(f"threshold = {best_config['threshold']}")
        print(f"w_code_reg = {best_config['w_code_reg']}")
        print(f"temp_reg = {best_config['temp_reg']}")
        print("\nUpdate these values in your train_deep_sdf.py file.")