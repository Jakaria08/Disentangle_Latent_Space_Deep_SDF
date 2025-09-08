import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import logging
import numpy as np
import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, data
import deep_sdf.workspace as ws
import networks.sdf_vae as vae

class SDFVAEBaseline(nn.Module):
    """
    Standard SDFVAE without approximator pipeline (baseline for comparison):
    Encoder → Latent → Decoder
    """
    def __init__(self, original_sdfvae):
        super().__init__()
        
        # Copy components from original SDFVAE
        self.encoder = original_sdfvae.encoder
        self.decoder = original_sdfvae.decoder
        self.num_samp_per_scene = original_sdfvae.num_samp_per_scene
        self.latent_size = original_sdfvae.latent_size
        self.kl_div_loss = original_sdfvae.kl_div_loss
    
    def forward(self, points, queries, train=True):
        """
        Standard forward pass: Encoder → Decoder (no approximator)
        """
        if points is not None:
            # Step 1: Encode surface points to latent
            if self.kl_div_loss:
                mu, logvar = self.encoder(points)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                if train:
                    z_latent = mu + eps * std
                else:
                    z_latent = mu
            else:
                z_latent = self.encoder(points)
            
            # Step 2: Prepare for decoder (same logic as original)
            if train:
                z_expanded = z_latent.unsqueeze(1).repeat(1, self.num_samp_per_scene, 1).view(-1, self.latent_size)
            else:
                batch_size = z_latent.shape[0]
                num_queries = queries.shape[0]
                queries_per_batch = num_queries // batch_size
                z_expanded = z_latent.unsqueeze(1).repeat(1, queries_per_batch, 1).view(-1, self.latent_size)
            
            # Step 3: Decode
            queries = queries.cuda()
            decoder_input = torch.cat([z_expanded, queries], dim=1)
            sdf = self.decoder(decoder_input)
            
            if self.kl_div_loss:
                return sdf, mu, logvar, z_latent
            else:
                return sdf, z_latent
        else:
            # Direct decoding without encoder
            sdf = self.decoder(queries)
            z_latent = None
            mu = None
            logvar = None
            
            if self.kl_div_loss:
                return sdf, mu, logvar, z_latent
            else:
                return sdf, z_latent

def finetune_baseline_model(experiment_directory, 
                          original_model_checkpoint="latest",
                          num_epochs=100,
                          lr_encoder=1e-4,
                          lr_decoder=1e-4,
                          batch_size=4,
                          latent_consistency_weight=0.01):
    """
    Fine-tune encoder and decoder WITHOUT approximator pipeline (baseline)
    """
    
    print("🔄 Step 1: Loading specifications and original model...")
    specs = ws.load_experiment_specifications(experiment_directory)
    
    # Load original SDFVAE
    latent_size = specs["CodeLength"]
    num_samp_per_scene = specs["SamplesPerScene"]
    decoder_specs = specs["NetworkSpecs"]
    kl_div_loss = False  # Set based on your training
    
    original_sdfvae = vae.SDFVAE(latent_size, num_samp_per_scene, decoder_specs, kl_div_loss).cuda()
    
    # Load original model weights
    model_path = os.path.join(experiment_directory, ws.model_params_subdir, f"{original_model_checkpoint}.pth")
    saved_model_state = torch.load(model_path)
    original_sdfvae.load_state_dict(saved_model_state["model_state_dict"])
    
    print("🔄 Step 2: Creating baseline model...")
    # Create the baseline model (standard encoder-decoder)
    baseline_model = SDFVAEBaseline(original_sdfvae).cuda()
    
    print("🔄 Step 3: Setting up data loader...")
    # Load data
    data_source = specs["DataSource"]
    data_source_mesh = specs["DataSourceMesh"]
    train_split_file = specs["TrainSplit"]
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, data_source_mesh, train_split, num_samp_per_scene, load_ram=False
    )
    
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    
    print("🔄 Step 4: Setting up optimizer and loss...")
    # Create optimizer for encoder and decoder
    trainable_params = []
    trainable_params.extend([
        {"params": baseline_model.encoder.parameters(), "lr": lr_encoder},
        {"params": baseline_model.decoder.parameters(), "lr": lr_decoder}
    ])
    
    optimizer = torch.optim.Adam(trainable_params)
    loss_l1 = torch.nn.L1Loss(reduction="sum")
    
    # Setup logging
    log_dir = os.path.join(experiment_directory, "finetune_baseline")
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=log_dir)
    
    print("🔄 Step 5: Starting baseline fine-tuning...")
    baseline_model.train()
    
    clamp_dist = specs["ClampingDistance"]
    minT, maxT = -clamp_dist, clamp_dist
    
    for epoch in range(1, num_epochs + 1):
        epoch_time_start = time.time()
        epoch_losses = []
        epoch_sdf_losses = []
        epoch_latent_losses = []
        
        print(f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, (sdf_data, indices, labels, filenames, surface_points) in enumerate(sdf_loader):
            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            
            xyz = sdf_data[:, 0:3]
            xyz.requires_grad = True
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)
            
            optimizer.zero_grad()
            
            # Forward pass through baseline model
            if kl_div_loss:
                pred_sdf, mu, logvar, z_latent = baseline_model(
                    surface_points, xyz, train=True
                )
            else:
                pred_sdf, z_latent = baseline_model(
                    surface_points, xyz, train=True
                )
            
            pred_sdf = torch.clamp(pred_sdf, minT, maxT)
            
            # Main SDF reconstruction loss
            sdf_loss = loss_l1(pred_sdf, sdf_gt.cuda()) / num_sdf_samples
            total_loss = sdf_loss
            
            # Latent consistency loss (optional)
            latent_loss = torch.tensor(0.0).cuda()
            if latent_consistency_weight > 0:
                # Encourage the latent to maintain reasonable magnitude
                latent_loss = torch.mean(torch.norm(z_latent, dim=1))
                total_loss += latent_consistency_weight * latent_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Logging
            epoch_losses.append(total_loss.item())
            epoch_sdf_losses.append(sdf_loss.item())
            epoch_latent_losses.append(latent_loss.item())
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: Total Loss: {total_loss.item():.6f}, "
                      f"SDF Loss: {sdf_loss.item():.6f}, "
                      f"Latent Loss: {latent_loss.item():.6f}")
        
        # Epoch logging
        avg_loss = np.mean(epoch_losses)
        avg_sdf_loss = np.mean(epoch_sdf_losses)
        avg_latent_loss = np.mean(epoch_latent_losses)
        
        summary_writer.add_scalar("Loss/Total", avg_loss, epoch)
        summary_writer.add_scalar("Loss/SDF", avg_sdf_loss, epoch)
        summary_writer.add_scalar("Loss/Latent", avg_latent_loss, epoch)
        
        epoch_time = time.time() - epoch_time_start
        summary_writer.add_scalar("Time/Epoch", epoch_time, epoch)
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s - "
              f"Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoints
        if epoch % 10 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(log_dir, f"baseline_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': baseline_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'specs': specs,
                'model_type': 'baseline'
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("🔄 Step 6: Saving final baseline model...")
    final_model_path = os.path.join(log_dir, "final_baseline_model.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': baseline_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'specs': specs,
        'model_type': 'baseline'
    }, final_model_path)
    
    summary_writer.close()
    print(f"✅ Baseline fine-tuning completed! Final model saved to: {final_model_path}")
    
    return baseline_model

def test_baseline_model(experiment_directory, checkpoint_name="final_baseline_model"):
    """Test the baseline fine-tuned model"""
    print("🔄 Testing baseline fine-tuned model...")
    
    # Load the fine-tuned baseline model
    log_dir = os.path.join(experiment_directory, "finetune_baseline")
    model_path = os.path.join(log_dir, f"{checkpoint_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path)
    specs = checkpoint['specs']
    
    # Recreate the model architecture
    latent_size = specs["CodeLength"]
    num_samp_per_scene = specs["SamplesPerScene"]
    decoder_specs = specs["NetworkSpecs"]
    kl_div_loss = False
    
    original_sdfvae = vae.SDFVAE(latent_size, num_samp_per_scene, decoder_specs, kl_div_loss).cuda()
    baseline_model = SDFVAEBaseline(original_sdfvae).cuda()
    
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    baseline_model.eval()
    
    print("✅ Baseline model loaded successfully!")
    
    # Test on a few samples
    data_source = specs["DataSource"]
    data_source_mesh = specs["DataSourceMesh"]
    train_split_file = specs["TrainSplit"]
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    test_filenames = train_split[:3]  # Test on first 3 samples
    
    for filename in test_filenames:
        print(f"\n🔄 Testing on {filename}...")
        
        # FIXED: Handle filename properly to avoid double .obj extension
        save_name = os.path.basename(filename)
        
        # Remove .npz extension if present
        if save_name.endswith(".npz"):
            save_name = save_name[:-4]
        
        # Check if save_name already has .obj extension
        if save_name.endswith(".obj"):
            mesh_path = os.path.join(data_source_mesh, save_name)
        else:
            mesh_path = os.path.join(data_source_mesh, save_name + ".obj")
        
        print(f"   Looking for mesh at: {mesh_path}")
        
        # Check if file exists before trying to load
        if not os.path.exists(mesh_path):
            print(f"❌ Mesh file not found: {mesh_path}")
            continue
        
        try:
            surface_points = data.get_surface_points(mesh_path)
            surface_points = torch.from_numpy(surface_points).unsqueeze(0).cuda()
            
            with torch.no_grad():
                # Test the baseline model
                test_mesh = mesh.create_mesh(
                    baseline_model,
                    kl_div_loss,
                    surface_points,
                    None,  # latent_vec
                    N=128,  # Lower resolution for testing
                    max_batch=int(2 ** 16),
                    return_trimesh=True
                )
                
                if test_mesh is not None:
                    print(f"✅ Successfully generated mesh for {save_name}")
                    print(f"   Vertices: {len(test_mesh.vertices)}")
                    print(f"   Faces: {len(test_mesh.faces)}")
                    
                    # Save the test mesh
                    test_output_dir = os.path.join(log_dir, "test_meshes")
                    os.makedirs(test_output_dir, exist_ok=True)
                    
                    # Clean save name for output (remove .obj if present)
                    clean_save_name = save_name.replace(".obj", "")
                    test_mesh_path = os.path.join(test_output_dir, f"{clean_save_name}_baseline.ply")
                    test_mesh.export(test_mesh_path)
                    print(f"   Saved mesh to: {test_mesh_path}")
                else:
                    print(f"❌ Failed to generate mesh for {save_name}")
                    
        except Exception as e:
            print(f"❌ Error testing {save_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return baseline_model

def test_baseline_components(experiment_directory):
    """Test baseline model components"""
    print("🔬 Testing baseline model components...")
    
    # Load specifications
    specs = ws.load_experiment_specifications(experiment_directory)
    latent_size = specs["CodeLength"]
    num_samp_per_scene = specs["SamplesPerScene"]
    decoder_specs = specs["NetworkSpecs"]
    kl_div_loss = False
    
    # Test 1: Load original model
    print("1. Testing original model loading...")
    original_sdfvae = vae.SDFVAE(latent_size, num_samp_per_scene, decoder_specs, kl_div_loss).cuda()
    model_path = os.path.join(experiment_directory, ws.model_params_subdir, "latest.pth")
    saved_model_state = torch.load(model_path)
    original_sdfvae.load_state_dict(saved_model_state["model_state_dict"])
    print("✅ Original model loaded successfully")
    
    # Test 2: Create baseline model
    print("2. Testing baseline model creation...")
    baseline_model = SDFVAEBaseline(original_sdfvae).cuda()
    print("✅ Baseline model created successfully")
    
    # Test 3: Test forward pass with dummy data
    print("3. Testing forward pass...")
    try:
        # Create dummy inputs with matching dimensions
        batch_size = 2
        num_points = 2048
        
        dummy_surface_points = torch.randn(batch_size, num_points, 3).cuda()
        # Create queries that match the expected pattern: batch_size * num_samp_per_scene
        dummy_queries = torch.randn(batch_size * num_samp_per_scene, 3).cuda()
        
        with torch.no_grad():
            if kl_div_loss:
                pred_sdf, mu, logvar, z_latent = baseline_model(
                    dummy_surface_points, dummy_queries, train=False
                )
            else:
                pred_sdf, z_latent = baseline_model(
                    dummy_surface_points, dummy_queries, train=False
                )
            
            print(f"✅ Forward pass successful!")
            print(f"   Predicted SDF shape: {pred_sdf.shape}")
            print(f"   Latent shape: {z_latent.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 All baseline components working correctly!")

def compare_training_losses(experiment_directory):
    """Compare training losses between baseline and approximator models"""
    print("📊 Comparing training losses...")
    
    # Load tensorboard logs
    baseline_log_dir = os.path.join(experiment_directory, "finetune_baseline")
    approximator_log_dir = os.path.join(experiment_directory, "finetune_with_approximator")
    
    baseline_exists = os.path.exists(baseline_log_dir)
    approximator_exists = os.path.exists(approximator_log_dir)
    
    print(f"Baseline logs exist: {baseline_exists}")
    print(f"Approximator logs exist: {approximator_exists}")
    
    if baseline_exists and approximator_exists:
        print("✅ Both training logs available for comparison!")
        print(f"📈 View comparison with: tensorboard --logdir {experiment_directory}")
        print(f"   Baseline logs: {baseline_log_dir}")
        print(f"   Approximator logs: {approximator_log_dir}")
    else:
        print("❌ Missing training logs for comparison")

if __name__ == "__main__":
    experiment_directory = "examples/torus_subgroup/only_pointnet_deep_sdf_no_KL"
    
    print("="*60)
    print("BASELINE FINE-TUNING (WITHOUT APPROXIMATOR PIPELINE)")
    print("="*60)
    
    # First test baseline components
    test_baseline_components(experiment_directory)
    
    print("\n" + "="*60)
    print("STARTING BASELINE FINE-TUNING")
    print("="*60)
    
    # Fine-tune the baseline model
    baseline_model = finetune_baseline_model(
        experiment_directory,
        original_model_checkpoint="latest",
        num_epochs=50,
        lr_encoder=1e-4,
        lr_decoder=1e-4,
        batch_size=4,
        latent_consistency_weight=0.01
    )
    
    # Test the baseline model
    print("\n" + "="*60)
    print("TESTING BASELINE MODEL")
    print("="*60)
    test_baseline_model(experiment_directory)
    
    # Compare with approximator training
    print("\n" + "="*60)
    print("LOSS COMPARISON")
    print("="*60)
    compare_training_losses(experiment_directory)
    
    print("\n🎯 SUMMARY:")
    print("1. Baseline model (without approximator) trained and saved")
    print("2. Test meshes generated and saved")
    print("3. Use TensorBoard to compare losses between baseline and approximator models")
    print(f"   Command: tensorboard --logdir {experiment_directory}")