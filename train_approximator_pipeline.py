import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import deep_sdf.workspace as ws
import networks.sdf_vae as vae
from networks.approximator_networks import LatentToHLLEICAApproximator, HLLEICAToLatentInverse

def load_frozen_model_and_latents(experiment_directory):
    """Load the frozen encoder-decoder model and extract latents"""
    
    # Load specifications
    specs = ws.load_experiment_specifications(experiment_directory)
    latent_size = specs["CodeLength"]
    num_samp_per_scene = specs["SamplesPerScene"]
    decoder_specs = specs["NetworkSpecs"]
    kl_div_loss = False
    
    # Load the trained model
    decoder = vae.SDFVAE(latent_size, num_samp_per_scene, decoder_specs, kl_div_loss).cuda()
    
    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, "latest.pth")
    )
    
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder.eval()
    
    # Freeze the model
    for param in decoder.parameters():
        param.requires_grad = False
    
    # Load precomputed latents
    metrics_dir = os.path.join(experiment_directory, "metrics")
    latents_path = os.path.join(metrics_dir, "all_latents.npy")
    all_latents_array = np.load(latents_path).squeeze()
    
    return decoder, all_latents_array, specs

def compute_target_hlle_ica(latents_array, n_neighbors=50, hlle_components=6, ica_components=6):
    """Compute target HLLE+ICA embeddings from latents"""
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latents_array)
    
    print(f"Computing HLLE with {n_neighbors} neighbors and {hlle_components} components...")
    
    # Apply HLLE
    hlle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=hlle_components,
        method='hessian',
        random_state=42,
        eigen_solver='dense'
    )
    hlle_embeddings = hlle.fit_transform(X_scaled)
    
    print(f"Computing ICA with {ica_components} components...")
    
    # Apply ICA to HLLE results
    ica = FastICA(
        n_components=ica_components,
        whiten='arbitrary-variance',
        random_state=42,
        max_iter=2000,
        tol=1e-6
    )
    hlle_ica_embeddings = ica.fit_transform(hlle_embeddings)
    
    return hlle_ica_embeddings, scaler

def add_jittering_to_latents(latents, jittering_std=0.01, jittering_probability=0.8):
    """Add jittering to latent codes during training"""
    if np.random.rand() < jittering_probability:
        noise = np.random.normal(0, jittering_std, latents.shape)
        return latents + noise
    return latents

def train_approximator_networks(experiment_directory, 
                              epochs_stage1=200, 
                              epochs_stage2=200,
                              lr=1e-3,
                              batch_size=32,
                              jittering_enabled=True):
    """Train the approximator networks in two stages"""
    
    print("🔄 Step 1: Loading frozen model and latents...")
    frozen_decoder, latents_array, specs = load_frozen_model_and_latents(experiment_directory)
    latent_dim = latents_array.shape[1]
    
    print("🔄 Step 2: Computing target HLLE+ICA embeddings...")
    target_hlle_ica, scaler = compute_target_hlle_ica(latents_array)
    hlle_ica_dim = target_hlle_ica.shape[1]
    
    print(f"Latent dimension: {latent_dim}")
    print(f"HLLE+ICA dimension: {hlle_ica_dim}")
    
    # Convert to tensors
    latents_tensor = torch.FloatTensor(latents_array).cuda()
    target_tensor = torch.FloatTensor(target_hlle_ica).cuda()
    
    # Create networks
    approximator = LatentToHLLEICAApproximator(latent_dim, hlle_ica_dim).cuda()
    inverse_net = HLLEICAToLatentInverse(hlle_ica_dim, latent_dim).cuda()
    
    # Create data loaders
    dataset = torch.utils.data.TensorDataset(latents_tensor, target_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("🔄 Step 3: Training Stage 1 - Latent → HLLE+ICA Approximator...")
    
    # Stage 1: Train approximator network
    optimizer_approx = torch.optim.Adam(approximator.parameters(), lr=lr)
    approximator.train()
    
    stage1_losses = []
    
    for epoch in range(epochs_stage1):
        epoch_loss = 0.0
        
        for batch_latents, batch_targets in dataloader:
            optimizer_approx.zero_grad()

            # Forward pass with original latents
            predicted_hlle_ica_orig = approximator(batch_latents)
            loss_orig = F.mse_loss(predicted_hlle_ica_orig, batch_targets)
            
            # Add jittering if enabled
            if jittering_enabled:
                jittered_latents = []
                for i in range(batch_latents.shape[0]):
                    lat_np = batch_latents[i].cpu().numpy()
                    jittered_lat = add_jittering_to_latents(lat_np)
                    jittered_latents.append(torch.FloatTensor(jittered_lat))
                batch_latents_jittered = torch.stack(jittered_latents).cuda()
            
                # Forward pass
                predicted_hlle_ica_jittered = approximator(batch_latents_jittered)
                loss_jittered = F.mse_loss(predicted_hlle_ica_jittered, batch_targets)

                # Combine losses
                total_loss = 0.5*loss_orig + 0.5*loss_jittered

            else:
                total_loss = loss_orig
            
            total_loss.backward()
            optimizer_approx.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        stage1_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Stage 1 - Epoch {epoch+1}/{epochs_stage1}, Loss: {avg_loss:.6f}")
    
    print("🔄 Step 4: Training Stage 2 - HLLE+ICA → Latent Inverse Network...")
    
    # Stage 2: Train inverse network
    optimizer_inverse = torch.optim.Adam(inverse_net.parameters(), lr=lr)
    inverse_net.train()
    approximator.eval()  # Keep approximator frozen
    
    stage2_losses = []
    
    for epoch in range(epochs_stage2):
        epoch_loss = 0.0
        
        for batch_latents, batch_targets in dataloader:
            optimizer_inverse.zero_grad()
            
            # Get HLLE+ICA embeddings from approximator
            with torch.no_grad():
                hlle_ica_embeddings = approximator(batch_latents)
            
            # Predict latents from HLLE+ICA embeddings
            reconstructed_latents = inverse_net(hlle_ica_embeddings)
            
            # Loss: MSE between reconstructed and original latents
            loss = F.mse_loss(reconstructed_latents, batch_latents)
            
            loss.backward()
            optimizer_inverse.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        stage2_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Stage 2 - Epoch {epoch+1}/{epochs_stage2}, Loss: {avg_loss:.6f}")
    
    print("🔄 Step 5: Evaluating full pipeline...")
    
    # Evaluate full pipeline
    approximator.eval()
    inverse_net.eval()
    
    with torch.no_grad():
        # Test on a subset
        test_latents = latents_tensor[:100]
        
        # Forward through approximator
        pred_hlle_ica = approximator(test_latents)
        
        # Forward through inverse
        reconstructed_latents = inverse_net(pred_hlle_ica)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed_latents, test_latents)
        
        print(f"Final reconstruction error: {reconstruction_error.item():.6f}")
        
        # Calculate correlation with target HLLE+ICA
        target_subset = target_tensor[:100]
        hlle_ica_error = F.mse_loss(pred_hlle_ica, target_subset)
        print(f"HLLE+ICA approximation error: {hlle_ica_error.item():.6f}")
    
    # Save models
    save_dir = os.path.join(experiment_directory, "approximator_models")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'approximator_state_dict': approximator.state_dict(),
        'inverse_net_state_dict': inverse_net.state_dict(),
        'scaler': scaler,
        'target_hlle_ica': target_hlle_ica,
        'specs': specs
    }, os.path.join(save_dir, "approximator_pipeline.pth"))
    
    print(f"✅ Models saved to {save_dir}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(stage1_losses)
    plt.title('Stage 1: Latent → HLLE+ICA Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(stage2_losses)
    plt.title('Stage 2: HLLE+ICA → Latent Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.show()
    
    return approximator, inverse_net, frozen_decoder


if __name__ == "__main__":
    experiment_directory = "examples/torus_subgroup/only_pointnet_deep_sdf_no_KL"
    
    approximator, inverse_net, frozen_decoder = train_approximator_networks(
        experiment_directory,
        epochs_stage1=1000,
        epochs_stage2=1000,
        lr=1e-3,
        batch_size=32,
        jittering_enabled=True
    )
