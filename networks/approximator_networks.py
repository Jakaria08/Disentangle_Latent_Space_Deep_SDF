import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentToHLLEICAApproximator(nn.Module):
    """Network to approximate HLLE+ICA transformation from latent codes"""
    def __init__(self, latent_dim=16, hlle_ica_dim=6, hidden_dims=[128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to HLLE+ICA dimensions
        layers.append(nn.Linear(prev_dim, hlle_ica_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, latent_codes):
        return self.network(latent_codes)

class HLLEICAToLatentInverse(nn.Module):
    """Network to map HLLE+ICA embeddings back to latent space"""
    def __init__(self, hlle_ica_dim=6, latent_dim=16, hidden_dims=[128, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = hlle_ica_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent dimensions
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, hlle_ica_embeddings):
        return self.network(hlle_ica_embeddings)