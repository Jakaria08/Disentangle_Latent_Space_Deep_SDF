import torch
import torch.nn as nn
from .pointnet_encoder import PointNetEncoder
from .deep_sdf_decoder import Decoder as SdfDecoder

class SDFVAE(nn.Module):
    def __init__(self, latent_size, num_samp_per_scene, decoder_specs):
        super(SDFVAE, self).__init__()
        
        self.encoder = PointNetEncoder(latent_size)
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = SdfDecoder(latent_size, **decoder_specs)
        self.decoder = nn.DataParallel(self.decoder)
        self.num_samp_per_scene = num_samp_per_scene
        self.latent_size = latent_size
        
    def forward(self, points, queries):
        #print(f"Shape of queries: {queries.shape}")
        mu, logvar = self.encoder(points)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #print(f"Shape of z: {z.shape}")
        z_expanded = z.unsqueeze(1).repeat(1, self.num_samp_per_scene, 1).view(-1, self.latent_size)
        #print(f"Shape of z_expanded: {z_expanded.shape}")
        #print(f"Shape of queries: {queries.shape}")
        #print(f"z_expanded device: {z_expanded.device}")
        #print(f"queries device: {queries.device}")
        queries = queries.cuda()
        decoder_input = torch.cat([z_expanded, queries], dim=1)
        sdf = self.decoder(decoder_input)
        return sdf, mu, logvar