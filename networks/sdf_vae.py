import torch
import torch.nn as nn
from .pointnet_encoder import PointNetEncoder, ResnetPointnet
from .deep_sdf_decoder import Decoder as SdfDecoder

class SDFVAE(nn.Module):
    def __init__(self, latent_size, num_samp_per_scene, decoder_specs, kl_div_loss):
        super(SDFVAE, self).__init__()
        
        self.encoder = ResnetPointnet(latent_size=latent_size, kl_div_loss=kl_div_loss)
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = SdfDecoder(latent_size, **decoder_specs)
        self.decoder = nn.DataParallel(self.decoder)
        self.num_samp_per_scene = num_samp_per_scene
        self.latent_size = latent_size
        self.kl_div_loss = kl_div_loss
        
    def forward(self, points, queries, train=True):
        #print(f"Shape of queries: {queries.shape}")
        if points is not None:
            if self.kl_div_loss:
                mu, logvar = self.encoder(points)
                #logvar = torch.clamp(logvar, min=-3, max=3)  # Clamp logvar to prevent numerical issues
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
            else:
                z = self.encoder(points)
            #print(f"Shape of z: {z.shape}")
            if train:
                if self.kl_div_loss:
                    z = mu + eps * std
                z_expanded = z.unsqueeze(1).repeat(1, self.num_samp_per_scene, 1).view(-1, self.latent_size)
            else:
                if self.kl_div_loss:
                    z = mu
                num_samples = queries.shape[0]  
                z_expanded = z.expand(num_samples, -1)
            #print(f"Shape of z_expanded: {z_expanded.shape}")
            #print(f"Shape of queries: {queries.shape}")
            #print(f"z_expanded device: {z_expanded.device}")
            #print(f"queries device: {queries.device}")
            queries = queries.cuda()
            decoder_input = torch.cat([z_expanded, queries], dim=1)
            sdf = self.decoder(decoder_input)
            if self.kl_div_loss:
                return sdf, mu, logvar, z
            else:
                return sdf, z