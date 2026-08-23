#!/usr/bin/env python3
"""PCA + neural-residual autoencoder.

    z_pca = (x - mu) @ P_k^T                 k dims, P_k FROZEN (the baseline PCA basis)
    x_pca = z_pca @ P_k + mu                 k = 0  =>  x_pca = mu
    r     = x - x_pca                        residual, mm
    z_res = ResidualEncoder(norm(r))         K - k dims
    out   = x_pca + denorm(ResidualDecoder(z_res))

Total latent is always K = k + (K - k), so comparisons against PCA-K are budget-matched.
The two endpoints of k are the controls that make this design worth running:

    k = K  ->  residual branch has zero latent; the model IS PCA-K exactly
    k = 0  ->  no PCA at all; a pure autoencoder on mean-centred shapes

so one search spans the whole ladder from pure-AE to pure-PCA, and the selected k reports
how much the neural branch actually earned.

Inputs and outputs are raw mm, matching the baseline PCA table.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPAE(nn.Module):
    """Plain dense autoencoder -- the control for 'does mesh structure matter at all?'.

    Consumes the flattened vertex vector directly. Legal here only because every scan shares
    one corresponded topology with a consistent vertex ordering.
    """

    def __init__(self, n_vertices, latent, hidden=1024, n_hidden=2, dropout=0.0, in_channels=3):
        super().__init__()
        self.n_input = int(n_vertices) * in_channels
        self.n_vertices = int(n_vertices)
        self.in_channels = in_channels
        self.latent = int(latent)

        def block(a, b):
            layers = [nn.Linear(a, b), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        enc, dim = [], self.n_input
        for _ in range(int(n_hidden)):
            enc += block(dim, hidden)
            dim = hidden
        enc.append(nn.Linear(dim, self.latent))
        self.encoder_net = nn.Sequential(*enc)

        dec, dim = [], self.latent
        for _ in range(int(n_hidden)):
            dec += block(dim, hidden)
            dim = hidden
        dec.append(nn.Linear(dim, self.n_input))
        self.decoder_net = nn.Sequential(*dec)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.encoder_net(x.reshape(x.size(0), -1))

    def decode(self, z):
        return self.decoder_net(z).view(-1, self.n_vertices, self.in_channels)

    def forward(self, x):
        return self.decode(self.encode(x))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualAE(nn.Module):
    def __init__(self, pca_mean, pca_components, residual_net, res_mean, res_std, latent_total):
        """pca_components: (k, 3V) float tensor, k may be 0. residual_net may be None when k == K."""
        super().__init__()
        self.k = 0 if pca_components is None else int(pca_components.shape[0])
        self.latent_total = int(latent_total)
        self.residual_latent = self.latent_total - self.k

        self.register_buffer("pca_mean", pca_mean)                       # (3V,)
        if pca_components is not None and self.k > 0:
            self.register_buffer("pca_components", pca_components)       # (k, 3V)
        else:
            self.pca_components = None
        self.register_buffer("res_mean", res_mean)                       # (V, 3)
        self.register_buffer("res_std", res_std)                         # (V, 3)

        self.residual_net = residual_net

    # -- PCA branch (frozen) ------------------------------------------------------------
    def pca_encode(self, x):
        flat = x.reshape(x.size(0), -1) - self.pca_mean
        if self.pca_components is None:
            return flat.new_zeros((x.size(0), 0))
        return flat @ self.pca_components.T

    def pca_decode(self, z_pca, shape):
        if self.pca_components is None or z_pca.shape[1] == 0:
            flat = self.pca_mean.unsqueeze(0).expand(shape[0], -1)
        else:
            flat = z_pca @ self.pca_components + self.pca_mean
        return flat.view(shape)

    # -- full model ---------------------------------------------------------------------
    def encode(self, x):
        """Returns the full K-dim latent [z_pca, z_res]."""
        z_pca = self.pca_encode(x)
        if self.residual_net is None:
            return z_pca
        x_pca = self.pca_decode(z_pca, x.shape)
        r_norm = (x - x_pca - self.res_mean) / self.res_std
        return torch.cat([z_pca, self.residual_net.encode(r_norm)], dim=1)

    def forward(self, x):
        z_pca = self.pca_encode(x)
        x_pca = self.pca_decode(z_pca, x.shape)
        if self.residual_net is None:
            return x_pca
        r_norm = (x - x_pca - self.res_mean) / self.res_std
        r_hat = self.residual_net(r_norm) * self.res_std + self.res_mean
        return x_pca + r_hat

    def residual_target(self, x):
        """Normalised residual the neural branch is trained to reproduce."""
        x_pca = self.pca_decode(self.pca_encode(x), x.shape)
        return (x - x_pca - self.res_mean) / self.res_std

    def num_parameters(self):
        """Trainable parameters only -- the frozen PCA basis is reported separately."""
        return 0 if self.residual_net is None else self.residual_net.num_parameters()


def residual_stats(x_train, pca_mean, pca_components):
    """Per-vertex mean/std of the residual on the training split.

    Residuals are ~30x smaller than the shapes and have a different per-vertex scale, so they
    need their own normalisation -- reusing the shape statistics leaves the residual network
    training on near-zero inputs.
    """
    flat = x_train.reshape(len(x_train), -1) - pca_mean
    if pca_components is None or pca_components.shape[0] == 0:
        recon = torch.zeros_like(flat)
    else:
        recon = (flat @ pca_components.T) @ pca_components
    r = (flat - recon).view(x_train.shape)
    mean = r.mean(dim=0)
    std = r.std(dim=0).clamp_min(1e-8)
    return mean, std
