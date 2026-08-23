#!/usr/bin/env python3
"""Deterministic spiral mesh autoencoder.

Structurally the SpiralNet++ autoencoder (guided_vae/reconstruction/network.py) with the
disentanglement machinery removed: no mu/log_var, no KL, no reparameterisation, no
classifier/regressor excitation heads. The encoder emits `latent_channels` directly, which is
what the downstream flow network needs.

Two things are configurable that the original hardcodes:
  * depth  -- len(out_channels) may be 3 or 4, which controls the pre-latent feature count
  * conv   -- per-level operator, so the ICCV adaptive op can be applied only to coarse levels
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from conv_layers import SpiralConv, build_conv


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    return scatter_add(out, row, dim, dim_size=trans.size(0))


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, conv_type="spiral", dynamic_indices=None):
        super().__init__()
        self.conv = build_conv(
            in_channels, out_channels, indices, conv_type=conv_type, dynamic_indices=dynamic_indices
        )
        self.act = nn.ELU()

    def forward(self, x, down_transform):
        return Pool(self.act(self.conv(x)), down_transform)


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, conv_type="spiral", dynamic_indices=None):
        super().__init__()
        self.conv = build_conv(
            in_channels, out_channels, indices, conv_type=conv_type, dynamic_indices=dynamic_indices
        )
        self.act = nn.ELU()

    def forward(self, x, up_transform):
        return self.act(self.conv(Pool(x, up_transform)))


def resolve_conv_types(n_levels: int, conv_type: str, adaptive_levels: int) -> list[str]:
    """Which levels use the adaptive operator.

    The paper applies plain SpiralConv at the fine levels and the adaptive op at the coarse
    ones (its CoMA config is [Spiral, Spiral, Adaptive, Adaptive]); `adaptive_levels` is the
    number of *coarsest* levels that get it.
    """
    if str(conv_type).strip().lower() == "spiral":
        return ["spiral"] * n_levels
    k = max(0, min(int(adaptive_levels), n_levels))
    return ["spiral"] * (n_levels - k) + ["adaptive"] * k


class SpiralAE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        spiral_indices,
        down_transform,
        up_transform,
        conv_types=None,
        dynamic_spiral_indices=None,
        dropout=0.0,
        linear_skip=False,
        n_input_vertices=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.latent_channels = int(latent_channels)
        self.n_levels = len(self.out_channels)
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = down_transform[-1].size(0)
        self.pre_latent_dim = self.num_vert * self.out_channels[-1]

        if conv_types is None:
            conv_types = ["spiral"] * self.n_levels
        self.conv_types = list(conv_types)
        if dynamic_spiral_indices is None:
            dynamic_spiral_indices = [None] * self.n_levels

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(self.n_levels):
            src = in_channels if idx == 0 else self.out_channels[idx - 1]
            self.en_layers.append(
                SpiralEnblock(
                    src,
                    self.out_channels[idx],
                    spiral_indices[idx],
                    conv_type=self.conv_types[idx],
                    dynamic_indices=dynamic_spiral_indices[idx],
                )
            )
        self.en_linear = nn.Linear(self.pre_latent_dim, self.latent_channels)

        # decoder
        self.de_linear = nn.Linear(self.latent_channels, self.pre_latent_dim)
        self.de_layers = nn.ModuleList()
        for idx in range(self.n_levels):
            src = self.out_channels[-1] if idx == 0 else self.out_channels[-idx]
            dst = self.out_channels[-idx - 1]
            self.de_layers.append(
                SpiralDeblock(
                    src,
                    dst,
                    spiral_indices[-idx - 1],
                    conv_type=self.conv_types[-idx - 1],
                    dynamic_indices=dynamic_spiral_indices[-idx - 1],
                )
            )
        self.out_conv = SpiralConv(self.out_channels[0], in_channels, indices=spiral_indices[0])

        # Regularisation. v1 studies overfit hard (train L1 falling while val RMSE rose), so
        # dropout sits on the pre-latent features and on the decoder's first expansion.
        self.dropout = float(dropout)
        self.drop_pre = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()
        self.drop_post = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        # Optional global linear branch through the same latent budget. PCA is the optimal
        # linear autoencoder at a given latent size, so a model that contains a linear path
        # can represent PCA exactly and spend the spiral branch on the nonlinear residual.
        self.linear_skip = bool(linear_skip)
        if self.linear_skip:
            self.n_input = int(n_input_vertices) * in_channels
            self.skip_encode = nn.Linear(self.n_input, self.latent_channels, bias=False)
            self.skip_decode = nn.Linear(self.latent_channels, self.n_input, bias=False)
            nn.init.xavier_uniform_(self.skip_encode.weight)
            nn.init.xavier_uniform_(self.skip_decode.weight)

        for module in (self.en_linear, self.de_linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def encode(self, x):
        h = x
        for idx, layer in enumerate(self.en_layers):
            h = layer(h, self.down_transform[idx])
        z = self.en_linear(self.drop_pre(h.reshape(h.size(0), -1)))
        if self.linear_skip:
            z = z + self.skip_encode(x.reshape(x.size(0), -1))
        return z

    def decode(self, z):
        x = self.drop_post(self.de_linear(z)).view(-1, self.num_vert, self.out_channels[-1])
        for idx, layer in enumerate(self.de_layers):
            x = layer(x, self.up_transform[self.n_levels - 1 - idx])
        out = self.out_conv(x)
        if self.linear_skip:
            out = out + self.skip_decode(z).view(out.shape)
        return out

    def forward(self, x):
        return self.decode(self.encode(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    transform,
    spiral_indices,
    dynamic_spiral_indices,
    down_transform,
    up_transform,
    out_channels,
    latent_channels,
    conv_type,
    adaptive_levels,
    in_channels=3,
    conv_types=None,
    dropout=0.0,
    linear_skip=False,
):
    """`conv_types`, when given, is authoritative -- callers that apply extra guards (e.g. a
    max-node limit for the adaptive op) must pass their resolved plan so the built model
    matches what they logged and saved."""
    if conv_types is None:
        conv_types = resolve_conv_types(len(out_channels), conv_type, adaptive_levels)
    if len(conv_types) != len(out_channels):
        raise ValueError(
            f"conv_types has {len(conv_types)} entries for {len(out_channels)} levels"
        )
    return SpiralAE(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        spiral_indices=spiral_indices,
        down_transform=down_transform,
        up_transform=up_transform,
        conv_types=conv_types,
        dynamic_spiral_indices=dynamic_spiral_indices,
        dropout=dropout,
        linear_skip=linear_skip,
        n_input_vertices=int(transform["vertices"][0].shape[0]),
    )
