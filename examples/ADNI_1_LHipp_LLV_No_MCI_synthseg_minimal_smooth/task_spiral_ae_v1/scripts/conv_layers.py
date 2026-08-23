#!/usr/bin/env python3
"""Spiral convolution operators.

`SpiralConv` is the SpiralNet++ operator (Gong et al., 2019).

`AdaptiveSpiralConv` is the ICCV'23 operator from Babiloni et al., "Adaptive Spiral Layers
for Efficient 3D Representation Learning on Meshes", ported from
https://github.com/FrancescaBabiloni/adaptive_spiral (conv/adaptive_spiralconv.py).

One deliberate deviation from upstream, recorded in every checkpoint as
`adaptive_reset_parameters_fixed`: upstream `Dynamic_spiral_pool.__init__` ends with a bare
`self.reset_parameters` (no call), so the spiral-length predictor `ro` never gets the
zero-initialisation the paper describes. Here it is called.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SUPPORTED_CONV_TYPES = ("spiral", "adaptive")


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices=None, dim=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1)).view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1)).view(bs, n_nodes, -1)
        else:
            raise RuntimeError(f"x.dim() expected 2 or 3, received {x.dim()}")
        return self.layer(x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"seq_length={self.seq_length})"
        )


class DynamicSpiralPool(nn.Module):
    """Learned spiral length: predicts s per vertex, then linearly interpolates the cumulative
    sum of the spiral sequence at that (fractional) length."""

    def __init__(self, in_channels, out_channels, dynamic_indices=None, dim=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.indices = dynamic_indices
        self.max_seq = dynamic_indices.size(1)
        self.ro = nn.Linear(in_channels, 1)
        groups = 1 if in_channels % 4 else 4
        self.norm = nn.GroupNorm(groups, in_channels)
        self.reset_parameters()  # upstream omits the call; see module docstring

    def reset_parameters(self):
        nn.init.constant_(self.ro.weight, 0)
        nn.init.constant_(self.ro.bias, 0)

    def dynamic_weighted_pool(self, x):
        b, n, k, c = x.size()
        assert k == self.max_seq
        s = torch.abs(self.ro(x.mean(2))).view(b, n, 1)
        s = torch.clamp(s * self.max_seq, max=self.max_seq - 1)
        pool = x.cumsum(2)

        top = s.ceil().long().detach()
        bot = s.floor().long().detach()
        frac = s - bot

        it = top.view(b, n, 1, 1).repeat(1, 1, 1, c)
        ib = bot.view(b, n, 1, 1).repeat(1, 1, 1, c)
        xt = torch.gather(pool, 2, it)
        xb = torch.gather(pool, 2, ib)

        pooled = xb + frac.unsqueeze(-1) * (xt - xb)
        y = pooled.view(b, n, -1)
        y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y, s

    def forward(self, x):
        assert x.dim() == 3
        bs, n_nodes, _ = x.size()
        x = torch.index_select(x, self.dim, self.indices.view(-1))
        x = x.view(bs, n_nodes, self.max_seq, -1)
        x, _ = self.dynamic_weighted_pool(x)
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"max_s={self.max_seq})"
        )


class GroupLinear(nn.Module):
    def __init__(self, in_channels, out_channels, g=2):
        super().__init__()
        self.in_ch = in_channels // g
        self.out_ch = out_channels // g
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(self.in_ch, self.out_ch))
        self.bias = nn.Parameter(torch.Tensor(self.out_ch))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        b, n, _ = x.size()
        x = x.view(b, n, self.g, -1)
        x = torch.einsum("bngc, cd -> bngd", x, self.weight) + self.bias
        return x.view(b, n, -1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_ch}, {self.out_ch}, g={self.g})"


class GatedSpiralDW(nn.Module):
    """Depthwise spiral filter with a per-vertex learned gate."""

    def __init__(self, in_channels, indices, dim):
        super().__init__()
        self.dim = dim
        self.indices = indices
        self.seq_length = indices.size(1)
        self.n_nodes = indices.size(0)
        self.ch = in_channels
        self.gate = nn.Linear(self.ch, self.ch, bias=True)
        self.weight = nn.Parameter(torch.Tensor(self.n_nodes, self.seq_length))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bs = x.size(0)
        gate = self.gate(x)
        neighbors = torch.index_select(x, self.dim, self.indices.view(-1))
        neighbors = neighbors.reshape(bs, self.n_nodes, self.seq_length, self.ch)
        x = torch.einsum("bvsf, vs -> bvf", neighbors, self.weight)
        return x * gate


class AdaptiveSpiralConv(nn.Module):
    """ICCV'23 adaptive spiral layer (Babiloni et al.)."""

    def __init__(self, in_channels, out_channels, indices=None, dynamic_indices=None, dim=1, **kw):
        super().__init__()
        if dynamic_indices is None:
            raise ValueError("AdaptiveSpiralConv requires dynamic_indices.")
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.SpiralConv = nn.Linear(in_channels * self.seq_length, out_channels)
        self.dynamic_pool = DynamicSpiralPool(in_channels, in_channels, dynamic_indices=dynamic_indices)
        self.u_cr = GroupLinear(in_channels, in_channels, g=_groups_for(in_channels))
        self.dynamic_dw = GatedSpiralDW(in_channels, self.indices, self.dim)
        self.u_dr = GroupLinear(in_channels, in_channels, g=_groups_for(in_channels))

        self.alpha = nn.Parameter(torch.ones([1]) * 0.1)
        self.beta = nn.Parameter(torch.ones([1]) * 1.0)
        self.gamma = nn.Parameter(torch.ones([in_channels]) * 0.1)
        self.delta = nn.Parameter(torch.ones([in_channels]) * 0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.SpiralConv.weight)
        nn.init.constant_(self.SpiralConv.bias, 0)

    def forward(self, x):
        assert x.dim() == 3
        n_nodes, _ = self.indices.size()
        bs = x.size(0)
        x = self.gamma.view(1, 1, -1) * self.u_cr(x) + x
        x = self.beta.view(1, 1, -1) * self.dynamic_dw(x) + x
        x = self.alpha * self.dynamic_pool(x) + x
        x = self.delta.view(1, 1, -1) * self.u_dr(x) + x

        x = torch.index_select(x, self.dim, self.indices.view(-1)).view(bs, n_nodes, -1)
        return self.SpiralConv(x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"seq_length={self.seq_length}, max_s={self.dynamic_pool.max_seq})"
        )


def _groups_for(channels: int) -> int:
    """Largest group count in {4,2,1} that divides `channels` (upstream hardcodes 4)."""
    for g in (4, 2):
        if channels % g == 0:
            return g
    return 1


def build_conv(in_channels, out_channels, indices, conv_type="spiral", dynamic_indices=None):
    conv_type = str(conv_type).strip().lower()
    if conv_type not in SUPPORTED_CONV_TYPES:
        raise ValueError(f"conv_type must be one of {SUPPORTED_CONV_TYPES}, got {conv_type!r}")
    if conv_type == "spiral":
        return SpiralConv(in_channels, out_channels, indices=indices)
    return AdaptiveSpiralConv(
        in_channels, out_channels, indices=indices, dynamic_indices=dynamic_indices
    )
