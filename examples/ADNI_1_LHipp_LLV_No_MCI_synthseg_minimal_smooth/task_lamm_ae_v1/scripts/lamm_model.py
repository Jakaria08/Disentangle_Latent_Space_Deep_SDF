#!/usr/bin/env python3
"""LAMM autoencoder for the ADNI left hippocampus.

Follows Tarasiou et al., "Locally Adaptive Neural 3D Morphable Models", CVPR 2024
(arXiv:2401.02937, github.com/michaeltrs/LAMM).

WHY THIS ARCHITECTURE. LAMM's Table 1 reports SpiralNet++ losing to PCA on all four of its
datasets (11.53 vs 10.42 on UHM12k, and similarly on +expr / UHM / Handy) -- independently
reproducing what we measure here (SpiralNet++ 0.036784 vs PCA-128 0.033668). LAMM is the only
method in that table that beats PCA, and the thing it changes is that BOTH ends are
transformer/MLPMixer: there is no graph or spiral convolution anywhere. Every MeshMAE run in
this project used the SpiralNet++ decoder as a fixed component, so "the spiral decoder is the
ceiling" has never been tested. This file tests it.

FAITHFUL TO THE PAPER
  * per-region tokenization by a linear map with NO parameter sharing:
        x_i^0 = W_i^in v_i^s,   W_i^in in R^{D x 3N_i}
  * a learnable identity token x_0^0 prepended to the region tokens; after L encoder layers
    only that token is kept and projected down:  z = W_down x_0^L
  * decoder starts from y_0^0 = W_up z plus K LEARNED region tokens, runs L_dec layers, then
    inverse-tokenizes per region:  v_hat_i = W_i^out y_i^L
  * pre-norm Transformer (heads=8, dim_head=64) or MLPMixer (token-mix expansion 4,
    channel-mix expansion 0.5), both returning per-layer outputs for deep supervision
  * L1 reconstruction loss, plus optional L1 on intermediate decoder layers

DELIBERATE DEVIATIONS, each forced by this cohort
  1. NORMALIZATION. LAMM mean-centres only. We keep this project's per-coordinate
     (x - mean) / std, because every number in this study -- PCA-128 0.033668, SpiralNet++
     0.036784 -- is computed in that space. Changing it would make the comparison meaningless.
  2. SCALE. LAMM uses D=512, 5+3 layers, latent 256, on 6,000-10,000 training meshes. We have
     2037 scans from 475 INDEPENDENT SUBJECTS (4.29 correlated visits each), so D and depth
     are searched over a smaller range instead of fixed.
  3. LATENT 128 by default, to sit alongside the rest of the study.
  4. NO CONTROL-POINT BRANCH. LAMM adds f_delta_i(dV_Ci) to the decoder tokens for local
     manipulation, with no bias terms so f(0)=0. For pure reconstruction dV=0 and the branch
     vanishes identically, so it is omitted rather than approximated.
  5. REGIONS ARE AUTOMATIC. LAMM hand-defines 11 semantic face regions; we reuse the
     decimation-centre patches already validated here (86 patches, 31.9 +- 10.2 vertices).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MESHMAE = Path("/home/jakaria/INR/Deep3DComp/examples/"
               "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_meshmae_ae_v1/scripts")
if str(MESHMAE) not in sys.path:
    sys.path.insert(0, str(MESHMAE))

from meshmae_model import build_patches, build_patch_members  # noqa: E402,F401


# --------------------------------------------------------------------------- backbones
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvFeedForward(nn.Module):
    """FeedForward with Conv1d(kernel_size=1) as the dense op -- LAMM's `chan_first` variant.

    On an input of (B, n_tokens, dim), Conv1d treats axis 1 as channels and axis 2 as length,
    so this mixes ACROSS TOKENS with `dim` as the batched spatial axis. No transpose: a
    transpose here would put LayerNorm(dim) on the wrong axis, which is exactly the bug the
    round-trip test caught.
    """

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(dropout),
                                 nn.Conv1d(hidden, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """LAMM's attention: qkv without bias, einsum scaled dot product, heads=8, dim_head=64."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.to_out((att @ v).transpose(1, 2).reshape(b, n, -1))


class TransformerBackbone(nn.Module):
    """Pre-norm transformer returning every layer's output (for deep supervision)."""

    def __init__(self, dim, depth, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList([PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                           PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout))])
            for _ in range(int(depth)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        outs = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            outs.append(x)
        outs[-1] = self.norm(x)
        return outs


class MixerBackbone(nn.Module):
    """MLPMixer, matching LAMM's module_mlpmixer: token-mixing over the token axis with
    expansion 4, channel-mixing over features with expansion 0.5, both pre-norm residual.

    Token mixing is a fixed-size map over the token axis, so the token count must be constant
    -- it is (K regions + 1 identity token) on both sides."""

    def __init__(self, dim, depth, n_tokens, expansion=4.0, expansion_token=0.5, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleList([
                PreNorm(dim, ConvFeedForward(int(n_tokens), int(n_tokens * expansion), dropout)),
                PreNorm(dim, FeedForward(dim, max(1, int(dim * expansion_token)), dropout)),
            ]) for _ in range(int(depth)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        outs = []
        for tok_mix, chan_mix in self.layers:
            x = tok_mix(x) + x                                    # mix across tokens
            x = chan_mix(x) + x                                   # mix across channels
            outs.append(x)
        outs[-1] = self.norm(x)
        return outs


def make_backbone(kind, dim, depth, n_tokens, heads, dropout, dim_head=64):
    """`dim_head` is INDEPENDENT of D in LAMM: inner width is heads*64 = 512 at heads=8,
    whatever D is. The first search here used dim_head = dim//heads, making inner ~= D --
    192 instead of 512 at the winning D=192, i.e. 2.7x narrower attention. It also made the
    head count meaningless, since both 4 and 8 then gave the same inner width, which is why
    that sweep's apparent preference for heads=4 measured nothing."""
    if kind == "transformer":
        return TransformerBackbone(dim, depth, heads=heads, dim_head=int(dim_head),
                                   dropout=dropout)
    if kind == "mlpmixer":
        return MixerBackbone(dim, depth, n_tokens, dropout=dropout)
    raise ValueError(f"unknown backbone {kind!r}")


# --------------------------------------------------------------- region (de)tokenization
class RegionTokenizer(nn.Module):
    """x_i^0 = W_i^in v_i^s with region-specific weights (LAMM shares NO parameters here;
    the paper's justification -- region weights respect the semantic meaning of each vertex --
    is exactly the fixed-correspondence argument that made the raw MeshMAE tokenizer work).

    Implemented as one batched einsum over padded regions rather than K small Linears. Padded
    slots are zeroed by the member mask, so the corresponding weight columns receive exactly
    zero gradient and never affect the output; they cost memory, not correctness.
    """

    def __init__(self, member_idx, member_mask, dim, in_channels=3, share=False,
                 mode="raw"):
        """`mode="both"` appends the region's mean/max/min/std (4*C dims) to the raw member
        coordinates. LAMM tokenizes raw coordinates only. On this cohort the identical change
        to the MeshMAE tokenizer bought -7.7% val with training error UNCHANGED to five
        decimals (0.017323 -> 0.017318) -- the signature of a shifted train/gap frontier
        rather than a slide along it, which is exactly what LAMM needs: its gap already
        equals SpiralNet++'s 1.41x, and only its fit falls short (0.029434 vs 0.026021)."""
        super().__init__()
        K, P = member_mask.shape
        self.K, self.P, self.C, self.share = int(K), int(P), int(in_channels), bool(share)
        self.mode = str(mode)
        if self.mode not in ("raw", "both"):
            raise ValueError(f"unknown region mode {mode!r}")
        self.register_buffer("member_idx", member_idx.reshape(-1).long())
        self.register_buffer("member_mask", member_mask.to(torch.bool))
        fan_in = self.P * self.C + (4 * self.C if self.mode == "both" else 0)
        if share:                                    # ViT-style shared patch embedding
            self.proj = nn.Linear(fan_in, dim)
        else:
            self.weight = nn.Parameter(torch.empty(self.K, fan_in, dim))
            self.bias = nn.Parameter(torch.zeros(self.K, dim))
            nn.init.trunc_normal_(self.weight, std=fan_in ** -0.5)

    def gather(self, x):
        """(B,V,C) -> (B,K,P,C) region members, pad slots zeroed."""
        b, _, c = x.shape
        g = x.index_select(1, self.member_idx).view(b, self.K, self.P, c)
        return g * self.member_mask.unsqueeze(0).unsqueeze(-1).to(g.dtype)

    def features(self, x):
        g = self.gather(x)
        if self.mode == "raw":
            return g.flatten(2)
        pad = ~self.member_mask.unsqueeze(0).unsqueeze(-1)          # (1,K,P,1)
        cnt = self.member_mask.sum(1).view(1, -1, 1).to(g.dtype)
        mean = g.sum(2) / cnt                                        # pads are already 0
        var = (g * g).sum(2) / cnt - mean * mean
        std = var.clamp_min(1e-12).sqrt()
        mx = g.masked_fill(pad, -1e4).amax(2)
        mn = g.masked_fill(pad, 1e4).amin(2)
        return torch.cat([g.flatten(2), mean, mx, mn, std], dim=-1)

    def forward(self, x):
        v = self.features(x)
        if self.share:
            return self.proj(v)
        return torch.einsum("bkp,kpd->bkd", v, self.weight) + self.bias


class RegionHeads(nn.Module):
    """v_hat_i = W_i^out y_i^L, then scatter regions back to vertex order."""

    def __init__(self, member_idx, member_mask, dim, n_vertices, in_channels=3, share=False):
        super().__init__()
        K, P = member_mask.shape
        self.K, self.P, self.C = int(K), int(P), int(in_channels)
        self.n_vertices = int(n_vertices)
        self.share = bool(share)
        flat_mask = member_mask.reshape(-1).to(torch.bool)
        self.register_buffer("slot_mask", flat_mask)
        self.register_buffer("vertex_of_slot", member_idx.reshape(-1)[flat_mask].long())
        out = self.P * self.C
        if share:
            self.proj = nn.Linear(dim, out)
        else:
            self.weight = nn.Parameter(torch.empty(self.K, dim, out))
            self.bias = nn.Parameter(torch.zeros(self.K, out))
            nn.init.trunc_normal_(self.weight, std=dim ** -0.5)

    def forward(self, y):                                  # y: (B,K,dim)
        b = y.size(0)
        v = self.proj(y) if self.share else (
            torch.einsum("bkd,kdp->bkp", y, self.weight) + self.bias)
        v = v.reshape(b, self.K * self.P, self.C)[:, self.slot_mask]     # (B,V,C) slot order
        out = torch.zeros(b, self.n_vertices, self.C, device=y.device, dtype=y.dtype)
        return out.index_copy_(1, self.vertex_of_slot, v)


# ------------------------------------------------------------------------------- model
class LAMMAutoencoder(nn.Module):
    def __init__(self, *, member_idx, member_mask, n_vertices, latent=128, dim=256,
                 enc_depth=5, dec_depth=3, heads=8, dropout=0.0, backbone="transformer",
                 in_channels=3, share_regions=False, latent_mode="id_token", dim_head=64,
                 region_mode="raw"):
        super().__init__()
        K = int(member_mask.shape[0])
        self.K, self.latent_mode = K, latent_mode
        self.tokenizer = RegionTokenizer(member_idx, member_mask, dim, in_channels,
                                         share=share_regions, mode=region_mode)
        self.id_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.id_token, std=0.02)
        self.encoder = make_backbone(backbone, dim, enc_depth, K + 1, heads, dropout,
                                     dim_head=dim_head)
        # LAMM keeps only the identity token: z = W_down x_0^L. `flatten` is an ablation --
        # this project measured flatten beating every aggregating head by a wide margin
        # (0.042143 vs 0.078685 pooled), so it is worth checking whether LAMM's CLS-style
        # extraction or flatten suits this cohort.
        self.w_down = nn.Linear(dim if latent_mode == "id_token" else dim * (K + 1), latent)
        self.w_up = nn.Linear(latent, dim)
        self.region_tokens = nn.Parameter(torch.zeros(1, K, dim))
        nn.init.trunc_normal_(self.region_tokens, std=0.02)
        self.decoder = make_backbone(backbone, dim, dec_depth, K + 1, heads, dropout,
                                     dim_head=dim_head)
        self.heads_out = RegionHeads(member_idx, member_mask, dim, n_vertices, in_channels,
                                     share=share_regions)

    def encode(self, x):
        tok = self.tokenizer(x)
        tok = torch.cat([self.id_token.expand(tok.size(0), -1, -1), tok], dim=1)
        h = self.encoder(tok)[-1]
        return self.w_down(h[:, 0] if self.latent_mode == "id_token" else h.flatten(1))

    def decode(self, z, all_layers=False):
        y0 = self.w_up(z).unsqueeze(1)
        y = torch.cat([y0, self.region_tokens.expand(z.size(0), -1, -1)], dim=1)
        outs = self.decoder(y)
        if not all_layers:
            return self.heads_out(outs[-1][:, 1:])
        return [self.heads_out(o[:, 1:]) for o in outs]

    def forward(self, x, all_layers=False):
        return self.decode(self.encode(x), all_layers=all_layers)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        g = lambda m: sum(p.numel() for p in m.parameters())
        return {"tokenizer": g(self.tokenizer), "encoder": g(self.encoder),
                "decoder": g(self.decoder), "heads_out": g(self.heads_out),
                "latent_proj": self.w_down.weight.numel() + self.w_up.weight.numel(),
                "total": self.num_parameters()}


# ------------------------------------------------------- hierarchical residual multiscale
class MultiScaleLAMM(nn.Module):
    """Simultaneous multiscale region tokens with an explicit residual decomposition.

        T = T_coarse u T_medium u T_fine          (one sequence; self-attention over the
                                                   union IS the cross-scale interaction)
        X = X_coarse + R_medium + R_fine          (each scale models what the coarser ones
                                                   left behind, not another copy of X)

    Two things motivate this over the flat single-scale model:

    1. LAMM splits 12k vertices into ELEVEN regions. The first search here used 86 on 2746
       vertices -- an 8x longer token sequence than the paper -- and that deviation was never
       examined. Region count is now explicit and searchable.
    2. Latents are split PER SCALE (z_c, z_m, z_f), which makes R^2(dz_scale, dt) measurable.
       If atrophy lives at one spatial scale, a single global latent spreads it across all
       128 dimensions where it is swamped -- the standing hypothesis for why every flow
       plateaus near 10% over a no-change baseline.

    With one scale this reduces exactly to the flat model with a flatten latent, so Stage A
    (region-count sweep) and Stage B (multiscale) share one code path.
    """

    def __init__(self, *, scales, n_vertices, latent=128, dim=256, enc_depth=5, dec_depth=3,
                 heads=8, dim_head=64, dropout=0.0, backbone="transformer", in_channels=3,
                 share_regions=False, residual=True, region_mode="raw", latent_split=None):
        super().__init__()
        self.n_scales = len(scales)
        self.residual = bool(residual)
        self.Ks = [int(mm.shape[0]) for _, mm in scales]
        self.tokenizers = nn.ModuleList(
            RegionTokenizer(mi, mm, dim, in_channels, share=share_regions, mode=region_mode)
            for mi, mm in scales)
        n_tokens = sum(self.Ks)
        self.encoder = make_backbone(backbone, dim, enc_depth, n_tokens, heads, dropout,
                                     dim_head=dim_head)
        # Latent budget across scales. The default even split compresses the scales very
        # unequally -- at 43,172 with D=192 the coarse scale maps 8,256 values into 64 dims
        # (129:1) while the fine scale maps 33,024 into the same 64 (516:1), 4x harder for
        # the same budget. `latent_split` makes that allocation explicit.
        if latent_split is not None:
            per = [int(v) for v in latent_split]
            if len(per) != self.n_scales or sum(per) != int(latent):
                raise ValueError(f"latent_split {per} must have {self.n_scales} entries "
                                 f"summing to {latent}")
        else:
            per = [int(latent) // self.n_scales] * self.n_scales
            per[-1] += int(latent) - sum(per)
        self.latent_split = per
        self.w_down = nn.ModuleList(nn.Linear(dim * k, d) for k, d in zip(self.Ks, per))
        self.w_up = nn.ModuleList(nn.Linear(d, dim) for d in per)
        self.region_tokens = nn.ParameterList(
            nn.Parameter(torch.zeros(1, k, dim)) for k in self.Ks)
        for t in self.region_tokens:
            nn.init.trunc_normal_(t, std=0.02)
        self.decoder = make_backbone(backbone, dim, dec_depth, n_tokens, heads, dropout,
                                     dim_head=dim_head)
        self.heads_out = nn.ModuleList(
            RegionHeads(mi, mm, dim, n_vertices, in_channels, share=share_regions)
            for mi, mm in scales)

    def encode(self, x, per_scale=False):
        tok = torch.cat([t(x) for t in self.tokenizers], dim=1)
        h = self.encoder(tok)[-1]
        zs, off = [], 0
        for k, w in zip(self.Ks, self.w_down):
            zs.append(w(h[:, off:off + k].flatten(1))); off += k
        return zs if per_scale else torch.cat(zs, dim=-1)

    def split_latent(self, z):
        out, off = [], 0
        for d in self.latent_split:
            out.append(z[:, off:off + d]); off += d
        return out

    def decode(self, z, all_layers=False):
        zs = z if isinstance(z, (list, tuple)) else self.split_latent(z)
        y = torch.cat([up(zi).unsqueeze(1) + rt.expand(zi.size(0), -1, -1)
                       for zi, up, rt in zip(zs, self.w_up, self.region_tokens)], dim=1)
        h = self.decoder(y)[-1]
        parts, off = [], 0
        for k, head in zip(self.Ks, self.heads_out):
            parts.append(head(h[:, off:off + k])); off += k
        if not self.residual:
            return parts[-1] if not all_layers else parts
        # partial sums: X_c, X_c+R_m, X_c+R_m+R_f -- supervising these is what forces each
        # scale to model a RESIDUAL instead of a second copy of the whole shape
        cum, running = [], 0
        for p_ in parts:
            running = running + p_
            cum.append(running)
        return cum if all_layers else cum[-1]

    def forward(self, x, all_layers=False):
        return self.decode(self.encode(x), all_layers=all_layers)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        g = lambda m: sum(p.numel() for p in m.parameters())
        return {"tokenizers": g(self.tokenizers), "encoder": g(self.encoder),
                "decoder": g(self.decoder), "heads_out": g(self.heads_out),
                "regions": self.Ks, "latent_split": self.latent_split,
                "total": self.num_parameters()}
