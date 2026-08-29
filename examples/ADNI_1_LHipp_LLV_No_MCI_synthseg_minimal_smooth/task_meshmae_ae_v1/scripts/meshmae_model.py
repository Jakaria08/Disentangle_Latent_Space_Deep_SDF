#!/usr/bin/env python3
"""Compact MeshMAE autoencoder for the ADNI left hippocampus.

Follows MeshMAE (Liang et al., ECCV 2022, arXiv:2207.10228) where it matters:

  * the mesh is split into NON-OVERLAPPING patches,
  * each patch becomes one token via a patch-feature MLP,
  * a positional embedding is derived from the patch centre's 3D position,
  * a Transformer encoder processes the token sequence,
  * optional masked-patch pretraining reconstructs deleted patches from visible ones.

Four deliberate deviations, each forced by this dataset:

1. PATCHES COME FROM THE DECIMATION HIERARCHY, NOT SUBDIVISION.
   MeshMAE needs faces = base * 4^k so it can take 64-face patches. This mesh has 5488
   faces = 343 * 4^2, and 5488/64 = 85.75, so MeshMAE's patching does not apply. MAPS
   remeshing would fix that but would destroy the vertex correspondence every metric in
   this project depends on. Instead patch centres are taken from an existing decimation
   level (172 centres -> 16.0 +- 4.6 vertices each) and vertices assigned to the nearest.

2. VERTEX FEATURES, NOT FACE FEATURES.
   MeshMAE's 13-dim face descriptor (coords, normal, area, inner angles) is tied to the
   subdivision patching. Here each patch is summarised by order-invariant statistics of its
   member vertices, which avoids ragged padding and is invariant to vertex ordering.

3. A GLOBAL BOTTLENECK IS ADDED.
   MeshMAE has no global latent -- its "representation" is the full token sequence. The
   cocycle flow needs a single 128-D vector, so mean+max pooling over tokens feeds a linear
   head. This is the main architectural addition.

4. THE DECODER IS THE EXISTING SPIRAL DECODER.
   MeshMAE's decoder reconstructs masked tokens from visible tokens; it cannot map a pooled
   latent back to 2746 vertices. Reusing SpiralAE's decoder keeps the comparison against
   spiralnet128/adaptive128 to a change of ENCODER ONLY, and allows warm-starting from
   trained decoder weights.

Loss is L1, matching every other model in this project (switching to L2 would make the
numbers non-comparable to the 9-cell flow matrix).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPIRAL_SCRIPTS = Path("/home/jakaria/INR/Deep3DComp/examples/"
                      "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts")
if str(SPIRAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SPIRAL_SCRIPTS))

from conv_layers import SpiralConv  # noqa: E402
from network import SpiralAE  # noqa: E402


def build_patches(template_vertices, centres):
    """Assign every vertex to its nearest patch centre. Returns (patch_id, n_patches)."""
    from scipy.spatial import cKDTree
    pid = cKDTree(np.asarray(centres)).query(np.asarray(template_vertices))[1]
    return torch.from_numpy(pid.astype(np.int64)), int(pid.max()) + 1


def build_patch_members(patch_id, n_patches):
    """Vertex ids of each patch, padded to a common width.

    Returns (member_idx (K,P), member_mask (K,P)). Pad slots point at vertex 0 and are zeroed
    after the gather, so they contribute nothing. Members are ordered by vertex index, which
    is FIXED across the cohort -- every mesh here shares one template topology, so slot j of
    patch k is the same anatomical landmark in all 2037 scans. That is what makes a raw,
    non-permutation-invariant patch embedding well defined on this data.
    """
    pid = np.asarray(patch_id)
    groups = [np.flatnonzero(pid == k) for k in range(int(n_patches))]
    width = max(len(g) for g in groups)
    idx = np.zeros((int(n_patches), width), dtype=np.int64)
    msk = np.zeros((int(n_patches), width), dtype=bool)
    for k, g in enumerate(groups):
        idx[k, :len(g)] = g
        msk[k, :len(g)] = True
    return torch.from_numpy(idx), torch.from_numpy(msk)


class PatchTokenizer(nn.Module):
    """Per-patch descriptor -> token embedding, in one of three modes.

    "moments" (original, default) summarises each patch by the mean, max, min and standard
    deviation of its member vertices (4 x 3 = 12 dims). Permutation invariant, so no padding
    is needed -- patches here hold 31.9 +- 10.2 vertices.

    "raw" embeds the member COORDINATES instead, gathered into a fixed padded layout.
    Permutation invariance buys nothing on this cohort (one shared template topology means
    vertex i is the same landmark in every scan), while the moments are an 8x lossy
    compression -- 86 x 12 = 1032 numbers standing in for 8238 input coordinates -- whose
    missing within-patch detail the decoder can only hallucinate. This mirrors the S2 result
    one level down: flatten beat pooling ACROSS patches, so it should beat pooling over
    vertices WITHIN a patch.

    "both" concatenates raw and moments, making the input a strict superset of "moments" so
    a regression against the S2 baseline can only come from optimisation, not lost signal.

    The projection is shared across patches (as in ViT/MeshMAE patch embedding); patch
    identity is supplied by the positional embedding, not by per-patch weights.
    """

    def __init__(self, n_patches, d_model, in_channels=3, mode="moments",
                 member_idx=None, member_mask=None, feat_channels=None):
        super().__init__()
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)
        self.mode = str(mode)
        if self.mode not in ("moments", "raw", "both"):
            raise ValueError(f"unknown tokenizer mode {mode!r}")
        # `feat_channels` is the width of the tensor being tokenized: 3 for bare coordinates,
        # 3+stem_channels once a spiral stem runs first. `in_channels` stays 3 because the
        # positional path always embeds the 3-D patch centres.
        feat_channels = int(in_channels if feat_channels is None else feat_channels)
        self.feat_channels = feat_channels
        in_dim = 4 * feat_channels if self.mode in ("moments", "both") else 0
        self.patch_width = 0
        if self.mode in ("raw", "both"):
            if member_idx is None or member_mask is None:
                raise ValueError(f"tokenizer mode {self.mode!r} needs member_idx/member_mask")
            self.patch_width = int(member_mask.shape[1])
            self.register_buffer("member_idx", member_idx.reshape(-1).long())
            self.register_buffer("member_mask", member_mask.to(torch.bool))
            in_dim += feat_channels * self.patch_width
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # MeshMAE derives positional embeddings from patch-centre coordinates
        self.pos = nn.Sequential(nn.Linear(in_channels, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def descriptors(self, x, patch_onehot, counts):
        """x: (B,V,3) -> (B,K,12) via scatter-mean/max/min/std over each patch."""
        b = x.size(0)
        s = torch.einsum("bvc,vk->bkc", x, patch_onehot)
        mean = s / counts.view(1, -1, 1)
        sq = torch.einsum("bvc,vk->bkc", x * x, patch_onehot) / counts.view(1, -1, 1)
        std = (sq - mean * mean).clamp_min(1e-12).sqrt()
        neg = torch.full((b, self.n_patches, x.size(2)), -1e4, device=x.device, dtype=x.dtype)
        pos = torch.full_like(neg, 1e4)
        idx = patch_onehot.argmax(1)                      # (V,) patch of each vertex
        mx = neg.index_reduce(1, idx, x, "amax", include_self=True)
        mn = pos.index_reduce(1, idx, x, "amin", include_self=True)
        return torch.cat([mean, mx, mn, std], dim=-1)

    def raw_members(self, x):
        """x: (B,V,C) -> (B,K,C*P) member coordinates with pad slots zeroed."""
        b, _, c = x.shape
        g = x.index_select(1, self.member_idx).view(b, self.n_patches, self.patch_width, c)
        return (g * self.member_mask.unsqueeze(0).unsqueeze(-1).to(g.dtype)).flatten(2)

    def features(self, x, patch_onehot, counts):
        if self.mode == "moments":
            return self.descriptors(x, patch_onehot, counts)
        if self.mode == "raw":
            return self.raw_members(x)
        return torch.cat([self.raw_members(x),
                          self.descriptors(x, patch_onehot, counts)], dim=-1)

    def forward(self, x, patch_onehot, counts, centres, mask=None):
        tok = self.embed(self.features(x, patch_onehot, counts))
        if mask is not None:                              # MAE-style patch deletion
            tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        return tok + self.pos(centres).unsqueeze(0)


class BiasedSelfAttention(nn.Module):
    """Explicit multi-head self-attention with an additive per-head bias.

    Written out rather than using nn.MultiheadAttention because torch 2.0 routes both
    nn.TransformerEncoderLayer and nn.MultiheadAttention through fused kernels
    (`_transformer_encoder_layer_fwd` / `_native_multi_head_attention`) that reject a 3-D
    float attention mask. This path cannot be intercepted.
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = int(heads)
        self.dk = d_model // self.h
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias=None):
        b, k, _ = x.shape
        q, kk, v = self.qkv(x).chunk(3, dim=-1)
        shape = lambda t: t.view(b, k, self.h, self.dk).transpose(1, 2)   # (B,H,K,dk)
        q, kk, v = shape(q), shape(kk), shape(v)
        att = (q @ kk.transpose(-2, -1)) / (self.dk ** 0.5)               # (B,H,K,K)
        if bias is not None:
            att = att + bias                                              # (1,H,K,K) broadcast
        att = self.drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).reshape(b, k, self.h * self.dk)
        return self.out(y)


class BiasedEncoderBlock(nn.Module):
    """Pre-norm Transformer block using BiasedSelfAttention."""

    def __init__(self, d_model, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = BiasedSelfAttention(d_model, heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        h = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, h), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias=None):
        x = x + self.drop(self.attn(self.n1(x), bias))
        return x + self.drop(self.mlp(self.n2(x)))


class LatentSetHead(nn.Module):
    """Cross-attention pooling with learned queries (3DShape2VecSet style).

    Mean pooling destroys spatial identity before the bottleneck: a mean over 86 tokens cannot
    express "channel 7 of patch 43", which is exactly what a reconstruction decoder needs.
    SpiralNet++ instead projects its full 86x192 = 16512 feature map, and generalises far
    better (gap 1.57x vs 2.34x) despite MORE parameters per training mesh -- so the deficit is
    architectural, not capacity.

    Serious transformer shape autoencoders do not mean-pool either: 3DShape2VecSet
    (arXiv:2301.11445) aggregates into a SET of latents via cross-attention with learned
    queries. Here `n_queries` learned queries each attend over the patch tokens and emit
    `latent // n_queries` channels, so the total latent budget is unchanged at 128.
    """

    def __init__(self, d_model, latent, n_queries=16, heads=4):
        super().__init__()
        if latent % n_queries:
            raise ValueError(f"latent {latent} must divide by n_queries {n_queries}")
        self.n_queries = int(n_queries)
        self.per_query = int(latent) // int(n_queries)
        self.queries = nn.Parameter(torch.zeros(1, self.n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, self.per_query)

    def forward(self, tokens):
        q = self.queries.expand(tokens.size(0), -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.proj(self.norm(out)).flatten(1)          # (B, latent)


class LatentSetProjHead(nn.Module):
    """Cross-attention latent set that does NOT starve the per-query projection.

    S1 used `latent // n_queries` channels per query -- 8 channels at 16 queries -- and ran
    far behind even the mean-pool baseline. Squeezing the per-query output to hold the 128-D
    budget conflates two independent axes:

        n_queries        = how many spatial slots the latent can address
        channels/query   = expressiveness within a slot

    Here each query emits the full d_model and the CONCATENATION is projected to `latent`,
    so slot count and latent budget are decoupled. Closer to 3DShape2VecSet, which keeps the
    whole latent set rather than compressing per query. Costs n_queries*d_model*latent
    (~262K at 16x128) on top of the attention.
    """

    def __init__(self, d_model, latent, n_queries=16, heads=4):
        super().__init__()
        self.n_queries = int(n_queries)
        self.queries = nn.Parameter(torch.zeros(1, self.n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(self.n_queries * d_model, int(latent))

    def forward(self, tokens):
        q = self.queries.expand(tokens.size(0), -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.proj(self.norm(out).flatten(1))


class FlattenHead(nn.Module):
    """Flatten-and-project: structurally what SpiralNet++ does at its bottleneck.

    Control for the latent-set head -- separates "mean pooling was lossy" from "learned
    queries specifically help". Costs n_patches*d_model*latent parameters (1.4M at K=86,
    d_model=128), bringing params/mesh to ~2750, the same ratio SpiralNet++ runs at.
    """

    def __init__(self, n_patches, d_model, latent):
        super().__init__()
        self.proj = nn.Linear(int(n_patches) * int(d_model), int(latent))

    def forward(self, tokens):
        return self.proj(tokens.flatten(1))


class SpiralStem(nn.Module):
    """Spiral convolutions on the full mesh, before patch tokenization.

    The measured deficit against SpiralNet++ is locality: identical decoder, identical latent
    size, generalisation gap 1.41x vs 1.87x. Spiral convolution computes a vertex's feature
    from its k-ring neighbourhood; global self-attention has no such constraint and, on 475
    independent subjects, exploits that freedom.

    S8 already tried to supply the prior as an attention-logit bias and got 0.14% -- but that
    only REWEIGHTS attention between tokens that were computed without any locality. A stem
    computes local features outright, which is the strong form of the same idea and what every
    successful mesh/surface transformer actually does (MeshMAE tokenizes local face
    descriptors, not raw geometry).

    Output is CONCATENATED with the input coordinates, so the tokenizer's input is a strict
    superset of what S7/S8 saw and this can only add information.
    """

    def __init__(self, in_channels, channels, indices, layers=2):
        super().__init__()
        dims = [int(in_channels)] + [int(channels)] * int(layers)
        self.convs = nn.ModuleList(
            SpiralConv(dims[i], dims[i + 1], indices=indices) for i in range(int(layers)))
        self.out_channels = int(channels)

    def forward(self, x):
        for conv in self.convs:
            x = F.elu(conv(x))
        return x


class GroupedHead(nn.Module):
    """Weight-shared per-patch projection, then concatenate. The middle ground flatten and
    pooling leave empty.

    S2 showed the win came from POSITIONAL IDENTITY: flatten preserves which patch a feature
    came from, mean+max pooling destroys it (0.042 vs 0.079). But flatten pays for that with
    n_patches*d_model*latent = 1.409M dense parameters -- 25% of the model and the only
    token->latent path, hence the obvious memorisation route now that locality is ruled out
    (attention bias +0.14%, spiral stem -0.08%).

    This projects every patch through the SAME small matrix -- the convolutional prior that
    makes spiral nets generalise -- and keeps each patch in its own slot of the concatenation,
    so positional identity survives. At rank r the dense path is 86*r*latent instead of
    86*d_model*latent: 31x smaller at r=4, converging back to plain flatten as r -> d_model.

    Distinct from LatentSetHead/LatentSetProjHead (0.116-0.144): those AGGREGATE tokens by
    cross-attention. This performs no aggregation at all.
    """

    def __init__(self, n_patches, d_model, latent, rank=4):
        super().__init__()
        self.rank = int(rank)
        self.proj = nn.Linear(int(d_model), int(rank))          # shared across patches
        self.out = nn.Linear(int(n_patches) * int(rank), int(latent))

    def forward(self, tokens):                                   # (B, K, d_model)
        return self.out(self.proj(tokens).flatten(1))


class CompactMeshMAEEncoder(nn.Module):
    def __init__(self, n_patches, d_model=128, depth=6, heads=4, mlp_ratio=4.0,
                 dropout=0.1, latent=128, in_channels=3, head="pool", n_queries=16,
                 distance_bias=False, patch_centres=None, bias_scale=1.0,
                 tokenizer_mode="moments", member_idx=None, member_mask=None,
                 stem_channels=0, stem_layers=2, stem_indices=None, head_rank=4):
        super().__init__()
        self.stem = None
        feat_channels = in_channels
        if int(stem_channels) > 0:
            if stem_indices is None:
                raise ValueError("stem_channels > 0 requires stem_indices (level-0 spiral)")
            self.stem = SpiralStem(in_channels, int(stem_channels), stem_indices,
                                   layers=int(stem_layers))
            feat_channels = in_channels + int(stem_channels)   # stem output is concatenated
        self.tokenizer = PatchTokenizer(n_patches, d_model, in_channels,
                                        mode=tokenizer_mode, member_idx=member_idx,
                                        member_mask=member_mask, feat_channels=feat_channels)
        self.head_kind = head
        self.n_heads = int(heads)
        self.use_biased_blocks = bool(distance_bias)
        if self.use_biased_blocks:
            self.blocks = nn.ModuleList(
                BiasedEncoderBlock(d_model, heads, mlp_ratio, dropout) for _ in range(int(depth)))
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(d_model)
        if head == "pool":            # original mean+max
            self.head = nn.Linear(2 * d_model, int(latent))
        elif head == "latentset":
            self.head = LatentSetHead(d_model, latent, n_queries, heads)
        elif head == "latentset_proj":
            self.head = LatentSetProjHead(d_model, latent, n_queries, heads)
        elif head == "flatten":
            self.head = FlattenHead(n_patches, d_model, latent)
        elif head == "grouped":
            self.head = GroupedHead(n_patches, d_model, latent, head_rank)
        else:
            raise ValueError(f"unknown head {head!r}")

        # Geometric locality bias. Global attention from layer 1 can learn spurious long-range
        # correlations on 2037 samples, whereas spiral convolutions are local by construction.
        # A learned monotone function of inter-patch distance injects that missing prior,
        # following MS-SiT's use of locality on cortical surfaces.
        self.distance_bias = bool(distance_bias)
        if self.distance_bias:
            if patch_centres is None:
                raise ValueError("distance_bias requires patch_centres")
            c = (patch_centres.detach().float().cpu() if torch.is_tensor(patch_centres)
                 else torch.as_tensor(np.asarray(patch_centres), dtype=torch.float32))
            d = torch.cdist(c, c)
            self.register_buffer("pair_distance", d / d.max().clamp_min(1e-8))
            self.bias_logit = nn.Parameter(torch.full((int(heads), 1, 1), float(bias_scale)))

    def _attn_bias(self):
        """Additive per-head bias -softplus(w) * normalised distance; (1,H,K,K) broadcast."""
        if not self.distance_bias:
            return None
        w = torch.nn.functional.softplus(self.bias_logit)                # (H,1,1)
        return (-w * self.pair_distance.unsqueeze(0)).unsqueeze(0)       # (1,H,K,K)

    def forward(self, x, patch_onehot, counts, centres, mask=None):
        if self.stem is not None:
            x = torch.cat([x, self.stem(x)], dim=-1)
        tok = self.tokenizer(x, patch_onehot, counts, centres, mask)
        if self.use_biased_blocks:
            m = self._attn_bias()
            h = tok
            for blk in self.blocks:
                h = blk(h, bias=m)
            t = self.norm(h)
        else:
            t = self.norm(self.blocks(tok))
        if self.head_kind == "pool":
            return self.head(torch.cat([t.mean(1), t.amax(1)], dim=-1)), t
        return self.head(t), t


class CompactMeshMAE(nn.Module):
    """Transformer encoder + frozen-topology spiral decoder."""

    def __init__(self, *, n_patches, patch_onehot, patch_counts, patch_centres,
                 spiral_indices, dynamic_spiral_indices, down_transform, up_transform,
                 out_channels, latent=128, d_model=128, depth=6, heads=4,
                 dropout=0.1, decoder_dropout=0.1, in_channels=3,
                 head="pool", n_queries=16, distance_bias=False, bias_scale=1.0,
                 tokenizer_mode="moments", member_idx=None, member_mask=None,
                 stem_channels=0, stem_layers=2, head_rank=4):
        super().__init__()
        self.register_buffer("patch_onehot", patch_onehot)
        self.register_buffer("patch_counts", patch_counts)
        self.register_buffer("patch_centres", patch_centres)
        self.encoder = CompactMeshMAEEncoder(n_patches, d_model, depth, heads,
                                             dropout=dropout, latent=latent,
                                             in_channels=in_channels, head=head,
                                             n_queries=n_queries,
                                             distance_bias=distance_bias,
                                             bias_scale=bias_scale,
                                             patch_centres=patch_centres,
                                             tokenizer_mode=tokenizer_mode,
                                             member_idx=member_idx,
                                             member_mask=member_mask,
                                             stem_channels=stem_channels,
                                             stem_layers=stem_layers,
                                             stem_indices=spiral_indices[0],
                                             head_rank=head_rank)
        # decoder reused verbatim so the comparison isolates the encoder
        self._ae = SpiralAE(in_channels, out_channels, latent, spiral_indices,
                            down_transform, up_transform,
                            conv_types=["spiral"] * len(out_channels),
                            dynamic_spiral_indices=dynamic_spiral_indices,
                            dropout=decoder_dropout, linear_skip=False)
        # SpiralAE builds an encoder we never call -- the Transformer replaces it. Left in
        # place its en_linear alone is 2.114M parameters (32% of the model) that AdamW would
        # optimise to no effect, inflating capacity on a 2037-sample problem. Drop it.
        del self._ae.en_layers
        del self._ae.en_linear
        self.n_patches = int(n_patches)

    def encode(self, x, mask=None):
        z, _ = self.encoder(x, self.patch_onehot, self.patch_counts, self.patch_centres, mask)
        return z

    def decode(self, z):
        return self._ae.decode(z)

    def forward(self, x, mask=None):
        return self.decode(self.encode(x, mask))

    def load_decoder_weights(self, state_dict, strict=True):
        """Warm-start from a trained SpiralAE. Only decoder tensors are taken."""
        sub = {k: v for k, v in state_dict.items() if k.startswith(("de_", "out_conv"))}
        missing = self._ae.load_state_dict(sub, strict=False)
        loaded = len(sub) - len(getattr(missing, "unexpected_keys", []))
        if strict and loaded == 0:
            raise ValueError("no decoder tensors matched; check the source checkpoint")
        return loaded

    def freeze_decoder(self, frozen=True):
        for n, p in self._ae.named_parameters():
            if n.startswith(("de_", "out_conv")):
                p.requires_grad = not frozen

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
