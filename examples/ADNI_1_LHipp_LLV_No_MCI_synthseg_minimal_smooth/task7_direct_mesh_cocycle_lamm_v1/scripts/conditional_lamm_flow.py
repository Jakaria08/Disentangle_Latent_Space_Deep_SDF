#!/usr/bin/env python3
"""Fully end-to-end, non-ODE, time/disease-conditioned direct LAMM surface flow."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from region_layout import RegionLayout


class RegionTokenizer(nn.Module):
    """LAMM's non-shared per-region linear tokenization of raw coordinates.

    This local form intentionally avoids importing MeshMAE/torch-scatter: neither graph
    convolution nor torch-scatter participates in a LAMM forward pass.
    """

    def __init__(self, member_idx: torch.Tensor, member_mask: torch.Tensor, dim: int) -> None:
        super().__init__()
        regions, width = member_mask.shape
        self.K, self.P, self.C = int(regions), int(width), 3
        self.register_buffer("member_idx", member_idx.reshape(-1).long())
        self.register_buffer("member_mask", member_mask.to(torch.bool))
        fan_in = self.P * self.C
        self.weight = nn.Parameter(torch.empty(self.K, fan_in, int(dim)))
        self.bias = nn.Parameter(torch.zeros(self.K, int(dim)))
        nn.init.trunc_normal_(self.weight, std=fan_in**-0.5)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        batch = vertices.shape[0]
        gathered = vertices.index_select(1, self.member_idx).view(
            batch, self.K, self.P, self.C
        )
        gathered = gathered * self.member_mask.unsqueeze(0).unsqueeze(-1).to(gathered)
        features = gathered.flatten(2)
        return torch.einsum("bkp,kpd->bkd", features, self.weight) + self.bias


class RegionHeads(nn.Module):
    """LAMM's non-shared inverse tokenization, here producing six velocity channels."""

    def __init__(
        self,
        member_idx: torch.Tensor,
        member_mask: torch.Tensor,
        dim: int,
        n_vertices: int,
        out_channels: int = 6,
    ) -> None:
        super().__init__()
        regions, width = member_mask.shape
        self.K, self.P, self.C = int(regions), int(width), int(out_channels)
        self.n_vertices = int(n_vertices)
        active = member_mask.reshape(-1).to(torch.bool)
        self.register_buffer("slot_mask", active)
        self.register_buffer("vertex_of_slot", member_idx.reshape(-1)[active].long())
        self.weight = nn.Parameter(torch.empty(self.K, int(dim), self.P * self.C))
        self.bias = nn.Parameter(torch.zeros(self.K, self.P * self.C))
        nn.init.trunc_normal_(self.weight, std=int(dim) ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        values = torch.einsum("bkd,kdp->bkp", tokens, self.weight) + self.bias
        values = values.reshape(batch, self.K * self.P, self.C)[:, self.slot_mask]
        output = torch.zeros(
            batch,
            self.n_vertices,
            self.C,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return output.index_copy_(1, self.vertex_of_slot, values)


class TimeDiseaseEmbedding(nn.Module):
    """The same age/diagnosis information used by the direct Spiral comparators."""

    def __init__(self, condition_dim: int, frequencies: int, age_mean: float, age_std: float) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.frequencies = int(frequencies)
        self.register_buffer("age_mean", torch.tensor(float(age_mean)))
        self.register_buffer("age_std", torch.tensor(max(float(age_std), 1.0e-6)))
        self.disease = nn.Embedding(2, 8)
        raw_dim = 6 + 4 * self.frequencies + 8
        self.network = nn.Sequential(
            nn.Linear(raw_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.SiLU(),
        )

    def forward(
        self, source_age: torch.Tensor, target_age: torch.Tensor, disease: torch.Tensor
    ) -> torch.Tensor:
        source = (source_age.reshape(-1, 1) - self.age_mean) / self.age_std
        target = (target_age.reshape(-1, 1) - self.age_mean) / self.age_std
        delta = (target_age.reshape(-1, 1) - source_age.reshape(-1, 1)) / self.age_std
        midpoint = 0.5 * (source + target)
        elapsed = (target_age.reshape(-1, 1) - source_age.reshape(-1, 1)).abs()
        raw = [source, target, delta, delta.abs(), midpoint, torch.log1p(elapsed)]
        for frequency in range(self.frequencies):
            scale = math.pi * float(2**frequency)
            raw.extend((torch.sin(scale * source), torch.cos(scale * source)))
            raw.extend((torch.sin(scale * target), torch.cos(scale * target)))
        label = disease.reshape(-1).round().long().clamp(0, 1)
        raw.append(self.disease(label))
        return self.network(torch.cat(raw, dim=1))


class ConditionalMixerBlock(nn.Module):
    """LAMM MLPMixer block with zero-initialized FiLM after each pre-normalization."""

    def __init__(
        self,
        token_dim: int,
        n_tokens: int,
        condition_dim: int,
        token_expansion: float,
        channel_expansion: float,
        dropout: float,
    ) -> None:
        super().__init__()
        token_hidden = max(1, int(n_tokens * float(token_expansion)))
        channel_hidden = max(1, int(token_dim * float(channel_expansion)))
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_film = nn.Linear(condition_dim, 2 * token_dim)
        self.token_mixer = nn.Sequential(
            nn.Conv1d(n_tokens, token_hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(token_hidden, n_tokens, 1),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_film = nn.Linear(condition_dim, 2 * token_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
            nn.Dropout(dropout),
        )
        for layer in (self.token_film, self.channel_film):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _conditioned(normed: torch.Tensor, condition: torch.Tensor, film: nn.Linear) -> torch.Tensor:
        gamma, beta = film(condition).chunk(2, dim=-1)
        return normed * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        mixed = self._conditioned(self.token_norm(tokens), condition, self.token_film)
        tokens = tokens + self.token_mixer(mixed)
        mixed = self._conditioned(self.channel_norm(tokens), condition, self.channel_film)
        return tokens + self.channel_mixer(mixed)


class ConditionalMixerBackbone(nn.Module):
    def __init__(
        self,
        token_dim: int,
        depth: int,
        n_tokens: int,
        condition_dim: int,
        token_expansion: float,
        channel_expansion: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ConditionalMixerBlock(
                token_dim,
                n_tokens,
                condition_dim,
                token_expansion,
                channel_expansion,
                dropout,
            )
            for _ in range(int(depth))
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, condition)
        return self.norm(tokens)


class LatentResidualBlock(nn.Module):
    def __init__(self, width: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(condition_dim, 2 * width)
        self.net = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm(value) * (1.0 + gamma) + beta
        return value + self.net(hidden)


class ConditionalLAMMFlow(nn.Module):
    """LAMM is the velocity backbone; the transported state remains the surface itself.

    The exact diagonal identity is supplied by the outer residual path:
        Phi(X,s,t,d) = X + (t-s) [V_CN(X,s,t) + d V_AD(X,s,t)].
    There is one optimizer and no pretrained/frozen autoencoder requirement.
    """

    operator = "lamm_mlpmixer"
    ode_used = False
    global_latent_bottleneck = True

    def __init__(
        self,
        layout: RegionLayout,
        faces: torch.Tensor | np.ndarray,
        config: dict[str, Any],
        statistics: dict[str, Any],
    ) -> None:
        super().__init__()
        model = config["model"]
        if str(model.get("operator")) != self.operator:
            raise ValueError(f"model.operator must be {self.operator!r}")
        expected_regions = [int(value) for value in model["region_scales"]]
        if layout.region_counts != expected_regions:
            raise ValueError(f"Layout {layout.region_counts} != configured {expected_regions}")
        self.layout_fingerprint = layout.fingerprint
        self.layout_source = layout.source
        self.n_vertices = int(layout.n_vertices)
        self.token_dim = int(model["token_dim"])
        self.latent_dim = int(model["latent_dim"])
        self.latent_split = [int(value) for value in model["latent_split"]]
        if len(self.latent_split) != len(layout.scales) or sum(self.latent_split) != self.latent_dim:
            raise ValueError("latent_split must be positive, per-scale, and sum to latent_dim")
        if any(value <= 0 for value in self.latent_split):
            raise ValueError("latent_split entries must be positive")
        condition_dim = int(model["condition_dim"])
        dropout = float(model["dropout"])
        token_expansion = float(model.get("token_expansion", 4.0))
        channel_expansion = float(model.get("channel_expansion", 0.5))
        scales = layout.scales
        n_tokens = sum(scale.regions for scale in scales)

        self.register_buffer("faces", torch.as_tensor(faces, dtype=torch.long))
        self.register_buffer(
            "template", torch.as_tensor(statistics["template_vertices_mm"], dtype=torch.float32)
        )
        self.register_buffer(
            "coordinate_scale",
            torch.tensor(max(float(statistics["coordinate_scale_mm"]), 1.0e-6)),
        )
        self.register_buffer(
            "velocity_scale",
            torch.tensor(max(float(statistics["velocity_scale_mm_per_year"]), 1.0e-8)),
        )
        self.condition = TimeDiseaseEmbedding(
            condition_dim,
            int(model.get("time_frequencies", 2)),
            float(statistics["age_mean_years"]),
            float(statistics["age_std_years"]),
        )
        self.tokenizers = nn.ModuleList(
            RegionTokenizer(
                scale.member_idx,
                scale.member_mask,
                self.token_dim,
            )
            for scale in scales
        )
        self.encoder = ConditionalMixerBackbone(
            self.token_dim,
            int(model["encoder_depth"]),
            n_tokens,
            condition_dim,
            token_expansion,
            channel_expansion,
            dropout,
        )
        self.w_down = nn.ModuleList(
            nn.Linear(self.token_dim * scale.regions, width)
            for scale, width in zip(scales, self.latent_split)
        )
        latent_width = int(model["latent_width"])
        self.latent_input = nn.Linear(self.latent_dim + condition_dim, latent_width)
        self.latent_blocks = nn.ModuleList(
            LatentResidualBlock(latent_width, condition_dim, dropout)
            for _ in range(int(model["latent_residual_blocks"]))
        )
        self.latent_output = nn.Linear(latent_width, self.latent_dim)
        self.w_up = nn.ModuleList(
            nn.Linear(width, self.token_dim) for width in self.latent_split
        )
        self.region_tokens = nn.ParameterList(
            nn.Parameter(torch.zeros(1, scale.regions, self.token_dim)) for scale in scales
        )
        for tokens in self.region_tokens:
            nn.init.trunc_normal_(tokens, std=0.02)
        self.decoder = ConditionalMixerBackbone(
            self.token_dim,
            int(model["decoder_depth"]),
            n_tokens,
            condition_dim,
            token_expansion,
            channel_expansion,
            dropout,
        )
        # Six channels give one 3-D CN field and one 3-D AD residual field.
        self.velocity_heads = nn.ModuleList(
            RegionHeads(
                scale.member_idx,
                scale.member_mask,
                self.token_dim,
                self.n_vertices,
                out_channels=6,
            )
            for scale in scales
        )
        # Match the direct Spiral experiment: the untrained model is exactly no-change.
        for head in self.velocity_heads:
            for parameter in head.parameters():
                nn.init.zeros_(parameter)

    def input_features(self, vertices: torch.Tensor) -> torch.Tensor:
        return (vertices - self.template.unsqueeze(0)) / self.coordinate_scale

    def encode(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.condition(source_age, target_age, disease)
        features = self.input_features(vertices)
        tokens = torch.cat([tokenizer(features) for tokenizer in self.tokenizers], dim=1)
        tokens = self.encoder(tokens, condition)
        latents = []
        offset = 0
        for tokenizer, projection in zip(self.tokenizers, self.w_down):
            count = tokenizer.K
            latents.append(projection(tokens[:, offset : offset + count].flatten(1)))
            offset += count
        return torch.cat(latents, dim=-1), condition

    def conditioned_latent(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.latent_input(torch.cat((latent, condition), dim=-1)))
        for block in self.latent_blocks:
            hidden = block(hidden, condition)
        return latent + self.latent_output(hidden)

    def velocity_components(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, condition = self.encode(vertices, source_age, target_age, disease)
        latent = self.conditioned_latent(latent, condition)
        pieces = torch.split(latent, self.latent_split, dim=-1)
        tokens = torch.cat(
            [
                projection(piece).unsqueeze(1) + regional.expand(piece.shape[0], -1, -1)
                for piece, projection, regional in zip(pieces, self.w_up, self.region_tokens)
            ],
            dim=1,
        )
        tokens = self.decoder(tokens, condition)
        fields = []
        offset = 0
        for tokenizer, head in zip(self.tokenizers, self.velocity_heads):
            count = tokenizer.K
            fields.append(head(tokens[:, offset : offset + count]))
            offset += count
        combined = torch.stack(fields, dim=0).sum(dim=0)
        cn, ad_residual = combined.chunk(2, dim=-1)
        return self.velocity_scale * cn, self.velocity_scale * ad_residual

    def average_velocity(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        cn, ad_residual = self.velocity_components(
            vertices, source_age, target_age, disease
        )
        label = disease.reshape(-1, 1, 1).to(vertices)
        return cn + label * ad_residual

    def instantaneous_velocity(
        self, vertices: torch.Tensor, age: torch.Tensor, disease: torch.Tensor
    ) -> torch.Tensor:
        return self.average_velocity(vertices, age, age, disease)

    def transport(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = (target_age.reshape(-1, 1, 1) - source_age.reshape(-1, 1, 1)).to(vertices)
        return vertices + elapsed * self.average_velocity(
            vertices, source_age, target_age, disease
        )

    def adaptive_support_report(self) -> list[dict[str, Any]]:
        return []

    def parameter_breakdown(self) -> dict[str, int]:
        count = lambda module: sum(value.numel() for value in module.parameters())
        return {
            "condition": count(self.condition),
            "tokenizers": count(self.tokenizers),
            "encoder": count(self.encoder),
            "down_projection": count(self.w_down),
            "latent_flow": count(self.latent_input) + count(self.latent_blocks) + count(self.latent_output),
            "up_projection": count(self.w_up),
            "decoder": count(self.decoder),
            "velocity_heads": count(self.velocity_heads),
            "total": sum(value.numel() for value in self.parameters()),
        }

    @torch.no_grad()
    def set_test_velocity_bias(self, cn: float, ad_residual: float) -> None:
        """Contract-test helper; production training never calls this."""
        for head in self.velocity_heads:
            if not hasattr(head, "bias"):
                raise RuntimeError("Contract helper requires non-shared region heads")
            values = head.bias.view(head.K, head.P, 6)
            values[..., :3].fill_(float(cn) / len(self.velocity_heads))
            values[..., 3:].fill_(float(ad_residual) / len(self.velocity_heads))
