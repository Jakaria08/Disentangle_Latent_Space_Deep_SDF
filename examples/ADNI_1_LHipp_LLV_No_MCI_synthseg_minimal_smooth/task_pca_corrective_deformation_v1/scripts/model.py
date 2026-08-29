#!/usr/bin/env python3
"""PCA-conditioned graph decoder with a hard orthogonal corrective displacement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PCAContract, SOURCE_SCRIPTS

if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))

from conv_layers import SpiralConv  # noqa: E402
import spiral_common as source_sc  # noqa: E402


def sparse_pool(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a sparse vertex transform to batched vertex features."""
    row, col = transform._indices()
    values = transform._values().to(dtype=x.dtype)
    gathered = x.index_select(1, col) * values.view(1, -1, 1)
    output = x.new_zeros((x.shape[0], transform.size(0), x.shape[2]))
    output.index_add_(1, row, gathered)
    return output


def orthogonal_project(delta_flat: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Project displacements outside the row span of an orthonormal PCA basis."""
    return delta_flat - (delta_flat @ components.T) @ components


class ConditionedSpiralBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indices: torch.Tensor,
        global_dim: int,
        dropout: float,
        film_scale: float,
    ):
        super().__init__()
        self.conv_in = SpiralConv(in_channels, out_channels, indices=indices)
        self.conv_residual = SpiralConv(out_channels, out_channels, indices=indices)
        self.film = nn.Linear(global_dim, 2 * out_channels)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.film_scale = float(film_scale)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        value = F.elu(self.conv_in(x))
        gamma, beta = self.film(global_feature).chunk(2, dim=-1)
        value = value * (1.0 + self.film_scale * torch.tanh(gamma).unsqueeze(1))
        value = value + self.film_scale * beta.unsqueeze(1)
        return value + self.dropout(F.elu(self.conv_residual(value)))


class CorrectiveSpiralDecoder(nn.Module):
    """Decode a fixed PCA code into a small displacement on its PCA reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        channels_fine_to_coarse: list[int],
        global_dim: int,
        global_hidden_dim: int,
        spiral_indices: list[torch.Tensor],
        down_transforms: list[torch.Tensor],
        up_transforms: list[torch.Tensor],
        dropout: float,
        film_scale: float,
    ):
        super().__init__()
        if len(channels_fine_to_coarse) != len(up_transforms):
            raise ValueError("One decoder channel is required per hierarchy transform")
        self.channels = list(int(value) for value in channels_fine_to_coarse)
        self.down_transforms = down_transforms
        self.up_transforms = up_transforms
        self.global_network = nn.Sequential(
            nn.Linear(int(latent_dim), int(global_hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(global_hidden_dim)),
            nn.Linear(int(global_hidden_dim), int(global_dim)),
            nn.GELU(),
        )
        coarse_vertices = int(down_transforms[-1].size(0))
        coarse_channels = self.channels[-1]
        self.seed = nn.Linear(int(global_dim), coarse_vertices * coarse_channels)
        self.coarse_coordinate = nn.Linear(3, coarse_channels, bias=False)

        blocks = []
        current_channels = coarse_channels
        for level in reversed(range(len(self.channels))):
            target_channels = self.channels[level]
            blocks.append(
                ConditionedSpiralBlock(
                    current_channels + 3,
                    target_channels,
                    indices=spiral_indices[level],
                    global_dim=int(global_dim),
                    dropout=float(dropout),
                    film_scale=float(film_scale),
                )
            )
            current_channels = target_channels
        self.blocks = nn.ModuleList(blocks)
        self.output = SpiralConv(current_channels, 3, indices=spiral_indices[0])
        nn.init.xavier_uniform_(self.seed.weight)
        nn.init.zeros_(self.seed.bias)
        nn.init.xavier_uniform_(self.coarse_coordinate.weight)
        # The complete model is exactly PCA before the first update.
        nn.init.zeros_(self.output.layer.weight)
        nn.init.zeros_(self.output.layer.bias)

    def forward(self, standardized_coefficients: torch.Tensor, pca_offset: torch.Tensor) -> torch.Tensor:
        global_feature = self.global_network(standardized_coefficients)
        coordinate_levels = [pca_offset]
        for transform in self.down_transforms:
            coordinate_levels.append(sparse_pool(coordinate_levels[-1], transform))

        coarse_vertices = coordinate_levels[-1].shape[1]
        value = self.seed(global_feature).view(
            standardized_coefficients.shape[0], coarse_vertices, self.channels[-1]
        )
        value = value + self.coarse_coordinate(coordinate_levels[-1])

        for block_index, level in enumerate(reversed(range(len(self.channels)))):
            value = sparse_pool(value, self.up_transforms[level])
            value = torch.cat((value, coordinate_levels[level]), dim=-1)
            value = self.blocks[block_index](value, global_feature)
        return self.output(value)


class PCACorrectiveModel(nn.Module):
    def __init__(
        self,
        contract: PCAContract,
        decoder: CorrectiveSpiralDecoder,
        hard_orthogonal_projection: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.hard_orthogonal_projection = bool(hard_orthogonal_projection)
        self.n_vertices = contract.n_vertices
        self.register_buffer("pca_mean", torch.from_numpy(contract.mean.copy()))
        self.register_buffer("pca_components", torch.from_numpy(contract.components.copy()))
        self.register_buffer(
            "coefficient_mean", torch.from_numpy(contract.coefficient_mean.copy())
        )
        self.register_buffer(
            "coefficient_std", torch.from_numpy(contract.coefficient_std.copy())
        )
        self.register_buffer(
            "residual_rms_mm", torch.tensor(float(contract.residual_rms_mm), dtype=torch.float32)
        )
        self.register_buffer(
            "coordinate_scale_mm",
            torch.tensor(float(contract.coordinate_scale_mm), dtype=torch.float32),
        )

    def pca_decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        flat = coefficients @ self.pca_components + self.pca_mean
        return flat.view(-1, self.n_vertices, 3)

    def pca_encode(self, vertices: torch.Tensor) -> torch.Tensor:
        flat = vertices.reshape(vertices.shape[0], -1) - self.pca_mean
        return flat @ self.pca_components.T

    def decode_with_details(self, coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
        pca_mesh = self.pca_decode(coefficients)
        standardized = (coefficients - self.coefficient_mean) / self.coefficient_std
        mean_mesh = self.pca_mean.view(1, self.n_vertices, 3)
        pca_offset = (pca_mesh - mean_mesh) / self.coordinate_scale_mm
        raw_normalized = self.decoder(standardized, pca_offset)
        raw_delta = raw_normalized * self.residual_rms_mm
        flat_delta = raw_delta.reshape(raw_delta.shape[0], -1)
        if self.hard_orthogonal_projection:
            flat_delta = orthogonal_project(flat_delta, self.pca_components)
        delta = flat_delta.view_as(raw_delta)
        prediction = pca_mesh + delta
        return {
            "prediction": prediction,
            "pca_mesh": pca_mesh,
            "delta": delta,
            "raw_delta": raw_delta,
        }

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.decode_with_details(coefficients)["prediction"]

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(config: dict, contract: PCAContract, device: torch.device) -> PCACorrectiveModel:
    hierarchy = config["hierarchy"]
    transform = source_sc.get_transform(hierarchy["downsample_factors"], rows=contract.rows)
    n_levels = len(transform["down_transform"])
    channels = list(config["model"]["channels_fine_to_coarse"])
    if len(channels) != n_levels:
        raise ValueError(f"Configured {len(channels)} channel levels for {n_levels} transforms")
    spirals, _dynamic, down, up = source_sc.build_spiral_stack(
        transform,
        seq_length=int(hierarchy["spiral_sequence_length"]),
        dilation=int(hierarchy["spiral_dilation"]),
        dynamic_seq_lengths=[1] * n_levels,
        device=device,
    )
    decoder = CorrectiveSpiralDecoder(
        latent_dim=int(config["latent_dim"]),
        channels_fine_to_coarse=channels,
        global_dim=int(config["model"]["global_dim"]),
        global_hidden_dim=int(config["model"]["global_hidden_dim"]),
        spiral_indices=spirals,
        down_transforms=down,
        up_transforms=up,
        dropout=float(config["model"]["dropout"]),
        film_scale=float(config["model"]["film_scale"]),
    )
    return PCACorrectiveModel(
        contract,
        decoder,
        hard_orthogonal_projection=bool(config["model"]["hard_orthogonal_projection"]),
    ).to(device)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)
