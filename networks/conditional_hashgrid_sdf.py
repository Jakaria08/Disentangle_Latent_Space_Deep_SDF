"""Conditional Instant-NGP SDF: shared hash grids + one latent code per shape.

This is the Compact-SDF paper's Instant-NGP baseline, made multi-shape the way
that paper describes: a shape-specific global latent code ``z`` is concatenated
to the hash-grid features and fed to a shared MLP, so ``z`` is constant across
every query point of a subject.

    SDF(x, i) = MLP( xyz | HashGrid(x) | z_i )

The only per-shape state is ``z``; hash tables and MLP are population-shared.
The MLP matches ``networks.shared_multires_grid_sdf`` exactly, so a run against
this module differs from the dense multiresolution baseline in the *encoder
alone*.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from networks.multires_hashgrid_encoding import MultiResolutionHashEncoding, hash_ladder


def build_mlp(
    input_dim: int,
    latent_size: int,
    hidden_dims: Sequence[int],
    latent_skip_layer: int,
    output_dim: int = 1,
) -> tuple[nn.ModuleList, nn.Linear]:
    """Skip-connected MLP trunk with the repository's standard initialisation.

    ``output_dim`` defaults to the scalar SDF head every caller here wants; the
    deformed-implicit-field decoder asks for three so the same trunk builder and
    the same near-zero head initialisation produce its warp field.
    """
    widths = tuple(int(value) for value in hidden_dims)
    if not widths or any(value < 1 for value in widths):
        raise ValueError("hidden_dims must contain positive widths.")
    if latent_skip_layer < 0 or latent_skip_layer >= len(widths):
        raise ValueError("latent_skip_layer must index a hidden layer.")
    hidden = nn.ModuleList()
    previous = input_dim
    for index, width in enumerate(widths):
        if index == latent_skip_layer and index > 0:
            previous += latent_size
        layer = nn.Linear(previous, width)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
        hidden.append(layer)
        previous = width
    if output_dim < 1:
        raise ValueError("output_dim must be positive.")
    output = nn.Linear(previous, int(output_dim))
    nn.init.normal_(output.weight, mean=0.0, std=1.0e-4)
    nn.init.zeros_(output.bias)
    return hidden, output


def make_activation(activation: str, softplus_beta: float) -> nn.Module:
    activation = str(activation).lower()
    if activation == "softplus":
        return nn.Softplus(beta=float(softplus_beta))
    if activation == "silu":
        return nn.SiLU()
    if activation == "relu":
        return nn.ReLU()
    raise ValueError("activation must be 'softplus', 'silu', or 'relu'.")


def resolve_ladder(
    resolutions: Sequence[int] | None,
    base_resolution: int,
    per_level_scale: float,
    num_levels: int,
) -> tuple[int, ...]:
    """Derive the ladder, and check any pinned ladder against the generator."""
    derived = hash_ladder(base_resolution, per_level_scale, num_levels)
    if resolutions is None:
        return derived
    pinned = tuple(int(value) for value in resolutions)
    if pinned != derived:
        raise ValueError(
            "grid_resolutions does not match base_resolution/per_level_scale/"
            f"num_levels. Config pins {pinned}, generator yields {derived}."
        )
    return pinned


class Decoder(nn.Module):
    """One shared SDF decoder conditioned on XYZ, hash features, and Z."""

    def __init__(
        self,
        latent_size: int,
        grid_resolutions: Sequence[int] | None = None,
        base_resolution: int = 16,
        per_level_scale: float = 1.2944,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        grid_aabb: Sequence[Sequence[float]] = (
            (-0.45, -0.85, -0.70),
            (0.45, 0.85, 0.70),
        ),
        feature_roi_taper_width: float = 0.04,
        grid_init_std: float = 1.0e-4,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        latent_skip_layer: int = 2,
        activation: str = "softplus",
        softplus_beta: float = 100.0,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        if self.latent_size < 1:
            raise ValueError("latent_size must be positive.")
        ladder = resolve_ladder(
            grid_resolutions, base_resolution, per_level_scale, num_levels
        )
        self.encoding = MultiResolutionHashEncoding(
            grid_aabb,
            resolutions=ladder,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            taper_width=feature_roi_taper_width,
            init_std=grid_init_std,
        )
        self.latent_skip_layer = int(latent_skip_layer)
        self.activation = make_activation(activation, softplus_beta)
        self.hidden, self.output = build_mlp(
            self.latent_size + 3 + self.encoding.output_dim,
            self.latent_size,
            hidden_dims,
            self.latent_skip_layer,
        )

    @property
    def grid_resolutions(self) -> tuple[int, ...]:
        return self.encoding.resolutions

    @property
    def grids(self) -> nn.ParameterList:
        return self.encoding.grids

    def forward(
        self,
        input_x: torch.Tensor,
        level_weights=None,
        return_parts: bool = False,
    ):
        expected = self.latent_size + 3
        if input_x.ndim != 2 or input_x.shape[1] != expected:
            raise ValueError(f"Expected [N,{expected}], got {tuple(input_x.shape)}.")
        latent = input_x[:, : self.latent_size]
        xyz = input_x[:, self.latent_size :]
        encoded, levels, taper, inside, weights = self.encoding(
            xyz, level_weights=level_weights, return_levels=True
        )
        value = torch.cat((xyz, encoded, latent), dim=1)
        for index, layer in enumerate(self.hidden):
            if index == self.latent_skip_layer and index > 0:
                value = torch.cat((value, latent), dim=1)
            value = self.activation(layer(value))
        sdf = self.output(value)
        if return_parts:
            return {
                "sdf": sdf,
                "features": encoded,
                "level_features": levels,
                "level_weights": weights,
                "roi_weight": taper,
                "inside_roi": inside,
            }
        return sdf

    def grid_regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoding.regularization(level_weights)
