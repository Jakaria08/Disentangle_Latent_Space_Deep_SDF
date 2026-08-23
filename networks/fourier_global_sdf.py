"""Grid-free population SDF: band-limited Fourier features plus one latent code.

    sdf(x, i) = MLP( xyz | FourierFeatures(x) | z_i )

This is the Compact-SDF generalization branch with its spectral bias removed and
nothing else added.  It exists to test one specific question: on CALSNIC cortex
the grid-free branch produces by far the most intact surface (about 32 connected
components against 300-1200 for every grid-based model, and the best volume
error) but carries only ~55% of the true pial area, because a ReLU MLP on raw
coordinates cannot represent folding.  If band-limited Fourier features recover
the folds while keeping that clean topology, no spatial grid is needed at all.

Because there is no grid, the field has no locally unbounded capacity and cannot
manufacture off-surface zero crossings.  The highest representable frequency is
set explicitly by ``fourier_min_wavelength_mm``.

The module still exposes the grid contract the shared trainer expects -- an
empty ``grids`` list, an empty resolution ladder, and a zero grid
regularization -- so it drops into the same optimizer, schedule and decode path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from networks.conditional_hashgrid_sdf import build_mlp, make_activation
from networks.fourier_features import BandLimitedFourierFeatures


class Decoder(nn.Module):
    """One shared SDF decoder over Fourier-encoded coordinates and Z."""

    def __init__(
        self,
        latent_size: int,
        fourier_bands: int = 10,
        fourier_directions: int = 12,
        fourier_min_wavelength_mm: float = 0.8,
        fourier_max_wavelength_mm: float = 120.0,
        mm_per_unit: float = 117.431971,
        hidden_dims: Sequence[int] = (512,) * 8,
        latent_skip_layer: int = 4,
        activation: str = "relu",
        softplus_beta: float = 100.0,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        if self.latent_size < 1:
            raise ValueError("latent_size must be positive.")
        self.encoding = BandLimitedFourierFeatures(
            num_bands=fourier_bands,
            directions_per_band=fourier_directions,
            min_wavelength_mm=fourier_min_wavelength_mm,
            max_wavelength_mm=fourier_max_wavelength_mm,
            mm_per_unit=mm_per_unit,
            include_input=True,
        )
        self.latent_skip_layer = int(latent_skip_layer)
        self.activation = make_activation(activation, softplus_beta)
        self.hidden, self.output = build_mlp(
            self.latent_size + self.encoding.output_dim,
            self.latent_size,
            hidden_dims,
            self.latent_skip_layer,
        )
        # The trainer builds a "grid" optimizer group from this; it stays empty.
        self._grids = nn.ParameterList()

    @property
    def grid_resolutions(self) -> tuple[int, ...]:
        return ()

    @property
    def grids(self) -> nn.ParameterList:
        return self._grids

    def grid_regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        zero = next(self.parameters()).new_zeros(())
        return zero, zero

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
        encoded = self.encoding(xyz)
        value = torch.cat((encoded, latent), dim=1)
        for index, layer in enumerate(self.hidden):
            if index == self.latent_skip_layer and index > 0:
                value = torch.cat((value, latent), dim=1)
            value = self.activation(layer(value))
        sdf = self.output(value)
        if return_parts:
            return {
                "sdf": sdf,
                "features": encoded,
                "level_features": [],
                "level_weights": xyz.new_zeros(0),
                "roi_weight": xyz.new_ones(len(xyz), 1),
                "inside_roi": xyz.new_ones(len(xyz), 1, dtype=torch.bool),
            }
        return sdf
