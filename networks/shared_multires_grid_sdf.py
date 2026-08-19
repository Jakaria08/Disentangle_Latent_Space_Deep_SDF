"""Single-field population SDF with dense multiresolution spatial features.

The only per-shape state is the latent vector.  Every dense feature grid and
the decoder are shared by the population.  Grid features are smoothly tapered
to zero at the ROI boundary, while raw XYZ and the shape code remain available
everywhere so the same field can learn both near- and far-surface distances.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aabb_tensor(value: Sequence[Sequence[float]]) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (2, 3) or not torch.all(result[1] > result[0]):
        raise ValueError("grid_aabb must be [[xmin,ymin,zmin],[xmax,ymax,zmax]].")
    return result


class DenseMultiResolutionEncoding(nn.Module):
    """Collision-free dense grids queried with trilinear interpolation."""

    def __init__(
        self,
        resolutions: Sequence[int],
        channels: int,
        grid_aabb: Sequence[Sequence[float]],
        taper_width: float = 0.04,
        init_std: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.resolutions = tuple(int(value) for value in resolutions)
        self.channels = int(channels)
        if not self.resolutions or any(value < 2 for value in self.resolutions):
            raise ValueError("grid_resolutions must contain integers >= 2.")
        if tuple(sorted(set(self.resolutions))) != self.resolutions:
            raise ValueError("grid_resolutions must be strictly increasing.")
        if self.channels < 1:
            raise ValueError("grid_channels_per_level must be positive.")
        if taper_width <= 0.0:
            raise ValueError("feature_roi_taper_width must be positive.")
        self.taper_width = float(taper_width)
        aabb = _aabb_tensor(grid_aabb)
        self.register_buffer("grid_aabb_min", aabb[0].clone())
        self.register_buffer("grid_aabb_max", aabb[1].clone())
        self.grids = nn.ParameterList()
        for resolution in self.resolutions:
            grid = nn.Parameter(
                torch.empty(1, self.channels, resolution, resolution, resolution)
            )
            nn.init.normal_(grid, mean=0.0, std=float(init_std))
            self.grids.append(grid)

    @property
    def output_dim(self) -> int:
        return len(self.resolutions) * self.channels

    @property
    def grid_aabb(self) -> torch.Tensor:
        return torch.stack((self.grid_aabb_min, self.grid_aabb_max), dim=0)

    def normalize(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        minimum = self.grid_aabb_min.to(device=xyz.device, dtype=xyz.dtype)
        maximum = self.grid_aabb_max.to(device=xyz.device, dtype=xyz.dtype)
        normalized = 2.0 * (xyz - minimum) / (maximum - minimum) - 1.0
        inside = torch.logical_and(xyz >= minimum, xyz <= maximum).all(dim=1, keepdim=True)
        return normalized, inside

    def roi_weight(self, xyz: torch.Tensor) -> torch.Tensor:
        """C1 smoothstep taper from one to zero inside the ROI boundary."""
        minimum = self.grid_aabb_min.to(device=xyz.device, dtype=xyz.dtype)
        maximum = self.grid_aabb_max.to(device=xyz.device, dtype=xyz.dtype)
        distance = torch.minimum(xyz - minimum, maximum - xyz)
        unit = torch.clamp(distance / self.taper_width, min=0.0, max=1.0)
        smooth = unit.square() * (3.0 - 2.0 * unit)
        return smooth.prod(dim=1, keepdim=True)

    def _weights(self, xyz: torch.Tensor, level_weights) -> torch.Tensor:
        if level_weights is None:
            return xyz.new_ones(len(self.resolutions))
        result = torch.as_tensor(level_weights, device=xyz.device, dtype=xyz.dtype)
        if result.shape != (len(self.resolutions),):
            raise ValueError(
                f"Expected {len(self.resolutions)} level weights, got {tuple(result.shape)}."
            )
        if torch.any(result < 0.0) or torch.any(result > 1.0):
            raise ValueError("level_weights must lie in [0,1].")
        return result

    def forward(
        self, xyz: torch.Tensor, level_weights=None, return_levels: bool = False
    ):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected XYZ [N,3], got {tuple(xyz.shape)}.")
        normalized, inside = self.normalize(xyz)
        taper = self.roi_weight(xyz)
        weights = self._weights(xyz, level_weights)
        query = normalized.reshape(1, -1, 1, 1, 3)
        levels = []
        for index, grid in enumerate(self.grids):
            sampled = F.grid_sample(
                grid,
                query,
                mode="bilinear",  # trilinear for a five-dimensional volume
                padding_mode="zeros",
                align_corners=True,
            )
            feature = sampled[0, :, :, 0, 0].transpose(0, 1)
            feature = feature * taper * weights[index]
            levels.append(feature)
        encoded = torch.cat(levels, dim=1)
        if return_levels:
            return encoded, levels, taper, inside, weights
        return encoded

    def regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.grids[0]
        weights = self._weights(reference.new_zeros(1, 3), level_weights)
        denominator = weights.sum().clamp_min(1.0e-12)
        l2_terms = []
        tv_terms = []
        for weight, grid in zip(weights, self.grids):
            l2_terms.append(weight * grid.square().mean())
            differences = []
            for dimension in (2, 3, 4):
                upper = grid.narrow(dimension, 1, grid.shape[dimension] - 1)
                lower = grid.narrow(dimension, 0, grid.shape[dimension] - 1)
                differences.append(torch.abs(upper - lower).mean())
            tv_terms.append(weight * torch.stack(differences).mean())
        return torch.stack(l2_terms).sum() / denominator, torch.stack(tv_terms).sum() / denominator


class Decoder(nn.Module):
    """One shared SDF decoder conditioned on XYZ, multigrid features, and Z."""

    def __init__(
        self,
        latent_size: int,
        grid_resolutions: Sequence[int] = (8, 16, 24, 32, 48, 64),
        grid_channels_per_level: int = 4,
        grid_aabb: Sequence[Sequence[float]] = (
            (-0.65, -0.95, -0.68),
            (0.65, 0.95, 0.68),
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
        self.encoding = DenseMultiResolutionEncoding(
            grid_resolutions,
            grid_channels_per_level,
            grid_aabb,
            taper_width=feature_roi_taper_width,
            init_std=grid_init_std,
        )
        widths = tuple(int(value) for value in hidden_dims)
        if not widths or any(value < 1 for value in widths):
            raise ValueError("hidden_dims must contain positive widths.")
        self.latent_skip_layer = int(latent_skip_layer)
        if self.latent_skip_layer < 0 or self.latent_skip_layer >= len(widths):
            raise ValueError("latent_skip_layer must index a hidden layer.")
        activation = str(activation).lower()
        if activation == "softplus":
            self.activation = nn.Softplus(beta=float(softplus_beta))
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError("activation must be 'softplus' or 'silu'.")

        input_dim = self.latent_size + 3 + self.encoding.output_dim
        self.hidden = nn.ModuleList()
        previous = input_dim
        for index, width in enumerate(widths):
            if index == self.latent_skip_layer and index > 0:
                previous += self.latent_size
            layer = nn.Linear(previous, width)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            previous = width
        self.output = nn.Linear(previous, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.output.bias)

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

