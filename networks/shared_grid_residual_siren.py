"""Flow-friendly global/local SDF decoder with a population-shared feature grid.

The only per-shape state is the global latent vector.  The spatial grid belongs
to the decoder and is shared by every shape, unlike the per-shape grids used by
``networks.local_decoder``.

The global and local branches both predict an SDF directly.  Their outputs are
combined with a continuous surface/ROI gate.  This follows Compact-SDF's
two-field design while retaining the project's pretrained skip-SIREN global
decoder and a much smaller skip-SIREN for the local field.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.siren_decoder import Decoder as GlobalSirenDecoder


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item < 2 for item in result):
        raise ValueError(f"{name} must contain three integers >= 2, got {value!r}.")
    return result


class LocalSkipSiren(nn.Module):
    """Compact SIREN with one configurable input skip connection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        skip_layer: int = 2,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        widths = [int(value) for value in hidden_dims]
        if not widths or any(value < 1 for value in widths):
            raise ValueError("hidden_dims must contain positive widths.")
        if skip_layer < 0 or skip_layer >= len(widths):
            raise ValueError(
                f"skip_layer must index a hidden layer, got {skip_layer} for {len(widths)} layers."
            )
        if omega_0 <= 0:
            raise ValueError("omega_0 must be positive.")
        self.input_dim = int(input_dim)
        self.skip_layer = int(skip_layer)
        self.omega_0 = float(omega_0)
        self.hidden = nn.ModuleList()
        previous = self.input_dim
        for index, width in enumerate(widths):
            if index == self.skip_layer and index > 0:
                previous += self.input_dim
            layer = nn.Linear(previous, width)
            self._initialize(layer, first=index == 0)
            self.hidden.append(layer)
            previous = width
        self.output = nn.Linear(previous, 1)
        self._initialize(self.output, first=False)

    def _initialize(self, layer: nn.Linear, first: bool) -> None:
        with torch.no_grad():
            fan_in = layer.weight.shape[1]
            bound = 1.0 / fan_in if first else (6.0 / fan_in) ** 0.5 / self.omega_0
            layer.weight.uniform_(-bound, bound)
            layer.bias.uniform_(-bound, bound)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        value = input_x
        for index, layer in enumerate(self.hidden):
            if index == self.skip_layer and index > 0:
                value = torch.cat((value, input_x), dim=1)
            value = torch.sin(self.omega_0 * layer(value))
        return self.output(value)


class Decoder(nn.Module):
    """Pretrained global SIREN plus a shared-grid-conditioned local SIREN."""

    def __init__(
        self,
        latent_size: int,
        global_network_specs: dict,
        grid_resolution: int | Sequence[int] = 32,
        grid_channels: int = 16,
        grid_aabb: Sequence[Sequence[float]] = (
            (-0.65, -0.95, -0.68),
            (0.65, 0.95, 0.68),
        ),
        local_hidden_dims: Sequence[int] = (128, 128, 128),
        local_skip_layer: int = 2,
        local_omega_0: float = 30.0,
        gate_tau: float = 0.05,
        roi_taper_width: float = 0.04,
        grid_init_std: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.grid_channels = int(grid_channels)
        if self.latent_size < 1 or self.grid_channels < 1:
            raise ValueError("latent_size and grid_channels must be positive.")
        self.grid_resolution_xyz = _triple(grid_resolution, "grid_resolution")
        if gate_tau <= 0:
            raise ValueError("gate_tau must be positive.")
        self.gate_tau = float(gate_tau)
        if roi_taper_width <= 0:
            raise ValueError("roi_taper_width must be positive.")
        self.roi_taper_width = float(roi_taper_width)

        aabb = torch.as_tensor(grid_aabb, dtype=torch.float32)
        if aabb.shape != (2, 3) or not torch.all(aabb[1] > aabb[0]):
            raise ValueError(
                "grid_aabb must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]."
            )
        self.register_buffer("grid_aabb_min", aabb[0].clone())
        self.register_buffer("grid_aabb_max", aabb[1].clone())

        self.global_decoder = GlobalSirenDecoder(
            self.latent_size, **dict(global_network_specs)
        )

        nx, ny, nz = self.grid_resolution_xyz
        # grid_sample stores volumes as [N,C,D(z),H(y),W(x)].
        self.shared_grid = nn.Parameter(
            torch.empty(1, self.grid_channels, nz, ny, nx)
        )
        nn.init.normal_(self.shared_grid, mean=0.0, std=float(grid_init_std))

        local_input_dim = self.latent_size + 3 + self.grid_channels
        self.local_decoder = LocalSkipSiren(
            local_input_dim,
            local_hidden_dims,
            skip_layer=int(local_skip_layer),
            omega_0=float(local_omega_0),
        )

    @property
    def grid_aabb(self) -> torch.Tensor:
        return torch.stack((self.grid_aabb_min, self.grid_aabb_max), dim=0)

    def normalize_grid_coordinates(
        self, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map world coordinates to grid_sample coordinates and return ROI mask."""
        minimum = self.grid_aabb_min.to(device=xyz.device, dtype=xyz.dtype)
        maximum = self.grid_aabb_max.to(device=xyz.device, dtype=xyz.dtype)
        normalized = 2.0 * (xyz - minimum) / (maximum - minimum) - 1.0
        inside = torch.logical_and(xyz >= minimum, xyz <= maximum).all(dim=-1, keepdim=True)
        return normalized, inside

    def roi_weight(self, xyz: torch.Tensor) -> torch.Tensor:
        """Continuous product taper that is zero on and outside the grid AABB."""
        minimum = self.grid_aabb_min.to(device=xyz.device, dtype=xyz.dtype)
        maximum = self.grid_aabb_max.to(device=xyz.device, dtype=xyz.dtype)
        distance = torch.minimum(xyz - minimum, maximum - xyz)
        unit = torch.clamp(distance / self.roi_taper_width, min=0.0, max=1.0)
        smooth = unit.square() * (3.0 - 2.0 * unit)
        return smooth.prod(dim=1, keepdim=True)

    def sample_shared_grid(
        self, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected xyz [N,3], got {tuple(xyz.shape)}.")
        normalized, inside = self.normalize_grid_coordinates(xyz)
        query = normalized.reshape(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.shared_grid,
            query,
            mode="bilinear",  # trilinear for a 5-D input
            padding_mode="zeros",
            align_corners=True,
        )
        features = sampled[0, :, :, 0, 0].transpose(0, 1)
        # The Boolean mask is used only to make feature padding exact.  Fusion uses
        # roi_weight(), which tapers continuously to zero before this boundary.
        features = features * inside.to(features.dtype)
        return features, inside

    def global_sdf(self, latent: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return self.global_decoder(torch.cat((latent, xyz), dim=1))

    def local_sdf(self, latent: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, inside = self.sample_shared_grid(xyz)
        prediction = self.local_decoder(torch.cat((latent, features, xyz), dim=1))
        return prediction, features, inside

    def forward(
        self,
        input_x: torch.Tensor,
        return_parts: bool = False,
        global_only: bool = False,
        fusion_alpha: float = 1.0,
    ):
        if input_x.ndim != 2 or input_x.shape[1] != self.latent_size + 3:
            raise ValueError(
                f"Expected [N,{self.latent_size + 3}], got {tuple(input_x.shape)}."
            )
        latent = input_x[:, : self.latent_size]
        xyz = input_x[:, self.latent_size :]
        global_sdf = self.global_sdf(latent, xyz)

        if global_only:
            if return_parts:
                zeros = torch.zeros_like(global_sdf)
                return {
                    "sdf": global_sdf,
                    "global_sdf": global_sdf,
                    "local_sdf": zeros,
                    "residual": zeros,
                    "correction": zeros,
                    "gate": zeros,
                    "surface_gate": zeros,
                    "roi_weight": zeros,
                    "inside_roi": torch.zeros_like(global_sdf, dtype=torch.bool),
                }
            return global_sdf

        local_sdf, features, inside = self.local_sdf(latent, xyz)
        surface_gate = torch.exp(-torch.square(global_sdf / self.gate_tau))
        roi_weight = self.roi_weight(xyz)
        gate = surface_gate * roi_weight * float(fusion_alpha)
        residual = local_sdf - global_sdf
        correction = gate * residual
        sdf = global_sdf + correction
        if return_parts:
            return {
                "sdf": sdf,
                "global_sdf": global_sdf,
                "local_sdf": local_sdf,
                "residual": residual,
                "correction": correction,
                "gate": gate,
                "surface_gate": surface_gate,
                "roi_weight": roi_weight,
                "features": features,
                "inside_roi": inside,
            }
        return sdf

    def grid_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean-square magnitude and anisotropic total variation."""
        grid_l2 = self.shared_grid.square().mean()
        differences = []
        for dimension in (2, 3, 4):
            upper = self.shared_grid.narrow(dimension, 1, self.shared_grid.shape[dimension] - 1)
            lower = self.shared_grid.narrow(dimension, 0, self.shared_grid.shape[dimension] - 1)
            differences.append(torch.abs(upper - lower).mean())
        return grid_l2, torch.stack(differences).mean()

    def set_train_stage(self, stage: str) -> None:
        """Select a trainable subset, including decoder-frozen latent adaptation."""
        if stage not in {"latent_adapt", "global_adapt", "local_warmup", "joint"}:
            raise ValueError(f"Unknown train stage {stage!r}.")
        global_trainable = stage in {"global_adapt", "joint"}
        local_trainable = stage in {"local_warmup", "joint"}
        for parameter in self.global_decoder.parameters():
            parameter.requires_grad_(global_trainable)
        for parameter in self.local_decoder.parameters():
            parameter.requires_grad_(local_trainable)
        self.shared_grid.requires_grad_(local_trainable)
