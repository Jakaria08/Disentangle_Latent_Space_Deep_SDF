"""Band-limited Fourier features for coordinate-based SDF decoders.

A ReLU or Softplus MLP applied to raw coordinates is strongly biased toward
low-frequency functions (Tancik et al. 2020, "Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains").  On CALSNIC cortex that
bias is the reason the grid-free generalization branch renders as a smooth
shrink-wrapped surface carrying only ~55% of the true pial area: it cannot
represent folding at all.

This module fixes that without importing the failure mode of a hash grid.  The
frequency set is **explicitly band limited**: wavelengths are log spaced between
``max_wavelength_mm`` and ``min_wavelength_mm``, so the decoder gains exactly the
spatial scales asked for and nothing finer.  A hash grid, by contrast, has
locally unbounded capacity, which is what produces off-surface zero crossings.

Directions are a fixed deterministic spherical set rather than the axis-aligned
powers of two used by NeRF, which avoids the axis-aligned striping artefacts of
that encoding.  ``B`` is a buffer, not a parameter: the frequencies are part of
the hypothesis being tested and must not drift during training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def fibonacci_directions(count: int) -> torch.Tensor:
    """`count` approximately uniform directions on the sphere, deterministic."""
    if count < 1:
        raise ValueError("count must be positive.")
    index = torch.arange(count, dtype=torch.float64) + 0.5
    # Fibonacci lattice: uniform in cos(polar), golden-angle in azimuth.
    cos_polar = 1.0 - 2.0 * index / count
    polar = torch.acos(torch.clamp(cos_polar, -1.0, 1.0))
    azimuth = math.pi * (1.0 + 5.0**0.5) * index
    return torch.stack(
        (
            torch.sin(polar) * torch.cos(azimuth),
            torch.sin(polar) * torch.sin(azimuth),
            torch.cos(polar),
        ),
        dim=1,
    ).to(torch.float32)


class BandLimitedFourierFeatures(nn.Module):
    """sin/cos features over log-spaced wavelengths and fixed directions.

    Wavelengths are given in millimetres and converted with ``mm_per_unit``, so a
    configuration states the physical scales it intends to represent rather than
    an opaque frequency constant.
    """

    def __init__(
        self,
        num_bands: int = 8,
        directions_per_band: int = 8,
        min_wavelength_mm: float = 1.0,
        max_wavelength_mm: float = 120.0,
        mm_per_unit: float = 117.431971,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if num_bands < 1 or directions_per_band < 1:
            raise ValueError("num_bands and directions_per_band must be positive.")
        if not 0.0 < min_wavelength_mm < max_wavelength_mm:
            raise ValueError("Require 0 < min_wavelength_mm < max_wavelength_mm.")
        self.num_bands = int(num_bands)
        self.directions_per_band = int(directions_per_band)
        self.min_wavelength_mm = float(min_wavelength_mm)
        self.max_wavelength_mm = float(max_wavelength_mm)
        self.mm_per_unit = float(mm_per_unit)
        self.include_input = bool(include_input)

        wavelengths_mm = torch.logspace(
            math.log10(self.max_wavelength_mm),
            math.log10(self.min_wavelength_mm),
            self.num_bands,
            dtype=torch.float32,
        )
        # A wavelength in millimetres is (lambda / mm_per_unit) in model units.
        wavelengths = wavelengths_mm / self.mm_per_unit
        directions = fibonacci_directions(self.directions_per_band)
        # frequency row = 2*pi * direction / wavelength
        rows = (
            2.0
            * math.pi
            * directions[None, :, :]
            / wavelengths[:, None, None]
        ).reshape(-1, 3)
        self.register_buffer("frequencies", rows)
        self.register_buffer("wavelengths_mm", wavelengths_mm)

    @property
    def output_dim(self) -> int:
        return 2 * self.frequencies.shape[0] + (3 if self.include_input else 0)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected XYZ [N,3], got {tuple(xyz.shape)}.")
        projected = xyz @ self.frequencies.to(xyz.dtype).T
        parts = [torch.sin(projected), torch.cos(projected)]
        if self.include_input:
            parts.insert(0, xyz)
        return torch.cat(parts, dim=1)

    def extra_repr(self) -> str:
        return (
            f"bands={self.num_bands}, directions={self.directions_per_band}, "
            f"wavelengths_mm=[{self.max_wavelength_mm:g} .. {self.min_wavelength_mm:g}], "
            f"output_dim={self.output_dim}"
        )
