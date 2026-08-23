"""Instant-NGP style multiresolution hash encoding for population SDF fields.

This is the hash-table counterpart of ``DenseMultiResolutionEncoding`` in
``networks.shared_multires_grid_sdf``.  It keeps that module's contract exactly
-- same ``grid_aabb``, same C1 smoothstep ROI taper, same ``level_weights``
coarse-to-fine gating, same ``regularization`` return shape -- so the two
encoders are interchangeable inside one decoder and one trainer.

Two deliberate departures from a literal Instant-NGP port:

* ``resolution`` counts grid **vertices** per axis, not voxels, matching the
  repository's dense convention (``align_corners=True`` grids of shape
  ``[1, C, R, R, R]`` span the AABB with ``R - 1`` cells).  Physical cell size
  is therefore ``extent / (R - 1)``.
* A level is stored **densely** whenever ``R ** 3 <= 2 ** log2_hashmap_size``.
  Hashing a level that fits in its table only manufactures collisions, so the
  coarse levels use bijective stride indexing.  tiny-cuda-nn does the same.

Features are tapered to exactly zero at the AABB boundary, so points outside the
ROI contribute nothing regardless of how their corner indices clamp.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

# Instant-NGP's spatial hash primes (Teschner et al. 2003; NGP eq. 4).
_HASH_PRIMES = (1, 2654435761, 805459861)


def _aabb_tensor(value: Sequence[Sequence[float]]) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (2, 3) or not torch.all(result[1] > result[0]):
        raise ValueError("grid_aabb must be [[xmin,ymin,zmin],[xmax,ymax,zmax]].")
    return result


def hash_ladder(
    base_resolution: int, per_level_scale: float, num_levels: int
) -> tuple[int, ...]:
    """Instant-NGP geometric resolution ladder ``floor(base * scale ** level)``."""
    base = int(base_resolution)
    scale = float(per_level_scale)
    levels = int(num_levels)
    if base < 2:
        raise ValueError("base_resolution must be >= 2.")
    if scale <= 1.0:
        raise ValueError("per_level_scale must be > 1.")
    if levels < 1:
        raise ValueError("num_levels must be positive.")
    ladder = tuple(int(base * (scale**level)) for level in range(levels))
    if tuple(sorted(set(ladder))) != ladder:
        raise ValueError(
            f"per_level_scale={scale} with base={base} produces a non-increasing "
            f"ladder {ladder}; raise the scale or lower the level count."
        )
    return ladder


class MultiResolutionHashEncoding(nn.Module):
    """Multiresolution hash grids queried with trilinear interpolation."""

    def __init__(
        self,
        grid_aabb: Sequence[Sequence[float]],
        resolutions: Sequence[int] | None = None,
        base_resolution: int = 16,
        per_level_scale: float = 1.2944,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        taper_width: float = 0.04,
        init_std: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if resolutions is None:
            self.resolutions = hash_ladder(base_resolution, per_level_scale, num_levels)
        else:
            self.resolutions = tuple(int(value) for value in resolutions)
            if not self.resolutions or any(value < 2 for value in self.resolutions):
                raise ValueError("grid_resolutions must contain integers >= 2.")
            if tuple(sorted(set(self.resolutions))) != self.resolutions:
                raise ValueError("grid_resolutions must be strictly increasing.")
        self.channels = int(features_per_level)
        if self.channels < 1:
            raise ValueError("features_per_level must be positive.")
        self.log2_hashmap_size = int(log2_hashmap_size)
        if not 4 <= self.log2_hashmap_size <= 30:
            raise ValueError("log2_hashmap_size must lie in [4, 30].")
        if taper_width <= 0.0:
            raise ValueError("feature_roi_taper_width must be positive.")
        self.taper_width = float(taper_width)
        self.base_resolution = int(self.resolutions[0])
        self.num_levels = len(self.resolutions)

        aabb = _aabb_tensor(grid_aabb)
        self.register_buffer("grid_aabb_min", aabb[0].clone())
        self.register_buffer("grid_aabb_max", aabb[1].clone())

        table_limit = 1 << self.log2_hashmap_size
        self.grids = nn.ParameterList()
        dense_flags: list[bool] = []
        for resolution in self.resolutions:
            vertices = resolution**3
            dense = vertices <= table_limit
            entries = vertices if dense else table_limit
            table = nn.Parameter(torch.empty(entries, self.channels))
            nn.init.uniform_(table, -float(init_std), float(init_std))
            self.grids.append(table)
            dense_flags.append(dense)
        self.dense_levels = tuple(dense_flags)

        corners = torch.stack(
            torch.meshgrid(
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(8, 3)
        self.register_buffer("corner_offsets", corners.to(torch.int64))
        self.register_buffer(
            "hash_primes", torch.tensor(_HASH_PRIMES, dtype=torch.int64)
        )

    # ------------------------------------------------------------------
    # Geometry helpers -- identical semantics to DenseMultiResolutionEncoding.
    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        return self.num_levels * self.channels

    @property
    def grid_aabb(self) -> torch.Tensor:
        return torch.stack((self.grid_aabb_min, self.grid_aabb_max), dim=0)

    @property
    def table_entries(self) -> tuple[int, ...]:
        return tuple(int(table.shape[0]) for table in self.grids)

    def cell_sizes(self) -> torch.Tensor:
        """Physical cell size per level, in the AABB's units, shaped [L, 3]."""
        extent = (self.grid_aabb_max - self.grid_aabb_min).to(torch.float64)
        counts = torch.tensor(
            [max(1, resolution - 1) for resolution in self.resolutions],
            dtype=torch.float64,
        )
        return (extent[None, :] / counts[:, None]).to(torch.float32)

    def normalize(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        minimum = self.grid_aabb_min.to(device=xyz.device, dtype=xyz.dtype)
        maximum = self.grid_aabb_max.to(device=xyz.device, dtype=xyz.dtype)
        normalized = (xyz - minimum) / (maximum - minimum)
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
            return xyz.new_ones(self.num_levels)
        result = torch.as_tensor(level_weights, device=xyz.device, dtype=xyz.dtype)
        if result.shape != (self.num_levels,):
            raise ValueError(
                f"Expected {self.num_levels} level weights, got {tuple(result.shape)}."
            )
        if torch.any(result < 0.0) or torch.any(result > 1.0):
            raise ValueError("level_weights must lie in [0,1].")
        return result

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _lookup_indices(self, corner_indices: torch.Tensor, level: int) -> torch.Tensor:
        """Map integer corner coordinates [N,8,3] to table rows [N,8]."""
        resolution = self.resolutions[level]
        if self.dense_levels[level]:
            return (
                corner_indices[..., 0] * resolution + corner_indices[..., 1]
            ) * resolution + corner_indices[..., 2]
        primes = self.hash_primes.to(corner_indices.device)
        hashed = torch.bitwise_xor(
            torch.bitwise_xor(
                corner_indices[..., 0] * primes[0], corner_indices[..., 1] * primes[1]
            ),
            corner_indices[..., 2] * primes[2],
        )
        return torch.bitwise_and(hashed, (1 << self.log2_hashmap_size) - 1)

    def forward(
        self, xyz: torch.Tensor, level_weights=None, return_levels: bool = False
    ):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected XYZ [N,3], got {tuple(xyz.shape)}.")
        unit, inside = self.normalize(xyz)
        taper = self.roi_weight(xyz)
        weights = self._weights(xyz, level_weights)
        # Clamp only the index path; the taper already zeroes outside-ROI points.
        clamped = unit.clamp(0.0, 1.0)
        offsets = self.corner_offsets.to(xyz.device)

        levels = []
        for level, table in enumerate(self.grids):
            resolution = self.resolutions[level]
            position = clamped * (resolution - 1)
            base = torch.floor(position)
            fraction = position - base
            base_index = base.to(torch.int64).clamp_(0, resolution - 2)
            corner_indices = base_index.unsqueeze(1) + offsets  # [N,8,3]
            rows = self._lookup_indices(corner_indices, level)  # [N,8]
            gathered = table[rows]  # [N,8,F]
            corner_fraction = offsets.unsqueeze(0).to(fraction.dtype)
            blend = corner_fraction * fraction.unsqueeze(1) + (
                1.0 - corner_fraction
            ) * (1.0 - fraction.unsqueeze(1))
            corner_weight = blend.prod(dim=-1, keepdim=True)  # [N,8,1]
            feature = (gathered * corner_weight).sum(dim=1)  # [N,F]
            levels.append(feature * taper * weights[level])

        encoded = torch.cat(levels, dim=1)
        if return_levels:
            return encoded, levels, taper, inside, weights
        return encoded

    def regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        """L2 over active tables; total variation is undefined for hashed storage."""
        reference = self.grids[0]
        weights = self._weights(reference.new_zeros(1, 3), level_weights)
        denominator = weights.sum().clamp_min(1.0e-12)
        l2_terms = [
            weight * table.square().mean() for weight, table in zip(weights, self.grids)
        ]
        l2 = torch.stack(l2_terms).sum() / denominator
        # Hash tables carry no spatial adjacency, so no meaningful TV term exists.
        return l2, reference.new_zeros(())

    def extra_repr(self) -> str:
        return (
            f"levels={self.num_levels}, features_per_level={self.channels}, "
            f"log2_hashmap_size={self.log2_hashmap_size}, "
            f"resolutions={self.resolutions}, dense_levels={sum(self.dense_levels)}"
        )
