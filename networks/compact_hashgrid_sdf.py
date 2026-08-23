"""Compact-SDF style two-branch SDF with an Instant-NGP local branch.

Follows the design of "Learning Compact Latent Space for Representing Neural
Signed Distance Functions with High-fidelity Geometry Details"
(arXiv:2511.14539): one *generalization* branch that sees only coordinates and
the shape code, and one *overfitting* branch that additionally sees an
interpolated spatial feature.  Both branches share a single 256-D code, so the
per-shape representation stays compact.

    global(x, z) = MLP_gen( xyz | z )                      # smooth, complete field
    local(x, z)  = MLP_ovf( xyz | HashGrid(x) | z )        # near-surface detail

Two departures from the paper, both recorded in the task README:

* The local branch uses a multiresolution **hash** encoding instead of a dense
  128**3 x 128 grid.  On CALSNIC cortex a 128**3 grid resolves only ~1.5 mm,
  which is coarser than the sulcal gaps the branch is supposed to sharpen.
* Besides the paper's hard narrow-band replacement (``hard_band``, applied on
  the reconstruction lattice by the decode routine, not here), this module also
  offers a continuous ``smooth_gate`` fusion.  The paper notes that bandwidth
  choice can leave residual noise or a discontinuity at the band boundary; the
  gated form removes that failure mode by construction and mirrors the working
  gate in ``networks.shared_grid_residual_siren``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from networks.conditional_hashgrid_sdf import build_mlp, make_activation, resolve_ladder
from networks.fourier_features import BandLimitedFourierFeatures
from networks.multires_hashgrid_encoding import MultiResolutionHashEncoding

FUSION_MODES = ("global", "local", "smooth_gate")


class Decoder(nn.Module):
    """Generalization branch + hash-grid overfitting branch over one shared Z."""

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
        global_hidden_dims: Sequence[int] = (512,) * 8,
        global_latent_skip_layer: int = 4,
        global_activation: str = "relu",
        global_fourier_enabled: bool = False,
        global_fourier_bands: int = 8,
        global_fourier_directions: int = 8,
        global_fourier_min_wavelength_mm: float = 1.0,
        global_fourier_max_wavelength_mm: float = 120.0,
        mm_per_unit: float = 117.431971,
        local_hidden_dims: Sequence[int] = (128, 128, 128, 128),
        local_latent_skip_layer: int = 2,
        local_activation: str = "softplus",
        softplus_beta: float = 100.0,
        gate_tau: float = 0.02,
        fusion_mode: str = "smooth_gate",
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        if self.latent_size < 1:
            raise ValueError("latent_size must be positive.")
        if gate_tau <= 0.0:
            raise ValueError("gate_tau must be positive.")
        self.gate_tau = float(gate_tau)

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

        # A raw-coordinate MLP cannot represent cortical folding at all; band
        # limited Fourier features give it exactly the wavelengths configured and
        # nothing finer, unlike the locally unbounded capacity of a hash grid.
        self.global_encoding = (
            BandLimitedFourierFeatures(
                num_bands=global_fourier_bands,
                directions_per_band=global_fourier_directions,
                min_wavelength_mm=global_fourier_min_wavelength_mm,
                max_wavelength_mm=global_fourier_max_wavelength_mm,
                mm_per_unit=mm_per_unit,
                include_input=True,
            )
            if global_fourier_enabled
            else None
        )
        global_input_dim = (
            self.global_encoding.output_dim if self.global_encoding is not None else 3
        )
        self.global_latent_skip_layer = int(global_latent_skip_layer)
        self.global_activation = make_activation(global_activation, softplus_beta)
        self.global_hidden, self.global_output = build_mlp(
            self.latent_size + global_input_dim,
            self.latent_size,
            global_hidden_dims,
            self.global_latent_skip_layer,
        )

        self.local_latent_skip_layer = int(local_latent_skip_layer)
        self.local_activation = make_activation(local_activation, softplus_beta)
        self.local_hidden, self.local_output = build_mlp(
            self.latent_size + 3 + self.encoding.output_dim,
            self.latent_size,
            local_hidden_dims,
            self.local_latent_skip_layer,
        )

        self.fusion_mode = "smooth_gate"
        self.set_fusion_mode(fusion_mode)

    # ------------------------------------------------------------------
    def set_fusion_mode(self, mode: str) -> None:
        """Select what a bare ``forward`` returns; ``hard_band`` is decode-time."""
        mode = str(mode)
        if mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES}; 'hard_band' is applied "
                "on the reconstruction lattice by decode_fused_to_mesh."
            )
        self.fusion_mode = mode

    @property
    def grid_resolutions(self) -> tuple[int, ...]:
        return self.encoding.resolutions

    @property
    def grids(self) -> nn.ParameterList:
        return self.encoding.grids

    def grid_regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoding.regularization(level_weights)

    def global_parameters(self):
        yield from self.global_hidden.parameters()
        yield from self.global_output.parameters()

    def local_parameters(self):
        yield from self.local_hidden.parameters()
        yield from self.local_output.parameters()

    # ------------------------------------------------------------------
    def _global_features(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz if self.global_encoding is None else self.global_encoding(xyz)

    def _split(self, input_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.latent_size + 3
        if input_x.ndim != 2 or input_x.shape[1] != expected:
            raise ValueError(f"Expected [N,{expected}], got {tuple(input_x.shape)}.")
        return input_x[:, : self.latent_size], input_x[:, self.latent_size :]

    def _run(
        self,
        trunk: nn.ModuleList,
        head: nn.Linear,
        activation: nn.Module,
        skip: int,
        value: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        for index, layer in enumerate(trunk):
            if index == skip and index > 0:
                value = torch.cat((value, latent), dim=1)
            value = activation(layer(value))
        return head(value)

    def global_sdf(self, input_x: torch.Tensor) -> torch.Tensor:
        """Generalization branch alone -- no grid input, so it cannot fragment."""
        latent, xyz = self._split(input_x)
        return self._run(
            self.global_hidden,
            self.global_output,
            self.global_activation,
            self.global_latent_skip_layer,
            torch.cat((self._global_features(xyz), latent), dim=1),
            latent,
        )

    def local_sdf(self, input_x: torch.Tensor, level_weights=None) -> torch.Tensor:
        """Overfitting branch alone."""
        latent, xyz = self._split(input_x)
        encoded = self.encoding(xyz, level_weights=level_weights)
        return self._run(
            self.local_hidden,
            self.local_output,
            self.local_activation,
            self.local_latent_skip_layer,
            torch.cat((xyz, encoded, latent), dim=1),
            latent,
        )

    def forward(
        self,
        input_x: torch.Tensor,
        level_weights=None,
        return_parts: bool = False,
    ):
        latent, xyz = self._split(input_x)
        encoded, levels, taper, inside, weights = self.encoding(
            xyz, level_weights=level_weights, return_levels=True
        )
        sdf_global = self._run(
            self.global_hidden,
            self.global_output,
            self.global_activation,
            self.global_latent_skip_layer,
            torch.cat((self._global_features(xyz), latent), dim=1),
            latent,
        )
        sdf_local = self._run(
            self.local_hidden,
            self.local_output,
            self.local_activation,
            self.local_latent_skip_layer,
            torch.cat((xyz, encoded, latent), dim=1),
            latent,
        )
        # Gaussian in the global distance: detail is admitted only where the
        # generalization branch already believes a surface is nearby.
        gate = torch.exp(-(sdf_global / self.gate_tau).square()) * taper
        fused = sdf_global + gate * (sdf_local - sdf_global)

        if self.fusion_mode == "global":
            sdf = sdf_global
        elif self.fusion_mode == "local":
            sdf = sdf_local
        else:
            sdf = fused

        if return_parts:
            return {
                "sdf": sdf,
                "sdf_global": sdf_global,
                "sdf_local": sdf_local,
                "sdf_smooth_gate": fused,
                "gate": gate,
                "features": encoded,
                "level_features": levels,
                "level_weights": weights,
                "roi_weight": taper,
                "inside_roi": inside,
            }
        return sdf
