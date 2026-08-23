"""Deformed implicit field: one canonical SDF plus a per-subject warp.

    sdf(x, z) = T( x + W(x, z) ) + s(x, z)

Follows the factorization of "Deformed Implicit Field: Modeling 3D Shapes with
Learned Dense Correspondence" (Deng et al., CVPR 2021), adapted to CALSNIC
cortex and to this repository's exact-SDF supervision.

Why this shape of model, on this data
-------------------------------------
Arms A-C regressed ``sdf(x, z)`` directly over Euclidean space and all plateaued
around 64% of true pial area against 75% for a PCA-172 oracle.  The measured
reason is not capacity: latent 256 -> 512 moved ASSD 4.6%, releasing
``code_bound`` 4x moved it 0.8%, a hash grid *hurt*, and MC 512 added 2 points of
area.  The reason is correspondence.  At a fixed point ``x``, 78.3% of training
subjects disagree on inside/outside, with a median across-subject SDF spread of
1.74 mm -- wider than the folds themselves.  A shared Euclidean field must
therefore implement a ``z``-dependent sign flip almost everywhere in the
cortical ribbon, and the L1-optimal smooth answer is the population median
surface.  That is exactly the smooth, under-folded surface those arms produce.

The factorization removes that requirement:

* ``T`` is the **canonical** field.  It sees no latent, so its whole capacity
  goes to representing folding *once*, in a frame where folds do not move.
* ``W`` is the **warp**.  It is conditioned on ``z`` and band limited to long
  wavelengths, so it can slide folds into subject position but cannot
  manufacture the sub-millimetre structure that fragmented the grid arms.  It is
  bounded by a tanh so the deformation cannot fold space over on itself.
* ``s`` is a small scalar **residual** for what a warp cannot express -- a fold
  that is present in one subject and absent in another changes topology, and no
  diffeomorphism does that.  It is held small by an L2 penalty rather than
  bounded, following the paper.

The module exposes the same contract as ``networks.fourier_global_sdf``: an
empty ``grids`` list, an empty resolution ladder and a zero grid
regularization, so it drops into the existing trainer, optimizer groups and
decode path unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from networks.conditional_hashgrid_sdf import build_mlp, make_activation
from networks.fourier_features import BandLimitedFourierFeatures

READOUT_MODES = ("full", "template")


class Decoder(nn.Module):
    """Canonical SDF + latent-conditioned warp + small scalar residual."""

    def __init__(
        self,
        latent_size: int,
        # -- canonical template field: no latent, fine wavelengths ------------
        template_bands: int = 10,
        template_directions: int = 12,
        template_min_wavelength_mm: float = 0.8,
        template_max_wavelength_mm: float = 120.0,
        template_hidden_dims: Sequence[int] = (512,) * 8,
        # -- warp field: latent conditioned, LONG wavelengths only ------------
        warp_bands: int = 4,
        warp_directions: int = 8,
        warp_min_wavelength_mm: float = 15.0,
        warp_max_wavelength_mm: float = 120.0,
        warp_hidden_dims: Sequence[int] = (256,) * 4,
        warp_latent_skip_layer: int = 2,
        warp_scale_mm: float = 6.0,
        # -- scalar residual: latent conditioned, penalised toward zero -------
        residual_enabled: bool = True,
        residual_bands: int = 6,
        residual_directions: int = 8,
        residual_min_wavelength_mm: float = 4.0,
        residual_max_wavelength_mm: float = 120.0,
        residual_hidden_dims: Sequence[int] = (128,) * 3,
        residual_latent_skip_layer: int = 1,
        # -- shared ------------------------------------------------------------
        mm_per_unit: float = 117.431971,
        activation: str = "relu",
        softplus_beta: float = 100.0,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        if self.latent_size < 1:
            raise ValueError("latent_size must be positive.")
        if warp_scale_mm <= 0.0:
            raise ValueError("warp_scale_mm must be positive.")
        self.mm_per_unit = float(mm_per_unit)
        # The warp is bounded in model units; the config states millimetres.
        self.warp_scale = float(warp_scale_mm) / self.mm_per_unit
        self.activation = make_activation(activation, softplus_beta)

        # -- canonical field ---------------------------------------------------
        self.template_encoding = BandLimitedFourierFeatures(
            num_bands=template_bands,
            directions_per_band=template_directions,
            min_wavelength_mm=template_min_wavelength_mm,
            max_wavelength_mm=template_max_wavelength_mm,
            mm_per_unit=self.mm_per_unit,
            include_input=True,
        )
        # latent_size 0 with skip layer 0 means "no latent anywhere in this MLP":
        # build_mlp only concatenates at a skip index strictly greater than zero.
        self.template_hidden, self.template_output = build_mlp(
            self.template_encoding.output_dim, 0, template_hidden_dims, 0
        )

        # -- warp field --------------------------------------------------------
        if not warp_min_wavelength_mm > template_min_wavelength_mm:
            raise ValueError(
                "warp_min_wavelength_mm must exceed template_min_wavelength_mm; a "
                "warp as fine as the template can re-create the high-frequency "
                "noise the canonical field is meant to hold still."
            )
        self.warp_encoding = BandLimitedFourierFeatures(
            num_bands=warp_bands,
            directions_per_band=warp_directions,
            min_wavelength_mm=warp_min_wavelength_mm,
            max_wavelength_mm=warp_max_wavelength_mm,
            mm_per_unit=self.mm_per_unit,
            include_input=True,
        )
        self.warp_latent_skip_layer = int(warp_latent_skip_layer)
        self.warp_hidden, self.warp_output = build_mlp(
            self.latent_size + self.warp_encoding.output_dim,
            self.latent_size,
            warp_hidden_dims,
            self.warp_latent_skip_layer,
            output_dim=3,
        )

        # -- scalar residual ---------------------------------------------------
        self.residual_enabled = bool(residual_enabled)
        if self.residual_enabled:
            self.residual_encoding = BandLimitedFourierFeatures(
                num_bands=residual_bands,
                directions_per_band=residual_directions,
                min_wavelength_mm=residual_min_wavelength_mm,
                max_wavelength_mm=residual_max_wavelength_mm,
                mm_per_unit=self.mm_per_unit,
                include_input=True,
            )
            self.residual_latent_skip_layer = int(residual_latent_skip_layer)
            self.residual_hidden, self.residual_output = build_mlp(
                self.latent_size + self.residual_encoding.output_dim,
                self.latent_size,
                residual_hidden_dims,
                self.residual_latent_skip_layer,
            )

        self.readout_mode = "full"
        # The trainer builds a "grid" optimizer group from this; it stays empty.
        self._grids = nn.ParameterList()

    # ------------------------------------------------------------------
    # Grid contract expected by the shared trainer and decode path
    # ------------------------------------------------------------------
    @property
    def grid_resolutions(self) -> tuple[int, ...]:
        return ()

    @property
    def grids(self) -> nn.ParameterList:
        return self._grids

    def grid_regularization(self, level_weights=None) -> tuple[torch.Tensor, torch.Tensor]:
        zero = next(self.parameters()).new_zeros(())
        return zero, zero

    # ------------------------------------------------------------------
    def set_readout_mode(self, mode: str) -> None:
        """Select what a bare ``forward`` returns.

        ``template`` evaluates the canonical field at undeformed coordinates,
        ignoring the latent entirely.  Decoding it is the headline diagnostic for
        this architecture: it shows whether the shared field learned a folded
        cortex, independently of whether the warp is placing it correctly.
        """
        mode = str(mode)
        if mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}.")
        self.readout_mode = mode

    def template_parameters(self):
        yield from self.template_hidden.parameters()
        yield from self.template_output.parameters()

    def warp_parameters(self):
        yield from self.warp_hidden.parameters()
        yield from self.warp_output.parameters()

    # ------------------------------------------------------------------
    def _run(
        self,
        trunk: nn.ModuleList,
        head: nn.Linear,
        skip: int,
        value: torch.Tensor,
        latent: torch.Tensor | None,
    ) -> torch.Tensor:
        for index, layer in enumerate(trunk):
            if latent is not None and index == skip and index > 0:
                value = torch.cat((value, latent), dim=1)
            value = self.activation(layer(value))
        return head(value)

    def _split(self, input_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.latent_size + 3
        if input_x.ndim != 2 or input_x.shape[1] != expected:
            raise ValueError(f"Expected [N,{expected}], got {tuple(input_x.shape)}.")
        return input_x[:, : self.latent_size], input_x[:, self.latent_size :]

    def template_sdf(self, xyz: torch.Tensor) -> torch.Tensor:
        """The canonical field, evaluated wherever asked. Carries no latent."""
        return self._run(
            self.template_hidden,
            self.template_output,
            0,
            self.template_encoding(xyz),
            None,
        )

    def warp_unit(self, input_x: torch.Tensor) -> torch.Tensor:
        """Warp in units of ``warp_scale``: every component lies in (-1, 1).

        Exposed separately because this, not the displacement, is what says
        whether the tanh is saturating.  A component at |unit| -> 1 has
        d(tanh)/d(raw) -> 0, so its gradient dies and the warp silently stops
        being able to move that direction any further.
        """
        latent, xyz = self._split(input_x)
        raw = self._run(
            self.warp_hidden,
            self.warp_output,
            self.warp_latent_skip_layer,
            torch.cat((self.warp_encoding(xyz), latent), dim=1),
            latent,
        )
        return torch.tanh(raw)

    def warp(self, input_x: torch.Tensor) -> torch.Tensor:
        """Per-subject displacement, bounded to +/- ``warp_scale`` per axis."""
        return self.warp_scale * self.warp_unit(input_x)

    def residual_sdf(self, input_x: torch.Tensor) -> torch.Tensor:
        """Scalar correction for structure no diffeomorphism can produce."""
        if not self.residual_enabled:
            latent, _xyz = self._split(input_x)
            return latent.new_zeros(len(latent), 1)
        latent, xyz = self._split(input_x)
        return self._run(
            self.residual_hidden,
            self.residual_output,
            self.residual_latent_skip_layer,
            torch.cat((self.residual_encoding(xyz), latent), dim=1),
            latent,
        )

    def template_only_sdf(self, input_x: torch.Tensor) -> torch.Tensor:
        """Branch entry point for the Eikonal term and for decoding."""
        _latent, xyz = self._split(input_x)
        return self.template_sdf(xyz)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_x: torch.Tensor,
        level_weights=None,
        return_parts: bool = False,
    ):
        latent, xyz = self._split(input_x)
        unit = self.warp_unit(input_x)
        displacement = self.warp_scale * unit
        canonical = self.template_sdf(xyz + displacement)
        residual = self.residual_sdf(input_x)
        full = canonical + residual

        if self.readout_mode == "template":
            sdf = self.template_sdf(xyz)
        else:
            sdf = full

        if return_parts:
            return {
                "sdf": sdf,
                "sdf_full": full,
                "sdf_canonical": canonical,
                "warp": displacement,
                "warp_unit": unit,
                "residual": residual,
                # Contract keys the shared trainer and tests expect. There is no
                # grid, so there is no region of interest to taper and no levels.
                "features": xyz,
                "level_features": [],
                "level_weights": xyz.new_zeros(0),
                "roi_weight": xyz.new_ones(len(xyz), 1),
                "inside_roi": xyz.new_ones(len(xyz), 1, dtype=torch.bool),
            }
        return sdf

    def extra_repr(self) -> str:
        return (
            f"latent_size={self.latent_size}, "
            f"warp_scale={self.warp_scale * self.mm_per_unit:g} mm, "
            f"residual={'on' if self.residual_enabled else 'off'}, "
            f"readout={self.readout_mode}"
        )
