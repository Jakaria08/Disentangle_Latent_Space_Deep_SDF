#!/usr/bin/env python3
"""Shared utilities for the CALSNIC Instant-NGP / Compact-SDF cortex experiment.

Everything that is already tested elsewhere is imported, not reimplemented:

* sampling, dataset, seeding, checkpoint helpers, latent fitting and marching
  cubes come from the ADNI ``task2_inr_multires_single_field_v1`` engine;
* manifest handling, the bulk-path guard and the SDF-space to millimetre
  transform come from the CALSNIC ``task_exact_multires_cortex_v1`` task.

This module adds only what hash grids and the two-branch model genuinely need:

1. a finite-difference epsilon pinned in **millimetres** rather than derived
   from the finest active grid resolution (at resolution 767 that derivation
   yields 0.26 mm, far too small for a stable central difference);
2. per-branch field callables and gradients for the two-branch decoder;
3. ``decode_fused_to_mesh``, the paper's narrow-band replacement, applied on the
   reconstruction lattice;
4. a hash capacity/collision report used by ``check_pipeline`` and provenance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
MULTIRES_TASK = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_inr_multires_single_field_v1"
)
GENERIC_SCRIPTS = MULTIRES_TASK / "scripts"
CALSNIC_TASK = REPO_ROOT / "examples" / "CALSNIC_control_L" / "task_exact_multires_cortex_v1"
CALSNIC_SCRIPTS = CALSNIC_TASK / "scripts"

for _path in (REPO_ROOT, GENERIC_SCRIPTS, CALSNIC_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from multires_common import (  # noqa: E402
    ContinuousSDFDataset,
    REFERENCE_TASK,
    append_csv,
    atomic_torch_save,
    choose_device,
    clamped_l1,
    config_public,
    decode_latent_to_mesh,
    effective_eikonal_weight,
    expand_codes,
    file_signature,
    fit_single_latent,
    level_weights_for_epoch,
    checkpoint_path,
    load_manifest,
    require_bulk_path,
    resolve_repo_path,
    seed_dataloader_worker,
    select_near_surface_points,
    select_stratified_rows,
    set_global_seed,
    sha256_file,
    stable_seed,
    validate_manifest_contract,
    write_csv,
    write_json,
)
from multires_common import finite_difference_epsilon as _resolution_epsilon  # noqa: E402
from multires_common import load_config as _load_multires_config  # noqa: E402

from calsnic_common import (  # noqa: E402
    exact_signed_distance,
    load_sdf_space_mesh,
    mesh_sdf_to_mm,
    read_manifest,
    sdf_vertices_to_mm,
)

# Every CALSNIC scan shares one scaled-OBJ similarity scale, so the SDF space is
# a uniform rescaling of millimetres and an isotropic epsilon stays isotropic.
MM_PER_UNIT_TOLERANCE = 1.0e-4

HASH_ARCHITECTURES = {
    "conditional_hashgrid_sdf": "conditional_instant_ngp_sdf",
    "compact_hashgrid_sdf": "compact_sdf_two_branch_instant_ngp",
    "fourier_global_sdf": "grid_free_band_limited_fourier_sdf",
    "deformed_implicit_field_sdf": "deformed_implicit_field",
}
# Architectures with no spatial grid: the ladder and collision checks do not apply.
GRID_FREE_ARCHITECTURES = {"fourier_global_sdf", "deformed_implicit_field_sdf"}
# Architectures carrying a latent-conditioned warp, so the deformation
# regularizers in the trainer apply and ``template_only`` is decodable.
DEFORMATION_ARCHITECTURES = {"deformed_implicit_field_sdf"}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MAX_MM_PER_UNIT_RELATIVE_SPREAD = 1.0e-6


def manifest_mm_per_unit(manifest_path: str | Path) -> float:
    """Millimetres per normalized unit, read back from the manifest itself.

    The per-scan values differ only in the last few digits of the fitted
    similarity transform, so the cohort shares one scale in practice.  The
    spread is checked rather than exact equality; a genuinely per-scan scale
    would break the assumption that one normalized epsilon is one millimetre
    distance for every subject.
    """
    rows = read_manifest(manifest_path)
    scales = np.asarray([float(row["scaled_from_mm_scale"]) for row in rows])
    if not np.all(scales > 0.0):
        raise ValueError("scaled_from_mm_scale must be positive for every scan.")
    values = 1.0 / scales
    median = float(np.median(values))
    spread = float(values.max() - values.min()) / median
    if spread > MAX_MM_PER_UNIT_RELATIVE_SPREAD:
        raise ValueError(
            f"scaled_from_mm_scale varies by {spread:.3e} across the cohort; an "
            "isotropic millimetre epsilon assumes a single shared scale."
        )
    return median


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a run config and enforce the hash-grid specific contract."""
    config = _load_multires_config(path)
    arch = str(config["network_arch"])
    if arch not in HASH_ARCHITECTURES:
        raise ValueError(
            f"network_arch {arch!r} is not a hash-grid architecture; expected one of "
            f"{sorted(HASH_ARCHITECTURES)}."
        )
    specs = config["network_specs"]
    if arch in GRID_FREE_ARCHITECTURES:
        if specs.get("grid_resolutions"):
            raise ValueError(f"{arch} has no spatial grid; grid_resolutions must be empty.")
        if config.get("level_schedule"):
            raise ValueError(f"{arch} has no grid levels; level_schedule must be empty.")
        _check_mm_scale(config)
        return config
    ladder = tuple(int(value) for value in specs["grid_resolutions"])
    from networks.multires_hashgrid_encoding import hash_ladder

    derived = hash_ladder(
        int(specs["base_resolution"]),
        float(specs["per_level_scale"]),
        int(specs["num_levels"]),
    )
    if ladder != derived:
        raise ValueError(
            f"grid_resolutions {ladder} does not match base_resolution/"
            f"per_level_scale/num_levels, which yield {derived}."
        )
    if int(specs["grid_resolution"]) != max(ladder):
        raise ValueError("network_specs.grid_resolution must equal max(grid_resolutions).")
    _check_mm_scale(config)
    return config


def _check_mm_scale(config: dict[str, Any]) -> None:
    expected_mm = manifest_mm_per_unit(config["manifest"])
    declared = float(config["mm_per_normalized_unit"])
    if abs(declared - expected_mm) > MM_PER_UNIT_TOLERANCE:
        raise ValueError(
            f"config mm_per_normalized_unit={declared} disagrees with the manifest "
            f"value {expected_mm}."
        )


def architecture_name(config: dict[str, Any]) -> str:
    return HASH_ARCHITECTURES[str(config["network_arch"])]


def is_two_branch(config: dict[str, Any]) -> bool:
    return str(config["network_arch"]) == "compact_hashgrid_sdf"


# ----------------------------------------------------------------------------
# Eikonal support
# ----------------------------------------------------------------------------
def finite_difference_epsilon(
    epoch: int, config: dict[str, Any], level_weights: list[float]
) -> np.ndarray:
    """Central-difference step, pinned in millimetres when the config asks.

    The resolution-derived rule in ``multires_common`` shrinks epsilon with the
    finest active level.  A hash ladder reaching resolution 767 would drive it to
    0.26 mm, where a central difference on a hash field is dominated by
    interpolation noise.  ``eikonal.epsilon_mm`` pins it instead.
    """
    settings = config.get("eikonal", {})
    millimetres = float(settings.get("epsilon_mm", 0.0))
    if millimetres <= 0.0:
        return _resolution_epsilon(epoch, config, level_weights)
    final_mm = float(settings.get("final_epsilon_mm", millimetres))
    final_start = int(
        settings.get("final_epsilon_start_epoch", int(config["total_epochs"]) + 1)
    )
    if epoch >= final_start:
        millimetres = final_mm
    if millimetres <= 0.0:
        raise ValueError("eikonal.epsilon_mm values must be positive.")
    step = millimetres / float(config["mm_per_normalized_unit"])
    return np.full(3, step, dtype=np.float32)


def field_callable(model, branch: str) -> Callable[..., torch.Tensor]:
    """Return ``f(input_x, level_weights) -> [N,1]`` for one branch of the model."""
    if branch == "fused":
        return lambda value, level_weights=None: model(value, level_weights=level_weights)
    if branch == "global":
        if not hasattr(model, "global_sdf"):
            raise ValueError("This architecture has no separate global branch.")
        return lambda value, level_weights=None: model.global_sdf(value)
    if branch == "local":
        if not hasattr(model, "local_sdf"):
            raise ValueError("This architecture has no separate local branch.")
        return lambda value, level_weights=None: model.local_sdf(
            value, level_weights=level_weights
        )
    if branch == "template":
        if not hasattr(model, "template_only_sdf"):
            raise ValueError("This architecture has no canonical template branch.")
        return lambda value, level_weights=None: model.template_only_sdf(value)
    raise ValueError(
        f"Unknown branch {branch!r}; expected fused, global, local, or template."
    )


def numerical_spatial_gradient_field(
    field: Callable[..., torch.Tensor],
    codes: torch.Tensor,
    xyz: torch.Tensor,
    epsilon_xyz: torch.Tensor,
    level_weights: list[float],
) -> torch.Tensor:
    """Central differences with one batched six-offset evaluation of ``field``.

    Identical arithmetic to ``multires_common.numerical_spatial_gradient``; it
    only takes a callable so a single checkpoint's global, local, and fused
    fields can each be constrained.
    """
    if epsilon_xyz.shape != (3,) or torch.any(epsilon_xyz <= 0.0):
        raise ValueError("epsilon_xyz must be a positive three-vector.")
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    plus = xyz[None, :, :] + identity[:, None, :] * epsilon_xyz[None, None, :]
    minus = xyz[None, :, :] - identity[:, None, :] * epsilon_xyz[None, None, :]
    queries = torch.cat((plus, minus), dim=0).reshape(-1, 3)
    repeated = codes[None, :, :].expand(6, -1, -1).reshape(-1, codes.shape[1])
    values = field(torch.cat((repeated, queries), dim=1), level_weights).reshape(
        6, len(xyz), 1
    )
    return (
        ((values[:3] - values[3:]) / (2.0 * epsilon_xyz[:, None, None]))
        .squeeze(-1)
        .transpose(0, 1)
    )


def numerical_gradient_and_laplacian(
    field: Callable[..., torch.Tensor],
    codes: torch.Tensor,
    xyz: torch.Tensor,
    epsilon_xyz: torch.Tensor,
    level_weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference gradient and Laplacian from one batched 7-point stencil.

    The six offsets the Eikonal term already needs also give the second
    derivatives, so the Laplacian costs a single extra evaluation (the centre)
    rather than the nine-point-per-axis stencil a full Hessian would need.

    The step size is the band selector.  A central second difference at step h
    responds to oscillation at scale ~h and is nearly blind to structure much
    larger than h, so penalising this Laplacian at h = 0.5 mm suppresses
    voxel-scale jitter while leaving 3-10 mm cortical folds untouched.  This is
    the opposite intent to the CAD literature, which drives curvature toward zero
    to obtain developable patches; cortex is curved everywhere and must stay so.
    """
    if epsilon_xyz.shape != (3,) or torch.any(epsilon_xyz <= 0.0):
        raise ValueError("epsilon_xyz must be a positive three-vector.")
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    plus = xyz[None, :, :] + identity[:, None, :] * epsilon_xyz[None, None, :]
    minus = xyz[None, :, :] - identity[:, None, :] * epsilon_xyz[None, None, :]
    queries = torch.cat((plus, minus, xyz[None, :, :]), dim=0).reshape(-1, 3)
    repeated = codes[None, :, :].expand(7, -1, -1).reshape(-1, codes.shape[1])
    values = field(torch.cat((repeated, queries), dim=1), level_weights).reshape(
        7, len(xyz), 1
    )
    forward, backward, centre = values[:3], values[3:6], values[6]
    gradient = (
        ((forward - backward) / (2.0 * epsilon_xyz[:, None, None]))
        .squeeze(-1)
        .transpose(0, 1)
    )
    second = (forward - 2.0 * centre[None] + backward) / (
        epsilon_xyz[:, None, None] ** 2
    )
    laplacian = second.sum(dim=0).squeeze(-1)
    return gradient, laplacian


def numerical_jacobian_squared(
    field: Callable[..., torch.Tensor],
    codes: torch.Tensor,
    xyz: torch.Tensor,
    epsilon_xyz: torch.Tensor,
) -> torch.Tensor:
    """Per-point squared Frobenius norm of a vector field's Jacobian.

    Central differences from one batched six-offset evaluation, the same stencil
    ``numerical_gradient_and_laplacian`` uses for the scalar case.  ``field``
    must map ``[N, latent+3] -> [N, 3]``.

    This is the smoothness measure for a deformation field: penalising it keeps
    the warp locally close to a rigid motion, which is what stops it from
    folding space over on itself or re-creating high-frequency detail that the
    canonical field is supposed to hold still.
    """
    if epsilon_xyz.shape != (3,) or torch.any(epsilon_xyz <= 0.0):
        raise ValueError("epsilon_xyz must be a positive three-vector.")
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    plus = xyz[None, :, :] + identity[:, None, :] * epsilon_xyz[None, None, :]
    minus = xyz[None, :, :] - identity[:, None, :] * epsilon_xyz[None, None, :]
    queries = torch.cat((plus, minus), dim=0).reshape(-1, 3)
    repeated = codes[None, :, :].expand(6, -1, -1).reshape(-1, codes.shape[1])
    values = field(torch.cat((repeated, queries), dim=1)).reshape(6, len(xyz), 3)
    forward, backward = values[:3], values[3:]
    # jacobian[a, n, b] = d(field_b) / d(x_a) at point n
    jacobian = (forward - backward) / (2.0 * epsilon_xyz[:, None, None])
    return jacobian.square().sum(dim=(0, 2))


def far_field_agreement(
    sdf_global: torch.Tensor,
    sdf_local: torch.Tensor,
    target: torch.Tensor,
    near_band: float,
    margin: float,
) -> torch.Tensor:
    """Penalize the local branch drifting from the global branch off-surface.

    Hash features are active everywhere inside the ROI, so the overfitting branch
    can invent zero crossings far from the surface.  Outside ``near_band`` the
    two branches are required to agree to within ``margin``; inside it the local
    branch is free to add detail.
    """
    mask = target.abs() > float(near_band)
    if not bool(mask.any()):
        return sdf_local.new_zeros(())
    deviation = (sdf_local - sdf_global).abs()[mask] - float(margin)
    return torch.clamp(deviation, min=0.0).mean()


# ----------------------------------------------------------------------------
# Narrow-band fused reconstruction
# ----------------------------------------------------------------------------
def _surface_band(volume: np.ndarray, band_cells: int) -> np.ndarray:
    """Cells straddling the zero level set, dilated by ``band_cells`` layers."""
    from scipy import ndimage

    positive = volume > 0.0
    mask = np.zeros_like(positive, dtype=bool)
    for axis in (0, 1, 2):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        change = positive[tuple(lower)] != positive[tuple(upper)]
        mask[tuple(lower)] |= change
        mask[tuple(upper)] |= change
    if band_cells > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(band_cells))
    return mask


def _lattice_xyz(index: torch.Tensor, resolution: int, step: float) -> torch.Tensor:
    x = torch.div(index, resolution * resolution, rounding_mode="floor")
    y = torch.div(index, resolution, rounding_mode="floor") % resolution
    z = index % resolution
    return torch.stack((x, y, z), dim=1).float() * step - 1.0


def _evaluate_field(
    field: Callable[..., torch.Tensor],
    latent_tensor: torch.Tensor,
    index: torch.Tensor,
    resolution: int,
    step: float,
    level_weights: list[float],
) -> torch.Tensor:
    xyz = _lattice_xyz(index, resolution, step)
    value = torch.cat((latent_tensor.expand(len(index), -1), xyz), dim=1)
    return field(value, level_weights)[:, 0]


def decode_fused_to_mesh(
    decoder,
    latent: np.ndarray,
    output_path: str | Path,
    resolution: int,
    max_batch: int,
    device: torch.device,
    band_cells: int = 3,
) -> dict[str, Any]:
    """Compact-SDF narrow-band fusion, evaluated on the reconstruction lattice.

    The generalization branch is evaluated everywhere; the cells its zero level
    set crosses are dilated by ``band_cells`` layers; the overfitting branch then
    replaces the field inside that band only.  This is the paper's stated
    inference procedure, which its released ``gl_sdf`` does not actually perform.
    """
    import trimesh
    from skimage.measure import marching_cubes

    output = require_bulk_path(output_path, "mesh output")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolution = int(resolution)
    max_batch = int(max_batch)
    latent_tensor = (
        torch.from_numpy(np.asarray(latent, dtype=np.float32)).reshape(1, -1).to(device)
    )
    level_weights = [1.0] * len(decoder.grid_resolutions)
    global_field = field_callable(decoder, "global")
    local_field = field_callable(decoder, "local")

    total = resolution**3
    step = 2.0 / (resolution - 1)
    values = np.empty(total, dtype=np.float32)
    decoder.eval()
    with torch.no_grad():
        for start in range(0, total, max_batch):
            stop = min(start + max_batch, total)
            index = torch.arange(start, stop, device=device)
            values[start:stop] = (
                _evaluate_field(
                    global_field, latent_tensor, index, resolution, step, level_weights
                )
                .detach()
                .cpu()
                .numpy()
            )

    volume = values.reshape(resolution, resolution, resolution)
    global_min, global_max = float(volume.min()), float(volume.max())
    if not global_min <= 0.0 <= global_max:
        raise RuntimeError(
            f"Global branch has no zero level set: [{global_min}, {global_max}]"
        )
    band = _surface_band(volume, band_cells)
    band_flat = np.flatnonzero(band.reshape(-1))
    band_count = int(len(band_flat))

    # Replace the field inside the band with the overfitting branch, in place.
    with torch.no_grad():
        for start in range(0, band_count, max_batch):
            stop = min(start + max_batch, band_count)
            index = torch.from_numpy(band_flat[start:stop]).to(device)
            values[band_flat[start:stop]] = (
                _evaluate_field(
                    local_field, latent_tensor, index, resolution, step, level_weights
                )
                .detach()
                .cpu()
                .numpy()
            )

    volume = values.reshape(resolution, resolution, resolution)
    value_min, value_max = float(volume.min()), float(volume.max())
    if not value_min <= 0.0 <= value_max:
        raise RuntimeError(f"No zero level set after fusion: [{value_min}, {value_max}]")
    vertices, faces, _normals, _values = marching_cubes(
        volume, level=0.0, spacing=(step, step, step), method="lewiner"
    )
    vertices += np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    mesh.export(temporary)
    os.replace(temporary, output)
    return {
        "mesh_path": str(output),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "coordinate_space": "normalized",
        "sdf_grid_min": value_min,
        "sdf_grid_max": value_max,
        "global_grid_min": global_min,
        "global_grid_max": global_max,
        "band_cells": int(band_cells),
        "band_lattice_points": band_count,
        "band_fraction": band_count / float(total),
    }


def decode_variant_to_mesh(
    decoder,
    latent: np.ndarray,
    output_path: str | Path,
    resolution: int,
    max_batch: int,
    device: torch.device,
    variant: str,
    band_cells: int = 3,
) -> dict[str, Any]:
    """Decode one readout of a checkpoint: single, global, local, or fused."""
    if variant == "fused_hard_band":
        return decode_fused_to_mesh(
            decoder, latent, output_path, resolution, max_batch, device, band_cells
        )
    # A deformation field switches readout instead of fusion; 'single' means the
    # warped canonical field, 'template_only' the canonical field on its own.
    if hasattr(decoder, "set_readout_mode"):
        readouts = {"single": "full", "template_only": "template"}
        if variant not in readouts:
            raise ValueError(
                f"Unknown reconstruction variant {variant!r} for a deformation "
                f"field; expected one of {sorted(readouts)}."
            )
        previous_readout = decoder.readout_mode
        decoder.set_readout_mode(readouts[variant])
        try:
            return decode_latent_to_mesh(
                decoder, latent, output_path, resolution, max_batch, device, scaling=None
            )
        finally:
            decoder.set_readout_mode(previous_readout)
    modes = {
        "single": None,
        "global_only": "global",
        "local_only": "local",
        "fused_smooth_gate": "smooth_gate",
    }
    if variant not in modes:
        raise ValueError(f"Unknown reconstruction variant {variant!r}.")
    mode = modes[variant]
    previous = getattr(decoder, "fusion_mode", None)
    if mode is not None:
        decoder.set_fusion_mode(mode)
    elif previous is not None:
        raise ValueError("Variant 'single' is only valid for a one-field architecture.")
    try:
        return decode_latent_to_mesh(
            decoder, latent, output_path, resolution, max_batch, device, scaling=None
        )
    finally:
        if previous is not None:
            decoder.set_fusion_mode(previous)


def variants_for(config: dict[str, Any], requested: list[str] | None = None) -> list[str]:
    if is_deformation_field(config):
        # 'template_only' decodes the canonical field with the latent ignored.
        return list(requested) if requested else ["single", "template_only"]
    if not is_two_branch(config):
        return ["single"]
    default = ["global_only", "local_only", "fused_hard_band", "fused_smooth_gate"]
    return list(requested) if requested else default


# ----------------------------------------------------------------------------
# Capacity / collision reporting
# ----------------------------------------------------------------------------
def build_decoder(config: dict[str, Any], device: torch.device):
    """Construct the decoder, dropping keys that are dataset-only for this arch.

    ``grid_aabb`` stays in the config because the sampler and the manifest
    contract check both need it, but a grid-free decoder takes no such argument.
    """
    import importlib

    module = importlib.import_module(f"networks.{config['network_arch']}")
    specs = dict(config["network_specs"])
    for key in ("grid_resolution", "sampling_balance_resolution"):
        specs.pop(key, None)
    if is_grid_free(config):
        for key in ("grid_resolutions", "grid_aabb"):
            specs.pop(key, None)
    return module.Decoder(int(config["latent_size"]), **specs).to(device)


def load_decoder_checkpoint(config: dict[str, Any], checkpoint: str | Path, device):
    """Load a checkpoint through this task's ``build_decoder``.

    The shared engine's version calls its own ``build_decoder``, which passes
    every ``network_specs`` key to the constructor and therefore fails on a
    grid-free architecture that takes no ``grid_resolutions`` or ``grid_aabb``.
    Redefining it here keeps evaluation, latent export and reconstruction all
    going through the one decoder factory that understands both cases.
    """
    path = checkpoint_path(config, checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    decoder = build_decoder(config, device)
    state = payload.get("model_state_dict", payload)
    decoder.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}
    )
    decoder.eval()
    return decoder, payload, path


def is_grid_free(config: dict[str, Any]) -> bool:
    return str(config["network_arch"]) in GRID_FREE_ARCHITECTURES


def is_deformation_field(config: dict[str, Any]) -> bool:
    return str(config["network_arch"]) in DEFORMATION_ARCHITECTURES


def hash_capacity_report(config: dict[str, Any]) -> dict[str, Any]:
    """Per-level storage, physical cell size, and near-surface collision load.

    ``surface_cells`` estimates how many cells one cortical surface crosses at a
    given level (area divided by cell area).  Dividing by the table size gives
    the load factor that determines how badly Instant-NGP's hash collides.
    """
    specs = config["network_specs"]
    mm = float(config["mm_per_normalized_unit"])
    if is_grid_free(config):
        # A deformation field carries two independent bands: the canonical field
        # sets the finest representable geometry, the warp sets how sharply the
        # deformation may vary. Reporting the warp band as "finest" would be
        # wrong by more than an order of magnitude.
        if is_deformation_field(config):
            finest = float(specs.get("template_min_wavelength_mm", 0.0))
            report = {
                "storage": "none: canonical Fourier field + latent-conditioned warp",
                "fourier_min_wavelength_mm": finest,
                "fourier_max_wavelength_mm": float(
                    specs.get("template_max_wavelength_mm", 0.0)
                ),
                "warp_min_wavelength_mm": float(specs.get("warp_min_wavelength_mm", 0.0)),
                "warp_max_wavelength_mm": float(specs.get("warp_max_wavelength_mm", 0.0)),
                "warp_scale_mm": float(specs.get("warp_scale_mm", 0.0)),
            }
        else:
            finest = float(specs.get("fourier_min_wavelength_mm", 0.0))
            report = {
                "storage": "none: grid-free band-limited Fourier encoding",
                "fourier_min_wavelength_mm": finest,
                "fourier_max_wavelength_mm": float(
                    specs.get("fourier_max_wavelength_mm", 0.0)
                ),
            }
        return {
            "levels": [],
            "grid_parameters": 0,
            "grid_fp32_mib": 0.0,
            **report,
            "finest_geometric_mean_cell_mm": finest / 2.0,
            "marching_cubes_voxel_mm": {str(v): 2.0 / (v - 1) * mm for v in (256, 512)},
        }
    ladder = [int(value) for value in specs["grid_resolutions"]]
    aabb = np.asarray(specs["grid_aabb"], dtype=np.float64)
    extent = aabb[1] - aabb[0]
    mm_per_unit = mm
    features = int(specs["features_per_level"])
    table_limit = 1 << int(specs["log2_hashmap_size"])
    area_mm2 = float(config.get("reference_surface_area_mm2", 100000.0))

    levels = []
    total = 0
    for resolution in ladder:
        vertices = resolution**3
        dense = vertices <= table_limit
        entries = vertices if dense else table_limit
        total += entries * features
        cell_mm = (extent / max(1, resolution - 1)) * mm_per_unit
        mean_cell_mm = float(np.exp(np.log(cell_mm).mean()))
        surface_cells = area_mm2 / (mean_cell_mm**2)
        levels.append(
            {
                "resolution": resolution,
                "storage": "dense" if dense else "hash",
                "entries": int(entries),
                "cell_mm": [float(value) for value in cell_mm],
                "geometric_mean_cell_mm": mean_cell_mm,
                "estimated_surface_cells": float(surface_cells),
                "collision_load": float(surface_cells / entries),
            }
        )
    return {
        "levels": levels,
        "features_per_level": features,
        "log2_hashmap_size": int(specs["log2_hashmap_size"]),
        "grid_parameters": int(total),
        "grid_fp32_mib": total * 4 / 2**20,
        "finest_geometric_mean_cell_mm": levels[-1]["geometric_mean_cell_mm"],
        "reference_surface_area_mm2": area_mm2,
        "marching_cubes_voxel_mm": {
            str(value): 2.0 / (value - 1) * mm_per_unit for value in (256, 512)
        },
    }
