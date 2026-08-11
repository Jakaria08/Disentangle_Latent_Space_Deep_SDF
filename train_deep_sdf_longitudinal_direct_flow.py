#!/usr/bin/env python3
"""Train the modular ADNI scan-to-scan direct continuous-age flow.

This is intentionally separate from ``train_deep_sdf_longitudinal.py``.  It
uses a frozen Task-2 DeepSDF decoder, one frozen Task-2 latent per real scan,
all chronological forward scan pairs, and the baseline enabled objectives:

* real source latent -> real target SDF prediction;
* observed-intermediate direct-vs-composed latent consistency;
* random virtual-intermediate direct-vs-composed latent consistency.

Backward virtual agreement and future extrapolation consistency can be enabled
with explicit spec switches while preserving older experiment specs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from adni_no_mci_direct_flow_data import (
    DirectFlowDataContract,
    DirectFlowPairDataset,
    DirectFlowSequenceDataset,
    build_contract,
    direct_flow_sequence_collate,
)
import deep_sdf.lr_scheduling as lr_scheduling
import deep_sdf.workspace as ws
from longitudinal_direct_flow import (
    DirectAgeFlow,
    LatentODEFlow,
    LocalDiseaseDecomposedFlow,
    MinimalDirectFlowLoss,
    MinimalLossConfig,
    per_row_latent_mse,
)


LOGGER = logging.getLogger("DirectFlowTrain")


def resolve_path(path_value: str | Path, experiment_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (experiment_dir / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(experiment_dir / "train.log", mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def load_specs(experiment_dir: Path) -> Dict[str, object]:
    specs = ws.load_experiment_specifications(str(experiment_dir))
    required = [
        "DataSource",
        "TrainSplit",
        "LongitudinalMetadataFile",
        "FrozenLatentFiles",
        "PretrainedDecoderCheckpoint",
        "NetworkArch",
        "NetworkSpecs",
        "CodeLength",
        "FlowHiddenDims",
    ]
    missing = [key for key in required if key not in specs]
    if missing:
        raise KeyError(f"Missing required specification keys: {missing}")
    return specs


def validate_loss_scope(specs: Mapping[str, object]) -> None:
    enabled = {
        "UseRealTargetSDFLoss": bool(specs.get("UseRealTargetSDFLoss", True)),
        "UseChangeWeightedSDFLoss": bool(
            specs.get("UseChangeWeightedSDFLoss", False)
        ),
        "UseDeltaSDFDirectionLoss": bool(
            specs.get("UseDeltaSDFDirectionLoss", False)
        ),
        "UseDeltaSDFRMAELoss": bool(
            specs.get("UseDeltaSDFRMAELoss", False)
        ),
        "UseNoChangeMarginLoss": bool(
            specs.get("UseNoChangeMarginLoss", False)
        ),
        "UseObservedCocycleLoss": bool(
            specs.get("UseObservedCocycleLoss", True)
        ),
        "UseVirtualCocycleLoss": bool(
            specs.get("UseVirtualCocycleLoss", True)
        ),
        "UseBackwardVirtualLatentLoss": bool(
            specs.get("UseBackwardVirtualLatentLoss", False)
        ),
        "UseBackwardVirtualShapeLoss": bool(
            specs.get("UseBackwardVirtualShapeLoss", False)
        ),
        "UseFutureExtrapolationCocycleLoss": bool(
            specs.get("UseFutureExtrapolationCocycleLoss", False)
        ),
        "UseFutureLatentCocycleLoss": bool(
            specs.get("UseFutureLatentCocycleLoss", False)
        ),
        "UseFutureExtrapolationShapeLoss": bool(
            specs.get("UseFutureExtrapolationShapeLoss", False)
        ),
        "UseSequenceRolloutSDFLoss": bool(
            specs.get("UseSequenceRolloutSDFLoss", False)
        ),
        "UseSequenceCocycleLoss": bool(
            specs.get("UseSequenceCocycleLoss", False)
        ),
        "UseLatentDirectionLoss": bool(
            specs.get("UseLatentDirectionLoss", False)
        ),
        "UseLatentDisplacementMagnitudeLoss": bool(
            specs.get("UseLatentDisplacementMagnitudeLoss", False)
        ),
        "UseSequenceDisplacementMagnitudeLoss": bool(
            specs.get("UseSequenceDisplacementMagnitudeLoss", False)
        ),
        "UseLatentManifoldGuardLoss": bool(
            specs.get("UseLatentManifoldGuardLoss", False)
        ),
        "UseSpeedGuardLoss": bool(
            specs.get("UseSpeedGuardLoss", False)
        ),
        "UseRelativeVolumeLoss": bool(
            specs.get("UseRelativeVolumeLoss", False)
        ),
        "UseDiseaseVolumeOrderingLoss": bool(
            specs.get("UseDiseaseVolumeOrderingLoss", False)
        ),
    }
    if not any(enabled.values()):
        raise ValueError(
            "At least one direct-flow objective must be enabled; "
            f"got {enabled}"
        )
    forbidden_true = [
        "UseRealTargetLatentLoss",
        "UseTargetLatentLoss",
        "UseNoChangeRankingLoss",
        "UseCocycleShapeLoss",
        "UseBackwardLoss",
        "UseClosureLoss",
        "UseGeneratorConsistencyLoss",
        "UseVelocityMagnitudeLoss",
        "UseVelocitySmoothnessLoss",
        "UseZeroDisplacementLoss",
        "UseEikonal",
        "CodeRegularization",
        "UseDisentanglement",
        "UseExtrapolationLoss",
    ]
    active_forbidden = [
        key for key in forbidden_true if bool(specs.get(key, False))
    ]
    if active_forbidden:
        raise ValueError(
            "Initial loss scope forbids these enabled options: "
            f"{active_forbidden}"
        )


def build_data_contract(
    specs: Mapping[str, object],
    experiment_dir: Path,
) -> DirectFlowDataContract:
    latent_files_raw = specs["FrozenLatentFiles"]
    if not isinstance(latent_files_raw, dict):
        raise TypeError("FrozenLatentFiles must be an object keyed by split")
    latent_files = {
        split: resolve_path(path_value, experiment_dir)
        for split, path_value in latent_files_raw.items()
    }
    return build_contract(
        resolve_path(specs["LongitudinalMetadataFile"], experiment_dir),
        latent_files,
        int(specs["CodeLength"]),
        pair_source_mode=str(specs.get("PairSourceMode", "all_forward_starts")),
        sequence_min_length=int(specs.get("SequenceMinLength", 2)),
        sequence_start_mode=str(
            specs.get("SequenceStartMode", "all_forward_starts")
        ),
        allow_extra_latents=bool(specs.get("AllowExtraLatents", False)),
    )


def load_frozen_decoder(
    specs: Mapping[str, object],
    experiment_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, int]:
    architecture = __import__(
        "networks." + str(specs["NetworkArch"]),
        fromlist=["Decoder"],
    )
    decoder = architecture.Decoder(
        int(specs["CodeLength"]),
        **specs["NetworkSpecs"],
    ).to(device)
    checkpoint_path = resolve_path(
        specs["PretrainedDecoderCheckpoint"],
        experiment_dir,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in payload:
        raise KeyError(
            f"Pretrained decoder checkpoint lacks model_state_dict: {checkpoint_path}"
        )
    state_dict = payload["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    decoder.load_state_dict(state_dict)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder, int(payload.get("epoch", -1))


def build_pca_parameters(
    contract: DirectFlowDataContract,
    *,
    split: str,
    latent_size: int,
    component_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if split not in contract.latent_maps:
        raise KeyError(f"Cannot fit PCA from missing split {split!r}")
    latents = torch.stack(
        [
            latent.float()
            for _, latent in sorted(contract.latent_maps[split].items())
        ],
        dim=0,
    )
    if latents.ndim != 2 or latents.shape[1] != int(latent_size):
        raise ValueError(
            f"Expected PCA latents [N,{latent_size}], got {tuple(latents.shape)}"
        )
    if not (0 < int(component_count) <= int(latent_size)):
        raise ValueError(
            f"PCA component count must be in [1,{latent_size}], got {component_count}"
        )
    mean = latents.mean(dim=0, keepdim=True)
    centered = latents - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[: int(component_count)].contiguous()
    return mean.contiguous(), components


def build_flow(
    specs: Mapping[str, object],
    device: torch.device,
    contract: Optional[DirectFlowDataContract] = None,
) -> torch.nn.Module:
    latent_condition_mode = str(specs.get("LatentConditionMode", "full")).lower()
    latent_condition_dim = specs.get("LatentConditionDim")
    pca_mean = None
    pca_components = None
    if latent_condition_mode == "pca":
        if contract is None:
            raise ValueError("PCA latent conditioning requires a data contract")
        if latent_condition_dim is None:
            raise KeyError("LatentConditionDim is required for PCA mode")
        pca_mean, pca_components = build_pca_parameters(
            contract,
            split=str(specs.get("PCAFitSplit", "train")),
            latent_size=int(specs["CodeLength"]),
            component_count=int(latent_condition_dim),
        )
    flow_type = str(specs.get("FlowType", "direct_age")).strip().lower()
    if flow_type in {"direct_age", "direct", "average_velocity"}:
        return DirectAgeFlow(
            latent_size=int(specs["CodeLength"]),
            hidden_dims=[int(value) for value in specs["FlowHiddenDims"]],
            condition_dim=int(specs.get("ConditionDim", 1)),
            activation=str(specs.get("FlowActivation", "relu")),
            dropout=float(specs.get("FlowDropout", 0.0)),
            zero_initialize_output=bool(
                specs.get("FlowZeroInitializeOutput", True)
            ),
            latent_condition_mode=latent_condition_mode,
            latent_condition_dim=(
                None if latent_condition_dim is None else int(latent_condition_dim)
            ),
            include_delta_time_input=bool(
                specs.get("FlowIncludeDeltaTimeInput", False)
            ),
            pca_mean=pca_mean,
            pca_components=pca_components,
        ).to(device)
    if flow_type in {
        "local_disease_decomposed",
        "local_decomposed",
        "composed_local_disease",
    }:
        age_spec = specs.get("AgeNormalization", {})
        age_range_years = float(age_spec.get("age_range_years", 1.0))
        local_step_years = float(specs.get("LocalStepYears", 0.5))
        max_step_norm = float(
            specs.get(
                "LocalMaxStepNorm",
                local_step_years / max(age_range_years, 1.0e-8),
            )
        )
        return LocalDiseaseDecomposedFlow(
            latent_size=int(specs["CodeLength"]),
            hidden_dims=[int(value) for value in specs["FlowHiddenDims"]],
            rank=int(specs.get("FlowRank", 32)),
            condition_dim=int(specs.get("ConditionDim", 1)),
            activation=str(specs.get("FlowActivation", "silu")),
            dropout=float(specs.get("FlowDropout", 0.0)),
            zero_initialize_output=bool(
                specs.get("FlowZeroInitializeOutput", True)
            ),
            latent_condition_mode=latent_condition_mode,
            latent_condition_dim=(
                None if latent_condition_dim is None else int(latent_condition_dim)
            ),
            include_time_input=bool(specs.get("FlowIncludeTimeInput", True)),
            max_step_norm=max_step_norm,
            basis_init_scale=float(specs.get("FlowBasisInitScale", 0.02)),
            pca_mean=pca_mean,
            pca_components=pca_components,
        ).to(device)
    if flow_type in {"latent_ode", "ode", "neural_ode", "siren_latent_ode"}:
        hidden_dims = specs.get("ODEHiddenDims", specs["FlowHiddenDims"])
        max_step_norm = specs.get("ODEMaxStepNorm")
        return LatentODEFlow(
            latent_size=int(specs["CodeLength"]),
            hidden_dims=[int(value) for value in hidden_dims],
            condition_dim=int(specs.get("ConditionDim", 1)),
            activation=str(specs.get("ODEActivation", specs.get("FlowActivation", "silu"))),
            dropout=float(specs.get("ODEDropout", specs.get("FlowDropout", 0.0))),
            zero_initialize_output=bool(
                specs.get("ODEZeroInitializeOutput", specs.get("FlowZeroInitializeOutput", True))
            ),
            latent_condition_mode=latent_condition_mode,
            latent_condition_dim=(
                None if latent_condition_dim is None else int(latent_condition_dim)
            ),
            integration_substeps=int(specs.get("ODEIntegrationSubsteps", 4)),
            max_step_norm=None if max_step_norm is None else float(max_step_norm),
            pca_mean=pca_mean,
            pca_components=pca_components,
        ).to(device)
    raise ValueError(f"Unsupported FlowType {specs.get('FlowType')!r}")


def build_loss(
    specs: Mapping[str, object],
    decoder: torch.nn.Module,
    flow: torch.nn.Module,
) -> MinimalDirectFlowLoss:
    real_weight = float(specs.get("RealTargetSDFLossLambda", 1.0))
    change_weighted_sdf_weight = float(
        specs.get("ChangeWeightedSDFLossLambda", 0.0)
    )
    delta_sdf_direction_weight = float(
        specs.get("DeltaSDFDirectionLossLambda", 0.0)
    )
    delta_sdf_rmae_weight = float(specs.get("DeltaSDFRMAELossLambda", 0.0))
    no_change_margin_weight = float(
        specs.get("NoChangeMarginLossLambda", 0.0)
    )
    observed_weight = float(specs.get("ObservedCocycleLossLambda", 0.01))
    virtual_weight = float(specs.get("VirtualCocycleLossLambda", 0.01))
    if not bool(specs.get("UseRealTargetSDFLoss", True)):
        real_weight = 0.0
    if not bool(specs.get("UseChangeWeightedSDFLoss", False)):
        change_weighted_sdf_weight = 0.0
    if not bool(specs.get("UseDeltaSDFDirectionLoss", False)):
        delta_sdf_direction_weight = 0.0
    if not bool(specs.get("UseDeltaSDFRMAELoss", False)):
        delta_sdf_rmae_weight = 0.0
    if not bool(specs.get("UseNoChangeMarginLoss", False)):
        no_change_margin_weight = 0.0
    if not bool(specs.get("UseObservedCocycleLoss", True)):
        observed_weight = 0.0
    if not bool(specs.get("UseVirtualCocycleLoss", True)):
        virtual_weight = 0.0
    backward_latent_weight = float(
        specs.get("BackwardVirtualLatentLossLambda", 0.0)
    )
    backward_shape_weight = float(
        specs.get("BackwardVirtualShapeLossLambda", 0.0)
    )
    future_latent_weight = float(
        specs.get(
            "FutureExtrapolationLatentLossLambda",
            specs.get("FutureLatentCocycleLossLambda", 0.0),
        )
    )
    if bool(specs.get("UseFutureLatentCocycleLoss", False)):
        future_latent_weight = float(
            specs.get("FutureLatentCocycleLossLambda", future_latent_weight)
        )
    future_shape_weight = float(
        specs.get("FutureExtrapolationShapeLossLambda", 0.0)
    )
    if not bool(specs.get("UseBackwardVirtualLatentLoss", False)):
        backward_latent_weight = 0.0
    if not bool(specs.get("UseBackwardVirtualShapeLoss", False)):
        backward_shape_weight = 0.0
    if not (
        bool(specs.get("UseFutureExtrapolationCocycleLoss", False))
        or bool(specs.get("UseFutureLatentCocycleLoss", False))
    ):
        future_latent_weight = 0.0
    if not bool(specs.get("UseFutureExtrapolationShapeLoss", False)):
        future_shape_weight = 0.0
    config = MinimalLossConfig(
        real_prediction_weight=real_weight,
        change_weighted_sdf_weight=change_weighted_sdf_weight,
        change_weighted_sdf_alpha=float(specs.get("ChangeWeightedSDFAlpha", 2.0)),
        change_weighted_sdf_eps=float(specs.get("ChangeWeightedSDFEps", 1.0e-5)),
        delta_sdf_direction_weight=delta_sdf_direction_weight,
        delta_sdf_min_norm=float(specs.get("DeltaSDFMinNorm", 1.0e-6)),
        delta_sdf_rmae_weight=delta_sdf_rmae_weight,
        delta_sdf_rmae_eps=float(specs.get("DeltaSDFRMAEEps", 1.0e-4)),
        delta_sdf_rmae_cap=float(specs.get("DeltaSDFRMAECap", 5.0)),
        no_change_margin_weight=no_change_margin_weight,
        no_change_margin=float(specs.get("NoChangeMargin", 1.0e-4)),
        observed_consistency_weight=observed_weight,
        virtual_consistency_weight=virtual_weight,
        virtual_ratio_min=float(specs.get("VirtualTimeRatioMin", 0.1)),
        virtual_ratio_max=float(specs.get("VirtualTimeRatioMax", 0.9)),
        backward_virtual_latent_weight=backward_latent_weight,
        backward_virtual_shape_weight=backward_shape_weight,
        backward_shape_samples=int(specs.get("BackwardShapeSamples", 1024)),
        future_extrapolation_latent_weight=future_latent_weight,
        future_extrapolation_shape_weight=future_shape_weight,
        future_extrapolation_ratio_min=float(
            specs.get(
                "FutureExtrapolationRatioMin",
                specs.get("FutureAlphaMin", 0.25),
            )
        ),
        future_extrapolation_ratio_max=float(
            specs.get(
                "FutureExtrapolationRatioMax",
                specs.get("FutureAlphaMax", 1.0),
            )
        ),
        future_shape_samples=int(specs.get("FutureShapeSamples", 1024)),
    )
    return MinimalDirectFlowLoss(
        decoder=decoder,
        flow=flow,
        clamp_distance=float(specs.get("ClampingDistance", 0.1)),
        config=config,
    )


def make_loader(
    contract: DirectFlowDataContract,
    split: str,
    specs: Mapping[str, object],
    *,
    training: bool,
) -> DataLoader:
    records = select_pair_records(contract, split, specs)
    dataset = DirectFlowPairDataset(
        records,
        contract.latent_maps[split],
        int(specs.get("SamplesPerTarget", 4096)),
        deterministic=not training,
        deterministic_seed=int(specs.get("ValidationSampleSeed", 991)),
    )
    worker_count = int(specs.get("DataLoaderThreads", 4))
    generator = torch.Generator()
    generator.manual_seed(
        int(specs.get("Seed", 42)) + (0 if training else 10000)
    )
    sampler = None
    shuffle = bool(training)
    if training and bool(specs.get("UseSubjectDiagnosisBalancedSampling", False)):
        subject_pair_counts: Dict[str, int] = {}
        diagnosis_subjects: Dict[str, set[str]] = {}
        for record in records:
            subject_pair_counts[record.subject_id] = (
                subject_pair_counts.get(record.subject_id, 0) + 1
            )
            diagnosis_subjects.setdefault(record.diagnosis, set()).add(record.subject_id)
        weights = []
        for record in records:
            diagnosis_count = max(len(diagnosis_subjects.get(record.diagnosis, ())), 1)
            pair_count = max(subject_pair_counts.get(record.subject_id, 1), 1)
            weights.append(1.0 / (diagnosis_count * pair_count))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=int(specs.get("PairsPerBatch", 16)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=worker_count,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(
            specs.get("PersistentWorkers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            specs.get("PrefetchFactor", 2)
        )
    return DataLoader(**loader_kwargs)


def _split_float_value(
    specs: Mapping[str, object],
    key: str,
    split: str,
) -> Optional[float]:
    split_values = specs.get(f"{key}BySplit")
    if isinstance(split_values, Mapping) and split in split_values:
        value = split_values[split]
    else:
        value = specs.get(key)
    if value is None:
        return None
    return float(value)


def _pair_gap_years(record, specs: Mapping[str, object]) -> float:
    age_spec = specs.get("AgeNormalization", {})
    age_range = float(age_spec.get("age_range_years", 1.0))
    return float(abs(record.age_gap_norm) * age_range)


def select_pair_records(
    contract: DirectFlowDataContract,
    split: str,
    specs: Mapping[str, object],
):
    """Apply optional protocol-level pair filters before creating a loader."""

    records = list(contract.pair_records[split])
    min_gap_years = _split_float_value(specs, "PairMinGapYears", split)
    max_gap_years = _split_float_value(specs, "PairMaxGapYears", split)
    if min_gap_years is None and max_gap_years is None:
        return records
    filtered = []
    for record in records:
        gap_years = _pair_gap_years(record, specs)
        if min_gap_years is not None and gap_years < min_gap_years:
            continue
        if max_gap_years is not None and gap_years > max_gap_years:
            continue
        filtered.append(record)
    if not filtered:
        raise RuntimeError(
            f"Pair gap filters removed every {split!r} pair: "
            f"PairMinGapYears={min_gap_years}, PairMaxGapYears={max_gap_years}"
        )
    return filtered


def sequence_losses_enabled(specs: Mapping[str, object]) -> bool:
    return bool(
        specs.get("UseSequenceRolloutSDFLoss", False)
        or specs.get("UseSequenceCocycleLoss", False)
        or specs.get("UseSequenceDisplacementMagnitudeLoss", False)
    )


def auxiliary_latent_losses_enabled(specs: Mapping[str, object]) -> bool:
    return bool(
        specs.get("UseLatentDirectionLoss", False)
        or specs.get("UseLatentDisplacementMagnitudeLoss", False)
        or specs.get("UseSequenceDisplacementMagnitudeLoss", False)
        or specs.get("UseLatentManifoldGuardLoss", False)
        or specs.get("UseSpeedGuardLoss", False)
    )


def make_sequence_loader(
    contract: DirectFlowDataContract,
    split: str,
    specs: Mapping[str, object],
    *,
    training: bool,
) -> DataLoader:
    dataset = DirectFlowSequenceDataset(
        contract.sequence_records[split],
        contract.latent_maps[split],
        int(specs.get("SequenceSamplesPerScan", 1024)),
        deterministic=not training,
        deterministic_seed=int(specs.get("ValidationSampleSeed", 991)) + 20000,
    )
    worker_count = int(specs.get("DataLoaderThreads", 4))
    generator = torch.Generator()
    generator.manual_seed(
        int(specs.get("Seed", 42)) + (2000 if training else 12000)
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=int(specs.get("SequenceSubjectsPerBatch", 4)),
        shuffle=training,
        num_workers=worker_count,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=direct_flow_sequence_collate,
    )
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(
            specs.get("PersistentWorkers", True)
        )
        loader_kwargs["prefetch_factor"] = int(
            specs.get("PrefetchFactor", 2)
        )
    return DataLoader(**loader_kwargs)


@dataclass(frozen=True)
class LatentRegularizerStats:
    mean: torch.Tensor
    std: torch.Tensor
    speed_percentile: float
    median_gap_norm: float


def build_latent_regularizer_stats(
    contract: DirectFlowDataContract,
    *,
    split: str = "train",
    speed_percentile: float = 95.0,
) -> LatentRegularizerStats:
    latents = torch.stack(
        [
            latent.float()
            for _, latent in sorted(contract.latent_maps[split].items())
        ],
        dim=0,
    )
    std = latents.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    speeds = []
    gaps = []
    for record in contract.pair_records[split]:
        source = contract.latent_maps[split][record.source_scan_id].float()
        target = contract.latent_maps[split][record.target_scan_id].float()
        gap = abs(float(record.target_time) - float(record.source_time))
        if gap <= 0.0:
            continue
        gaps.append(gap)
        speeds.append(float(torch.linalg.vector_norm(target - source).item() / gap))
    if not speeds:
        raise RuntimeError(f"No positive-gap pairs found for split {split!r}")
    return LatentRegularizerStats(
        mean=latents.mean(dim=0, keepdim=True).contiguous(),
        std=std.view(1, -1).contiguous(),
        speed_percentile=float(np.percentile(np.asarray(speeds), speed_percentile)),
        median_gap_norm=float(np.median(np.asarray(gaps))),
    )


def _column_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(1)
    return value


def _gap_weights(
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    specs: Mapping[str, object],
    stats: LatentRegularizerStats,
) -> torch.Tensor:
    source_time = _column_tensor(source_time)
    target_time = _column_tensor(target_time)
    if not bool(specs.get("UseGapWeightedLoss", False)):
        return torch.ones_like(source_time)
    reference = max(float(stats.median_gap_norm), 1.0e-6)
    weights = torch.abs(target_time - source_time) / reference
    return torch.clamp(
        weights,
        min=float(specs.get("GapWeightMin", 0.5)),
        max=float(specs.get("GapWeightMax", 2.5)),
    )


def _sequence_gap_weights(
    source_time: torch.Tensor,
    previous_time: torch.Tensor,
    target_time: torch.Tensor,
    specs: Mapping[str, object],
    stats: LatentRegularizerStats,
) -> torch.Tensor:
    mode = str(specs.get("SequenceWeightMode", "step_gap")).strip().lower()
    if mode in {"step_gap", "adjacent_gap"}:
        return _gap_weights(previous_time, target_time, specs, stats)
    if mode in {"gap_from_source", "source_gap"}:
        return _gap_weights(source_time, target_time, specs, stats)
    raise ValueError(
        "SequenceWeightMode must be one of step_gap, adjacent_gap, "
        f"gap_from_source, source_gap; got {mode!r}"
    )


def _weighted_mean(
    values: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    values = values.view(-1)
    if weights is None:
        weights = torch.ones_like(values)
    else:
        weights = weights.to(device=values.device, dtype=values.dtype).view(-1)
    if mask is not None:
        mask = mask.to(device=values.device, dtype=torch.bool).view(-1)
        values = values[mask]
        weights = weights[mask]
    if values.numel() == 0:
        return values.sum() * 0.0
    denominator = weights.sum().clamp_min(1.0e-8)
    return torch.sum(values * weights) / denominator


def _displacement_norms(
    predicted_latent: torch.Tensor,
    base_latent: torch.Tensor,
    real_latent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    predicted_norm = torch.linalg.vector_norm(
        predicted_latent - base_latent,
        dim=1,
    )
    real_norm = torch.linalg.vector_norm(real_latent - base_latent, dim=1)
    return predicted_norm, real_norm


def displacement_magnitude_row_loss(
    predicted_latent: torch.Tensor,
    base_latent: torch.Tensor,
    real_latent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_norm, real_norm = _displacement_norms(
        predicted_latent,
        base_latent,
        real_latent,
    )
    row_loss = F.smooth_l1_loss(
        predicted_norm,
        real_norm,
        reduction="none",
    )
    return row_loss, predicted_norm, real_norm


def latent_zscore_summary(
    latents: torch.Tensor,
    stats: LatentRegularizerStats,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latents.ndim == 3:
        latents = latents.reshape(-1, latents.shape[-1])
    if latents.ndim != 2:
        raise ValueError(f"Expected latent tensor [N,D], got {tuple(latents.shape)}")
    mean = stats.mean.to(device=latents.device, dtype=latents.dtype)
    std = stats.std.to(device=latents.device, dtype=latents.dtype)
    z_score = torch.abs((latents - mean) / std)
    return (
        torch.quantile(z_score.detach().reshape(-1), 0.95),
        z_score.detach().max(),
    )


def latent_manifold_guard_loss(
    latents: torch.Tensor,
    stats: LatentRegularizerStats,
    *,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latents.ndim == 3:
        latents = latents.reshape(-1, latents.shape[-1])
    if latents.ndim != 2:
        raise ValueError(f"Expected latent tensor [N,D], got {tuple(latents.shape)}")
    mean = stats.mean.to(device=latents.device, dtype=latents.dtype)
    std = stats.std.to(device=latents.device, dtype=latents.dtype)
    z_score = torch.abs((latents - mean) / std)
    hinge = F.relu(z_score - float(threshold))
    loss = torch.mean(hinge ** 2)
    z_p95, z_max = latent_zscore_summary(latents, stats)
    return (loss, z_p95, z_max)


def speed_guard_loss(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    stats: LatentRegularizerStats,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_time = _column_tensor(source_time).to(
        device=source_latent.device,
        dtype=source_latent.dtype,
    )
    target_time = _column_tensor(target_time).to(
        device=source_latent.device,
        dtype=source_latent.dtype,
    )
    gap = torch.abs(target_time - source_time).clamp_min(1.0e-6)
    speed = torch.linalg.vector_norm(target_latent - source_latent, dim=1) / gap.view(-1)
    threshold = max(float(stats.speed_percentile), 1.0e-6)
    normalized_excess = F.relu(speed - threshold) / threshold
    return torch.mean(normalized_excess ** 2), torch.quantile(speed.detach(), 0.95)


def volume_losses_enabled(specs: Mapping[str, object]) -> bool:
    return bool(
        specs.get("UseRelativeVolumeLoss", False)
        or specs.get("UseDiseaseVolumeOrderingLoss", False)
    )


def make_volume_probe_points(
    specs: Mapping[str, object],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not volume_losses_enabled(specs):
        return None
    count = int(specs.get("VolumeProbePoints", 4096))
    if count <= 0:
        raise ValueError("VolumeProbePoints must be positive")
    raw_bounds = specs.get("VolumeProbeBounds", [-1.05, 1.05])
    bounds = torch.as_tensor(raw_bounds, dtype=torch.float32)
    if bounds.shape == (2,):
        low = torch.full((3,), float(bounds[0]), dtype=torch.float32)
        high = torch.full((3,), float(bounds[1]), dtype=torch.float32)
    elif bounds.shape == (3, 2):
        low = bounds[:, 0].contiguous()
        high = bounds[:, 1].contiguous()
    else:
        raise ValueError(
            "VolumeProbeBounds must be [low, high] or [[x0,x1],[y0,y1],[z0,z1]]"
        )
    if not bool(torch.all(high > low).item()):
        raise ValueError(f"Invalid VolumeProbeBounds: {raw_bounds}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(specs.get("VolumeProbeSeed", 12345)))
    points = torch.rand(count, 3, generator=generator, dtype=torch.float32)
    points = low.view(1, 3) + points * (high - low).view(1, 3)
    return points.to(device=device)


def soft_volume_proxy(
    loss_module: MinimalDirectFlowLoss,
    latent: torch.Tensor,
    probe_points: torch.Tensor,
    *,
    temperature: float,
    inside_sdf_sign: float,
    chunk_size: int,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("VolumeProbeTemperature must be positive")
    if int(chunk_size) <= 0:
        raise ValueError("VolumeProbeChunkSize must be positive")
    occupancies = []
    for start in range(0, int(probe_points.shape[0]), int(chunk_size)):
        xyz = probe_points[start : start + int(chunk_size)].to(
            device=latent.device,
            dtype=latent.dtype,
        )
        xyz = xyz.unsqueeze(0).expand(latent.shape[0], -1, -1)
        sdf = loss_module.decode_sdf_at_xyz(latent, xyz)
        occupancies.append(
            torch.sigmoid(float(inside_sdf_sign) * sdf / float(temperature))
        )
    occupancy = torch.cat(occupancies, dim=1)
    return occupancy.mean(dim=(1, 2)).clamp_min(1.0e-8)


def compute_pair_volume_losses(
    *,
    batch: Mapping[str, torch.Tensor],
    output,
    loss_module: MinimalDirectFlowLoss,
    flow: torch.nn.Module,
    specs: Mapping[str, object],
    stats: LatentRegularizerStats,
    volume_probe_points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = output.total * 0.0
    metrics: Dict[str, torch.Tensor] = {
        "relative_volume": zero,
        "disease_volume_ordering": zero,
        "predicted_log_volume_ratio": zero.detach(),
        "real_log_volume_ratio": zero.detach(),
        "cn_condition_log_volume_ratio": zero.detach(),
        "ad_condition_log_volume_ratio": zero.detach(),
    }
    if not volume_losses_enabled(specs):
        return zero, metrics
    if volume_probe_points is None:
        raise ValueError("Volume losses are enabled but volume_probe_points is None")

    temperature = float(specs.get("VolumeProbeTemperature", 0.01))
    inside_sdf_sign = float(specs.get("VolumeProbeInsideSDFSign", -1.0))
    chunk_size = int(specs.get("VolumeProbeChunkSize", 4096))
    source_latent = batch["source_latent"]
    source_time = _column_tensor(batch["source_time"])
    target_time = _column_tensor(batch["target_time"])
    condition = batch["condition"]

    with torch.no_grad():
        source_proxy = soft_volume_proxy(
            loss_module,
            source_latent,
            volume_probe_points,
            temperature=temperature,
            inside_sdf_sign=inside_sdf_sign,
            chunk_size=chunk_size,
        )
    target_proxy = soft_volume_proxy(
        loss_module,
        output.direct_target_latent,
        volume_probe_points,
        temperature=temperature,
        inside_sdf_sign=inside_sdf_sign,
        chunk_size=chunk_size,
    )
    predicted_log_ratio = torch.log(target_proxy / source_proxy.detach())
    metrics["predicted_log_volume_ratio"] = predicted_log_ratio.detach().mean()

    total = zero
    if bool(specs.get("UseRelativeVolumeLoss", False)):
        if "source_mesh_volume_mm3" not in batch or "target_mesh_volume_mm3" not in batch:
            raise KeyError(
                "UseRelativeVolumeLoss requires source_mesh_volume_mm3 and "
                "target_mesh_volume_mm3 in the pair batch."
            )
        source_real = batch["source_mesh_volume_mm3"].to(
            device=source_latent.device,
            dtype=source_latent.dtype,
        ).view(-1)
        target_real = batch["target_mesh_volume_mm3"].to(
            device=source_latent.device,
            dtype=source_latent.dtype,
        ).view(-1)
        valid = torch.isfinite(source_real) & torch.isfinite(target_real)
        valid = valid & (source_real > 0.0) & (target_real > 0.0)
        real_log_ratio = torch.zeros_like(predicted_log_ratio)
        real_log_ratio[valid] = torch.log(target_real[valid] / source_real[valid])
        metrics["real_log_volume_ratio"] = (
            real_log_ratio[valid].detach().mean()
            if bool(valid.any().item())
            else zero.detach()
        )
        row_loss = F.smooth_l1_loss(
            predicted_log_ratio,
            real_log_ratio,
            reduction="none",
            beta=float(specs.get("RelativeVolumeSmoothL1Beta", 0.02)),
        )
        weights = _gap_weights(source_time, target_time, specs, stats)
        relative_volume = _weighted_mean(row_loss, weights, valid)
        metrics["relative_volume"] = relative_volume
        total = total + float(specs.get("RelativeVolumeLossLambda", 0.0)) * relative_volume

    if bool(specs.get("UseDiseaseVolumeOrderingLoss", False)):
        cn_condition = torch.zeros_like(condition)
        ad_condition = torch.ones_like(condition)
        cn_latent = flow.transport(source_latent, source_time, target_time, cn_condition)
        ad_latent = flow.transport(source_latent, source_time, target_time, ad_condition)
        cn_proxy = soft_volume_proxy(
            loss_module,
            cn_latent,
            volume_probe_points,
            temperature=temperature,
            inside_sdf_sign=inside_sdf_sign,
            chunk_size=chunk_size,
        )
        ad_proxy = soft_volume_proxy(
            loss_module,
            ad_latent,
            volume_probe_points,
            temperature=temperature,
            inside_sdf_sign=inside_sdf_sign,
            chunk_size=chunk_size,
        )
        cn_log_ratio = torch.log(cn_proxy / source_proxy.detach())
        ad_log_ratio = torch.log(ad_proxy / source_proxy.detach())
        age_spec = specs.get("AgeNormalization", {})
        age_range_years = float(age_spec.get("age_range_years", 1.0))
        gap_years = torch.abs(target_time - source_time).view(-1) * age_range_years
        margin = float(specs.get("DiseaseVolumeMarginPerYear", 0.005)) * gap_years
        row_loss = F.relu(ad_log_ratio - cn_log_ratio + margin) ** 2
        weights = _gap_weights(source_time, target_time, specs, stats)
        disease_volume_ordering = _weighted_mean(row_loss, weights)
        metrics["disease_volume_ordering"] = disease_volume_ordering
        metrics["cn_condition_log_volume_ratio"] = cn_log_ratio.detach().mean()
        metrics["ad_condition_log_volume_ratio"] = ad_log_ratio.detach().mean()
        total = total + float(
            specs.get("DiseaseVolumeOrderingLossLambda", 0.0)
        ) * disease_volume_ordering

    return total, metrics


def compute_pair_auxiliary_losses(
    *,
    batch: Mapping[str, torch.Tensor],
    output,
    flow: torch.nn.Module,
    specs: Mapping[str, object],
    stats: LatentRegularizerStats,
    loss_module: Optional[MinimalDirectFlowLoss] = None,
    volume_probe_points: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = output.total * 0.0
    metrics: Dict[str, torch.Tensor] = {
        "latent_direction": zero,
        "latent_displacement_magnitude": zero,
        "latent_manifold_guard": zero,
        "speed_guard": zero,
        "latent_zscore_p95": zero.detach(),
        "latent_zscore_max": zero.detach(),
        "predicted_speed_p95": zero.detach(),
    }
    total = zero
    source_latent = batch["source_latent"]
    target_latent = batch["target_latent"]
    source_time = _column_tensor(batch["source_time"])
    target_time = _column_tensor(batch["target_time"])
    condition = batch["condition"]
    direct_target = output.direct_target_latent

    if bool(specs.get("UseLatentDirectionLoss", False)):
        real_delta = target_latent - source_latent
        predicted_delta = direct_target - source_latent
        row_loss = 1.0 - F.cosine_similarity(
            predicted_delta,
            real_delta,
            dim=1,
            eps=1.0e-8,
        )
        weights = _gap_weights(source_time, target_time, specs, stats)
        direction_loss = _weighted_mean(row_loss, weights)
        metrics["latent_direction"] = direction_loss
        total = total + float(
            specs.get("LatentDirectionLossLambda", 0.0)
        ) * direction_loss

    if bool(specs.get("UseLatentDisplacementMagnitudeLoss", False)):
        row_loss, _, _ = displacement_magnitude_row_loss(
            direct_target,
            source_latent,
            target_latent,
        )
        weights = _gap_weights(source_time, target_time, specs, stats)
        displacement_loss = _weighted_mean(row_loss, weights)
        metrics["latent_displacement_magnitude"] = displacement_loss
        total = total + float(
            specs.get("LatentDisplacementMagnitudeLossLambda", 0.0)
        ) * displacement_loss

    diagnostic_latents = [
        direct_target,
        output.virtual_intermediate_latent,
        output.virtual_composed_target_latent,
    ]
    guard_latents = list(diagnostic_latents)
    future_ratio = getattr(output, "future_ratio", None)
    if future_ratio is not None and (
        bool(specs.get("UseFutureExtrapolationCocycleLoss", False))
        or bool(specs.get("UseFutureLatentCocycleLoss", False))
    ):
        future_ratio = _column_tensor(future_ratio).to(
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
        future_time = target_time + future_ratio * (target_time - source_time)
        future_direct = flow.transport(
            source_latent,
            source_time,
            future_time,
            condition,
        )
        future_composed = flow.transport(
            direct_target,
            target_time,
            future_time,
            condition,
        )
        diagnostic_latents.extend([future_direct, future_composed])
        if (
            str(specs.get("ApplyLatentGuardTo", "all")).strip().lower()
            == "future_only"
        ):
            guard_latents = [future_direct, future_composed]
        else:
            guard_latents.extend([future_direct, future_composed])
    elif str(specs.get("ApplyLatentGuardTo", "all")).strip().lower() == "future_only":
        guard_latents = []

    if diagnostic_latents:
        z_p95, z_max = latent_zscore_summary(
            torch.cat(diagnostic_latents, dim=0),
            stats,
        )
        metrics["latent_zscore_p95"] = z_p95
        metrics["latent_zscore_max"] = z_max

    if bool(specs.get("UseLatentManifoldGuardLoss", False)) and guard_latents:
        guard_loss, _, _ = latent_manifold_guard_loss(
            torch.cat(guard_latents, dim=0),
            stats,
            threshold=float(specs.get("LatentGuardThreshold", 3.0)),
        )
        metrics["latent_manifold_guard"] = guard_loss
        total = total + float(
            specs.get("LatentManifoldGuardLossLambda", 0.0)
        ) * guard_loss

    if bool(specs.get("UseSpeedGuardLoss", False)):
        speed_loss, speed_p95 = speed_guard_loss(
            source_latent,
            direct_target,
            source_time,
            target_time,
            stats,
        )
        metrics["speed_guard"] = speed_loss
        metrics["predicted_speed_p95"] = speed_p95
        total = total + float(specs.get("SpeedGuardLossLambda", 0.0)) * speed_loss

    if volume_losses_enabled(specs):
        if loss_module is None:
            raise ValueError("Volume losses require loss_module")
        volume_total, volume_metrics = compute_pair_volume_losses(
            batch=batch,
            output=output,
            loss_module=loss_module,
            flow=flow,
            specs=specs,
            stats=stats,
            volume_probe_points=volume_probe_points,
        )
        total = total + volume_total
        metrics.update(volume_metrics)

    return total, metrics


def to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tensor_keys = (
        "source_latent",
        "target_latent",
        "source_time",
        "target_time",
        "condition",
        "target_samples",
        "observed_intermediate_time",
        "observed_intermediate_mask",
    )
    result = {
        key: batch[key].to(device=device, non_blocking=True)
        for key in tensor_keys
    }
    optional_tensor_keys = (
        "source_mesh_volume_mm3",
        "target_mesh_volume_mm3",
    )
    for key in optional_tensor_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, non_blocking=True)
    return result


def sequence_to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    tensor_keys = (
        "visit_orders",
        "age_years",
        "times",
        "latents",
        "condition",
        "target_samples",
        "scan_mask",
        "step_mask",
    )
    result: Dict[str, object] = {
        key: batch[key].to(device=device, non_blocking=True)  # type: ignore[union-attr]
        for key in tensor_keys
    }
    result["subject_id"] = batch["subject_id"]
    result["diagnosis"] = batch["diagnosis"]
    result["scan_ids"] = batch["scan_ids"]
    return result


def no_change_sdf_loss(
    loss_module: MinimalDirectFlowLoss,
    source_latent: torch.Tensor,
    target_samples: torch.Tensor,
) -> torch.Tensor:
    return loss_module.decode_target_sdf_loss(
        source_latent,
        target_samples,
    )


def run_loader(
    *,
    loader: DataLoader,
    loss_module: MinimalDirectFlowLoss,
    flow: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    gradient_clip_norm: Optional[float],
    specs: Mapping[str, object],
    regularizer_stats: LatentRegularizerStats,
    volume_probe_points: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    flow.train(training)
    loss_module.decoder.eval()
    sums = {"no_change": 0.0}
    row_count = 0
    observed_row_count = 0
    start_time = time.time()

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            output = loss_module(
                source_latent=batch["source_latent"],
                target_latent=batch["target_latent"],
                source_time=batch["source_time"],
                target_time=batch["target_time"],
                condition=batch["condition"],
                target_samples=batch["target_samples"],
                observed_intermediate_time=batch[
                    "observed_intermediate_time"
                ],
                observed_intermediate_mask=batch[
                    "observed_intermediate_mask"
                ],
            )
            auxiliary_total, auxiliary_metrics = compute_pair_auxiliary_losses(
                batch=batch,
                output=output,
                flow=flow,
                specs=specs,
                stats=regularizer_stats,
                loss_module=loss_module,
                volume_probe_points=volume_probe_points,
            )
            training_total = output.total + auxiliary_total
            training_total.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    flow.parameters(),
                    float(gradient_clip_norm),
                )
            optimizer.step()
            no_change = torch.zeros_like(output.real_prediction)
        else:
            with torch.no_grad():
                fixed_ratio = torch.full(
                    (batch["source_latent"].shape[0], 1),
                    0.5,
                    device=device,
                    dtype=batch["source_latent"].dtype,
                )
                output = loss_module(
                    source_latent=batch["source_latent"],
                    target_latent=batch["target_latent"],
                    source_time=batch["source_time"],
                    target_time=batch["target_time"],
                    condition=batch["condition"],
                    target_samples=batch["target_samples"],
                    observed_intermediate_time=batch[
                        "observed_intermediate_time"
                    ],
                    observed_intermediate_mask=batch[
                        "observed_intermediate_mask"
                    ],
                    virtual_ratio=fixed_ratio,
                    future_ratio=fixed_ratio,
                )
                auxiliary_total, auxiliary_metrics = compute_pair_auxiliary_losses(
                    batch=batch,
                    output=output,
                    flow=flow,
                    specs=specs,
                    stats=regularizer_stats,
                    loss_module=loss_module,
                    volume_probe_points=volume_probe_points,
                )
                no_change = no_change_sdf_loss(
                    loss_module,
                    batch["source_latent"],
                    batch["target_samples"],
                )

        batch_size = int(batch["source_latent"].shape[0])
        scalars = output.detached_scalars()
        scalars["total"] = float(
            (output.total + auxiliary_total).detach().cpu().item()
        )
        for key, value in auxiliary_metrics.items():
            scalars[key] = float(value.detach().cpu().item())
        for key, value in scalars.items():
            sums.setdefault(key, 0.0)
            sums[key] += value * batch_size
        sums["no_change"] += float(no_change.detach().cpu().item()) * batch_size
        row_count += batch_size
        observed_row_count += int(
            batch["observed_intermediate_mask"].sum().item()
        )

    if row_count == 0:
        raise RuntimeError("Data loader produced no batches")
    result = {key: value / row_count for key, value in sums.items()}
    result["model_minus_no_change"] = (
        result["real_prediction"] - result["no_change"]
    )
    result["sdf_l1_improvement"] = (
        result["no_change"] - result["real_prediction"]
    )
    result.update(
        {
            "rows": float(row_count),
            "observed_rows": float(observed_row_count),
            "seconds": float(time.time() - start_time),
        }
    )
    return result


def compute_sequence_losses(
    *,
    batch: Mapping[str, object],
    loss_module: MinimalDirectFlowLoss,
    flow: torch.nn.Module,
    specs: Mapping[str, object],
    stats: LatentRegularizerStats,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    latents = batch["latents"]  # type: ignore[index]
    times = batch["times"]  # type: ignore[index]
    condition = batch["condition"]  # type: ignore[index]
    target_samples = batch["target_samples"]  # type: ignore[index]
    step_mask = batch["step_mask"]  # type: ignore[index]
    if not isinstance(latents, torch.Tensor):
        raise TypeError("Sequence batch latents must be a tensor")
    if not isinstance(times, torch.Tensor) or not isinstance(condition, torch.Tensor):
        raise TypeError("Sequence batch times/condition must be tensors")
    if not isinstance(target_samples, torch.Tensor) or not isinstance(step_mask, torch.Tensor):
        raise TypeError("Sequence batch samples/mask must be tensors")
    if latents.ndim != 3:
        raise ValueError(f"Expected sequence latents [B,K,D], got {tuple(latents.shape)}")
    batch_size, sequence_length, _ = latents.shape
    if sequence_length < 2:
        zero = latents.sum() * 0.0
        return zero, {
            "sequence_total": zero,
            "sequence_rollout_sdf": zero,
            "sequence_one_shot_sdf": zero,
            "sequence_no_change_sdf": zero,
            "sequence_cocycle": zero,
            "sequence_displacement_magnitude": zero,
            "sequence_latent_manifold_guard": zero,
            "sequence_speed_guard": zero,
            "sequence_latent_zscore_p95": zero.detach(),
            "sequence_latent_zscore_max": zero.detach(),
            "sequence_predicted_speed_p95": zero.detach(),
            "sequence_rollout_minus_no_change": zero.detach(),
        }

    source_latent = latents[:, 0, :]
    source_time = times[:, 0]
    current_latent = source_latent
    previous_time = source_time
    rollout_sdf_values = []
    one_shot_sdf_values = []
    no_change_sdf_values = []
    rollout_masks = []
    rollout_weights = []
    cocycle_values = []
    cocycle_masks = []
    cocycle_weights = []
    displacement_values = []
    displacement_masks = []
    displacement_weights = []
    diagnostic_latents = []
    speed_losses = []
    speed_p95_values = []

    for step_index in range(sequence_length - 1):
        target_time = times[:, step_index + 1]
        mask = step_mask[:, step_index].to(dtype=torch.bool)
        samples = target_samples[:, step_index]
        rollout_latent = flow.transport(
            current_latent,
            previous_time,
            target_time,
            condition,
        )
        one_shot_latent = flow.transport(
            source_latent,
            source_time,
            target_time,
            condition,
        )
        rollout_sdf_values.append(
            loss_module.decode_target_sdf_loss_per_row(
                rollout_latent,
                samples,
            )
        )
        one_shot_sdf_values.append(
            loss_module.decode_target_sdf_loss_per_row(
                one_shot_latent,
                samples,
            )
        )
        no_change_sdf_values.append(
            loss_module.decode_target_sdf_loss_per_row(
                source_latent,
                samples,
            )
        )
        rollout_masks.append(mask)
        step_weights = _sequence_gap_weights(
            source_time,
            previous_time,
            target_time,
            specs,
            stats,
        ).view(-1)
        rollout_weights.append(step_weights)
        if step_index >= 1:
            cocycle_values.append(
                per_row_latent_mse(one_shot_latent, rollout_latent)
            )
            cocycle_masks.append(mask)
            cocycle_weights.append(step_weights)
        if bool(specs.get("UseSequenceDisplacementMagnitudeLoss", False)):
            row_loss, _, _ = displacement_magnitude_row_loss(
                rollout_latent,
                source_latent,
                latents[:, step_index + 1, :],
            )
            displacement_values.append(row_loss)
            displacement_masks.append(mask)
            displacement_weights.append(step_weights)
        diagnostic_latents.extend([rollout_latent, one_shot_latent])
        if bool(specs.get("UseSpeedGuardLoss", False)):
            speed_loss, speed_p95 = speed_guard_loss(
                current_latent,
                rollout_latent,
                previous_time,
                target_time,
                stats,
            )
            speed_losses.append(speed_loss)
            speed_p95_values.append(speed_p95)
        current_latent = rollout_latent
        previous_time = target_time

    rollout_values = torch.stack(rollout_sdf_values, dim=1)
    one_shot_values = torch.stack(one_shot_sdf_values, dim=1)
    no_change_values = torch.stack(no_change_sdf_values, dim=1)
    mask_tensor = torch.stack(rollout_masks, dim=1)
    weight_tensor = torch.stack(rollout_weights, dim=1)
    rollout_sdf = _weighted_mean(rollout_values, weight_tensor, mask_tensor)
    one_shot_sdf = _weighted_mean(one_shot_values, weight_tensor, mask_tensor)
    no_change_sdf = _weighted_mean(no_change_values, weight_tensor, mask_tensor)

    zero = rollout_sdf * 0.0
    if cocycle_values:
        cocycle = _weighted_mean(
            torch.stack(cocycle_values, dim=1),
            torch.stack(cocycle_weights, dim=1),
            torch.stack(cocycle_masks, dim=1),
        )
    else:
        cocycle = zero

    if displacement_values:
        displacement_magnitude = _weighted_mean(
            torch.stack(displacement_values, dim=1),
            torch.stack(displacement_weights, dim=1),
            torch.stack(displacement_masks, dim=1),
        )
    else:
        displacement_magnitude = zero

    latent_guard = zero
    latent_zscore_p95 = zero.detach()
    latent_zscore_max = zero.detach()
    if diagnostic_latents:
        latent_zscore_p95, latent_zscore_max = latent_zscore_summary(
            torch.cat(diagnostic_latents, dim=0),
            stats,
        )
    if (
        bool(specs.get("UseLatentManifoldGuardLoss", False))
        and diagnostic_latents
        and str(specs.get("ApplyLatentGuardTo", "all")).strip().lower() != "future_only"
    ):
        latent_guard, _, _ = latent_manifold_guard_loss(
            torch.cat(diagnostic_latents, dim=0),
            stats,
            threshold=float(specs.get("LatentGuardThreshold", 3.0)),
        )

    speed_guard = zero
    predicted_speed_p95 = zero.detach()
    if speed_losses:
        speed_guard = torch.stack(speed_losses).mean()
        predicted_speed_p95 = torch.stack(speed_p95_values).mean().detach()

    total = zero
    if bool(specs.get("UseSequenceRolloutSDFLoss", False)):
        total = total + float(
            specs.get("SequenceRolloutSDFLossLambda", 0.0)
        ) * rollout_sdf
    if bool(specs.get("UseSequenceCocycleLoss", False)):
        total = total + float(
            specs.get("SequenceCocycleLossLambda", 0.0)
        ) * cocycle
    if bool(specs.get("UseSequenceDisplacementMagnitudeLoss", False)):
        total = total + float(
            specs.get("SequenceDisplacementMagnitudeLossLambda", 0.0)
        ) * displacement_magnitude
    if bool(specs.get("UseLatentManifoldGuardLoss", False)):
        total = total + float(
            specs.get("LatentManifoldGuardLossLambda", 0.0)
        ) * latent_guard
    if bool(specs.get("UseSpeedGuardLoss", False)):
        total = total + float(specs.get("SpeedGuardLossLambda", 0.0)) * speed_guard

    metrics = {
        "sequence_total": total,
        "sequence_rollout_sdf": rollout_sdf,
        "sequence_one_shot_sdf": one_shot_sdf,
        "sequence_no_change_sdf": no_change_sdf,
        "sequence_rollout_minus_no_change": rollout_sdf - no_change_sdf,
        "sequence_cocycle": cocycle,
        "sequence_displacement_magnitude": displacement_magnitude,
        "sequence_latent_manifold_guard": latent_guard,
        "sequence_speed_guard": speed_guard,
        "sequence_latent_zscore_p95": latent_zscore_p95,
        "sequence_latent_zscore_max": latent_zscore_max,
        "sequence_predicted_speed_p95": predicted_speed_p95,
        "sequence_valid_steps": mask_tensor.to(dtype=torch.float32).sum(),
    }
    return total, metrics


def run_sequence_loader(
    *,
    loader: DataLoader,
    loss_module: MinimalDirectFlowLoss,
    flow: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    gradient_clip_norm: Optional[float],
    specs: Mapping[str, object],
    regularizer_stats: LatentRegularizerStats,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    flow.train(training)
    loss_module.decoder.eval()
    sums: Dict[str, float] = {}
    step_count = 0.0
    sequence_count = 0
    start_time = time.time()

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = sequence_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            total, metrics = compute_sequence_losses(
                batch=batch,
                loss_module=loss_module,
                flow=flow,
                specs=specs,
                stats=regularizer_stats,
            )
            total.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    flow.parameters(),
                    float(gradient_clip_norm),
                )
            optimizer.step()
        else:
            with torch.no_grad():
                total, metrics = compute_sequence_losses(
                    batch=batch,
                    loss_module=loss_module,
                    flow=flow,
                    specs=specs,
                    stats=regularizer_stats,
                )
        valid_steps = float(metrics["sequence_valid_steps"].detach().cpu().item())
        if valid_steps <= 0:
            continue
        for key, value in metrics.items():
            if key == "sequence_valid_steps":
                continue
            sums.setdefault(key, 0.0)
            sums[key] += float(value.detach().cpu().item()) * valid_steps
        step_count += valid_steps
        sequence_count += int(batch["latents"].shape[0])  # type: ignore[index,union-attr]

    if step_count == 0:
        raise RuntimeError("Sequence data loader produced no valid steps")
    result = {key: value / step_count for key, value in sums.items()}
    result["sequence_steps"] = float(step_count)
    result["sequences"] = float(sequence_count)
    result["sequence_seconds"] = float(time.time() - start_time)
    return result


def checkpoint_payload(
    *,
    epoch: int,
    decoder: torch.nn.Module,
    flow: torch.nn.Module,
    specs: Mapping[str, object],
    best_validation_real_prediction: float,
    best_validation_selection_metric: float,
) -> Dict[str, object]:
    loss_scope = enabled_loss_scope(specs)
    return {
        "epoch": int(epoch),
        "model_state_dict": decoder.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "decoder_frozen": True,
        "frozen_latent_files": dict(specs["FrozenLatentFiles"]),
        "best_validation_real_prediction": float(
            best_validation_real_prediction
        ),
        "best_validation_selection_metric": float(
            best_validation_selection_metric
        ),
        "latent_condition_mode": str(specs.get("LatentConditionMode", "full")),
        "latent_condition_dim": specs.get("LatentConditionDim", int(specs["CodeLength"])),
        "flow_type": str(specs.get("FlowType", "direct_age")),
        "flow_rank": specs.get("FlowRank"),
        "local_step_years": specs.get("LocalStepYears"),
        "local_max_step_norm": getattr(flow, "max_step_norm", None),
        "flow_include_delta_time_input": bool(
            specs.get("FlowIncludeDeltaTimeInput", False)
        ),
        "ode_integration_substeps": specs.get("ODEIntegrationSubsteps"),
        "ode_max_step_norm": specs.get("ODEMaxStepNorm"),
        "ode_hidden_dims": specs.get("ODEHiddenDims"),
        "loss_scope": loss_scope,
    }


def enabled_loss_scope(specs: Mapping[str, object]) -> list[str]:
    scope = []
    if bool(specs.get("UseRealTargetSDFLoss", True)):
        scope.append("real_target_sdf_prediction")
    if bool(specs.get("UseChangeWeightedSDFLoss", False)):
        scope.append("change_weighted_target_sdf_prediction")
    if bool(specs.get("UseDeltaSDFDirectionLoss", False)):
        scope.append("delta_sdf_direction_alignment")
    if bool(specs.get("UseDeltaSDFRMAELoss", False)):
        scope.append("delta_sdf_relative_change_error")
    if bool(specs.get("UseNoChangeMarginLoss", False)):
        scope.append("no_change_margin_ranking")
    if bool(specs.get("UseObservedCocycleLoss", True)):
        scope.append("observed_intermediate_latent_consistency")
    if bool(specs.get("UseVirtualCocycleLoss", True)):
        scope.append("virtual_intermediate_latent_consistency")
    if bool(specs.get("UseBackwardVirtualLatentLoss", False)):
        scope.append("backward_source_target_virtual_latent_agreement")
    if bool(specs.get("UseBackwardVirtualShapeLoss", False)):
        scope.append("backward_source_target_virtual_shape_agreement")
    if bool(specs.get("UseFutureExtrapolationCocycleLoss", False)):
        scope.append("future_extrapolation_latent_cocycle_consistency")
    if bool(specs.get("UseFutureLatentCocycleLoss", False)):
        scope.append("future_latent_cocycle_consistency")
    if bool(specs.get("UseFutureExtrapolationShapeLoss", False)):
        scope.append("future_extrapolation_shape_cocycle_consistency")
    if bool(specs.get("UseSequenceRolloutSDFLoss", False)):
        scope.append("sequence_rollout_sdf_prediction")
    if bool(specs.get("UseSequenceCocycleLoss", False)):
        scope.append("sequence_direct_vs_rollout_cocycle_consistency")
    if bool(specs.get("UseLatentDirectionLoss", False)):
        scope.append("latent_displacement_direction")
    if bool(specs.get("UseLatentDisplacementMagnitudeLoss", False)):
        scope.append("latent_displacement_magnitude")
    if bool(specs.get("UseLatentManifoldGuardLoss", False)):
        scope.append("latent_manifold_guard")
    if bool(specs.get("UseSpeedGuardLoss", False)):
        scope.append("latent_speed_guard")
    if bool(specs.get("UseSequenceDisplacementMagnitudeLoss", False)):
        scope.append("sequence_displacement_magnitude")
    if bool(specs.get("UseRelativeVolumeLoss", False)):
        scope.append("relative_soft_volume_change")
    if bool(specs.get("UseDiseaseVolumeOrderingLoss", False)):
        scope.append("cn_ad_counterfactual_volume_ordering")
    return scope


def save_checkpoint(
    experiment_dir: Path,
    name: str,
    *,
    epoch: int,
    decoder: torch.nn.Module,
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    specs: Mapping[str, object],
    best_validation_real_prediction: float,
    best_validation_selection_metric: float,
) -> None:
    model_dir = experiment_dir / ws.model_params_subdir
    optimizer_dir = experiment_dir / ws.optimizer_params_subdir
    model_dir.mkdir(parents=True, exist_ok=True)
    optimizer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            epoch=epoch,
            decoder=decoder,
            flow=flow,
            specs=specs,
            best_validation_real_prediction=best_validation_real_prediction,
            best_validation_selection_metric=best_validation_selection_metric,
        ),
        model_dir / f"{name}.pth",
    )
    torch.save(
        {
            "epoch": int(epoch),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        optimizer_dir / f"{name}.pth",
    )


def load_checkpoint(
    experiment_dir: Path,
    name: str,
    *,
    flow: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float, float]:
    model_path = experiment_dir / ws.model_params_subdir / f"{name}.pth"
    optimizer_path = (
        experiment_dir / ws.optimizer_params_subdir / f"{name}.pth"
    )
    model_payload = torch.load(model_path, map_location="cpu")
    optimizer_payload = torch.load(optimizer_path, map_location="cpu")
    flow.load_state_dict(model_payload["flow_state_dict"])
    optimizer.load_state_dict(optimizer_payload["optimizer_state_dict"])
    model_epoch = int(model_payload["epoch"])
    optimizer_epoch = int(optimizer_payload["epoch"])
    if model_epoch != optimizer_epoch:
        raise RuntimeError(
            f"Checkpoint epoch mismatch: model={model_epoch}, optimizer={optimizer_epoch}"
        )
    return (
        model_epoch,
        float(
            model_payload.get(
                "best_validation_real_prediction",
                float("inf"),
            )
        ),
        float(
            model_payload.get(
                "best_validation_selection_metric",
                model_payload.get("best_validation_real_prediction", float("inf")),
            )
        ),
    )


def save_logs(
    experiment_dir: Path,
    *,
    epoch: int,
    history: Mapping[str, object],
) -> None:
    torch.save(
        {"epoch": int(epoch), **history},
        experiment_dir / ws.logs_filename,
    )


def load_logs(experiment_dir: Path, epoch: int) -> Dict[str, object]:
    path = experiment_dir / ws.logs_filename
    if not path.is_file():
        return {
            "train": [],
            "validation": [],
            "learning_rate": [],
        }
    payload = torch.load(path, map_location="cpu")
    history = {
        "train": list(payload.get("train", [])),
        "validation": list(payload.get("validation", [])),
        "learning_rate": list(payload.get("learning_rate", [])),
    }
    history["train"] = [
        row for row in history["train"] if int(row["epoch"]) <= int(epoch)
    ]
    history["validation"] = [
        row
        for row in history["validation"]
        if int(row["epoch"]) <= int(epoch)
    ]
    history["learning_rate"] = history["learning_rate"][: int(epoch)]
    return history


def determine_device(gpu: Optional[int], validate_only: bool) -> torch.device:
    if validate_only:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Use --validate-only for CPU contract checks."
        )
    if gpu is not None:
        torch.cuda.set_device(int(gpu))
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def build_optimizer(
    specs: Mapping[str, object],
    flow: torch.nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    optimizer_name = str(specs.get("Optimizer", "Adam")).strip().lower()
    weight_decay = float(specs.get("WeightDecay", 0.0))
    if optimizer_name == "adam":
        return torch.optim.Adam(
            flow.parameters(),
            lr=float(learning_rate),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            flow.parameters(),
            lr=float(learning_rate),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported Optimizer {specs.get('Optimizer')!r}")


def validation_selection_metric(
    metrics: Mapping[str, float],
    *,
    require_beat_no_change: bool,
    specs: Optional[Mapping[str, object]] = None,
) -> float:
    mode = str(
        (specs or {}).get(
            "ValidationSelectionMode",
            "pair_model_minus_no_change",
        )
    ).strip().lower()
    if mode == "pair_sequence_minus_no_change":
        pair_score = float(metrics["model_minus_no_change"])
        if "sequence_rollout_minus_no_change" in metrics:
            return pair_score + float(
                (specs or {}).get("SequenceValidationSelectionWeight", 0.5)
            ) * float(metrics["sequence_rollout_minus_no_change"])
        return pair_score
    if mode == "pair_sdf_plus_volume":
        return (
            float(metrics["model_minus_no_change"])
            + float((specs or {}).get("VolumeValidationSelectionWeight", 1.0))
            * float(metrics.get("relative_volume", 0.0))
            + float(
                (specs or {}).get("DiseaseVolumeValidationSelectionWeight", 0.0)
            )
            * float(metrics.get("disease_volume_ordering", 0.0))
        )
    if mode == "pair_sdf_plus_change_volume":
        return (
            float(metrics["model_minus_no_change"])
            + float(
                (specs or {}).get(
                    "ChangeWeightedSDFValidationSelectionWeight",
                    0.0,
                )
            )
            * float(metrics.get("change_weighted_sdf", 0.0))
            + float(
                (specs or {}).get(
                    "DeltaDirectionValidationSelectionWeight",
                    (specs or {}).get("ChangeValidationSelectionWeight", 0.25),
                )
            )
            * float(metrics.get("delta_sdf_direction", 0.0))
            + float(
                (specs or {}).get(
                    "DeltaRMAEValidationSelectionWeight",
                    (specs or {}).get("ChangeRmaeValidationSelectionWeight", 0.1),
                )
            )
            * float(metrics.get("delta_sdf_rmae", 0.0))
            + float(
                (specs or {}).get("NoChangeMarginValidationSelectionWeight", 0.0)
            )
            * float(metrics.get("no_change_margin", 0.0))
            + float((specs or {}).get("VolumeValidationSelectionWeight", 1.0))
            * float(metrics.get("relative_volume", 0.0))
            + float(
                (specs or {}).get("DiseaseVolumeValidationSelectionWeight", 0.0)
            )
            * float(metrics.get("disease_volume_ordering", 0.0))
        )
    if require_beat_no_change:
        return float(metrics["model_minus_no_change"])
    return float(metrics["real_prediction"])


def validation_satisfies_best_rule(
    metrics: Mapping[str, float],
    *,
    require_beat_no_change: bool,
    specs: Optional[Mapping[str, object]] = None,
) -> bool:
    pair_ok = True if not require_beat_no_change else (
        float(metrics["model_minus_no_change"]) < 0.0
    )
    if not pair_ok:
        return False
    if bool((specs or {}).get("RequireSequenceBeatNoChangeForBest", False)):
        return float(metrics.get("sequence_rollout_minus_no_change", 1.0)) < 0.0
    return True


def write_contract_report(
    experiment_dir: Path,
    report: Mapping[str, object],
) -> None:
    analysis_dir = experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "input_contract_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def experiment_report(
    contract: DirectFlowDataContract,
    specs: Mapping[str, object],
) -> Dict[str, object]:
    report = dict(contract.report)
    report["loss_scope"] = enabled_loss_scope(specs)
    return report


def train(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment).resolve()
    setup_logging(experiment_dir)
    specs = load_specs(experiment_dir)
    validate_loss_scope(specs)
    seed_everything(int(specs.get("Seed", 42)))
    contract = build_data_contract(specs, experiment_dir)
    report = experiment_report(contract, specs)
    write_contract_report(experiment_dir, report)
    LOGGER.info("Input contract:\n%s", json.dumps(report, indent=2))

    device = determine_device(args.gpu, args.validate_only)
    decoder, decoder_epoch = load_frozen_decoder(
        specs,
        experiment_dir,
        device,
    )
    LOGGER.info(
        "Loaded and froze Task-2 decoder at epoch %d on %s",
        decoder_epoch,
        device,
    )
    flow = build_flow(specs, device, contract)
    loss_module = build_loss(specs, decoder, flow)
    volume_probe_points = make_volume_probe_points(specs, device)
    regularizer_stats = build_latent_regularizer_stats(
        contract,
        split="train",
        speed_percentile=float(specs.get("SpeedGuardPercentile", 95.0)),
    )
    LOGGER.info(
        "Train latent regularizer stats: speed_p%.1f=%.6g median_gap_norm=%.6g",
        float(specs.get("SpeedGuardPercentile", 95.0)),
        regularizer_stats.speed_percentile,
        regularizer_stats.median_gap_norm,
    )

    if args.validate_only:
        train_loader = make_loader(
            contract,
            "train",
            specs,
            training=False,
        )
        raw_batch = next(iter(train_loader))
        batch = to_device(raw_batch, device)
        with torch.no_grad():
            output = loss_module(
                source_latent=batch["source_latent"],
                target_latent=batch["target_latent"],
                source_time=batch["source_time"],
                target_time=batch["target_time"],
                condition=batch["condition"],
                target_samples=batch["target_samples"],
                observed_intermediate_time=batch[
                    "observed_intermediate_time"
                ],
                observed_intermediate_mask=batch[
                    "observed_intermediate_mask"
                ],
                virtual_ratio=torch.full(
                    (batch["source_latent"].shape[0], 1),
                    0.5,
                    device=device,
                ),
                future_ratio=torch.full(
                    (batch["source_latent"].shape[0], 1),
                    0.5,
                    device=device,
                ),
            )
            auxiliary_total, auxiliary_metrics = compute_pair_auxiliary_losses(
                batch=batch,
                output=output,
                flow=flow,
                specs=specs,
                stats=regularizer_stats,
                loss_module=loss_module,
                volume_probe_points=volume_probe_points,
            )
        LOGGER.info(
            "CPU validation forward pass succeeded: %s",
            json.dumps(
                {
                    **output.detached_scalars(),
                    **{
                        key: float(value.detach().cpu().item())
                        for key, value in auxiliary_metrics.items()
                    },
                    "total_with_auxiliary": float(
                        (output.total + auxiliary_total).detach().cpu().item()
                    ),
                },
                indent=2,
            ),
        )
        if sequence_losses_enabled(specs):
            sequence_loader = make_sequence_loader(
                contract,
                "train",
                specs,
                training=False,
            )
            sequence_batch = sequence_to_device(next(iter(sequence_loader)), device)
            with torch.no_grad():
                _, sequence_metrics = compute_sequence_losses(
                    batch=sequence_batch,
                    loss_module=loss_module,
                    flow=flow,
                    specs=specs,
                    stats=regularizer_stats,
                )
            LOGGER.info(
                "CPU sequence validation forward pass succeeded: %s",
                json.dumps(
                    {
                        key: float(value.detach().cpu().item())
                        for key, value in sequence_metrics.items()
                    },
                    indent=2,
                ),
            )
        return

    train_loader = make_loader(contract, "train", specs, training=True)
    validation_loader = make_loader(
        contract,
        "val",
        specs,
        training=False,
    )
    train_sequence_loader = (
        make_sequence_loader(contract, "train", specs, training=True)
        if sequence_losses_enabled(specs)
        else None
    )
    validation_sequence_loader = (
        make_sequence_loader(contract, "val", specs, training=False)
        if sequence_losses_enabled(specs)
        else None
    )
    schedules = lr_scheduling.get_learning_rate_schedules(specs)
    if len(schedules) != 1:
        raise ValueError(
            "Direct-flow training expects exactly one learning-rate schedule"
        )
    optimizer = build_optimizer(
        specs,
        flow,
        float(schedules[0].get_learning_rate(0)),
    )
    history: Dict[str, object] = {
        "train": [],
        "validation": [],
        "learning_rate": [],
    }
    start_epoch = 1
    best_validation_real_prediction = float("inf")
    best_validation_selection = float("inf")
    best_checkpoint_saved = False
    require_beat_no_change = bool(specs.get("RequireBeatNoChangeForBest", False))
    early_stopping_patience = specs.get("EarlyStoppingPatience")
    early_stopping_min_delta = float(specs.get("EarlyStoppingMinDelta", 0.0))
    epochs_since_selection_improved = 0
    if args.continue_from:
        (
            loaded_epoch,
            best_validation_real_prediction,
            best_validation_selection,
        ) = load_checkpoint(
            experiment_dir,
            args.continue_from,
            flow=flow,
            optimizer=optimizer,
        )
        history = load_logs(experiment_dir, loaded_epoch)
        start_epoch = loaded_epoch + 1
        best_checkpoint_saved = (
            experiment_dir / ws.model_params_subdir / "best.pth"
        ).is_file()
        LOGGER.info(
            "Resumed checkpoint %s at epoch %d",
            args.continue_from,
            loaded_epoch,
        )

    tensorboard_dir = experiment_dir / ws.tb_logs_dir
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    num_epochs = int(specs.get("NumEpochs", 1000))
    validation_frequency = int(specs.get("ValidationFrequency", 25))
    snapshot_frequency = int(specs.get("SnapshotFrequency", 100))
    latest_frequency = int(specs.get("LatestFrequency", 10))
    gradient_clip = specs.get("GradientClipNorm", 1.0)
    max_batches = args.smoke_batches

    for epoch in range(start_epoch, num_epochs + 1):
        stop_after_epoch = False
        learning_rate = float(
            schedules[0].get_learning_rate(
                epoch - 1,
                [row["total"] for row in history["train"]],
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        history["learning_rate"].append(learning_rate)

        train_metrics = run_loader(
            loader=train_loader,
            loss_module=loss_module,
            flow=flow,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=(
                None if gradient_clip is None else float(gradient_clip)
            ),
            specs=specs,
            regularizer_stats=regularizer_stats,
            volume_probe_points=volume_probe_points,
            max_batches=max_batches,
        )
        if train_sequence_loader is not None:
            sequence_metrics = run_sequence_loader(
                loader=train_sequence_loader,
                loss_module=loss_module,
                flow=flow,
                device=device,
                optimizer=optimizer,
                gradient_clip_norm=(
                    None if gradient_clip is None else float(gradient_clip)
                ),
                specs=specs,
                regularizer_stats=regularizer_stats,
                max_batches=max_batches,
            )
            train_metrics.update(sequence_metrics)
            train_metrics["total"] = (
                float(train_metrics["total"])
                + float(sequence_metrics["sequence_total"])
            )
        train_metrics["epoch"] = int(epoch)
        history["train"].append(train_metrics)
        for key, value in train_metrics.items():
            if key not in ("epoch", "rows", "observed_rows"):
                writer.add_scalar(f"Loss/train_{key}", value, epoch)
        writer.add_scalar("LearningRate/flow", learning_rate, epoch)

        LOGGER.info(
            "epoch=%d lr=%.6g total=%.6g real=%.6g change_w=%.6g "
            "delta_dir=%.6g delta_rmae=%.6g nochg_margin=%.6g "
            "observed=%.6g virtual=%.6g backward_lat=%.6g backward_shape=%.6g "
            "future_lat=%.6g future_shape=%.6g direction=%.6g "
            "disp_mag=%.6g latent_guard=%.6g speed_guard=%.6g "
            "rel_vol=%.6g disease_vol=%.6g seq_rollout=%.6g "
            "seq_cocycle=%.6g seq_disp_mag=%.6g seconds=%.1f",
            epoch,
            learning_rate,
            train_metrics["total"],
            train_metrics["real_prediction"],
            train_metrics.get("change_weighted_sdf", 0.0),
            train_metrics.get("delta_sdf_direction", 0.0),
            train_metrics.get("delta_sdf_rmae", 0.0),
            train_metrics.get("no_change_margin", 0.0),
            train_metrics["observed_consistency"],
            train_metrics["virtual_consistency"],
            train_metrics.get("backward_virtual_latent", 0.0),
            train_metrics.get("backward_virtual_shape", 0.0),
            train_metrics.get("future_extrapolation_latent", 0.0),
            train_metrics.get("future_extrapolation_shape", 0.0),
            train_metrics.get("latent_direction", 0.0),
            train_metrics.get("latent_displacement_magnitude", 0.0),
            train_metrics.get("latent_manifold_guard", 0.0),
            train_metrics.get("speed_guard", 0.0),
            train_metrics.get("relative_volume", 0.0),
            train_metrics.get("disease_volume_ordering", 0.0),
            train_metrics.get("sequence_rollout_sdf", 0.0),
            train_metrics.get("sequence_cocycle", 0.0),
            train_metrics.get("sequence_displacement_magnitude", 0.0),
            train_metrics["seconds"],
        )

        should_validate = (
            epoch == 1
            or epoch % validation_frequency == 0
            or epoch == num_epochs
        )
        if should_validate:
            validation_metrics = run_loader(
                loader=validation_loader,
                loss_module=loss_module,
                flow=flow,
                device=device,
                optimizer=None,
                gradient_clip_norm=None,
                specs=specs,
                regularizer_stats=regularizer_stats,
                volume_probe_points=volume_probe_points,
                max_batches=max_batches,
            )
            if validation_sequence_loader is not None:
                sequence_metrics = run_sequence_loader(
                    loader=validation_sequence_loader,
                    loss_module=loss_module,
                    flow=flow,
                    device=device,
                    optimizer=None,
                    gradient_clip_norm=None,
                    specs=specs,
                    regularizer_stats=regularizer_stats,
                    max_batches=max_batches,
                )
                validation_metrics.update(sequence_metrics)
                validation_metrics["total"] = (
                    float(validation_metrics["total"])
                    + float(sequence_metrics["sequence_total"])
                )
            validation_metrics["epoch"] = int(epoch)
            history["validation"].append(validation_metrics)
            for key, value in validation_metrics.items():
                if key not in ("epoch", "rows", "observed_rows"):
                    writer.add_scalar(f"Loss/validation_{key}", value, epoch)
            LOGGER.info(
                "validation epoch=%d real=%.6g change_w=%.6g "
                "delta_dir=%.6g delta_rmae=%.6g nochg_margin=%.6g "
                "no_change=%.6g "
                "model_minus_no_change=%.6g observed=%.6g virtual=%.6g "
                "backward_lat=%.6g backward_shape=%.6g "
                "future_lat=%.6g future_shape=%.6g direction=%.6g "
                "disp_mag=%.6g latent_guard=%.6g speed_guard=%.6g "
                "rel_vol=%.6g disease_vol=%.6g "
                "seq_rollout=%.6g seq_no_change=%.6g seq_minus=%.6g "
                "seq_cocycle=%.6g seq_disp_mag=%.6g",
                epoch,
                validation_metrics["real_prediction"],
                validation_metrics.get("change_weighted_sdf", 0.0),
                validation_metrics.get("delta_sdf_direction", 0.0),
                validation_metrics.get("delta_sdf_rmae", 0.0),
                validation_metrics.get("no_change_margin", 0.0),
                validation_metrics["no_change"],
                validation_metrics["model_minus_no_change"],
                validation_metrics["observed_consistency"],
                validation_metrics["virtual_consistency"],
                validation_metrics.get("backward_virtual_latent", 0.0),
                validation_metrics.get("backward_virtual_shape", 0.0),
                validation_metrics.get("future_extrapolation_latent", 0.0),
                validation_metrics.get("future_extrapolation_shape", 0.0),
                validation_metrics.get("latent_direction", 0.0),
                validation_metrics.get("latent_displacement_magnitude", 0.0),
                validation_metrics.get("latent_manifold_guard", 0.0),
                validation_metrics.get("speed_guard", 0.0),
                validation_metrics.get("relative_volume", 0.0),
                validation_metrics.get("disease_volume_ordering", 0.0),
                validation_metrics.get("sequence_rollout_sdf", 0.0),
                validation_metrics.get("sequence_no_change_sdf", 0.0),
                validation_metrics.get("sequence_rollout_minus_no_change", 0.0),
                validation_metrics.get("sequence_cocycle", 0.0),
                validation_metrics.get("sequence_displacement_magnitude", 0.0),
            )
            selection_metric = validation_selection_metric(
                validation_metrics,
                require_beat_no_change=require_beat_no_change,
                specs=specs,
            )
            selection_improved = (
                selection_metric
                < best_validation_selection - early_stopping_min_delta
            )
            if selection_improved:
                best_validation_selection = float(selection_metric)
                best_validation_real_prediction = float(
                    validation_metrics["real_prediction"]
                )
                save_checkpoint(
                    experiment_dir,
                    "best_candidate",
                    epoch=epoch,
                    decoder=decoder,
                    flow=flow,
                    optimizer=optimizer,
                    specs=specs,
                    best_validation_real_prediction=best_validation_real_prediction,
                    best_validation_selection_metric=best_validation_selection,
                )
                epochs_since_selection_improved = 0
                LOGGER.info(
                    "Saved best_candidate checkpoint: epoch=%d "
                    "selection=%.6g validation_real=%.6g",
                    epoch,
                    best_validation_selection,
                    best_validation_real_prediction,
                )
            else:
                epochs_since_selection_improved += validation_frequency

            if selection_improved and validation_satisfies_best_rule(
                validation_metrics,
                require_beat_no_change=require_beat_no_change,
                specs=specs,
            ):
                save_checkpoint(
                    experiment_dir,
                    "best",
                    epoch=epoch,
                    decoder=decoder,
                    flow=flow,
                    optimizer=optimizer,
                    specs=specs,
                    best_validation_real_prediction=best_validation_real_prediction,
                    best_validation_selection_metric=best_validation_selection,
                )
                best_checkpoint_saved = True
                LOGGER.info(
                    "Saved best checkpoint: epoch=%d selection=%.6g "
                    "validation_real=%.6g",
                    epoch,
                    best_validation_selection,
                    best_validation_real_prediction,
                )

            if (
                early_stopping_patience is not None
                and int(early_stopping_patience) > 0
                and epochs_since_selection_improved >= int(early_stopping_patience)
            ):
                LOGGER.info(
                    "Early stopping at epoch %d after %d validation epochs "
                    "without selection improvement.",
                    epoch,
                    epochs_since_selection_improved,
                )
                save_checkpoint(
                    experiment_dir,
                    "latest",
                    epoch=epoch,
                    decoder=decoder,
                    flow=flow,
                    optimizer=optimizer,
                    specs=specs,
                    best_validation_real_prediction=best_validation_real_prediction,
                    best_validation_selection_metric=best_validation_selection,
                )
                stop_after_epoch = True

        if epoch % latest_frequency == 0 or epoch == num_epochs:
            save_checkpoint(
                experiment_dir,
                "latest",
                epoch=epoch,
                decoder=decoder,
                flow=flow,
                optimizer=optimizer,
                specs=specs,
                best_validation_real_prediction=best_validation_real_prediction,
                best_validation_selection_metric=best_validation_selection,
            )
        if epoch % snapshot_frequency == 0:
            save_checkpoint(
                experiment_dir,
                str(epoch),
                epoch=epoch,
                decoder=decoder,
                flow=flow,
                optimizer=optimizer,
                specs=specs,
                best_validation_real_prediction=best_validation_real_prediction,
                best_validation_selection_metric=best_validation_selection,
            )
        save_logs(experiment_dir, epoch=epoch, history=history)
        writer.flush()

        if stop_after_epoch:
            break
        if args.smoke_batches is not None:
            LOGGER.info(
                "Smoke training requested; stopping after epoch %d",
                epoch,
            )
            break
    if require_beat_no_change and not best_checkpoint_saved:
        LOGGER.warning(
            "No validation checkpoint beat no-change. Use latest.pth or "
            "best_candidate.pth for diagnostics; best.pth was not created."
        )
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen-decoder scan-to-scan ADNI direct flow."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help="Experiment directory containing specs.json.",
    )
    parser.add_argument(
        "-c",
        "--continue-from",
        default=None,
        help="Checkpoint name to resume, such as latest or 500.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and run one CPU forward pass without training.",
    )
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=None,
        help="Run only this many batches and stop after one epoch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
