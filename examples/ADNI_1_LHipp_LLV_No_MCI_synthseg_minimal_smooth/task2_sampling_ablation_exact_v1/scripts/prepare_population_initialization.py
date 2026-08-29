#!/usr/bin/env python3
"""Create or verify the epoch-0 population initialization on the 10-TB disk."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPTS = SCRIPT_DIR.parent.parent / "task2_inr_multires_single_field_v1" / "scripts"
for path in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_common import (  # noqa: E402
    atomic_write_json,
    initialization_path,
    load_json,
    require_bulk_path,
    sha256_file,
)
from multires_common import (  # noqa: E402
    atomic_torch_save,
    build_decoder,
    config_public,
    load_config,
    load_manifest,
    set_global_seed,
)
from train_multires_sdf import make_optimizer  # noqa: E402


EXPECTED_ARCHITECTURE = "single_field_dense_multiresolution_sdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_digest(state[key]).encode("ascii"))
    return digest.hexdigest()


def _normalized_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    source = payload.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError("Source checkpoint has no model_state_dict.")
    return {key.removeprefix("module."): value.detach().cpu() for key, value in source.items()}


def _assert_model_compatible(
    config: dict[str, Any], source: dict[str, torch.Tensor]
) -> torch.nn.Module:
    model = build_decoder(config, torch.device("cpu"))
    target = model.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    mismatched = sorted(
        key for key in set(target).intersection(source)
        if tuple(target[key].shape) != tuple(source[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Population model is incompatible: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} mismatched={mismatched[:5]}."
        )
    model.load_state_dict(source, strict=True)
    return model


def _probe_prediction(
    model: torch.nn.Module, latent_table: torch.Tensor, latent_size: int
) -> torch.Tensor:
    xyz = torch.tensor(
        [
            [-0.40, -0.50, -0.30],
            [-0.10, 0.20, 0.15],
            [0.00, 0.00, 0.00],
            [0.25, -0.35, 0.40],
            [0.50, 0.70, -0.45],
        ],
        dtype=torch.float32,
    )
    codes = latent_table[:5, :latent_size].float()
    weights = [1.0] * len(model.grid_resolutions)
    model.eval()
    with torch.no_grad():
        return model(torch.cat((codes, xyz), dim=1), level_weights=weights).detach().cpu()


def validate_source(
    config: dict[str, Any], source_path: Path, expected_hash: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[str], torch.Tensor, dict[str, Any]]:
    actual_hash = sha256_file(source_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Source checkpoint checksum mismatch: expected {expected_hash}, found {actual_hash}."
        )
    payload = torch.load(source_path, map_location="cpu")
    settings = config["population_initialization"]
    if payload.get("architecture") != settings["expected_architecture"]:
        raise ValueError(
            f"Source architecture {payload.get('architecture')!r} does not match "
            f"{settings['expected_architecture']!r}."
        )
    if int(payload.get("epoch", -1)) != int(settings["expected_source_epoch"]):
        raise ValueError("Source checkpoint epoch does not match the pinned epoch.")
    source_ids = list(payload.get("training_scan_ids", []))
    rows = load_manifest(config["manifest"])
    train_ids = [row["scan_id"] for row in rows if row["split"] == "train"]
    if source_ids != train_ids:
        raise ValueError("Source checkpoint training scan IDs/order differ from the exact manifest.")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training scan IDs are not unique.")
    latents = payload.get("latent_codes")
    expected_shape = (len(train_ids), int(settings["expected_latent_size"]))
    if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != expected_shape:
        raise ValueError(f"Source latent shape must be {expected_shape}, found {getattr(latents, 'shape', None)}.")
    latents = latents.detach().cpu().float()
    if not torch.isfinite(latents).all():
        raise ValueError("Source latent table contains non-finite values.")
    source = _normalized_state(payload)
    model = _assert_model_compatible(config, source)
    probe = _probe_prediction(model, latents, int(config["latent_size"]))
    report = {
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": actual_hash,
        "source_epoch": int(payload["epoch"]),
        "source_architecture": payload["architecture"],
        "training_scan_count": len(train_ids),
        "training_scan_order_exact_match": True,
        "latent_shape": list(latents.shape),
        "latent_finite": True,
        "latent_mean": float(latents.mean()),
        "latent_std": float(latents.std()),
        "latent_max_abs": float(latents.abs().max()),
        "latent_sha256": tensor_digest(latents),
        "model_tensor_count": len(source),
        "model_state_sha256": state_digest(source),
        "model_state_key_and_shape_match": True,
        "probe_prediction": probe[:, 0].tolist(),
    }
    return payload, source, train_ids, latents, report


def verify_initialization(
    config: dict[str, Any], path: Path, source_report: dict[str, Any]
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("architecture") != EXPECTED_ARCHITECTURE or int(payload.get("epoch", -1)) != 0:
        raise ValueError("Initialization must be an epoch-0 multiresolution checkpoint.")
    if payload.get("training_scan_ids") != source_report["training_scan_ids"]:
        raise ValueError("Initialization training scan order changed.")
    latents = payload["latent_codes"].detach().cpu().float()
    source_latents = source_report["source_latents"]
    if not torch.equal(latents, source_latents):
        raise ValueError("Initialization latent table is not byte-identical to the source.")
    state = _normalized_state(payload)
    if state_digest(state) != source_report["model_state_sha256"]:
        raise ValueError("Initialization model state differs from the source.")
    optimizer = payload.get("optimizer_state_dict", {})
    if optimizer.get("state"):
        raise ValueError("Initialization optimizer contains inherited state/momentum.")
    observed_lrs = {
        group["name"]: float(group["lr"]) for group in optimizer.get("param_groups", [])
    }
    expected_lrs = {key: float(value) for key, value in config["learning_rates"].items()}
    if observed_lrs != expected_lrs:
        raise ValueError(f"Fresh optimizer LRs differ: {observed_lrs} != {expected_lrs}.")
    if not math.isinf(float(payload.get("best_validation_l1", 0.0))):
        raise ValueError("Initialization best validation metric was not reset.")
    if not math.isinf(float(payload.get("best_mesh_assd", 0.0))):
        raise ValueError("Initialization best mesh metric was not reset.")
    model = _assert_model_compatible(config, state)
    prediction = _probe_prediction(model, latents, int(config["latent_size"]))
    source_prediction = torch.tensor(source_report["probe_prediction"]).reshape(-1, 1)
    max_difference = float(torch.max(torch.abs(prediction - source_prediction)))
    if max_difference != 0.0:
        raise ValueError(f"Initialization prediction differs from source by {max_difference}.")
    return {
        "passed": True,
        "initialization_checkpoint": str(path),
        "initialization_checkpoint_sha256": sha256_file(path),
        "epoch_reset_to_zero": True,
        "model_state_byte_identical": True,
        "training_latents_byte_identical": True,
        "training_scan_order_exact_match": True,
        "source_optimizer_loaded": False,
        "source_rng_loaded": False,
        "fresh_optimizer_state_entries": len(optimizer.get("state", {})),
        "fresh_learning_rates": observed_lrs,
        "best_metrics_reset_to_infinity": True,
        "probe_prediction_max_abs_difference": max_difference,
    }


def prepare(config_path: str | Path, verify_only: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    settings = config.get("population_initialization", {})
    if settings.get("mode") != "model_and_training_latents_fresh_optimizer":
        raise ValueError("Config does not request the controlled population initialization mode.")
    source_path = Path(settings["checkpoint"]).resolve()
    source_payload, source_state, train_ids, latents, source = validate_source(
        config, source_path, settings["checkpoint_sha256"]
    )
    source_internal = {
        **source,
        "training_scan_ids": train_ids,
        "source_latents": latents,
    }
    output = require_bulk_path(config["output_dir"])
    initialization = output / "checkpoints" / "population_initialization.pth"
    if initialization.is_file():
        verification = verify_initialization(config, initialization, source_internal)
        return {"source": source, "initialization": verification, "existing_verified": True}
    if verify_only:
        raise FileNotFoundError(f"Initialization has not been prepared: {initialization}")

    model = _assert_model_compatible(config, source_state)
    embedding = torch.nn.Embedding(len(train_ids), int(config["latent_size"]), device="cpu")
    embedding.weight.data.copy_(latents)
    optimizer = make_optimizer(model, embedding, config)
    if optimizer.state:
        raise RuntimeError("A newly constructed Adam optimizer unexpectedly has state entries.")
    set_global_seed(int(config["seed"]))
    transfer_report = {
        "mode": "model_and_training_latents_fresh_optimizer",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": settings["checkpoint_sha256"],
        "source_epoch": int(source_payload["epoch"]),
        "target_epoch": 0,
        "model_loaded": True,
        "training_latents_loaded": True,
        "source_optimizer_loaded": False,
        "source_rng_loaded": False,
        "best_metrics_reset": True,
        "training_scan_order_exact_match": True,
        "model_state_sha256": source["model_state_sha256"],
        "latent_sha256": source["latent_sha256"],
    }
    payload = {
        "format_version": 1,
        "architecture": EXPECTED_ARCHITECTURE,
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "latent_codes": embedding.weight.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_l1": math.inf,
        "best_mesh_assd": math.inf,
        "config": config_public(config),
        "training_scan_ids": train_ids,
        "level_weights": [1.0] * len(config["network_specs"]["grid_resolutions"]),
        "warm_start_report": transfer_report,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    atomic_torch_save(payload, initialization)
    verification = verify_initialization(config, initialization, source_internal)
    report = {"source": source, "initialization": verification, "existing_verified": False}
    atomic_write_json(output / "population_initialization_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    report = prepare(args.config, args.verify_only)
    import json

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
