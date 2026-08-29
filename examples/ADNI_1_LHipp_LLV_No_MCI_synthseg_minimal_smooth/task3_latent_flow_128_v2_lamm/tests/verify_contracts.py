#!/usr/bin/env python3
"""Read-only contract test for the pinned LAMM/SpiralNet C4 experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from _bootstrap import activate

activate()
import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Strictly reconstruct both models and run a one-mesh CPU encode/decode.",
    )
    return parser.parse_args()


def comparable_config(path: Path) -> dict:
    config = C.read_json(path)
    if config["method"] != "direct_c4":
        raise ValueError(f"Non-C4 method in {path}")
    if int(config["model"]["latent_dim"]) != C.LATENT_DIM:
        raise ValueError(f"Wrong latent dimension in {path}")
    if bool(config["model"]["ode_used"]) or bool(config["model"]["attention_used"]):
        raise ValueError(f"Unexpected ODE/attention in {path}")
    if float(config["loss"]["coboundary_weight"]) != 0.0:
        raise ValueError(f"Coboundary is not zero in {path}")
    if bool(config["selection"]["test_used_for_selection"]):
        raise ValueError(f"Test selection enabled in {path}")
    normalized = copy.deepcopy(config)
    normalized.pop("name")
    normalized.pop("representation")
    normalized["training"].pop("run_name")
    return normalized


def verify_checkpoint_metadata(registry: dict) -> dict:
    result = {}
    for name in ("spiralnet128", "lamm128", "lamm128_n3_s2", "lamm128_n3_s3"):
        spec = C.representation_spec(name, registry)
        path = C.resolve_path(spec["checkpoint"])
        if C.sha256(path) != spec["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if float(payload["best_val_rmse_mm"]) != float(spec["source_validation_rmse_mm"]):
            raise ValueError(f"Pinned validation metric mismatch: {name}")
        if not isinstance(payload.get("model_state_dict"), dict):
            raise ValueError(f"Missing model state: {name}")
        result[name] = {
            "checkpoint": str(path),
            "sha256": spec["checkpoint_sha256"],
            "best_validation_rmse_mm": float(payload["best_val_rmse_mm"]),
        }
        if name.startswith("lamm128"):
            args = payload["args"]
            info = payload["info"]
            expected = {
                "latent": 128,
                "backbone": "mlpmixer",
                "dim": 256,
                "enc_depth": 8,
                "dec_depth": 6,
                "heads": 4,
                "dropout": 0.05,
                "residual": True,
                "mixup_alpha": 0.8,
                "mixup_prob": 0.5,
                "loss": "l1",
                "norm_mode": "std",
                "metric_weighted_loss": True,
            }
            for key, value in expected.items():
                if args.get(key) != value:
                    raise ValueError(f"LAMM {key} drift: {args.get(key)!r} != {value!r}")
            if info.get("scales") != [43, 86] or info.get("params", {}).get("latent_split") != [64, 64]:
                raise ValueError("LAMM scale/latent split drift")
    return result


@torch.no_grad()
def verify_models(registry: dict) -> dict:
    device = torch.device("cpu")
    vertices = np.asarray(C.cached_vertices("train", registry)[0:1], dtype=np.float32).copy()
    mean, std = C.ae_normalization(registry)
    normalized = torch.from_numpy((vertices - mean) / std)
    output = {}
    for name in ("spiralnet128", "lamm128", "lamm128_n3_s2", "lamm128_n3_s3"):
        model, _ = C.load_ae_model(name, device, registry)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError(f"{name} is not frozen/eval")
        latent = model.encode(normalized)
        decoded = model.decode(latent)
        if latent.shape != (1, C.LATENT_DIM) or decoded.shape != normalized.shape:
            raise RuntimeError(
                f"{name} encode/decode shape mismatch: {tuple(latent.shape)}, {tuple(decoded.shape)}"
            )
        if not torch.isfinite(latent).all() or not torch.isfinite(decoded).all():
            raise RuntimeError(f"{name} produced non-finite values")
        output[name] = {
            "latent_shape": list(latent.shape),
            "decoded_shape": list(decoded.shape),
            "trainable_parameters": C.parameter_count(model),
        }
    return output


def verify_source_splits(registry: dict) -> dict:
    archives = {split: C.load_source_archive(split, registry) for split in C.SPLITS}
    subjects = {split: set(archive["subject_ids"].astype(str)) for split, archive in archives.items()}
    scans = {split: set(archive["visit_scan_ids"].astype(str)) for split, archive in archives.items()}
    for first_index, first in enumerate(C.SPLITS):
        for second in C.SPLITS[first_index + 1 :]:
            if subjects[first] & subjects[second] or scans[first] & scans[second]:
                raise ValueError(f"Source split leakage: {first}/{second}")
    return {
        split: {"subjects": len(subjects[split]), "visits": len(scans[split])}
        for split in C.SPLITS
    }


def main() -> int:
    args = parse_args()
    registry = C.load_registry()
    expected_representations = {
        "spiralnet128", "lamm128", "lamm128_n3_s2", "lamm128_n3_s3"
    }
    if set(registry["representations"]) != expected_representations:
        raise ValueError("This task must pin SpiralNet and all three N3 LAMM seeds")
    spiral = comparable_config(TASK_ROOT / "configs" / "spiralnet128_direct_c4_s42.json")
    lamm = comparable_config(TASK_ROOT / "configs" / "lamm128_direct_c4_s42.json")
    lamm_s2 = comparable_config(TASK_ROOT / "configs" / "lamm128_n3_s2_direct_c4_s42.json")
    lamm_s3 = comparable_config(TASK_ROOT / "configs" / "lamm128_n3_s3_direct_c4_s42.json")
    if not (spiral == lamm == lamm_s2 == lamm_s3):
        raise ValueError("SpiralNet and all N3 LAMM C4 configurations are not matched")
    report = {
        "status": "PASS",
        "registry": str(C.REGISTRY_PATH),
        "checkpoint_metadata": verify_checkpoint_metadata(registry),
        "source_splits": verify_source_splits(registry),
        "matched_c4_configuration": True,
        "models_loaded": False,
        "source_meshes_or_checkpoints_modified": False,
    }
    if args.load_models:
        report["model_checks"] = verify_models(registry)
        report["models_loaded"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
