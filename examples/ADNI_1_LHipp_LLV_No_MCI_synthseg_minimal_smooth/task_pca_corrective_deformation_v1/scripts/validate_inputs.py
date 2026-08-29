#!/usr/bin/env python3
"""Validate data leakage, PCA reproduction, topology, and model invariants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import common
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = common.load_config(args.config)
    device = common.choose_device(args.device)
    output = common.require_bulk_path(
        args.output or Path(config["output_root"]) / "validation" / "input_validation.json",
        "validation report",
    )
    common.makedirs(output.parent)

    contract = common.load_pca_contract(config)
    common.assert_subject_disjoint(contract.rows)
    expected = config["expected"]
    checks = {
        "vertex_count": contract.n_vertices == int(expected["vertices"]),
        "feature_count": contract.mean.size == int(expected["features"]),
        "train_count": contract.summary()["split_counts"]["train"] == int(expected["train_scans"]),
        "val_count": contract.summary()["split_counts"]["val"] == int(expected["val_scans"]),
        "test_count": contract.summary()["split_counts"]["test"] == int(expected["test_scans"]),
        "faces_in_range": int(contract.faces.min()) == 0
        and int(contract.faces.max()) == contract.n_vertices - 1,
        "bulk_output": str(common.require_bulk_path(config["output_root"])).startswith(
            str(common.BULK_ROOT)
        ),
    }

    val = common.load_split_tensors("val", contract, device)
    pca = (
        val.coefficients @ torch.from_numpy(contract.components).to(device)
        + torch.from_numpy(contract.mean).to(device)
    ).view_as(val.vertices_mm)
    per_scan = torch.sqrt((pca - val.vertices_mm).square().flatten(1).mean(dim=1))
    reproduced_val = float(per_scan.mean())
    tolerance = float(expected["pca_reproduction_tolerance_mm"])
    checks["pca_validation_reproduced"] = (
        abs(reproduced_val - float(expected["pca_val_rmse_mm"])) <= tolerance
    )

    model = build_model(config, contract, device)
    coefficients = val.coefficients[:2]
    with torch.no_grad():
        details = model.decode_with_details(coefficients)
        zero_delta_max = float(details["delta"].abs().max())
        pca_difference_max = float((details["prediction"] - details["pca_mesh"]).abs().max())
        reencoded = model.pca_encode(details["prediction"])
        latent_roundtrip_max = float((reencoded - coefficients).abs().max())
    checks["zero_initialized_delta"] = zero_delta_max <= 1e-9
    checks["epoch_zero_is_exact_pca"] = pca_difference_max <= 1e-9
    checks["pca_latent_roundtrip"] = latent_roundtrip_max <= 2e-4

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "pca_validation_rmse_mm": reproduced_val,
        "expected_pca_validation_rmse_mm": float(expected["pca_val_rmse_mm"]),
        "zero_initialized_delta_max_abs_mm": zero_delta_max,
        "epoch_zero_prediction_minus_pca_max_abs_mm": pca_difference_max,
        "pca_latent_roundtrip_max_abs": latent_roundtrip_max,
        "parameters_trainable": model.num_parameters(),
        "device": str(device),
        "data_contract": contract.summary(),
        "test_vertices_loaded_for_model_validation": False,
    }
    common.atomic_write_json(output, report)
    if report["status"] != "pass":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Validation failed: {failed}; report={output}")
    print(
        f"[pass] PCA val={reproduced_val:.9f} mm; params={model.num_parameters():,}; "
        f"report={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
