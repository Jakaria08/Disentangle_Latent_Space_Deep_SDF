#!/usr/bin/env python3
"""Contract checks for the PCA-corrective latent and frozen decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

import common as C  # noqa: E402
import train_c4  # noqa: E402


def main() -> int:
    registry = C.load_registry()
    assert C.output_root(registry).is_relative_to(C.BULK_ROOT)
    config = C.read_json(TASK_ROOT / "configs" / "pca_corrective128_direct_c4_s42.json")
    train_c4.validate_config(config)
    contract = C.load_corrective_contract(registry)
    assert contract.latent_dim == 128
    assert contract.n_vertices == 2746
    assert contract.faces.shape == (5488, 3)
    assert contract.summary()["split_counts"] == {"train": 2037, "val": 269, "test": 277}

    model, checkpoint, loaded_contract = C.load_corrective_model(torch.device("cpu"), registry)
    assert checkpoint["best_epoch"] == 5
    assert loaded_contract.summary() == contract.summary()
    coefficients = torch.from_numpy(
        C.corrective_codes(np.asarray(C.cached_vertices("val", registry)[:3]), contract)
    )
    details = model.decode_with_details(coefficients)
    assert details["prediction"].shape == (3, 2746, 3)
    assert torch.isfinite(details["prediction"]).all()
    pca_roundtrip = model.pca_encode(details["pca_mesh"])
    corrected_roundtrip = model.pca_encode(details["prediction"])
    assert float(torch.max(torch.abs(pca_roundtrip - coefficients))) < 2.0e-4
    assert float(torch.max(torch.abs(corrected_roundtrip - coefficients))) < 2.0e-4
    projected = details["delta"].reshape(3, -1) @ model.pca_components.T
    assert float(torch.max(torch.abs(projected))) < 2.0e-4
    assert not any(parameter.requires_grad for parameter in model.parameters())

    train_mean = contract.coefficient_mean
    train_std = contract.coefficient_std
    standardized = ((coefficients.numpy() - train_mean) / train_std).astype(np.float32)
    archive = {
        "train_latent_mean_128": train_mean,
        "train_latent_std_128": train_std,
    }
    geometry = C.build_geometry("pca_corrective128", archive, torch.device("cpu"), registry)
    latent = torch.from_numpy(standardized).requires_grad_(True)
    volume = geometry.volume(latent).mean()
    volume.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert float(torch.linalg.vector_norm(latent.grad)) > 0.0
    assert not any(parameter.grad is not None for parameter in geometry.parameters())

    print("CONTRACT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
