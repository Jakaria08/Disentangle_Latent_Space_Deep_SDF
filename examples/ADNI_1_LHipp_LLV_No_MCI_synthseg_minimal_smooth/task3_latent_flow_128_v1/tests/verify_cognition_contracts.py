#!/usr/bin/env python3
"""Synthetic tests for PCA128 voxel-cognition scientific contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

import brainode_cognition as B  # noqa: E402
from train_brainode_cognition import validate_config  # noqa: E402


def cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float32)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return vertices, faces


def main() -> int:
    config = json.loads((TASK_ROOT / "configs" / "pca128_brainode_cognition_s42.json").read_text())
    validate_config(config)
    assert config["representation"] == "pca128"
    assert config["ode_model"]["latent_dim"] == 128
    assert not config["scientific_contract"]["mci_used"]
    assert not config["scientific_contract"]["converter_supervision"]
    assert not config["scientific_contract"]["pseudo_cognition_sampling"]

    vertices, faces = cube()
    training = np.stack((vertices, vertices * 1.1))
    grid = B.VoxelGrid.from_training_vertices(training, resolution=16, padding_voxels=2)
    mask, quality = grid.voxelize(vertices, faces)
    assert mask.shape == (16, 16, 16)
    assert int(mask.sum()) > 0
    assert quality["clipped_local_voxels"] == 0.0
    packed = B.pack_masks(mask[None])
    restored = B.unpack_masks(packed, 16)
    assert np.array_equal(mask[None], restored)

    subjects = np.asarray(["cn1", "cn1", "cn2", "ad1", "ad1", "ad1", "ad2"])
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1])
    weights = B.subject_balanced_weights(subjects, labels)
    totals = {subject: float(weights[subjects == subject].sum()) for subject in np.unique(subjects)}
    assert np.isclose(totals["cn1"], totals["cn2"])
    assert np.isclose(totals["ad1"], totals["ad2"])
    assert np.isclose(totals["cn1"], totals["ad1"])

    model = B.VoxelCognitionCNN(base_channels=4, dropout=0.0).eval()
    output = model(torch.from_numpy(np.stack((mask, mask))))
    assert output.shape == (2,) and torch.isfinite(output).all()
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = B.scan_and_subject_metrics(
        np.asarray([0, 0, 1, 1]), probabilities, np.asarray(["a", "b", "c", "d"])
    )
    assert metrics["subject"]["auroc"] == 1.0
    print("COGNITION CONTRACTS PASSED: train-bounded solid voxels, loss balancing, shape-only CNN, no MCI/converter/pseudo targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
