"""Original-cohort contract adapter for the shared DeepSDF calibrator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARED_SCRIPTS = ROOT.parent.parent / "ADNI_1_L_No_MCI_large_strict_left" / "task3_longitudinal_prediction" / "inr_anchored_ad_residual_cocycle_qc_v1" / "scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from inr_anchored_common import ensure_paths_exist  # noqa: E402


def experiment_dir() -> Path:
    return ROOT


def load_metadata_and_latents(config: Mapping[str, Any], base: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    paths = ensure_paths_exist(config, base)
    frame = pd.read_csv(paths["metadata"])
    required = {
        "split", "scan_id", "subject_id", "visit_order", "diagnosis", "label_ad",
        "continuous_age_norm", "continuous_age_years", "mesh_path", "sdf_npz_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Original metadata missing columns: {missing}")
    frame = frame.copy()
    for column in ("scan_id", "subject_id", "split"):
        frame[column] = frame[column].astype(str)
    frame["visit_order"] = frame["visit_order"].astype(int)
    frame["label_ad"] = frame["label_ad"].astype(int)
    frame["continuous_age_norm"] = frame["continuous_age_norm"].astype(float)
    frame["continuous_age_years"] = frame["continuous_age_years"].astype(float)
    if set(frame.split.unique()) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test, got {sorted(frame.split.unique())}")
    if set(frame.diagnosis.unique()) != {"CN", "AD"} or set(frame.label_ad.unique()) != {0, 1}:
        raise ValueError("Expected fixed CN/AD labels 0/1.")
    latent_map: dict[str, np.ndarray] = {}
    latent_size = int(config["LatentSize"])
    for split, path_value in config["LatentArchives"].items():
        archive = np.load((base or ROOT) / path_value, allow_pickle=False)
        scan_ids, latents = archive["scan_ids"], archive["latents"]
        if latents.ndim != 2 or latents.shape[1] != latent_size or len(scan_ids) != len(latents):
            raise ValueError(f"Invalid {split} latent archive: {path_value}")
        for scan_id, latent in zip(scan_ids, latents):
            key = Path(scan_id.decode() if isinstance(scan_id, bytes) else str(scan_id)).stem
            if key in latent_map:
                raise ValueError(f"Duplicate latent scan ID {key}")
            latent_map[key] = np.asarray(latent, dtype=np.float32)
    if set(frame.scan_id) != set(latent_map):
        raise ValueError("Original metadata and frozen latent archive IDs differ.")
    frame = frame.sort_values(["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]).reset_index(drop=True)
    frame.insert(0, "global_index", np.arange(len(frame), dtype=np.int64))
    return frame, np.stack([latent_map[scan_id] for scan_id in frame.scan_id], axis=0)
