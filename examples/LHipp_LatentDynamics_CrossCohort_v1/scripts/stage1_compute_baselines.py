#!/usr/bin/env python3
"""Stage 1d: training-free baselines on every view, task and representation.

Baselines (one prediction of each subject's latest shape):

  nochange_raw          the most recent observed mesh itself (representation-free)
  nochange_decoded      the most recent observed code, decoded
  linear_raw            least-squares line through the observed meshes (k >= 2 only)
  linear_decoded        least-squares line through the observed codes, decoded (k >= 2 only)
  population_drift      most recent code + mean train velocity of the subject's label x horizon

Population drift is fitted on the view's train rows only. Metrics compare against the real
target mesh (end-to-end) and, for decoded baselines, also against the decoded target code,
which removes the representation floor. Volumes are signed mesh volumes from the shared faces.

Run with the pytorch_geo environment; decoding uses GPU 0 or 2 only.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import stage1_encode_cohort_latents as encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--representations", nargs="+", default=list(bc.REPRESENTATIONS))
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    v0, v1, v2 = v[:, faces[:, 0]], v[:, faces[:, 1]], v[:, faces[:, 2]]
    return np.abs(np.einsum("nfi,nfi->nf", v0, np.cross(v1, v2)).sum(axis=1) / 6.0)


class VertexStore:
    """Real meshes by cohort-qualified scan key, from the stage 1b caches."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.index: dict[str, dict[str, int]] = {}

    def get(self, scan_keys) -> np.ndarray:
        out = []
        for key in scan_keys:
            cohort = key.split(bc.ID_SEPARATOR, 1)[0]
            if cohort not in self.arrays:
                self.arrays[cohort] = np.load(bc.STAGE1_ROOT / "vertices" / f"{cohort}_vertices_mm.npy", mmap_mode="r")
                keys = bc.read_json(bc.STAGE1_ROOT / "vertices" / f"{cohort}_vertices_scans.json")["scan_keys"]
                self.index[cohort] = {k: i for i, k in enumerate(keys)}
            out.append(self.arrays[cohort][self.index[cohort][key]])
        return np.stack(out).astype(np.float32)


def row_metrics(predicted: np.ndarray, target: np.ndarray, decoded_target: np.ndarray | None) -> dict[str, np.ndarray]:
    delta = predicted.astype(np.float64) - target.astype(np.float64)
    metrics = {
        "euclidean_mm": np.linalg.norm(delta, axis=2).mean(axis=1),
        "coordinate_mae_mm": np.abs(delta).mean(axis=(1, 2)),
        "coordinate_rmse_mm": np.sqrt(np.square(delta).mean(axis=(1, 2))),
    }
    if decoded_target is not None:
        floor_delta = predicted.astype(np.float64) - decoded_target.astype(np.float64)
        metrics["euclidean_vs_decoded_target_mm"] = np.linalg.norm(floor_delta, axis=2).mean(axis=1)
        metrics["coordinate_mae_vs_decoded_target_mm"] = np.abs(floor_delta).mean(axis=(1, 2))
    return metrics


def evaluate_view_split(view: str, split: str, representations, encoders, store: VertexStore, faces, tasks, batch_size) -> list[pd.DataFrame]:
    root = bc.STAGE1_ROOT / "views" / view
    frames = []
    for task in tasks["tasks"]:
        # Prefix columns hold ';'-joined lists; a one-visit prefix would otherwise parse as a number.
        table = pd.read_csv(root / "tasks" / f"{split}_{task}.csv",
                            dtype={"subject_id": str, "prefix_indices": str, "prefix_scan_ids": str, "prefix_years": str})
        if table.empty:
            continue
        prefix_idx = [[int(i) for i in value.split(";")] for value in table["prefix_indices"]]
        prefix_years = [[float(t) for t in value.split(";")] for value in table["prefix_years"]]
        target_years = table["target_years"].to_numpy(dtype=np.float64)
        last_years = np.asarray([years[-1] for years in prefix_years])
        horizon = target_years - last_years
        dataset = bc.load_npz(root / "dataset" / f"{split}_subject_sequences.npz")
        scan_keys = dataset["visit_scan_ids"].astype(str)
        target_mesh = store.get(scan_keys[table["target_index"].to_numpy()])
        last_mesh = store.get([scan_keys[idx[-1]] for idx in prefix_idx])
        target_volume = mesh_volume(target_mesh, faces)
        last_volume = mesh_volume(last_mesh, faces)
        base = table.loc[:, ["subject_id", "cohort", "diagnosis", "diagnosis_source", "label_ad", "n_visits", "k",
                             "horizon_from_last_prefix_years", "horizon_from_first_prefix_years"]].copy()
        base.insert(0, "task", task)
        base.insert(0, "split", split)
        base.insert(0, "view", view)

        predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]] = {}
        predictions[("raw_mesh", "nochange_raw")] = (last_mesh, None)
        multi = np.asarray([len(idx) >= 2 for idx in prefix_idx])
        if multi.any():
            linear = np.full_like(target_mesh, np.nan)
            for row in np.flatnonzero(multi):
                meshes = store.get([scan_keys[i] for i in prefix_idx[row]]).reshape(len(prefix_idx[row]), -1)
                linear[row] = bc.linear_extrapolation(prefix_years[row], meshes, target_years[row]).reshape(-1, 3)
            predictions[("raw_mesh", "linear_raw")] = (linear, None)

        for rep in representations:
            archive = bc.load_npz(root / "representations" / rep / f"{split}_subject_sequences_128.npz")
            train = bc.load_npz(root / "representations" / rep / "train_subject_sequences_128.npz")
            codes = archive["visit_latent_raw_128"].astype(np.float64)
            drift = bc.population_drift(
                train["visit_latent_raw_128"].astype(np.float64), train["visit_subject_ids"].astype(str),
                train["visit_time_years_from_baseline"], train["visit_label_ad"],
            )
            last_codes = np.stack([codes[idx[-1]] for idx in prefix_idx])
            labels = table["label_ad"].to_numpy(dtype=np.int64)
            drift_codes = last_codes + np.stack([drift[int(label)] for label in labels]) * horizon[:, None]
            code_sets = {"nochange_decoded": last_codes, "population_drift": drift_codes}
            if multi.any():
                linear_codes = np.stack([
                    bc.linear_extrapolation(prefix_years[r], codes[prefix_idx[r]], target_years[r]) if multi[r] else last_codes[r]
                    for r in range(len(table))
                ])
                code_sets["linear_decoded"] = linear_codes
            decoded_target = encoders[rep].decode(codes[table["target_index"].to_numpy()].astype(np.float32), batch_size)
            for baseline, chosen in code_sets.items():
                decoded = encoders[rep].decode(chosen.astype(np.float32), batch_size)
                if baseline == "linear_decoded":
                    decoded[~multi] = np.nan
                predictions[(rep, baseline)] = (decoded, decoded_target)

        for (rep, baseline), (predicted, decoded_target) in predictions.items():
            valid = ~np.isnan(predicted.reshape(len(predicted), -1)).any(axis=1)
            if not valid.any():
                continue
            metrics = row_metrics(predicted[valid], target_mesh[valid], None if decoded_target is None else decoded_target[valid])
            volume = mesh_volume(predicted[valid], faces)
            frame = base.loc[valid].copy()
            frame.insert(4, "representation", rep)
            frame.insert(5, "baseline", baseline)
            for name, values in metrics.items():
                frame[name] = values
            years = np.maximum(horizon[valid], 1.0e-6)
            frame["volume_relative_error"] = np.abs(volume - target_volume[valid]) / target_volume[valid]
            frame["predicted_log_volume_rate"] = (np.log(volume) - np.log(last_volume[valid])) / years
            frame["observed_log_volume_rate"] = (np.log(target_volume[valid]) - np.log(last_volume[valid])) / years
            frames.append(frame)
    return frames


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["view", "split", "task", "representation", "baseline"]
    out = []
    for group_keys, frame in rows.groupby(keys, sort=True):
        for group_name, members in [("overall", frame), *[(f"cohort={c}", g) for c, g in frame.groupby("cohort")],
                                    *[(f"label={d}", g) for d, g in frame.groupby("diagnosis")]]:
            record = dict(zip(keys, group_keys)) | {"group": group_name, "subjects": int(len(members))}
            for metric in ("euclidean_mm", "coordinate_mae_mm", "coordinate_rmse_mm", "euclidean_vs_decoded_target_mm", "volume_relative_error"):
                if metric in members and members[metric].notna().any():
                    record[f"{metric}_mean"] = float(members[metric].mean())
                    record[f"{metric}_sd"] = float(members[metric].std(ddof=1)) if len(members) > 1 else 0.0
            record["predicted_log_volume_rate_mean"] = float(members["predicted_log_volume_rate"].mean())
            record["observed_log_volume_rate_mean"] = float(members["observed_log_volume_rate"].mean())
            out.append(record)
    return pd.DataFrame(out)


def main() -> int:
    args = parse_args()
    bc.require_allowed_gpu(args.device)
    device = torch.device(args.device)
    registry = bc.load_registry()
    tasks = bc.load_tasks()
    views = args.views or list(bc.expand_views())
    encoding.prepare_imports(registry)
    encoders = {rep: encoding.Encoder(rep, registry, device) for rep in args.representations}
    faces = np.load(registry["faces_path"]).astype(np.int64)
    store = VertexStore()
    destination = bc.require_bulk(bc.STAGE1_ROOT / "baselines")
    all_rows = []
    for view in views:
        for split in args.splits:
            frames = evaluate_view_split(view, split, args.representations, encoders, store, faces, tasks, args.batch_size)
            if not frames:
                continue
            rows = pd.concat(frames, ignore_index=True)
            bc.atomic_csv(destination / view / f"{split}_baseline_rows.csv", rows)
            all_rows.append(rows)
            primary = rows.loc[(rows["task"] == "one_shot_first") & (rows["baseline"].isin(["nochange_raw", "nochange_decoded"]))]
            brief = primary.groupby(["representation", "baseline"])["euclidean_mm"].mean().round(4).to_dict()
            print(f"[{view}/{split}] rows={len(rows)} one_shot_first no-change Euclidean {brief}", flush=True)
    summary = summarize(pd.concat(all_rows, ignore_index=True))
    bc.atomic_csv(destination / "baseline_summary.csv", summary)
    print(json.dumps({"views": len(views), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
