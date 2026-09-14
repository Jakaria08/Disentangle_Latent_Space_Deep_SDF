#!/usr/bin/env python3
"""Evaluate one trained dynamics checkpoint on one protocol view split.

Two families of results are written:

1. **Pair metrics (August-compatible).** The same categories and quantities as
   task3/August evaluate.py (all/adjacent/nonadjacent/first-last pairs, forward and backward,
   horizon bins, cocycle defects, representation floor, subject bootstrap). They are computed
   with task3's own functions, so numbers are directly comparable with every earlier summary.

2. **BrainODE-style tasks** (configs/evaluation_tasks.json, target = each subject's latest
   visit): one_shot_first, one_shot_prev, four_shot, all_prior_k. Every model is scored with
   BrainODE's aggregation - transport each observed visit to the target time, decode, average
   the k predicted meshes. The Latent ODE is also scored natively, encoding all k observations
   at once. Metrics per subject: per-vertex Euclidean and coordinate MAE/RMSE against the real
   mesh and against the decoded target, relative volume error, and predicted vs observed
   log-volume rate. Summaries give mean, SD, subject-bootstrap 95% CI, and AD capture.

Normalization guard: the evaluation view must share the training view's train split
statistics (P1 external views do, by construction). Outputs never overwrite silently.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import dynamics_core as D

# The Adaptive decoder's dynamic pooling needs ~4 GB extra at batch 256 and ran out of memory with
# several evaluations sharing a GPU with training (stage 3, 2026-09-13), so it gets a small batch.
EVAL_BATCH = {"pca128": 256, "spiralnet128": 96, "adaptive128": 32, "lamm128": 96, "pca128_pooled": 256}
TASK_METRICS = ("euclidean_mm", "coordinate_mae_mm", "coordinate_rmse_mm", "euclidean_vs_decoded_target_mm",
                "coordinate_mae_vs_decoded_target_mm", "volume_relative_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view", required=True, help="Evaluation view.")
    parser.add_argument("--trained-view", default=None, help="Training view; read from the checkpoint when it records one.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Inference batch size. Default depends on the representation (EVAL_BATCH); results do not depend on it.")
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--condition-override", type=int, choices=(0, 1), default=None,
                        help="Feed every subject this condition instead of its label (e.g. CALSNIC ALS scored as control). "
                             "Groups and observed rates still use the true diagnosis.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalization_guard(trained_view: str, eval_view: str, representation: str) -> None:
    if trained_view == eval_view:
        return
    a = bc.load_npz(D.view_root(trained_view) / "representations" / representation / "train_subject_sequences_128.npz")
    b = bc.load_npz(D.view_root(eval_view) / "representations" / representation / "train_subject_sequences_128.npz")
    same_codes = np.array_equal(a["train_latent_mean_128"], b["train_latent_mean_128"]) and np.array_equal(a["train_latent_std_128"], b["train_latent_std_128"])
    na = bc.read_json(D.view_root(trained_view) / "view_manifest.json")["normalization"]
    nb = bc.read_json(D.view_root(eval_view) / "view_manifest.json")["normalization"]
    if not same_codes or na["age_min_years"] != nb["age_min_years"] or na["age_max_years"] != nb["age_max_years"]:
        raise ValueError(f"{eval_view} does not share {trained_view}'s train normalization; evaluating there would be invalid")


def bootstrap_ci(values: np.ndarray, draws: int, seed: int = 12345) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if draws <= 0 or len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


# --------------------------------------------------------------------------------------
# pair metrics
# --------------------------------------------------------------------------------------


def pair_metrics(transport, geometry, values, archive, train_archive, raw, registry, split, payload, batch_size, draws):
    parts = D.core()
    C, O, E = parts["C"], parts["O"], parts["E"]
    method = payload["config"]["method"]
    if D.transport_requires_context(method):
        # Same metrics; the August coboundary objective additionally passes each subject's first visit as context.
        O = D.coboundary_modules(method)["objective"]
    forward = C.load_pairs(split, archive, registry)
    first_last = C.first_last_pairs(archive)
    backward = E.reverse_rows(forward)
    categories = {
        "all_forward": forward,
        "adjacent_forward": [r for r in forward if r.pair_type == "adjacent"],
        "nonadjacent_forward": [r for r in forward if r.pair_type == "nonadjacent"],
        "first_last_forward": first_last,
        "all_backward": backward,
        "adjacent_backward": [r for r in backward if r.pair_type.endswith("adjacent") and "nonadjacent" not in r.pair_type],
        "nonadjacent_backward": [r for r in backward if r.pair_type.endswith("nonadjacent")],
        "first_last_backward": E.reverse_rows(first_last),
    }
    categories.update({f"horizon_{k}_forward": v for k, v in E.horizon_groups(forward).items() if v})
    categories.update({f"horizon_{k}_backward": v for k, v in E.horizon_groups(backward).items() if v})
    results = {name: O.evaluate_pairs(transport, geometry, values, rows, raw, batch_size, include_rows=(name == "all_forward"))
               for name, rows in categories.items() if rows}
    rows = results["all_forward"].pop("row_metrics", [])
    statistics = payload.get("statistics")
    if statistics is None:
        train_pairs = C.load_pairs("train", train_archive, registry)
        z = train_archive["visit_latent_standardized_128"]
        displacement = np.asarray([np.sqrt(np.mean((z[r.target] - z[r.source]) ** 2)) for r in train_pairs])
        statistics = {"normalization_scales": {"displacement": float(max(np.median(displacement), 1e-6))}}
    defects = O.cocycle_defects(transport, values, forward, statistics, batch_size)
    floor = E.representation_floor(values, raw, geometry, batch_size)
    return {"pair_metrics": results, "consistency_defects": defects, "representation_floor": floor,
            "subject_bootstrap": E.subject_bootstrap(rows, draws)}, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# BrainODE-style tasks
# --------------------------------------------------------------------------------------


@torch.no_grad()
def task_rows(transport, method, geometry, values, raw, view, split, batch_size, predict=None, before_task=None) -> pd.DataFrame:
    """Per-subject task metrics. ``predict(source_visits, target_visits)`` replaces the transport call and
    ``before_task(task, table)`` runs before each task (stage 5 converter evaluation sets per-task conditions)."""
    root = D.view_root(view)
    tasks = bc.load_tasks()["tasks"]
    device = values["z"].device
    z, age, label = values["z"], values["age"], values["label"]
    frames = []
    for task in tasks:
        table = pd.read_csv(root / "tasks" / f"{split}_{task}.csv",
                            dtype={"subject_id": str, "prefix_indices": str, "prefix_scan_ids": str, "prefix_years": str})
        if table.empty:
            continue
        if before_task is not None:
            before_task(task, table)
        prefixes = [[int(i) for i in value.split(";")] for value in table["prefix_indices"]]
        targets = torch.as_tensor(table["target_index"].to_numpy(), device=device)
        real_target = torch.from_numpy(np.asarray(raw[table["target_index"].to_numpy()], dtype=np.float32)).to(device)
        decoded_target = torch.cat([geometry.vertices(z[targets[i : i + batch_size]]) for i in range(0, len(targets), batch_size)])
        last_real = torch.from_numpy(np.asarray(raw[[p[-1] for p in prefixes]], dtype=np.float32)).to(device)
        target_volume = geometry.volume_from_vertices(real_target)
        last_volume = geometry.volume_from_vertices(last_real)
        years = torch.as_tensor(table["horizon_from_last_prefix_years"].to_numpy(dtype=np.float32), device=device).clamp_min(1e-6)

        variants: dict[str, torch.Tensor] = {}
        flat_subject = np.concatenate([[row] * len(p) for row, p in enumerate(prefixes)])
        flat_source = torch.as_tensor(np.concatenate(prefixes), device=device)
        flat_target = targets[torch.as_tensor(flat_subject, device=device)]
        summed = torch.zeros_like(real_target)
        for start in range(0, len(flat_source), batch_size):
            src, tgt = flat_source[start : start + batch_size], flat_target[start : start + batch_size]
            predicted = predict(src, tgt) if predict is not None else D.transport_call(transport, method, values, src, tgt)
            summed.index_add_(0, torch.as_tensor(flat_subject[start : start + batch_size], device=device), geometry.vertices(predicted))
        counts = torch.as_tensor([len(p) for p in prefixes], device=device, dtype=summed.dtype)
        variants["averaged"] = summed / counts[:, None, None]
        if method in D.LATENT_ODE_METHODS:
            native = torch.zeros_like(real_target)
            for k in sorted({len(p) for p in prefixes}):
                rows = np.asarray([r for r, p in enumerate(prefixes) if len(p) == k])
                for start in range(0, len(rows), batch_size):
                    chunk = rows[start : start + batch_size]
                    index = torch.as_tensor(np.stack([prefixes[r] for r in chunk]), device=device)
                    obs, times = z[index], age[index]
                    mask = torch.ones(index.shape, dtype=torch.bool, device=device)
                    predicted = transport.predict(obs, times, mask, label[index[:, 0]], age[targets[torch.as_tensor(chunk, device=device)]][:, None])[:, 0, :]
                    native[torch.as_tensor(chunk, device=device)] = geometry.vertices(predicted)
            variants["native"] = native

        for variant, predicted in variants.items():
            delta = predicted - real_target
            floor_delta = predicted - decoded_target
            volume = geometry.volume_from_vertices(predicted)
            frame = table.loc[:, ["task", "subject_id", "cohort", "diagnosis", "diagnosis_source", "label_ad", "n_visits", "k",
                                  "horizon_from_last_prefix_years", "horizon_from_first_prefix_years"]].copy()
            frame.insert(1, "variant", variant)
            metrics = {
                "euclidean_mm": torch.linalg.vector_norm(delta, dim=2).mean(dim=1),
                "coordinate_mae_mm": delta.abs().mean(dim=(1, 2)),
                "coordinate_rmse_mm": delta.square().mean(dim=(1, 2)).sqrt(),
                "euclidean_vs_decoded_target_mm": torch.linalg.vector_norm(floor_delta, dim=2).mean(dim=1),
                "coordinate_mae_vs_decoded_target_mm": floor_delta.abs().mean(dim=(1, 2)),
                "volume_relative_error": (volume - target_volume).abs() / target_volume,
                "predicted_log_volume_rate": (volume.log() - last_volume.log()) / years,
                "observed_log_volume_rate": (target_volume.log() - last_volume.log()) / years,
            }
            for name, tensor in metrics.items():
                frame[name] = tensor.double().cpu().numpy()
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_tasks(rows: pd.DataFrame, draws: int) -> list[dict[str, Any]]:
    output = []
    if rows.empty:
        return output
    for (task, variant), frame in rows.groupby(["task", "variant"], sort=True):
        groups = [("overall", frame)]
        groups += [(f"label={d}", g) for d, g in frame.groupby("diagnosis")]
        groups += [(f"cohort={c}", g) for c, g in frame.groupby("cohort")]
        groups += [(f"cohort={c}|label={d}", g) for (c, d), g in frame.groupby(["cohort", "diagnosis"])]
        for name, members in groups:
            record: dict[str, Any] = {"task": task, "variant": variant, "group": name, "subjects": int(len(members))}
            for metric in TASK_METRICS:
                values = members[metric].to_numpy()
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for metric in ("euclidean_mm", "coordinate_mae_mm"):
                low, high = bootstrap_ci(members[metric].to_numpy(), draws)
                record[f"{metric}_ci95_low"], record[f"{metric}_ci95_high"] = low, high
            predicted = float(members["predicted_log_volume_rate"].mean())
            observed = float(members["observed_log_volume_rate"].mean())
            record.update({"predicted_log_volume_rate_mean": predicted, "observed_log_volume_rate_mean": observed,
                           "rate_capture_ratio": predicted / observed if abs(observed) > 1e-9 else float("nan")})
            output.append(record)
    return output


def main() -> int:
    args = parse_args()
    started = time.time()
    device = D.device(args.device)
    parts = D.core()
    C, O = parts["C"], parts["O"]
    checkpoint = args.checkpoint.resolve()
    transport, config, payload = D.load_trained_transport(checkpoint, device)
    method, representation = config["method"], config["representation"]
    args.batch_size = args.batch_size or EVAL_BATCH.get(representation, 64)
    trained_view = args.trained_view or payload.get("view")
    if trained_view is None:
        raise ValueError("checkpoint does not record its training view; pass --trained-view")
    normalization_guard(trained_view, args.view, representation)
    output_dir = args.output_dir or checkpoint.parent.parent / f"{args.name}__{args.view}" / args.split
    output_dir = bc.require_bulk(output_dir, "evaluation output")
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    registry = D.view_registry(args.view)
    train_archive = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    values = C.values_on_device(archive, device)
    if args.condition_override is not None:
        # Only the condition fed to the model changes; pair rows and task rows keep the true
        # diagnosis for grouping, so observed rates and group means stay those of the real labels.
        values["label"] = torch.full_like(values["label"], float(args.condition_override))
    O.attach_reference_geometry(values, geometry, args.batch_size)
    raw = D.view_vertices(archive)

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": bc.sha256_file(checkpoint), "checkpoint_epoch": payload.get("epoch"),
        "method": method, "representation": representation, "trained_view": trained_view, "evaluation_view": args.view,
        "split": args.split, "condition_override": args.condition_override,
        "test_loaded_during_training": bool(payload.get("test_data_loaded", False)),
        "git_commit": D.git_commit(),
    }
    if not args.skip_pairs:
        pairs, pair_rows = pair_metrics(transport, geometry, values, archive, train_archive, raw, registry, args.split, payload, args.batch_size, args.bootstrap)
        report.update(pairs)
    rows = task_rows(transport, method, geometry, values, raw, args.view, args.split, args.batch_size)
    report["tasks"] = summarize_tasks(rows, args.bootstrap)
    report["seconds"] = round(time.time() - started, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_csv(output_dir / "task_rows.csv", rows)
    if not args.skip_pairs:
        bc.atomic_csv(output_dir / "pair_rows_all_forward.csv", pair_rows)
    bc.atomic_json(output_dir / "summary.json", report)
    primary = [r for r in report["tasks"] if r["task"] == "one_shot_first" and r["variant"] == "averaged" and r["group"] == "overall"]
    if primary:
        print(f"one_shot_first: Euclidean {primary[0]['euclidean_mm_mean']:.4f} mm, MAE {primary[0]['coordinate_mae_mm_mean']:.4f} mm, n={primary[0]['subjects']}")
    print(f"WROTE {output_dir / 'summary.json'} ({report['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
