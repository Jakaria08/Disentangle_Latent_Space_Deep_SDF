#!/usr/bin/env python3
"""Stage 5 step 2: evaluate a converter cocycle (C1) checkpoint on p5_converter_pooled (E-C1, E-C3, E-C5; G5.5).

Every BrainODE-style task (target = each subject's latest visit) is scored twice. A subject's condition always
comes from a rule; learned onsets belong to train subjects and are never used here:

  prefix (primary)      the onset is the midpoint of the conversion window visible in the task's own prefix; a
                        prefix without a conversion keeps its last label (true forecasting)
  oracle (upper bound)  the window observed over all of the subject's visits

Per-subject metrics are evaluate_dynamics.task_rows' (same code path), each row tagged with its trajectory group
and rule. Consistency defects are reported for stable and for AD-converter pairs (oracle onsets). Gate G5.5: the
learned onset parameters hash identically before and after evaluation, and no evaluation table has a learned onset.
Comparators C0 and BrainODE-core are scored with evaluate_dynamics.py on the same view; the stage 5 report
regroups their rows by trajectory group.
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
import converter_line as L
import dynamics_core as D
import evaluate_dynamics as EV
import train_converter_cocycle as TC

GROUPS = ("CN-stable", "AD-stable", "CN->AD", "MCI->AD", "CN->MCI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device) -> tuple[L.ConverterCocycle, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if config.get("method") != "converter_cocycle":
        raise ValueError(f"{checkpoint} is not a converter cocycle checkpoint")
    flow = D.build_transport(config["base_config"], device)
    empty = L.ConditionTable(np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32),
                             np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, bool), len(payload["theta_subjects"]), [])
    model = L.ConverterCocycle(flow, empty, float(config["width_normalized"]), bool(config["variant_spec"]["mci_partial_dose"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), payload


def summarize(rows: pd.DataFrame, draws: int) -> list[dict[str, Any]]:
    output = []
    for (rule, task, variant), frame in rows.groupby(["rule", "task", "variant"], sort=True):
        members = [("all", frame), ("stable", frame[frame["group"].isin(["CN-stable", "AD-stable"])]),
                   ("AD converters", frame[frame["group"].isin(L.AD_CONVERTERS)])] + [(g, frame[frame["group"] == g]) for g in GROUPS]
        for name, part in members:
            if part.empty:
                continue
            record: dict[str, Any] = {"rule": rule, "task": task, "variant": variant, "group": name, "subjects": int(len(part))}
            for metric in EV.TASK_METRICS:
                record[f"{metric}_mean"] = float(part[metric].mean())
                record[f"{metric}_sd"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
            record["euclidean_mm_ci95_low"], record["euclidean_mm_ci95_high"] = EV.bootstrap_ci(part["euclidean_mm"].to_numpy(), draws)
            predicted, observed = float(part["predicted_log_volume_rate"].mean()), float(part["observed_log_volume_rate"].mean())
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
    model, payload = load_model(checkpoint, device)
    config, view = payload["config"], payload["view"]
    representation = config["representation"]
    batch_size = args.batch_size or EV.EVAL_BATCH.get(representation, 64)
    output_dir = bc.require_bulk(args.output_dir or checkpoint.parent.parent / f"evaluation__{view}" / args.split, "evaluation output")
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    theta_before = bc.sha256_bytes(model.theta.detach().cpu().numpy().tobytes())

    registry = D.view_registry(view)
    train = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train, device, registry)
    values = TC.prepared(archive, geometry, device, batch_size)
    raw = D.view_vertices(archive)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_index = {str(s): i for i, s in enumerate(archive["subject_ids"].astype(str))}
    group_of = dict(zip(archive["subject_ids"].astype(str), archive["subject_trajectory_groups"].astype(str)))
    partial = bool(config["variant_spec"]["mci_partial_dose"])
    theta_count = model.theta.numel()

    def before_task_for(rule: str):
        def hook(_task: str, table: pd.DataFrame) -> None:
            prefixes = {}
            for subject, indices in zip(table["subject_id"].astype(str), table["prefix_indices"]):
                index = subject_index[subject]
                prefixes[index] = [int(i) - int(offsets[index]) for i in str(indices).split(";")]
            condition_table = L.build_condition_table(archive, rule, prefixes, partial_dose=partial, theta_count=theta_count)
            if condition_table.learned_onset_visits():
                raise AssertionError("gate G5.5: an evaluation table contains a learned onset")
            model.set_table(condition_table)
        return hook

    predict = lambda src, tgt: model.transport_visits(values["z"][src], values["age"][src], values["age"][tgt], src)
    frames = []
    with torch.no_grad():
        for rule in ("prefix", "oracle"):
            rows = EV.task_rows(model, config["method"], geometry, values, raw, view, args.split, batch_size, predict=predict, before_task=before_task_for(rule))
            frames.append(rows.assign(rule=rule, group=rows["subject_id"].astype(str).map(group_of)))
        rows = pd.concat(frames, ignore_index=True)
        model.set_table(L.build_condition_table(archive, "oracle", partial_dose=partial, theta_count=theta_count))
        pairs = L.pair_rows(archive)
        stable_subjects = {s for s, g in group_of.items() if g in ("CN-stable", "AD-stable")}
        converter_subjects = {s for s, g in group_of.items() if g in L.AD_CONVERTERS}
        statistics = payload["statistics"]
        defects = {name: L.cocycle_defects(model.flow, values, [r for r in pairs if r.subject in members], statistics, batch_size, model.condition)
                   for name, members in (("stable", stable_subjects), ("AD converters", converter_subjects)) if any(r.subject in members for r in pairs)}
    theta_after = bc.sha256_bytes(model.theta.detach().cpu().numpy().tobytes())
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": bc.sha256_file(checkpoint), "checkpoint_epoch": payload.get("epoch"),
        "method": "converter_cocycle", "variant": config["variant"], "representation": representation, "seed": config["seed"],
        "view": view, "split": args.split, "tasks": summarize(rows, args.bootstrap), "consistency_defects": defects,
        "gate_G5_5": {"theta_sha256_before": theta_before, "theta_sha256_after": theta_after, "passed": theta_before == theta_after},
        "test_loaded_during_training": bool(payload.get("test_data_loaded", False)), "git_commit": D.git_commit(),
        "seconds": round(time.time() - started, 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_csv(output_dir / "task_rows.csv", rows)
    bc.atomic_json(output_dir / "summary.json", report)
    primary = [r for r in report["tasks"] if r["rule"] == "prefix" and r["task"] == "one_shot_first" and r["group"] in ("stable", "AD converters")]
    print(" | ".join(f"{r['group']}: {r['euclidean_mm_mean']:.4f} mm (n={r['subjects']})" for r in primary), f"| G5.5 {'PASS' if report['gate_G5_5']['passed'] else 'FAIL'}")
    print(f"WROTE {output_dir / 'summary.json'} ({report['seconds']} s)")
    return 0 if report["gate_G5_5"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
