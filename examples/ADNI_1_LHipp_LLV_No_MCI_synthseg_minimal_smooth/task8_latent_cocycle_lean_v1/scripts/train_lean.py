#!/usr/bin/env python3
"""Step-based trainer for the latent cocycle flow.

Differences from the published direct_c4 trainer, all of them deliberate:

  * an "epoch" is a fixed number of OPTIMIZER STEPS, not a fixed number of samples, so the
    unit means the same thing for every representation (the baseline held
    samples_per_epoch=4096 while batch_size varied 96/24/8, making one epoch 43, 171 or 512
    steps and putting the true optimum inside epoch 1 for two of the three runs);
  * validation runs on a step interval and the best checkpoint is selected per step, so an
    optimum that falls mid-epoch can actually be captured;
  * both warm-up ramps are expressed in steps;
  * every optimizer step is logged.

Evaluation, the selection score and the feasibility gates are imported unchanged from the
baseline objective, so the reported score is directly comparable to the published numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lean_objective as L  # noqa: E402
from _bootstrap import core  # noqa: E402
from lean_models import build_flow  # noqa: E402

C, O = core()


class PairDataset(Dataset):
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def infinite_pairs(rows: list, batch_size: int, seed: int, epoch: int):
    loader = DataLoader(
        PairDataset(rows),
        batch_size=batch_size,
        sampler=O.balanced_pair_sampler(rows, seed, epoch, batch_size * 4096),
        collate_fn=C.collate_pairs,
        num_workers=0,
    )
    return iter(loader)


@torch.no_grad()
def validate(flow, geometry, values, archive, pair_rows, raw_vertices, statistics, selection, batch_size, limit):
    flow.eval()
    rows = O.balanced_subset(pair_rows, limit)
    first_last = O.balanced_first_last_subset(C.first_last_pairs(archive), limit)
    all_metrics = O.evaluate_pairs(flow, geometry, values, rows, raw_vertices, batch_size)
    first_last_metrics = O.evaluate_pairs(flow, geometry, values, first_last, raw_vertices, batch_size)
    defects = O.cocycle_defects(flow, values, rows, statistics, batch_size)
    score, feasible, ratios = O.validation_score(all_metrics, first_last_metrics, defects, selection)
    flow.train()
    groups = {}
    for name in ("CN", "AD", "overall"):
        source = first_last_metrics["groups"]
        if name in source:
            groups[name] = {
                key: source[name][key]
                for key in (
                    "coordinate_mean",
                    "euclidean_mean",
                    "nochange_coordinate_mean",
                    "nochange_euclidean_mean",
                    "volume_relative_mean",
                    "nochange_volume_relative_mean",
                    "predicted_signed_rate_mean",
                    "observed_signed_rate_mean",
                    "end_to_end_euclidean_mean",
                )
                if key in source[name]
            }
    return {
        "score": float(score),
        "feasible": bool(feasible),
        "ratios": ratios,
        "defects": defects,
        "first_last_groups": groups,
        "all_pairs_overall": {
            key: all_metrics["groups"]["overall"][key]
            for key in (
                "coordinate_mean",
                "euclidean_mean",
                "nochange_coordinate_mean",
                "nochange_euclidean_mean",
                "volume_relative_mean",
                "nochange_volume_relative_mean",
            )
        },
    }


def main() -> int:
    args = parse_args()
    config = C.read_json(C.resolve_path(args.config))
    registry = C.load_registry()

    representation = str(config["representation"])
    loss_set = str(config["loss_set"])
    if loss_set not in L.LOSS_SETS:
        raise ValueError(f"Unknown loss_set {loss_set!r}; expected one of {sorted(L.LOSS_SETS)}")
    active = L.LOSS_SETS[loss_set]
    if active == ("baseline",):
        raise ValueError("full13 is provided by the baseline trainer; not implemented here")

    training = dict(config["train"])
    seed = int(training["seed"])
    device = C.choose_device(args.device)
    C.set_seed(seed)

    train_archive = C.load_archive(representation, "train", registry)
    val_archive = C.load_archive(representation, "val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Train/validation subject leakage")

    geometry = C.build_geometry(representation, train_archive, device, registry)
    train_values = C.values_on_device(train_archive, device)
    val_values = C.values_on_device(val_archive, device)
    decoder_batch = int(training["decoder_batch_size"])
    O.attach_reference_geometry(train_values, geometry, decoder_batch)
    O.attach_reference_geometry(val_values, geometry, decoder_batch)
    train_pairs = C.load_pairs("train", train_archive, registry)
    val_pairs = C.load_pairs("val", val_archive, registry)
    smoke_limit = 16 if args.smoke else None
    statistics = O.training_statistics(train_values, train_pairs, train_archive, smoke_limit)

    flow = build_flow(C.LATENT_DIM, config["model"]).to(device)
    parameters = C.parameter_count(flow)

    steps_per_epoch = int(training["steps_per_epoch"])
    epochs = 1 if args.smoke else int(training["epochs"])
    total_steps = steps_per_epoch * epochs
    eval_every = int(training["eval_every_steps"])
    batch_size = int(training["batch_size"])
    if args.smoke:
        steps_per_epoch, total_steps, eval_every = 6, 6, 3

    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=float(training["min_lr"])
    )

    output = Path(args.output_dir) if args.output_dir else (
        C.output_root(registry) / "runs" / str(config["id"]) / representation
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "device": str(device),
                "parameters": parameters,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "eval_every_steps": eval_every,
                "active_terms": list(active),
                "test_data_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    steps_log = (output / "steps.jsonl").open("w", encoding="utf-8")
    evals_log = (output / "evals.jsonl").open("w", encoding="utf-8")

    selection = config["selection"]
    eval_batch = int(training["eval_batch_size"])
    raw_val = C.cached_vertices("val", registry)
    weights = config["loss"]
    families = tuple(config["loss"].get("comp_families", L.COMP_FAMILIES))
    need_sequence = ("seq_vtx" in active) or ("rollout" in families)
    need_trend = "trend" in active

    baseline = validate(
        flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics,
        selection, eval_batch, smoke_limit,
    )
    best = {"step": 0, "score": float(baseline["score"]), "feasible": bool(baseline["feasible"])}
    evals_log.write(json.dumps({"step": 0, "epoch": 0, **baseline}, sort_keys=True) + "\n")
    evals_log.flush()
    torch.save(
        {"step": 0, "flow_state_dict": flow.state_dict(), "config": config,
         "statistics": statistics, "validation": baseline, "parameters": parameters,
         "test_data_loaded": False},
        output / "best.pt",
    )

    started = time.time()
    step = 0
    flow.train()
    for epoch in range(1, epochs + 1):
        pairs = infinite_pairs(train_pairs, batch_size, seed, epoch)
        sequence_starts = O.balanced_sequence_starts(train_archive, seed, epoch)
        for local in range(steps_per_epoch):
            raw = next(pairs)
            optimizer.zero_grad(set_to_none=True)
            pair = L.pair_terms(
                flow, geometry, train_values, raw, statistics, O, families, need_trend
            )
            sequence = (
                L.sequence_terms(
                    flow, geometry, train_values, train_archive,
                    sequence_starts[step % len(sequence_starts)], statistics, O,
                    with_rollout_comp=("rollout" in families),
                )
                if need_sequence
                else None
            )
            comp_ramp = L.ramp(step + 1, int(training["comp_ramp_steps"]))
            anatomy_ramp = L.ramp(step + 1, int(training["anatomy_ramp_steps"]))
            loss, report = L.total_loss(pair, sequence, weights, active, comp_ramp, anatomy_ramp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(flow.parameters(), float(training["grad_clip"]))
            )
            optimizer.step()
            scheduler.step()
            step += 1
            steps_log.write(
                json.dumps(
                    {"step": step, "epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
                     "grad_norm": grad_norm, **report},
                    sort_keys=True,
                )
                + "\n"
            )
            if step % eval_every == 0 or step == total_steps:
                current = validate(
                    flow, geometry, val_values, val_archive, val_pairs, raw_val, statistics,
                    selection, eval_batch, smoke_limit,
                )
                improved = current["feasible"] and current["score"] < best["score"] - 1.0e-5
                evals_log.write(
                    json.dumps(
                        {"step": step, "epoch": epoch, "improved": improved,
                         "elapsed_minutes": (time.time() - started) / 60.0, **current},
                        sort_keys=True,
                    )
                    + "\n"
                )
                evals_log.flush()
                steps_log.flush()
                if improved:
                    best = {"step": step, "score": current["score"], "feasible": True}
                    torch.save(
                        {"step": step, "flow_state_dict": flow.state_dict(), "config": config,
                         "statistics": statistics, "validation": current,
                         "parameters": parameters, "test_data_loaded": False},
                        output / "best.pt",
                    )
                print(
                    f"[{config['id']}/{representation}] step {step:5d}/{total_steps} "
                    f"loss={report['total']:.5f} val={current['score']:.5f} "
                    f"feas={current['feasible']} best={best['score']:.5f}@{best['step']}",
                    flush=True,
                )
    steps_log.close()
    evals_log.close()

    best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "status": "complete",
        "id": str(config["id"]),
        "representation": representation,
        "arch": str(config["model"].get("arch")),
        "loss_set": loss_set,
        "active_terms": list(active),
        "comp_families": list(families),
        "parameters": parameters,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "best_step": int(best["step"]),
        "best_epoch_equivalent": best["step"] / max(steps_per_epoch, 1),
        "best_score": float(best["score"]),
        "best_validation": best_payload["validation"],
        "elapsed_minutes": (time.time() - started) / 60.0,
        "test_data_loaded": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("id", "representation", "best_step", "best_score", "elapsed_minutes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
