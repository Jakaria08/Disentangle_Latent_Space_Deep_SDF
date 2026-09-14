#!/usr/bin/env python3
"""Stage 5 step 2: train the converter cocycle C1 on p5_converter_pooled (PLAN Part 5B).

Starts from the P3 pooled direct_c4 checkpoint of the same representation and seed (trained on stable subjects
only) and continues on the converter view's train split:

* stable subjects keep a constant condition (CN 0, AD 1), so for them C1 is exactly direct_c4 (checked at start);
* every train converter (CN->AD, MCI->AD; CN->MCI in the partial-dose variant) gets an onset
  tau_i = a_i + (b_i - a_i) sigmoid(theta_i) on its observed window, penalized by lambda * theta_i^2;
* the objective is the direct_c4 recipe's 13 leaves with the per-leg average dose as condition
  (converter_line.pair_terms / sequence_terms); group-rate and disease-gap leaves use stable subjects only and the
  P3 checkpoint's train statistics, so population priors come from stable subjects only;
* heads are fine-tuned at a reduced learning rate (c1, c1_mci, c1_w050) or frozen (c1_frozen);
* selection on val: macro over {CN-stable, AD-stable, AD converters} of the first-to-last decoded-shape ratio to
  no-change + 0.2 x the all-pair ratio, val converters at their oracle-window midpoint (validation labels only),
  feasible when semigroup and inverse defects are <= 0.25.

Only train and val are opened. Settings come from configs/converter_line.json.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import converter_line as L
import dynamics_core as D

CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(CONFIG["variants"]))
    parser.add_argument("--representation", required=True, choices=CONFIG["representations"])
    parser.add_argument("--seed", type=int, required=True, choices=CONFIG["seeds"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def init_checkpoint(representation: str, seed: int) -> Path:
    path = D.RUNS_ROOT / CONFIG["base_view"] / representation / "direct_c4" / f"{representation}_direct_c4_s{seed}" / "checkpoints" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"P3 initialization checkpoint missing: {path}")
    return path


def prepared(archive, geometry, device, batch_size: int) -> dict[str, torch.Tensor]:
    C, O = (D.core()[key] for key in ("C", "O"))
    values = C.values_on_device(archive, device)
    O.attach_reference_geometry(values, geometry, batch_size)
    values["stable_group"] = torch.as_tensor(L.stable_groups(archive), device=device)
    return values


def group_rows(archive) -> dict[str, list]:
    C = D.core()["C"]
    groups = archive["subject_trajectory_groups"].astype(str)
    first_last = C.first_last_pairs(archive)
    return {"CN-stable": [r for r, g in zip(first_last, groups) if g == "CN-stable"],
            "AD-stable": [r for r, g in zip(first_last, groups) if g == "AD-stable"],
            "AD converters": [r for r, g in zip(first_last, groups) if g in L.AD_CONVERTERS]}


def shape_ratio(metrics: dict[str, float]) -> float:
    return 0.5 * (metrics["coordinate_mean"] / max(metrics["nochange_coordinate_mean"], 1e-8)
                  + metrics["euclidean_mean"] / max(metrics["nochange_euclidean_mean"], 1e-8))


def validate(model, geometry, values, archive, raw, statistics, tables, batch_size: int, limit: int | None) -> dict[str, Any]:
    O = D.core()["O"]
    model.set_table(tables["val"])
    try:
        groups = {name: rows[:limit] if limit else rows for name, rows in group_rows(archive).items()}
        metrics = {name: L.evaluate_rows(model.flow, geometry, values, rows, raw, batch_size, model.condition) for name, rows in groups.items() if rows}
        pairs = L.pair_rows(archive)
        pairs = O.balanced_subset(pairs, limit) if limit else pairs
        all_pairs = L.evaluate_rows(model.flow, geometry, values, pairs, raw, batch_size, model.condition)
        defects = L.cocycle_defects(model.flow, values, pairs, statistics, batch_size, model.condition)
    finally:
        model.set_table(tables["train"])
    ratios = {name: shape_ratio(value) for name, value in metrics.items()}
    selection = CONFIG["selection"]
    score = float(np.mean(list(ratios.values()))) + 0.2 * shape_ratio(all_pairs) + 0.001 * all_pairs["volume_relative_mean"] / max(all_pairs["nochange_volume_relative_mean"], 1e-8)
    feasible = (math.isfinite(score) and defects["relative_semigroup_defect_mean"] <= selection["max_relative_semigroup_defect"]
                and defects["relative_inverse_defect_mean"] <= selection["max_relative_inverse_defect"])
    return {"score": score, "feasible": bool(feasible), "group_shape_ratios": ratios, "all_pair_shape_ratio": shape_ratio(all_pairs),
            "groups": metrics, "all_pairs": all_pairs, "defects": defects}


def epoch_indices(rows, converter_mask: np.ndarray, seed: int, epoch: int, samples: int, fraction: float) -> list[int]:
    O = D.core()["O"]
    converter_count = int(round(samples * fraction)) if converter_mask.any() else 0
    indices = list(O.balanced_pair_sampler(rows, seed, epoch, samples - converter_count))
    generator = np.random.default_rng(seed * 1009 + epoch)
    indices += generator.choice(np.flatnonzero(converter_mask), size=converter_count, replace=True).tolist()
    generator.shuffle(indices)
    return indices


def main() -> int:
    args = parse_args()
    variant = CONFIG["variants"][args.variant]
    if args.representation not in variant["representations"]:
        raise ValueError(f"{args.variant} is defined for {variant['representations']} only")
    parts = D.core()
    C, O = parts["C"], parts["O"]
    device = D.device(args.device)
    view, fit = CONFIG["view"], CONFIG["fit"]
    registry = D.view_registry(view)
    train_archive = C.load_archive(args.representation, "train", registry)
    val_archive = C.load_archive(args.representation, "val", registry)
    D.assert_no_test_leakage(view, [train_archive, val_archive])

    start_checkpoint = init_checkpoint(args.representation, args.seed)
    flow, base_config, base_payload = D.load_trained_transport(start_checkpoint, device)
    if base_config["method"] != "direct_c4":
        raise ValueError("C1 starts from a direct_c4 checkpoint")
    C.set_seed(args.seed)
    statistics = base_payload["statistics"]
    training = base_config["training"]
    batch_size = int(training["batch_size"])
    geometry = C.build_geometry(args.representation, train_archive, device, registry)
    train_values = prepared(train_archive, geometry, device, int(training["decoder_batch_size"]))
    val_values = prepared(val_archive, geometry, device, int(training["decoder_batch_size"]))
    raw_val = D.view_vertices(val_archive)

    normalization = bc.read_json(D.view_root(view) / "view_manifest.json")["normalization"]
    age_range = float(normalization["age_max_years"]) - float(normalization["age_min_years"])
    width = float(variant["width_years"]) / age_range
    partial = bool(variant["mci_partial_dose"])
    tables = {"train": L.build_condition_table(train_archive, "learned", partial_dose=partial)}
    tables["val"] = L.build_condition_table(val_archive, "oracle", partial_dose=partial, theta_count=tables["train"].theta_count)
    model = L.ConverterCocycle(flow, tables["train"], width, partial_dose=partial).to(device)

    # Gate G5.4 (runtime half): on stable train visits C1 must be direct_c4 with the binary label.
    stable = torch.as_tensor(np.flatnonzero(L.stable_groups(train_archive) >= 0)[:256], device=device)
    with torch.no_grad():
        model.eval()
        s = train_values["age"][stable]
        reduced = model.transport_visits(train_values["z"][stable], s, s + 0.05, stable)
        direct = flow.transport(train_values["z"][stable], s, s + 0.05, train_values["label"][stable])
        reduction_gap = float((reduced - direct).abs().max())
    if reduction_gap > 1e-6:
        raise RuntimeError(f"C1 does not reduce to direct_c4 on stable subjects (max gap {reduction_gap:.3e})")

    rows = L.pair_rows(train_archive)
    groups = train_archive["subject_trajectory_groups"].astype(str)
    subject_of = {str(s): g for s, g in zip(train_archive["subject_ids"].astype(str), groups)}
    converter_groups = set(L.AD_CONVERTERS) | ({"CN->MCI"} if partial else set())
    converter_mask = np.asarray([subject_of[row.subject] in converter_groups for row in rows])
    offsets = train_archive["subject_visit_offsets"].astype(np.int64)
    converter_starts = [int(offsets[i]) for i, g in enumerate(groups) if g in converter_groups]

    heads = [p for p in model.flow.parameters()]
    for parameter in heads:
        parameter.requires_grad_(variant["heads"] != "frozen")
    groups_for_optimizer = [{"params": [model.theta] + ([model.kappa_logit] if partial else []), "lr": float(fit["onset_learning_rate"]), "weight_decay": 0.0}]
    if variant["heads"] != "frozen":
        groups_for_optimizer.append({"params": heads, "lr": float(training["learning_rate"]) * float(fit["head_learning_rate_multiplier"]),
                                     "weight_decay": float(training["weight_decay"])})
    optimizer = torch.optim.AdamW(groups_for_optimizer)
    epochs = 1 if (args.smoke or args.dry_run) else int(args.epochs if args.epochs is not None else fit["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, epochs) / max(epochs, 1))))
    ramped_epoch = max(int(training["consistency_ramp_epochs"]), int(training["anatomy_ramp_epochs"]))
    limit = int(training.get("smoke_pair_limit", 16)) if (args.smoke or args.dry_run) else None
    trainable = [p for group in groups_for_optimizer for p in group["params"]]

    run_name = args.run_name or f"{args.representation}_{args.variant}_s{args.seed}"
    run_dir = D.run_directory(view, args.representation, args.variant, run_name, args.runs_root)
    config = {"method": "converter_cocycle", "variant": args.variant, "variant_spec": variant, "representation": args.representation,
              "seed": args.seed, "width_normalized": width, "age_range_years": age_range, "fit": fit, "selection": CONFIG["selection"],
              "init_checkpoint": str(start_checkpoint), "init_checkpoint_sha256": bc.sha256_file(start_checkpoint), "base_config": base_config}
    print("=" * 96)
    print(f"C1 {args.variant} | {args.representation} s{args.seed} | device={device} | train converters={tables['train'].theta_count} | "
          f"width={variant['width_years']} y | heads={variant['heads']} | reduction gap={reduction_gap:.1e}", flush=True)

    def payload(epoch: int, current: dict[str, Any]) -> dict[str, Any]:
        return {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "statistics": statistics,
                "theta_subjects": tables["train"].theta_subjects, "view": view, "seed": args.seed, "validation": current,
                "test_data_loaded": False}

    model.eval()
    baseline = validate(model, geometry, val_values, val_archive, raw_val, statistics, tables, int(training["evaluation_batch_size"]), limit)
    if args.dry_run:
        model.train()
        raw = C.collate_pairs([rows[i] for i in epoch_indices(rows, converter_mask, args.seed, 1, batch_size, float(fit["converter_sampling_fraction"]))])
        pair = L.pair_terms(model.flow, geometry, train_values, raw, statistics, model.condition)
        sequence = L.sequence_terms(model.flow, geometry, train_values, train_archive, converter_starts[0], statistics, model.condition)
        loss = O.total_loss(pair, sequence, base_config, ramped_epoch)[0] + float(fit["onset_penalty_lambda"]) * model.onset_penalty()
        loss.backward()
        norm = math.sqrt(sum(float(p.grad.square().sum()) for p in trainable if p.grad is not None))
        print(f"DRY RUN PASSED: loss={float(loss):.5f} gradient={norm:.3e} val score={baseline['score']:.5f} feasible={baseline['feasible']}")
        return 0 if math.isfinite(norm) and norm > 0 else 1

    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    checkpoints = run_dir / "checkpoints"
    bc.atomic_json(run_dir / "resolved_config.json", D.provenance(view, args.representation, config, args.seed, {"trainer": "train_converter_cocycle.py"}))
    C.atomic_torch_save(checkpoints / "epoch_0000_init.pt", payload(0, baseline))
    C.atomic_torch_save(checkpoints / "best.pt", payload(0, baseline))
    best_score, best_epoch, best_any, stale = float(baseline["score"]), 0, float(baseline["score"]), 0
    started = time.time()
    samples = batch_size * 2 if args.smoke else int(fit["samples_per_epoch"])
    fraction = float(fit["converter_sampling_fraction"])
    for epoch in range(1, epochs + 1):
        model.train()
        indices = epoch_indices(rows, converter_mask, args.seed, epoch, samples, fraction)
        sequence_starts = O.balanced_sequence_starts(train_archive, args.seed, epoch)
        totals: dict[str, float] = {}
        batches = 0
        for step, begin in enumerate(range(0, len(indices), batch_size), start=1):
            raw = C.collate_pairs([rows[i] for i in indices[begin:begin + batch_size]])
            start = converter_starts[(step - 1) % len(converter_starts)] if (converter_starts and step % 4 == 0) else sequence_starts[(step - 1) % len(sequence_starts)]
            optimizer.zero_grad(set_to_none=True)
            pair = L.pair_terms(model.flow, geometry, train_values, raw, statistics, model.condition)
            sequence = L.sequence_terms(model.flow, geometry, train_values, train_archive, start, statistics, model.condition)
            objective, terms = O.total_loss(pair, sequence, base_config, ramped_epoch)
            penalty = model.onset_penalty()
            loss = objective + float(fit["onset_penalty_lambda"]) * penalty
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C1 loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
            for name, value in (terms | {"onset_penalty": float(penalty)}).items():
                totals[name] = totals.get(name, 0.0) + value
        scheduler.step()
        model.eval()
        current = validate(model, geometry, val_values, val_archive, raw_val, statistics, tables, int(training["evaluation_batch_size"]), limit)
        score, feasible = float(current["score"]), bool(current["feasible"])
        improved_any = score < best_any - float(training["early_stopping_min_delta"])
        best_any = min(best_any, score)
        stale = 0 if improved_any else stale + 1
        state = payload(epoch, current)
        C.atomic_torch_save(checkpoints / "latest.pt", state)
        if feasible and score < best_score - float(training["early_stopping_min_delta"]):
            best_score, best_epoch = score, epoch
            C.atomic_torch_save(checkpoints / "best.pt", state)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(D.dumps({"epoch": epoch, "elapsed_minutes": (time.time() - started) / 60, "batches": batches,
                                  **{f"train_{k}": v / max(batches, 1) for k, v in totals.items()}, "validation": current}).replace("\n", " ") + "\n")
        bc.atomic_json(run_dir / "training_status.json", {"status": "running", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch,
                                                          "best_validation_selection_score": best_score, "stale_epochs": stale, "test_data_loaded": False})
        print(f"epoch {epoch:03d}/{epochs} loss={totals['total'] / max(batches, 1):.5f} val={score:.5f} feasible={feasible} "
              f"groups={ {k: round(v, 4) for k, v in current['group_shape_ratios'].items()} } best={best_score:.5f}@{best_epoch}", flush=True)
        if not args.smoke and stale >= int(fit["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch}")
            break

    best = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    offsets_years = train_archive["visit_time_years_from_baseline"].astype(np.float64)
    ages = train_archive["visit_age_norm_train"].astype(np.float64)
    theta = model.theta.detach().cpu().numpy().astype(np.float64)
    onsets = []
    for subject, index in zip(train_archive["subject_ids"].astype(str), range(len(offsets) - 1)):
        first = int(offsets[index])
        row = int(tables["train"].theta_index[first])
        if row < 0:
            continue
        a, b = float(tables["train"].window_a[first]), float(tables["train"].window_b[first])
        tau = a + (b - a) / (1 + math.exp(-theta[row]))
        to_years = lambda age: float(offsets_years[first] + (age - ages[first]) * age_range)
        onsets.append({"subject_id": subject, "group": groups[index], "theta": theta[row], "window_a_years": to_years(a),
                       "window_b_years": to_years(b), "onset_years": to_years(tau), "onset_fraction_of_window": (tau - a) / max(b - a, 1e-12)})
    bc.atomic_csv(run_dir / "learned_onsets.csv", pd.DataFrame(onsets))
    bc.atomic_json(run_dir / "training_status.json", {
        "status": "complete", "epoch": epoch, "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_selection_score": best_score,
        "selected_checkpoint": str(checkpoints / "best.pt"), "theta_sha256": bc.sha256_bytes(theta.tobytes()),
        "kappa_mci": float(torch.sigmoid(model.kappa_logit).detach()) if partial else None, "reduction_gap": reduction_gap,
        "elapsed_minutes": (time.time() - started) / 60, "test_data_loaded": False})
    print(f"COMPLETE: {checkpoints / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
