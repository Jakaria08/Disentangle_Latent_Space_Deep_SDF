#!/usr/bin/env python3
"""Train one matched latent-only loss-reduction job."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import objective as R  # noqa: E402
from _bootstrap import core  # noqa: E402
from configuration import validate_job  # noqa: E402
from model import DirectLatentCocycle  # noqa: E402

C, O = core()


class PairDataset(Dataset):
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def learning_rate(step: int, total_steps: int, training: dict[str, Any]) -> float:
    initial = float(training["initial_learning_rate"])
    peak = float(training["peak_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    warmup = int(training["warmup_steps"])
    if warmup > 0 and step <= warmup:
        return initial + (peak - initial) * float(step) / float(warmup)
    progress = float(step - warmup) / float(max(total_steps - warmup, 1))
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    flow: DirectLatentCocycle,
    geometry: Any,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    rows: list[Any],
    raw_vertices: np.ndarray,
    statistics: dict[str, Any],
    selection: dict[str, Any],
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    flow.eval()
    selected = O.balanced_subset(rows, limit)
    first_last = O.balanced_first_last_subset(C.first_last_pairs(archive), limit)
    all_metrics = O.evaluate_pairs(
        flow, geometry, values, selected, raw_vertices, batch_size
    )
    first_last_metrics = O.evaluate_pairs(
        flow, geometry, values, first_last, raw_vertices, batch_size
    )
    defects = O.cocycle_defects(flow, values, selected, statistics, batch_size)
    score, feasible, ratios = O.validation_score(
        all_metrics, first_last_metrics, defects, selection
    )
    flow.train()
    return {
        "score": float(score),
        "feasible": bool(feasible),
        "ratios": ratios,
        "defects": defects,
        "all_pairs": all_metrics,
        "first_last": first_last_metrics,
    }


def fixed_train_diagnostic(
    flow: DirectLatentCocycle,
    geometry: Any,
    values: dict[str, torch.Tensor],
    rows: list[Any],
    raw_vertices: np.ndarray,
    batch_size: int,
    limit: int,
) -> dict[str, Any]:
    selected = O.balanced_subset(rows, min(int(limit), len(rows)))
    flow.eval()
    metrics = O.evaluate_pairs(flow, geometry, values, selected, raw_vertices, batch_size)
    flow.train()
    return metrics


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def truncate_log(path: Path, last_step: int) -> None:
    """Drop partial-epoch records that are newer than the resumable checkpoint."""
    if not path.is_file():
        return
    kept = [
        line
        for line in path.read_text().splitlines()
        if int(json.loads(line)["step"]) <= int(last_step)
    ]
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")


def checkpoint(
    path: Path,
    flow: DirectLatentCocycle,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    step: int,
    validation: dict[str, Any],
    best_any: dict[str, Any],
    best_mature: dict[str, Any] | None,
    statistics: dict[str, Any],
) -> None:
    C.atomic_torch_save(
        path,
        {
            "epoch": int(epoch),
            "step": int(step),
            "flow_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "validation": validation,
            "best_any": best_any,
            "best_mature": best_mature,
            "statistics": statistics,
            "rng_state": rng_state(),
            "test_data_loaded": False,
        },
    )


def better(validation: dict[str, Any], record: dict[str, Any] | None, min_delta: float) -> bool:
    return bool(validation["feasible"]) and (
        record is None or float(validation["score"]) < float(record["score"]) - min_delta
    )


def main() -> int:
    args = parse_args()
    config = json.loads(C.resolve_path(args.config).read_text())
    validate_job(config)
    training = dict(config["training"])
    representation = str(config["representation"])
    loss_set = str(config["loss_set"])
    seed = int(training["seed"])
    device = C.choose_device(args.device)
    C.set_seed(seed)

    registry = C.load_registry()
    train_archive = C.load_archive(representation, "train", registry)
    val_archive = C.load_archive(representation, "val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(
        val_archive["subject_ids"].astype(str)
    ):
        raise ValueError("Train/validation subject leakage")
    geometry = C.build_geometry(representation, train_archive, device, registry)
    train_values = C.values_on_device(train_archive, device)
    val_values = C.values_on_device(val_archive, device)
    decoder_batch = int(training["decoder_batch_size"])
    O.attach_reference_geometry(train_values, geometry, decoder_batch)
    O.attach_reference_geometry(val_values, geometry, decoder_batch)
    train_rows = C.load_pairs("train", train_archive, registry)
    val_rows = C.load_pairs("val", val_archive, registry)
    smoke_limit = int(training["smoke_pair_limit"]) if args.smoke else None
    statistics = O.training_statistics(train_values, train_rows, train_archive, smoke_limit)
    raw_train = C.cached_vertices("train", registry)
    raw_val = C.cached_vertices("val", registry)

    model_config = config["model"]
    flow = DirectLatentCocycle(
        int(model_config["latent_dim"]),
        int(model_config["width"]),
        int(model_config["residual_blocks"]),
        float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(training["initial_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    epochs = int(training["epochs"])
    steps_per_epoch = int(training["steps_per_epoch"])
    total_steps = epochs * steps_per_epoch
    selection_start = int(training["selection_start_step"])
    if args.smoke:
        epochs = 1
        steps_per_epoch = int(training["smoke_steps"])
        total_steps = steps_per_epoch
        selection_start = max(1, total_steps // 2)

    output = Path(args.output_dir) if args.output_dir else (
        C.output_root(registry)
        / "runs"
        / representation
        / loss_set
        / f"seed_{seed}"
    )
    latest_path = output / "latest.pt"
    if (output / "resolved_config.json").exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run {output}")
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {latest_path}")
    output.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    step = 0
    stale_evaluations = 0
    best_any: dict[str, Any] | None = None
    best_mature: dict[str, Any] | None = None
    mode = "a" if args.resume else "w"
    if args.resume:
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        flow.load_state_dict(payload["flow_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        step = int(payload["step"])
        best_any = payload["best_any"]
        best_mature = payload["best_mature"]
        restore_rng(payload["rng_state"])
        truncate_log(output / "steps.jsonl", step)
        truncate_log(output / "evaluations.jsonl", step)

    if not args.resume:
        C.atomic_json(
            output / "resolved_config.json",
            {
                "config": config,
                "device": str(device),
                "parameters": C.parameter_count(flow),
                "total_steps": total_steps,
                "selection_start_step": selection_start,
                "all_raw_terms_computed_for_matching": True,
                "test_data_loaded": False,
            },
        )
        C.atomic_json(output / "training_statistics.json", statistics)
        initial = evaluate(
            flow,
            geometry,
            val_values,
            val_archive,
            val_rows,
            raw_val,
            statistics,
            config["selection"],
            int(training["evaluation_batch_size"]),
            smoke_limit,
        )
        train_fixed = fixed_train_diagnostic(
            flow,
            geometry,
            train_values,
            train_rows,
            raw_train,
            int(training["evaluation_batch_size"]),
            smoke_limit or int(training["fixed_train_evaluation_pairs"]),
        )
        best_any = {"step": 0, "epoch": 0, "score": initial["score"]}
        with (output / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"step": 0, "epoch": 0, "eligible": False, "train_fixed": train_fixed, **initial},
                    sort_keys=True,
                )
                + "\n"
            )
        checkpoint(
            output / "best_any.pt",
            flow,
            optimizer,
            config,
            0,
            0,
            initial,
            best_any,
            None,
            statistics,
        )

    assert best_any is not None
    started = time.time()
    stopped_early = False
    for epoch in range(start_epoch, epochs + 1):
        loader = DataLoader(
            PairDataset(train_rows),
            batch_size=int(training["batch_size"]),
            sampler=O.balanced_pair_sampler(
                train_rows, seed, epoch, int(training["batch_size"]) * steps_per_epoch
            ),
            collate_fn=C.collate_pairs,
            num_workers=0,
        )
        sequence_starts = O.balanced_sequence_starts(train_archive, seed, epoch)
        flow.train()
        for local_step, raw in enumerate(loader, start=1):
            step += 1
            lr = learning_rate(step, total_steps, training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            pair = O.pair_terms(flow, geometry, train_values, raw, statistics)
            sequence = O.sequence_terms(
                flow,
                geometry,
                train_values,
                train_archive,
                sequence_starts[(local_step - 1) % len(sequence_starts)],
                statistics,
            )
            consistency = R.ramp(step, int(training["consistency_ramp_steps"]))
            anatomy = R.ramp(step, int(training["anatomy_ramp_steps"]))
            loss, report = R.total_loss(
                pair,
                sequence,
                config["loss_weights"],
                loss_set,
                consistency,
                anatomy,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    flow.parameters(), float(training["gradient_clip_norm"])
                )
            )
            optimizer.step()
            with (output / "steps.jsonl").open(mode, encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "local_step": local_step,
                            "learning_rate": lr,
                            "gradient_norm": gradient_norm,
                            **report,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            mode = "a"

        current = evaluate(
            flow,
            geometry,
            val_values,
            val_archive,
            val_rows,
            raw_val,
            statistics,
            config["selection"],
            int(training["evaluation_batch_size"]),
            smoke_limit,
        )
        train_fixed = fixed_train_diagnostic(
            flow,
            geometry,
            train_values,
            train_rows,
            raw_train,
            int(training["evaluation_batch_size"]),
            smoke_limit or int(training["fixed_train_evaluation_pairs"]),
        )
        eligible = step >= selection_start
        min_delta = float(training["early_stopping_min_delta"])
        improved_any = better(current, best_any, 0.0)
        improved_mature = eligible and better(current, best_mature, min_delta)
        if improved_any:
            best_any = {"step": step, "epoch": epoch, "score": current["score"]}
            checkpoint(
                output / "best_any.pt",
                flow,
                optimizer,
                config,
                epoch,
                step,
                current,
                best_any,
                best_mature,
                statistics,
            )
        if improved_mature:
            best_mature = {"step": step, "epoch": epoch, "score": current["score"]}
            stale_evaluations = 0
            checkpoint(
                output / "best.pt",
                flow,
                optimizer,
                config,
                epoch,
                step,
                current,
                best_any,
                best_mature,
                statistics,
            )
        elif eligible:
            stale_evaluations += 1

        with (output / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "epoch": epoch,
                        "eligible": eligible,
                        "improved_any": improved_any,
                        "improved_mature": improved_mature,
                        "elapsed_minutes": (time.time() - started) / 60.0,
                        "train_fixed": train_fixed,
                        **current,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        checkpoint(
            latest_path,
            flow,
            optimizer,
            config,
            epoch,
            step,
            current,
            best_any,
            best_mature,
            statistics,
        )
        print(
            f"[{representation}/{loss_set}/s{seed}] epoch {epoch:03d}/{epochs} "
            f"step {step:04d}/{total_steps} val={current['score']:.6f} "
            f"best_any={best_any['score']:.6f}@e{best_any['epoch']} "
            f"best_mature={(best_mature or {}).get('score', float('nan')):.6f}",
            flush=True,
        )
        if (
            bool(training["early_stopping_enabled"])
            and step >= int(training["early_stopping_min_step"])
            and stale_evaluations >= int(training["early_stopping_patience_evaluations"])
        ):
            stopped_early = True
            break

    if best_mature is None:
        raise RuntimeError("No eligible feasible checkpoint was produced")
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    best_any_payload = torch.load(
        output / "best_any.pt", map_location="cpu", weights_only=False
    )
    summary = {
        "status": "complete",
        "representation": representation,
        "loss_set": loss_set,
        "loss_count": int(config["loss_count"]),
        "seed": seed,
        "epochs_completed": epoch,
        "steps_completed": step,
        "stopped_early": stopped_early,
        "selection_start_step": selection_start,
        "best_any": best_any,
        "best_mature": best_mature,
        "best_any_validation": best_any_payload["validation"],
        "best_mature_validation": selected["validation"],
        "early_checkpoint_better": float(best_any["score"]) + min_delta
        < float(best_mature["score"]),
        "parameters": C.parameter_count(flow),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "test_data_loaded": False,
    }
    C.atomic_json(output / "summary.json", summary)
    C.atomic_json(
        output / "training_status.json",
        {
            "status": "complete",
            "best_checkpoint": str(output / "best.pt"),
            "best_any_checkpoint": str(output / "best_any.pt"),
            "latest_checkpoint": str(latest_path),
            **{key: summary[key] for key in (
                "representation", "loss_set", "seed", "epochs_completed",
                "steps_completed", "best_any", "best_mature", "early_checkpoint_better",
                "test_data_loaded"
            )},
        },
    )
    print(json.dumps({"output": str(output), **best_mature}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
